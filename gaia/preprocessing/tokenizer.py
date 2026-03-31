"""
Step [6]: Corpus Conversion — MGM-compatible Tokenization

Converts abundance profiles into tokenized sequences:
1. Sort genera by abundance (descending)
2. Map genus names to vocabulary indices
3. Pad/truncate to fixed sequence length (512)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN]


class MicrobiomeTokenizer:
    """
    Tokenizer for soil microbiome abundance profiles.

    Converts genus-level abundance profiles into token sequences
    compatible with the MGM transformer architecture.
    """

    def __init__(
        self,
        max_length: int = 512,
        vocab: dict[str, int] | None = None,
    ):
        self.max_length = max_length
        self.vocab = vocab or {}
        self.id_to_genus: dict[int, str] = {}

        if vocab:
            self.id_to_genus = {v: k for k, v in vocab.items()}

    def build_vocab(
        self,
        abundance_df: pd.DataFrame,
        id_cols: list[str] | None = None,
        min_prevalence: int = 1,
    ) -> dict[str, int]:
        """
        Build vocabulary from abundance data.

        Assigns token IDs to genera ordered by total abundance
        across all samples (most abundant first).
        """
        if id_cols is None:
            id_cols = ["sample_id"]

        genus_cols = [c for c in abundance_df.columns if c not in id_cols]
        genus_data = abundance_df[genus_cols]

        # Calculate total abundance across all samples
        total_abundance = genus_data.sum(axis=0)

        # Filter by minimum prevalence
        prevalence = (genus_data > 0).sum(axis=0)
        valid_genera = prevalence[prevalence >= min_prevalence].index

        # Sort by total abundance (descending)
        sorted_genera = (
            total_abundance[valid_genera].sort_values(ascending=False).index.tolist()
        )

        # Build vocabulary: special tokens first, then genera
        self.vocab = {}
        for i, token in enumerate(SPECIAL_TOKENS):
            self.vocab[token] = i

        for genus in sorted_genera:
            self.vocab[genus] = len(self.vocab)

        self.id_to_genus = {v: k for k, v in self.vocab.items()}

        logger.info(
            f"Built vocabulary: {len(self.vocab)} tokens "
            f"({len(SPECIAL_TOKENS)} special + {len(sorted_genera)} genera)"
        )
        return self.vocab

    def encode(self, sample: pd.Series, genus_cols: list[str]) -> np.ndarray:
        """
        Encode a single sample's abundance profile into token sequence.

        Args:
            sample: Series with genus abundances
            genus_cols: List of genus column names

        Returns:
            Array of token IDs, length = max_length
        """
        # Get non-zero genera sorted by abundance (descending)
        abundances = sample[genus_cols]
        nonzero = abundances[abundances > 0].sort_values(ascending=False)

        tokens = [self.vocab.get(CLS_TOKEN, 0)]

        for genus in nonzero.index:
            if len(tokens) >= self.max_length - 1:  # Reserve space for SEP
                break
            token_id = self.vocab.get(genus, self.vocab.get(UNK_TOKEN, 1))
            tokens.append(token_id)

        tokens.append(self.vocab.get(SEP_TOKEN, 0))

        # Pad to max_length
        pad_id = self.vocab.get(PAD_TOKEN, 0)
        while len(tokens) < self.max_length:
            tokens.append(pad_id)

        return np.array(tokens[: self.max_length], dtype=np.int64)

    def encode_batch(
        self,
        abundance_df: pd.DataFrame,
        id_cols: list[str] | None = None,
    ) -> np.ndarray:
        """
        Encode all samples in a DataFrame.

        Returns:
            Array of shape (n_samples, max_length) with token IDs
        """
        if id_cols is None:
            id_cols = ["sample_id"]

        genus_cols = [c for c in abundance_df.columns if c not in id_cols]
        encoded = []

        for _, row in abundance_df.iterrows():
            tokens = self.encode(row, genus_cols)
            encoded.append(tokens)

        result = np.stack(encoded)
        logger.info(f"Encoded {result.shape[0]} samples to sequences of length {result.shape[1]}")
        return result

    def decode(self, token_ids: np.ndarray) -> list[str]:
        """Convert token IDs back to genus names."""
        return [
            self.id_to_genus.get(int(tid), UNK_TOKEN)
            for tid in token_ids
            if int(tid) not in {
                self.vocab.get(PAD_TOKEN, 0),
                self.vocab.get(CLS_TOKEN, 3),
                self.vocab.get(SEP_TOKEN, 4),
            }
        ]

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, "w") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "max_length": self.max_length,
                    "special_tokens": SPECIAL_TOKENS,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved tokenizer to {path}")

    @classmethod
    def load(cls, path: str) -> "MicrobiomeTokenizer":
        """Load vocabulary from JSON file."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(
            max_length=data["max_length"],
            vocab=data["vocab"],
        )
        logger.info(
            f"Loaded tokenizer from {path}: {len(tokenizer.vocab)} tokens"
        )
        return tokenizer
