"""
Dataset classes for Gaia training.

Supports:
- Causal Language Modeling (pre-training)
- Masked Language Modeling (pre-training)
- Supervised fine-tuning with labels
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MicrobiomeDataset(Dataset):
    """
    Dataset for microbiome token sequences.

    Loads pre-tokenized corpus and prepares batches for
    causal or masked language modeling.
    """

    def __init__(
        self,
        corpus_path: str,
        mask_ratio: float = 0.0,
        mask_token_id: int = 2,
        pad_token_id: int = 0,
    ):
        """
        Args:
            corpus_path: Path to gaia-corpus-v1.pkl
            mask_ratio: Fraction of tokens to mask (0.0 for causal LM)
            mask_token_id: Token ID for [MASK]
            pad_token_id: Token ID for [PAD]
        """
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        self.token_sequences = corpus["token_sequences"]  # (N, seq_len)
        self.sample_ids = corpus.get("sample_ids", list(range(len(self.token_sequences))))
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = torch.tensor(self.token_sequences[idx], dtype=torch.long)

        if self.mask_ratio > 0:
            return self._get_masked_item(tokens)
        else:
            return self._get_causal_item(tokens)

    def _get_causal_item(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare item for causal LM (next token prediction)."""
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
        }

    def _get_masked_item(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare item for masked LM (token reconstruction)."""
        labels = tokens.clone()
        input_ids = tokens.clone()

        # Create mask for non-special tokens
        maskable = (tokens != self.pad_token_id) & (tokens > 4)  # Skip special tokens
        maskable_indices = maskable.nonzero(as_tuple=True)[0]

        n_mask = max(1, int(len(maskable_indices) * self.mask_ratio))
        mask_indices = maskable_indices[
            torch.randperm(len(maskable_indices))[:n_mask]
        ]

        input_ids[mask_indices] = self.mask_token_id

        # Only compute loss on masked positions
        labels[~torch.zeros_like(labels, dtype=torch.bool).scatter_(0, mask_indices, True)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class SupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning tasks.

    Pairs microbiome token sequences with continuous or
    categorical labels.
    """

    def __init__(
        self,
        corpus_path: str,
        labels_path: str,
        label_column: str,
        task_type: str = "regression",
    ):
        """
        Args:
            corpus_path: Path to corpus pickle
            labels_path: Path to CSV with labels
            label_column: Column name for the target variable
            task_type: "regression" or "classification"
        """
        import pandas as pd

        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        labels_df = pd.read_csv(labels_path)

        self.token_sequences = corpus["token_sequences"]
        sample_ids = corpus.get("sample_ids", [])

        # Match samples between corpus and labels
        labels_df = labels_df.set_index("sample_id")
        self.labels = []
        self.valid_indices = []

        for i, sid in enumerate(sample_ids):
            if sid in labels_df.index:
                value = labels_df.loc[sid, label_column]
                if pd.notna(value):
                    self.valid_indices.append(i)
                    self.labels.append(value)

        if task_type == "classification":
            # Encode categorical labels
            unique_labels = sorted(set(self.labels))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.labels = [self.label_map[l] for l in self.labels]
            self.n_classes = len(unique_labels)

        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        corpus_idx = self.valid_indices[idx]
        tokens = torch.tensor(
            self.token_sequences[corpus_idx], dtype=torch.long
        )

        if self.task_type == "regression":
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": tokens,
            "label": label,
        }
