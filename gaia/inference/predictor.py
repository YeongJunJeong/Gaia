"""
Inference Module for Gaia.

Provides a high-level API for:
- Soil health diagnosis (chemical property prediction)
- Biome classification
- Sample embedding extraction
- Synthetic profile generation
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gaia.models.transformer import GaiaConfig, GaiaTransformer
from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Result of soil health diagnosis."""

    sample_id: str
    biome: str
    biome_confidence: float
    predicted_ph: float | None = None
    predicted_organic_c: float | None = None
    predicted_total_n: float | None = None
    keystone_genera: list[tuple[str, float]] | None = None
    embedding: np.ndarray | None = None

    @property
    def soil_health_report(self) -> str:
        lines = [f"Soil Health Diagnosis — {self.sample_id}"]
        lines.append(f"  Biome: {self.biome} (confidence: {self.biome_confidence:.2%})")
        if self.predicted_ph is not None:
            lines.append(f"  Predicted pH: {self.predicted_ph:.2f}")
        if self.predicted_organic_c is not None:
            lines.append(f"  Predicted Organic C: {self.predicted_organic_c:.2f}%")
        if self.predicted_total_n is not None:
            lines.append(f"  Predicted Total N: {self.predicted_total_n:.4f}%")
        if self.keystone_genera:
            lines.append("  Top Keystone Genera:")
            for genus, weight in self.keystone_genera[:5]:
                lines.append(f"    - {genus} (attention: {weight:.3f})")
        return "\n".join(lines)


class GaiaPredictor:
    """
    High-level predictor for soil microbiome analysis.

    Usage:
        predictor = GaiaPredictor.from_pretrained("path/to/checkpoint")
        result = predictor.diagnose("path/to/abundance.csv")
    """

    def __init__(
        self,
        model: GaiaTransformer,
        tokenizer: MicrobiomeTokenizer,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        # Fine-tuned heads (loaded separately if available)
        self.classification_head = None
        self.regression_heads = {}

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str) -> "GaiaPredictor":
        """Load predictor from a checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load model
        model_path = checkpoint_dir / "best.pt"
        if not model_path.exists():
            model_path = checkpoint_dir / "gaia-v0.1.pt"

        checkpoint = torch.load(model_path, map_location="cpu")
        config = GaiaConfig.from_dict(checkpoint["config"])
        model = GaiaTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load tokenizer
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            # Try parent directory
            tokenizer_path = checkpoint_dir.parent / "tokenizer.json"

        tokenizer = MicrobiomeTokenizer.load(str(tokenizer_path))

        return cls(model, tokenizer)

    @torch.no_grad()
    def get_embedding(self, abundance_df: pd.DataFrame) -> np.ndarray:
        """
        Get sample embeddings from abundance profiles.

        Args:
            abundance_df: DataFrame with sample_id + genus columns

        Returns:
            (n_samples, d_model) embedding array
        """
        tokens = self.tokenizer.encode_batch(abundance_df)
        input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
        embeddings = self.model.get_sample_embedding(input_ids)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def diagnose(
        self,
        abundance_path: str | pd.DataFrame,
    ) -> list[DiagnosisResult]:
        """
        Run soil health diagnosis on abundance profiles.

        Args:
            abundance_path: Path to CSV or DataFrame with abundance data

        Returns:
            List of DiagnosisResult objects
        """
        if isinstance(abundance_path, str):
            df = pd.read_csv(abundance_path)
        else:
            df = abundance_path

        id_cols = ["sample_id"]
        genus_cols = [c for c in df.columns if c not in id_cols]

        results = []
        for _, row in df.iterrows():
            sample_id = row.get("sample_id", "unknown")

            # Tokenize
            tokens = self.tokenizer.encode(row, genus_cols)
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(
                self.device
            )

            # Get embedding
            embedding = self.model.get_sample_embedding(input_ids)

            result = DiagnosisResult(
                sample_id=str(sample_id),
                biome="unknown",
                biome_confidence=0.0,
                embedding=embedding.cpu().numpy().squeeze(),
            )
            results.append(result)

        return results

    @torch.no_grad()
    def generate(
        self,
        prompt_genera: list[str],
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> list[str]:
        """
        Generate a synthetic microbial abundance profile.

        Given a list of "prompt" genera, generates additional genera
        that would likely co-occur in the same soil sample.

        Args:
            prompt_genera: Starting genera names
            n_tokens: Number of genera to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            List of generated genus names
        """
        # Encode prompt
        prompt_ids = [self.tokenizer.vocab.get("[CLS]", 3)]
        for genus in prompt_genera:
            token_id = self.tokenizer.vocab.get(genus)
            if token_id is not None:
                prompt_ids.append(token_id)

        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        generated = []
        for _ in range(n_tokens):
            output = self.model(input_ids)
            next_token_logits = output["logits"][:, -1, :] / temperature

            # Top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(1, next_token_idx)

            token_id = next_token.item()
            genus = self.tokenizer.id_to_genus.get(token_id, "[UNK]")

            if genus in ("[PAD]", "[SEP]"):
                break
            if genus not in ("[UNK]", "[CLS]", "[MASK]"):
                generated.append(genus)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Truncate if exceeding max length
            if input_ids.shape[1] >= self.tokenizer.max_length:
                break

        return generated
