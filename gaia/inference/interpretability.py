"""
Interpretability Tools for Gaia.

Identifies keystone genera using attention weight analysis
and provides ecological interpretation of model predictions.
"""

import logging

import numpy as np
import pandas as pd
import torch

from gaia.models.transformer import GaiaTransformer
from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

logger = logging.getLogger(__name__)

# Known ecological roles for common soil genera
GENUS_ROLES: dict[str, str] = {
    "Bradyrhizobium": "Nitrogen fixation",
    "Rhizobium": "Nitrogen fixation (legume symbiont)",
    "Azotobacter": "Free-living nitrogen fixation",
    "Nitrospira": "Nitrite oxidation",
    "Nitrosomonas": "Ammonia oxidation",
    "Glomus": "Arbuscular mycorrhizal network",
    "Rhizophagus": "Arbuscular mycorrhizal fungi",
    "Trichoderma": "Pathogen suppression, biocontrol",
    "Bacillus": "Plant growth promotion, biocontrol",
    "Pseudomonas": "Plant growth promotion, siderophore production",
    "Streptomyces": "Antibiotic production, organic matter decomposition",
    "Mycobacterium": "Organic matter decomposition",
    "Acidobacterium": "Acidic soil indicator",
    "Sphingomonas": "Aromatic compound degradation",
    "Burkholderia": "Heavy metal tolerance, plant association",
    "Methanobacterium": "Methanogenesis (anaerobic indicator)",
    "Geobacter": "Iron reduction, electron transfer",
    "Candidatus Udaeobacter": "Abundant soil bacterium, minimal genome",
}


def identify_keystone_genera(
    model: GaiaTransformer,
    tokenizer: MicrobiomeTokenizer,
    input_ids: torch.Tensor,
    top_k: int = 10,
) -> list[dict]:
    """
    Identify keystone genera from attention weights.

    For each sample, aggregates attention weights across all layers
    and heads to determine which genera the model focuses on most.

    Args:
        model: Gaia model
        tokenizer: Tokenizer for ID-to-genus mapping
        input_ids: (batch_size, seq_len) token IDs
        top_k: Number of top genera to return

    Returns:
        List of dicts with genus name, attention weight, and ecological role
    """
    attention_weights = model.get_attention_weights(input_ids)

    if not attention_weights:
        logger.warning("No attention weights captured")
        return []

    # Average attention across all layers and heads
    # Each weight: (batch, heads, seq, seq)
    all_attention = torch.stack(attention_weights)  # (layers, batch, heads, seq, seq)
    avg_attention = all_attention.mean(dim=(0, 2))  # (batch, seq, seq)

    # Sum attention received by each token position (column-wise)
    token_importance = avg_attention.sum(dim=1)  # (batch, seq)

    results = []
    for batch_idx in range(input_ids.shape[0]):
        sample_importance = token_importance[batch_idx].cpu().numpy()
        sample_tokens = input_ids[batch_idx].cpu().numpy()

        # Map token IDs to genera with importance scores
        genera_scores = []
        for pos, (token_id, importance) in enumerate(
            zip(sample_tokens, sample_importance)
        ):
            genus = tokenizer.id_to_genus.get(int(token_id), "")
            if genus and genus not in ("[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"):
                genera_scores.append(
                    {
                        "genus": genus,
                        "attention": float(importance),
                        "position": pos,
                        "role": GENUS_ROLES.get(genus, "Unknown"),
                    }
                )

        # Sort by attention and take top_k
        genera_scores.sort(key=lambda x: x["attention"], reverse=True)
        results.append(genera_scores[:top_k])

    return results


def format_keystone_report(
    keystone_genera: list[dict],
    sample_id: str = "sample",
) -> str:
    """Format keystone genera analysis as a readable report."""
    lines = [f"Keystone Genera Analysis — {sample_id}", "-" * 50]

    for i, genus_info in enumerate(keystone_genera, 1):
        lines.append(
            f"  {i}. {genus_info['genus']} "
            f"(attention: {genus_info['attention']:.4f})"
        )
        if genus_info["role"] != "Unknown":
            lines.append(f"     Role: {genus_info['role']}")

    return "\n".join(lines)
