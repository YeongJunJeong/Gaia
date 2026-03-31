"""
Synthetic Microbiome Profile Generation.

Uses the pre-trained Gaia model to generate realistic
soil microbial abundance profiles for specified conditions.

Applications:
  - Data augmentation
  - Scenario simulation
  - "Microbiome Turing Test"
"""

import logging

import numpy as np
import pandas as pd
import torch

from gaia.models.transformer import GaiaTransformer
from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

logger = logging.getLogger(__name__)

# Seed genera for different soil conditions
CONDITION_SEEDS: dict[str, list[str]] = {
    "healthy_wheat": [
        "Bradyrhizobium",
        "Bacillus",
        "Pseudomonas",
        "Streptomyces",
        "Rhizophagus",
    ],
    "healthy_rice": [
        "Azospirillum",
        "Anabaena",
        "Burkholderia",
        "Methylobacterium",
        "Sphingomonas",
    ],
    "forest_temperate": [
        "Acidobacterium",
        "Mycobacterium",
        "Russula",
        "Cortinarius",
        "Piloderma",
    ],
    "drought_stressed": [
        "Actinobacteria",
        "Streptomyces",
        "Rubrobacter",
        "Geodermatophilus",
    ],
    "nitrogen_rich": [
        "Nitrosomonas",
        "Nitrospira",
        "Bradyrhizobium",
        "Azotobacter",
        "Rhizobium",
    ],
}


def generate_profile(
    model: GaiaTransformer,
    tokenizer: MicrobiomeTokenizer,
    condition: str | list[str],
    n_genera: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    n_profiles: int = 1,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic microbial abundance profiles.

    Args:
        model: Pre-trained Gaia model
        tokenizer: Tokenizer
        condition: Condition name (from CONDITION_SEEDS) or list of seed genera
        n_genera: Number of genera per profile
        temperature: Sampling temperature (lower = more conservative)
        top_k: Top-k sampling
        n_profiles: Number of profiles to generate
        device: Torch device

    Returns:
        DataFrame with generated abundance profiles
    """
    if device is None:
        device = next(model.parameters()).device

    # Get seed genera
    if isinstance(condition, str):
        seed_genera = CONDITION_SEEDS.get(condition, [])
        if not seed_genera:
            logger.warning(f"Unknown condition '{condition}', using empty seed")
    else:
        seed_genera = condition

    model.eval()
    profiles = []

    for profile_idx in range(n_profiles):
        # Encode seed genera
        input_ids = [tokenizer.vocab.get("[CLS]", 3)]
        for genus in seed_genera:
            token_id = tokenizer.vocab.get(genus)
            if token_id is not None:
                input_ids.append(token_id)

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        generated_genera = list(seed_genera)

        with torch.no_grad():
            for _ in range(n_genera - len(seed_genera)):
                output = model(input_tensor)
                next_logits = output["logits"][:, -1, :] / temperature

                # Top-k filtering
                topk_logits, topk_indices = torch.topk(next_logits, top_k)
                probs = torch.softmax(topk_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
                next_token = topk_indices.gather(1, next_idx)

                token_id = next_token.item()
                genus = tokenizer.id_to_genus.get(token_id, "")

                if genus in ("[PAD]", "[SEP]"):
                    break
                if genus and genus not in ("[UNK]", "[CLS]", "[MASK]"):
                    generated_genera.append(genus)

                input_tensor = torch.cat([input_tensor, next_token], dim=1)
                if input_tensor.shape[1] >= tokenizer.max_length:
                    break

        # Convert to abundance profile (rank-based synthetic abundances)
        n = len(generated_genera)
        abundances = np.exp(-np.arange(n) * 0.05)  # Log-normal-like decay
        abundances += np.random.exponential(0.01, n)  # Add noise
        abundances /= abundances.sum()

        profile = {"sample_id": f"synthetic_{profile_idx}"}
        for genus, abundance in zip(generated_genera, abundances):
            profile[genus] = profile.get(genus, 0) + abundance
        profiles.append(profile)

    result = pd.DataFrame(profiles).fillna(0)
    logger.info(
        f"Generated {len(profiles)} synthetic profiles "
        f"with {result.shape[1] - 1} unique genera"
    )
    return result


def validate_synthetic_profiles(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> dict[str, float]:
    """
    Compare synthetic profiles against real data.

    Metrics:
    - Distribution similarity (Jensen-Shannon divergence)
    - Diversity index comparison (Shannon, Simpson)
    - Genera overlap ratio
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import entropy

    if id_cols is None:
        id_cols = ["sample_id"]

    # Get common genera
    synth_genera = set(synthetic_df.columns) - set(id_cols)
    real_genera = set(real_df.columns) - set(id_cols)
    common = synth_genera & real_genera
    overlap_ratio = len(common) / max(len(real_genera), 1)

    # Compute mean profiles
    synth_mean = synthetic_df[list(common)].mean(axis=0).values
    real_mean = real_df[list(common)].mean(axis=0).values

    # Normalize
    synth_mean = synth_mean / max(synth_mean.sum(), 1e-10)
    real_mean = real_mean / max(real_mean.sum(), 1e-10)

    # Jensen-Shannon divergence
    js_div = jensenshannon(synth_mean, real_mean)

    # Shannon diversity
    synth_diversity = entropy(synth_mean[synth_mean > 0])
    real_diversity = entropy(real_mean[real_mean > 0])
    diversity_diff = abs(synth_diversity - real_diversity)

    metrics = {
        "genera_overlap": overlap_ratio,
        "js_divergence": float(js_div),
        "shannon_diversity_diff": float(diversity_diff),
        "synth_shannon": float(synth_diversity),
        "real_shannon": float(real_diversity),
    }

    logger.info("Synthetic profile validation:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics
