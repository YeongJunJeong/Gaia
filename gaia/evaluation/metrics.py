"""
Evaluation Metrics for Gaia.

Pre-training quality metrics:
  1. Reconstruction accuracy (masked token recovery)
  2. Embedding clustering quality (silhouette score)
  3. Taxonomic structure preservation
  4. Ecological relationship encoding

Downstream task metrics:
  - Classification: ROC-AUC, F1, Accuracy, Kappa
  - Regression: R², RMSE, MAE
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def reconstruction_cosine_similarity(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    """
    Compute mean cosine similarity between original and reconstructed
    abundance profiles.

    Target: > 0.85 at 30% masking, > 0.75 at 50% masking
    """
    from scipy.spatial.distance import cosine

    similarities = []
    for orig, recon in zip(original, reconstructed):
        if np.any(orig) and np.any(recon):
            sim = 1 - cosine(orig, recon)
            similarities.append(sim)

    mean_sim = np.mean(similarities) if similarities else 0.0
    logger.info(f"Reconstruction cosine similarity: {mean_sim:.4f}")
    return mean_sim


def embedding_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score for sample embeddings grouped by biome.

    Target: > 0.5
    """
    if len(set(labels)) < 2:
        logger.warning("Need at least 2 clusters for silhouette score")
        return 0.0

    score = silhouette_score(embeddings, labels)
    logger.info(f"Embedding silhouette score: {score:.4f}")
    return score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute classification metrics.

    Returns dict with accuracy, f1, kappa, and optionally roc_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] > 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
            elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression metrics.

    Returns dict with r2, rmse, mae.
    """
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }

    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


@torch.no_grad()
def evaluate_reconstruction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_ratios: list[float] = [0.2, 0.3, 0.5],
) -> dict[str, float]:
    """
    Evaluate masked token reconstruction at multiple mask ratios.

    Returns dict mapping mask_ratio -> cosine similarity.
    """
    model.eval()
    results = {}

    for mask_ratio in mask_ratios:
        all_original = []
        all_reconstructed = []

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            batch_size, seq_len = input_ids.shape

            # Create masked input
            masked_input = input_ids.clone()
            maskable = (input_ids > 4)  # Non-special tokens
            mask = torch.rand_like(input_ids.float()) < mask_ratio
            mask = mask & maskable
            masked_input[mask] = 2  # [MASK] token

            # Get predictions
            output = model(masked_input)
            logits = output["logits"]
            predicted_ids = logits.argmax(dim=-1)

            # Compare only at masked positions
            for i in range(batch_size):
                mask_positions = mask[i]
                if mask_positions.any():
                    orig = input_ids[i][mask_positions].cpu().numpy()
                    pred = predicted_ids[i][mask_positions].cpu().numpy()
                    all_original.append(orig)
                    all_reconstructed.append(pred)

        if all_original:
            orig_flat = np.concatenate(all_original)
            pred_flat = np.concatenate(all_reconstructed)
            accuracy = (orig_flat == pred_flat).mean()
            results[f"reconstruction_acc_{mask_ratio}"] = float(accuracy)
            logger.info(
                f"Mask {mask_ratio:.0%}: reconstruction accuracy = {accuracy:.4f}"
            )

    return results
