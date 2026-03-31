"""
Step [2]: Abundance Normalization

Provides TSS (Total Sum Scaling) and CLR (Centered Log-Ratio)
normalization methods for compositional microbiome data.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def tss_normalize(
    abundance_df: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Total Sum Scaling (TSS) normalization.

    Converts raw counts to relative abundances (proportions summing to 1.0).

    Args:
        abundance_df: DataFrame with sample IDs + genus count columns
        id_cols: Non-abundance columns to preserve

    Returns:
        DataFrame with relative abundances
    """
    if id_cols is None:
        id_cols = ["sample_id"]

    id_data = abundance_df[id_cols].copy()
    genus_data = abundance_df.drop(columns=id_cols).astype(float)

    row_sums = genus_data.sum(axis=1).replace(0, 1)  # Avoid division by zero
    normalized = genus_data.div(row_sums, axis=0)

    result = pd.concat([id_data, normalized], axis=1)
    logger.info(f"TSS normalization complete: {normalized.shape}")
    return result


def clr_normalize(
    abundance_df: pd.DataFrame,
    id_cols: list[str] | None = None,
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """
    Centered Log-Ratio (CLR) transformation.

    Addresses compositionality of microbiome data by applying
    log-ratio transformation with the geometric mean as reference.

    Args:
        abundance_df: DataFrame with sample IDs + genus count columns
        id_cols: Non-abundance columns to preserve
        pseudocount: Small value added to zeros before log transform

    Returns:
        DataFrame with CLR-transformed values
    """
    if id_cols is None:
        id_cols = ["sample_id"]

    id_data = abundance_df[id_cols].copy()
    genus_data = abundance_df.drop(columns=id_cols).astype(float)

    # Add pseudocount to handle zeros
    genus_data = genus_data + pseudocount

    # Compute geometric mean per sample (row)
    log_data = np.log(genus_data)
    geometric_mean = log_data.mean(axis=1)

    # CLR = log(x) - mean(log(x))
    clr_data = log_data.sub(geometric_mean, axis=0)

    result = pd.concat([id_data, clr_data], axis=1)
    logger.info(f"CLR normalization complete: {clr_data.shape}")
    return result


def normalize(
    abundance_df: pd.DataFrame,
    method: str = "tss",
    id_cols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Normalize abundance data using the specified method.

    Args:
        abundance_df: Raw abundance DataFrame
        method: "tss" or "clr"
        id_cols: Non-abundance columns
        **kwargs: Additional arguments passed to the normalization function

    Returns:
        Normalized DataFrame
    """
    if method == "tss":
        return tss_normalize(abundance_df, id_cols=id_cols)
    elif method == "clr":
        return clr_normalize(abundance_df, id_cols=id_cols, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'tss' or 'clr'")
