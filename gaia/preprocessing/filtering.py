"""
Step [3]: Sparsity Filtering

Removes rare genera present in fewer than a threshold percentage
of samples to reduce noise and manage vocabulary size.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_sparse_genera(
    abundance_df: pd.DataFrame,
    id_cols: list[str] | None = None,
    min_prevalence: float = 0.001,
) -> pd.DataFrame:
    """
    Remove genera that appear in fewer than min_prevalence fraction of samples.

    Args:
        abundance_df: DataFrame with sample IDs + genus columns
        id_cols: Non-abundance columns to preserve
        min_prevalence: Minimum fraction of samples a genus must appear in
                       (default: 0.001 = 0.1%)

    Returns:
        Filtered DataFrame with sparse genera removed
    """
    if id_cols is None:
        id_cols = ["sample_id"]

    genus_cols = [c for c in abundance_df.columns if c not in id_cols]
    genus_data = abundance_df[genus_cols]

    n_samples = len(genus_data)
    min_count = max(1, int(n_samples * min_prevalence))

    # Count non-zero occurrences per genus
    prevalence = (genus_data > 0).sum(axis=0)
    keep_genera = prevalence[prevalence >= min_count].index.tolist()
    removed = len(genus_cols) - len(keep_genera)

    result = abundance_df[id_cols + keep_genera].copy()

    logger.info(
        f"Sparsity filtering: removed {removed} genera "
        f"(prevalence < {min_prevalence:.1%} = {min_count} samples), "
        f"kept {len(keep_genera)} genera"
    )

    return result
