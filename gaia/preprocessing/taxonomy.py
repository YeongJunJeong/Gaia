"""
Step [1]: Taxonomy Unification

Maps diverse taxonomic databases (SILVA, Greengenes, NCBI, etc.)
to GTDB r220 standard nomenclature.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Common genus-level name mappings from various databases to GTDB r220.
# This is a starter mapping; the full mapping should be loaded from
# an external reference file generated from GTDB metadata.
_KNOWN_RENAMES: dict[str, str] = {
    # SILVA -> GTDB renames (examples)
    "Candidatus Udaeobacter": "Udaeobacter",
    "Candidatus Solibacter": "Solibacter",
    "Burkholderia-Caballeronia-Paraburkholderia": "Paraburkholderia",
    # Greengenes legacy names
    "Rubrobacter": "Rubrobacter",
}


def load_gtdb_mapping(mapping_file: str | None = None) -> dict[str, str]:
    """
    Load a full GTDB taxonomy mapping from a TSV file.

    Expected format: old_name<TAB>gtdb_name
    """
    if mapping_file is None or not Path(mapping_file).exists():
        logger.info("Using built-in genus name mapping (limited)")
        return _KNOWN_RENAMES

    mapping = {}
    df = pd.read_csv(mapping_file, sep="\t", header=None, names=["old", "new"])
    for _, row in df.iterrows():
        mapping[row["old"]] = row["new"]

    logger.info(f"Loaded {len(mapping)} genus name mappings from {mapping_file}")
    return mapping


def unify_taxonomy(
    abundance_df: pd.DataFrame,
    id_cols: list[str] | None = None,
    mapping_file: str | None = None,
) -> pd.DataFrame:
    """
    Unify genus names to GTDB r220 standard.

    Steps:
    1. Strip whitespace and normalize casing
    2. Apply known name mappings
    3. Merge columns with the same GTDB name (sum abundances)

    Args:
        abundance_df: DataFrame with sample IDs + genus columns
        id_cols: Columns that are NOT genus names (e.g. sample_id)
        mapping_file: Optional path to full GTDB mapping TSV

    Returns:
        DataFrame with unified genus names
    """
    if id_cols is None:
        id_cols = ["sample_id"]

    mapping = load_gtdb_mapping(mapping_file)
    genus_cols = [c for c in abundance_df.columns if c not in id_cols]

    # Step 1: Normalize names
    rename_map = {}
    for col in genus_cols:
        clean = col.strip()
        # Apply GTDB mapping if available
        mapped = mapping.get(clean, clean)
        rename_map[col] = mapped

    # Step 2: Rename columns
    abundance_df = abundance_df.rename(columns=rename_map)

    # Step 3: Merge duplicate genus columns (sum)
    id_data = abundance_df[id_cols]
    genus_data = abundance_df.drop(columns=id_cols)

    # Group columns with the same name and sum
    genus_data = genus_data.T.groupby(level=0).sum().T

    result = pd.concat([id_data, genus_data], axis=1)
    n_merged = len(genus_cols) - genus_data.shape[1]
    if n_merged > 0:
        logger.info(f"Merged {n_merged} duplicate genus columns after GTDB mapping")

    logger.info(f"Taxonomy unification complete: {genus_data.shape[1]} genera")
    return result
