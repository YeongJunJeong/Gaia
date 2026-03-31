"""
Step [4]: Metadata Standardization

Standardizes biome classification (ENVO ontology), geographic
coordinates, and flags missing/suspicious data.

Step [5]: Batch Effect Tagging

Records sequencing platform, DNA extraction kit, and analysis
pipeline as metadata for downstream batch correction.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ENVO ontology mapping for common biome descriptions
BIOME_TO_ENVO: dict[str, str] = {
    "agricultural soil": "ENVO:00002259",
    "agricultural": "ENVO:00002259",
    "cropland": "ENVO:00002259",
    "farm": "ENVO:00002259",
    "forest soil": "ENVO:01001198",
    "forest": "ENVO:01001198",
    "woodland": "ENVO:01001198",
    "grassland soil": "ENVO:00005750",
    "grassland": "ENVO:00005750",
    "prairie": "ENVO:00005750",
    "meadow": "ENVO:00005750",
    "desert soil": "ENVO:01001357",
    "desert": "ENVO:01001357",
    "arid": "ENVO:01001357",
    "wetland soil": "ENVO:00002044",
    "wetland": "ENVO:00002044",
    "bog": "ENVO:00002044",
    "marsh": "ENVO:00002044",
    "permafrost": "ENVO:01001526",
    "tundra": "ENVO:01001526",
    "rhizosphere": "ENVO:01000999",
}


def standardize_biome(biome_str: str | None) -> str:
    """Map a biome description string to ENVO ontology term."""
    if pd.isna(biome_str) or not biome_str:
        return "unknown"

    biome_lower = str(biome_str).lower().strip()

    # Direct match
    if biome_lower in BIOME_TO_ENVO:
        return BIOME_TO_ENVO[biome_lower]

    # Substring match
    for key, envo in BIOME_TO_ENVO.items():
        if key in biome_lower:
            return envo

    return "unknown"


def standardize_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize metadata fields.

    Steps:
    1. Map biome descriptions to ENVO ontology
    2. Validate and standardize geographic coordinates
    3. Flag missing/suspicious data
    4. Tag batch effect variables
    """
    df = metadata_df.copy()

    # Step 4a: Biome classification (ENVO ontology)
    if "biome" in df.columns:
        df["biome_envo"] = df["biome"].apply(standardize_biome)
        n_unknown = (df["biome_envo"] == "unknown").sum()
        logger.info(
            f"Biome standardization: {n_unknown}/{len(df)} mapped to 'unknown'"
        )

    # Step 4b: Geographic coordinate validation
    if "latitude" in df.columns:
        invalid_lat = ~df["latitude"].between(-90, 90) & df["latitude"].notna()
        if invalid_lat.any():
            logger.warning(
                f"Found {invalid_lat.sum()} samples with invalid latitude"
            )
            df.loc[invalid_lat, "latitude"] = None

    if "longitude" in df.columns:
        invalid_lon = ~df["longitude"].between(-180, 180) & df["longitude"].notna()
        if invalid_lon.any():
            logger.warning(
                f"Found {invalid_lon.sum()} samples with invalid longitude"
            )
            df.loc[invalid_lon, "longitude"] = None

    # Step 4c: Flag missing data
    required_fields = ["biome_envo", "latitude", "longitude"]
    existing_required = [f for f in required_fields if f in df.columns]
    df["metadata_complete"] = ~df[existing_required].isna().any(axis=1)

    # Step 5: Batch effect tagging
    batch_fields = ["sequencing_platform", "extraction_kit", "analysis_pipeline"]
    existing_batch = [f for f in batch_fields if f in df.columns]
    df["batch_correctable"] = ~df[existing_batch].isna().any(axis=1) if existing_batch else False

    n_complete = df["metadata_complete"].sum()
    n_batch = df["batch_correctable"].sum() if "batch_correctable" in df.columns else 0
    logger.info(
        f"Metadata standardization complete: "
        f"{n_complete}/{len(df)} complete, "
        f"{n_batch}/{len(df)} batch-correctable"
    )

    return df
