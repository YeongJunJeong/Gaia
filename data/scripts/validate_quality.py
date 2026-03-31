"""
Data Quality Validation for Gaia Project.

Validates soil microbiome samples against the quality checklist
defined in docs/data_standard.md.

Quality Criteria:
  - Total reads > 10,000
  - Classified genera > 20
  - Metadata completeness (biome + location required)
  - Top-1 genus share < 90% (contamination check)
  - Sequencing platform info present
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    total_samples: int = 0
    passed: int = 0
    failed_reads: int = 0
    failed_genera: int = 0
    failed_metadata: int = 0
    flagged_contamination: int = 0
    flagged_no_platform: int = 0
    removed_sample_ids: list = field(default_factory=list)
    flagged_sample_ids: list = field(default_factory=list)


def validate_abundance(
    abundance_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
    min_total_reads: int = 10_000,
    min_genera: int = 20,
    max_top1_share: float = 0.90,
) -> tuple[pd.DataFrame, QualityReport]:
    """
    Validate abundance data against quality criteria.

    Returns:
        Tuple of (filtered DataFrame, QualityReport)
    """
    report = QualityReport(total_samples=len(abundance_df))

    # Identify abundance columns (non-metadata)
    id_cols = {"sample_id", "analysis_id"}
    genus_cols = [c for c in abundance_df.columns if c not in id_cols]

    abundance_values = abundance_df[genus_cols]

    # Criterion 1: Total reads > min_total_reads
    total_reads = abundance_values.sum(axis=1)
    mask_reads = total_reads > min_total_reads
    report.failed_reads = (~mask_reads).sum()

    # Criterion 2: Classified genera > min_genera
    nonzero_genera = (abundance_values > 0).sum(axis=1)
    mask_genera = nonzero_genera > min_genera
    report.failed_genera = (~mask_genera).sum()

    # Criterion 4: Top-1 genus share < max_top1_share (contamination)
    max_abundance = abundance_values.max(axis=1)
    top1_share = max_abundance / total_reads.replace(0, 1)
    mask_contamination = top1_share < max_top1_share
    report.flagged_contamination = (~mask_contamination).sum()

    # Combined filter: remove samples failing reads or genera
    mask_keep = mask_reads & mask_genera
    report.removed_sample_ids = abundance_df.loc[
        ~mask_keep, "sample_id"
    ].tolist()

    # Flag (but don't remove) contamination suspects
    report.flagged_sample_ids = abundance_df.loc[
        ~mask_contamination & mask_keep, "sample_id"
    ].tolist()

    # Criterion 3 & 5: Metadata checks
    if metadata_df is not None:
        missing_biome = metadata_df["biome"].isna() | (
            metadata_df["biome"] == ""
        )
        missing_location = metadata_df["latitude"].isna() | metadata_df[
            "longitude"
        ].isna()
        report.failed_metadata = (missing_biome | missing_location).sum()

        if "sequencing_platform" in metadata_df.columns:
            missing_platform = metadata_df["sequencing_platform"].isna()
            report.flagged_no_platform = missing_platform.sum()

    filtered_df = abundance_df[mask_keep].reset_index(drop=True)
    report.passed = len(filtered_df)

    return filtered_df, report


def print_report(report: QualityReport):
    """Print a formatted quality report."""
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"Total samples:             {report.total_samples}")
    logger.info(f"Passed:                    {report.passed}")
    logger.info(f"Failed (low reads):        {report.failed_reads}")
    logger.info(f"Failed (few genera):       {report.failed_genera}")
    logger.info(f"Failed (metadata):         {report.failed_metadata}")
    logger.info(f"Flagged (contamination):   {report.flagged_contamination}")
    logger.info(f"Flagged (no platform):     {report.flagged_no_platform}")
    logger.info(
        f"Pass rate:                 "
        f"{report.passed / max(report.total_samples, 1) * 100:.1f}%"
    )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate soil microbiome data quality"
    )
    parser.add_argument(
        "abundance_file",
        help="Path to abundance CSV file",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for filtered output CSV",
    )
    parser.add_argument("--min-reads", type=int, default=10_000)
    parser.add_argument("--min-genera", type=int, default=20)
    parser.add_argument("--max-top1-share", type=float, default=0.90)
    args = parser.parse_args()

    abundance_df = pd.read_csv(args.abundance_file)

    metadata_df = None
    if args.metadata:
        metadata_df = pd.read_csv(args.metadata)

    filtered_df, report = validate_abundance(
        abundance_df,
        metadata_df,
        min_total_reads=args.min_reads,
        min_genera=args.min_genera,
        max_top1_share=args.max_top1_share,
    )

    print_report(report)

    if args.output:
        filtered_df.to_csv(args.output, index=False)
        logger.info(f"Saved filtered data to {args.output}")


if __name__ == "__main__":
    main()
