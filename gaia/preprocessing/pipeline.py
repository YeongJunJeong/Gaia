"""
Full Preprocessing Pipeline for Gaia.

Orchestrates all 6 preprocessing steps:
  [1] Taxonomy unification (GTDB r220)
  [2] Abundance normalization (TSS or CLR)
  [3] Sparsity filtering
  [4] Metadata standardization (ENVO ontology)
  [5] Batch effect tagging
  [6] Corpus conversion (MGM-compatible tokenization)
"""

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from gaia.preprocessing.filtering import filter_sparse_genera
from gaia.preprocessing.metadata import standardize_metadata
from gaia.preprocessing.normalization import normalize
from gaia.preprocessing.taxonomy import unify_taxonomy
from gaia.preprocessing.tokenizer import MicrobiomeTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    abundance_path: str,
    metadata_path: str,
    output_dir: str,
    normalization_method: str = "tss",
    min_prevalence: float = 0.001,
    max_length: int = 512,
    gtdb_mapping_file: str | None = None,
) -> dict:
    """
    Run the full preprocessing pipeline.

    Args:
        abundance_path: Path to raw abundance CSV
        metadata_path: Path to raw metadata CSV
        output_dir: Directory for output files
        normalization_method: "tss" or "clr"
        min_prevalence: Min fraction of samples for sparsity filtering
        max_length: Token sequence length
        gtdb_mapping_file: Optional GTDB name mapping file

    Returns:
        Dictionary with pipeline statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    # Load data
    logger.info("Loading raw data...")
    abundance_df = pd.read_csv(abundance_path)
    metadata_df = pd.read_csv(metadata_path)
    stats["raw_samples"] = len(abundance_df)
    stats["raw_genera"] = len(
        [c for c in abundance_df.columns if c not in ["sample_id", "analysis_id"]]
    )

    id_cols = [c for c in ["sample_id", "analysis_id"] if c in abundance_df.columns]

    # Step 1: Taxonomy unification
    logger.info("[1/6] Unifying taxonomy to GTDB r220...")
    abundance_df = unify_taxonomy(
        abundance_df, id_cols=id_cols, mapping_file=gtdb_mapping_file
    )

    # Step 2: Abundance normalization
    logger.info(f"[2/6] Normalizing abundances ({normalization_method})...")
    abundance_norm = normalize(
        abundance_df, method=normalization_method, id_cols=id_cols
    )

    # Step 3: Sparsity filtering
    logger.info("[3/6] Filtering sparse genera...")
    abundance_filtered = filter_sparse_genera(
        abundance_norm, id_cols=id_cols, min_prevalence=min_prevalence
    )
    genus_cols = [c for c in abundance_filtered.columns if c not in id_cols]
    stats["filtered_genera"] = len(genus_cols)

    # Step 4 & 5: Metadata standardization + batch tagging
    logger.info("[4/6] Standardizing metadata...")
    logger.info("[5/6] Tagging batch effect variables...")
    metadata_std = standardize_metadata(metadata_df)

    # Step 6: Corpus conversion
    logger.info("[6/6] Building vocabulary and tokenizing corpus...")
    tokenizer = MicrobiomeTokenizer(max_length=max_length)
    tokenizer.build_vocab(abundance_filtered, id_cols=id_cols)
    token_sequences = tokenizer.encode_batch(abundance_filtered, id_cols=id_cols)
    stats["vocab_size"] = len(tokenizer.vocab)
    stats["corpus_samples"] = token_sequences.shape[0]
    stats["sequence_length"] = token_sequences.shape[1]

    # Save outputs
    logger.info("Saving outputs...")

    # Corpus pickle
    corpus = {
        "token_sequences": token_sequences,
        "sample_ids": abundance_filtered[id_cols[0]].tolist(),
        "normalization": normalization_method,
    }
    corpus_path = output_dir / "gaia-corpus-v1.pkl"
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)

    # Metadata CSV
    metadata_path_out = output_dir / "gaia-metadata-v1.csv"
    metadata_std.to_csv(metadata_path_out, index=False)

    # Normalized abundance (for downstream analysis)
    abundance_path_out = output_dir / "gaia-abundance-v1.csv"
    abundance_filtered.to_csv(abundance_path_out, index=False)

    # Tokenizer vocabulary
    tokenizer.save(str(output_dir / "tokenizer.json"))

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  Output directory: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run Gaia preprocessing pipeline"
    )
    parser.add_argument("abundance", help="Path to raw abundance CSV")
    parser.add_argument("metadata", help="Path to raw metadata CSV")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--normalization",
        choices=["tss", "clr"],
        default="tss",
        help="Normalization method",
    )
    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.001,
        help="Min prevalence for sparsity filtering",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Token sequence length",
    )
    parser.add_argument(
        "--gtdb-mapping",
        default=None,
        help="Path to GTDB name mapping TSV",
    )
    args = parser.parse_args()

    run_pipeline(
        abundance_path=args.abundance,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        normalization_method=args.normalization,
        min_prevalence=args.min_prevalence,
        max_length=args.max_length,
        gtdb_mapping_file=args.gtdb_mapping,
    )


if __name__ == "__main__":
    main()
