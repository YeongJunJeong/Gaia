"""
Run All Gaia Benchmarks.

Usage:
    python -m benchmarks.run_all \
        --abundance data/processed/gaia-abundance-v1.csv \
        --metadata data/processed/gaia-metadata-v1.csv \
        --model-path checkpoints/pretrain/best.pt \
        --output benchmarks/results/
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from benchmarks.tasks import (
    task1_biome_classification,
    task2_soil_chemistry,
    task3_tillage_classification,
    task4_drought_detection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run all Gaia benchmarks")
    parser.add_argument("--abundance", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="benchmarks/results")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="Which tasks to run (1-5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Task 1: Biome Classification
    if "1" in args.tasks:
        try:
            results = task1_biome_classification(args.abundance, args.metadata)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Task 1 failed: {e}")

    # Task 2: Soil Chemistry Prediction
    if "2" in args.tasks:
        for target in ["ph", "organic_c", "total_n"]:
            try:
                results = task2_soil_chemistry(
                    args.abundance, args.metadata, target=target
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Task 2 ({target}) failed: {e}")

    # Task 3: Tillage Classification
    if "3" in args.tasks:
        try:
            results = task3_tillage_classification(args.abundance, args.metadata)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Task 3 failed: {e}")

    # Task 4: Drought Stress Detection
    if "4" in args.tasks:
        try:
            results = task4_drought_detection(args.abundance, args.metadata)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Task 4 failed: {e}")

    # Task 5: Abundance Reconstruction (requires model)
    if "5" in args.tasks and args.model_path:
        try:
            import torch
            from gaia.models.transformer import GaiaConfig, GaiaTransformer
            from gaia.training.dataset import MicrobiomeDataset
            from torch.utils.data import DataLoader
            from benchmarks.tasks import task5_abundance_reconstruction

            checkpoint = torch.load(args.model_path, map_location="cpu")
            config = GaiaConfig.from_dict(checkpoint["config"])
            model = GaiaTransformer(config)
            model.load_state_dict(checkpoint["model_state_dict"])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Use test split for reconstruction
            corpus_path = str(
                Path(args.abundance).parent / "gaia-corpus-v1.pkl"
            )
            dataset = MicrobiomeDataset(corpus_path)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            results = task5_abundance_reconstruction(model, loader, device)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Task 5 failed: {e}")

    # Save results
    if all_results:
        rows = []
        for r in all_results:
            row = {"task": r.task_name, "model": r.model_name}
            row.update(r.metrics)
            rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_dir / "benchmark_results.csv", index=False)

        # Pretty print
        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 70)
        logger.info(results_df.to_string(index=False))

        # Also save as JSON
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(rows, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
