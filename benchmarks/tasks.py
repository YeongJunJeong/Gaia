"""
Benchmark Task Definitions for Gaia.

5 standard evaluation tasks:
  1. Biome Classification (multi-class)
  2. Soil Chemistry Prediction (regression)
  3. Tillage Classification (multi-class)
  4. Drought Stress Detection (binary)
  5. Abundance Reconstruction (self-supervised)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from benchmarks.baselines import run_classification_baselines, run_regression_baselines
from gaia.evaluation.metrics import (
    classification_metrics,
    evaluate_reconstruction,
    regression_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""

    task_name: str
    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray | None = None


def _load_abundance_and_labels(
    abundance_path: str,
    metadata_path: str,
    label_column: str,
    id_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data and split into train/test."""
    if id_cols is None:
        id_cols = ["sample_id"]

    abundance_df = pd.read_csv(abundance_path)
    metadata_df = pd.read_csv(metadata_path)

    # Merge on sample_id
    merged = abundance_df.merge(
        metadata_df[["sample_id", label_column]].dropna(), on="sample_id"
    )

    genus_cols = [c for c in merged.columns if c not in id_cols + [label_column]]
    X = merged[genus_cols].values
    y = merged[label_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.dtype == object else None
    )
    return X_train, X_test, y_train, y_test


def task1_biome_classification(
    abundance_path: str,
    metadata_path: str,
) -> list[BenchmarkResult]:
    """
    Task 1: Soil Biome Classification.

    Input: microbial abundance profile
    Output: biome type (agricultural/forest/grassland/desert/wetland)
    Metrics: ROC-AUC, F1
    """
    logger.info("=== Task 1: Biome Classification ===")

    X_train, X_test, y_train, y_test = _load_abundance_and_labels(
        abundance_path, metadata_path, "biome_envo"
    )

    # Run baselines
    baseline_results = run_classification_baselines(
        X_train, y_train, X_test, y_test
    )

    results = []
    for name, res in baseline_results.items():
        metrics = classification_metrics(y_test, res["y_pred"], res["y_proba"])
        results.append(
            BenchmarkResult(
                task_name="biome_classification",
                model_name=name,
                metrics=metrics,
                predictions=res["y_pred"],
            )
        )

    return results


def task2_soil_chemistry(
    abundance_path: str,
    metadata_path: str,
    target: str = "ph",
) -> list[BenchmarkResult]:
    """
    Task 2: Soil Chemistry Prediction.

    Input: microbial abundance profile
    Output: pH, organic C, total N (continuous)
    Metrics: R², RMSE
    """
    column_map = {
        "ph": "soilInWaterpH",
        "organic_c": "organicCPercent",
        "total_n": "nitrogenPercent",
    }
    label_col = column_map.get(target, target)

    logger.info(f"=== Task 2: Soil Chemistry Prediction ({target}) ===")

    X_train, X_test, y_train, y_test = _load_abundance_and_labels(
        abundance_path, metadata_path, label_col
    )

    baseline_results = run_regression_baselines(
        X_train, y_train, X_test, y_test
    )

    results = []
    for name, res in baseline_results.items():
        metrics = regression_metrics(y_test, res["y_pred"])
        results.append(
            BenchmarkResult(
                task_name=f"soil_chemistry_{target}",
                model_name=name,
                metrics=metrics,
                predictions=res["y_pred"],
            )
        )

    return results


def task3_tillage_classification(
    abundance_path: str,
    metadata_path: str,
) -> list[BenchmarkResult]:
    """
    Task 3: Tillage Practice Classification.

    Input: microbial abundance profile
    Output: no-till / minimum-till / conventional-till
    Metrics: Accuracy, Kappa
    """
    logger.info("=== Task 3: Tillage Classification ===")

    X_train, X_test, y_train, y_test = _load_abundance_and_labels(
        abundance_path, metadata_path, "tillage_type"
    )

    baseline_results = run_classification_baselines(
        X_train, y_train, X_test, y_test
    )

    results = []
    for name, res in baseline_results.items():
        metrics = classification_metrics(y_test, res["y_pred"], res["y_proba"])
        results.append(
            BenchmarkResult(
                task_name="tillage_classification",
                model_name=name,
                metrics=metrics,
            )
        )

    return results


def task4_drought_detection(
    abundance_path: str,
    metadata_path: str,
) -> list[BenchmarkResult]:
    """
    Task 4: Drought Stress Detection.

    Input: microbial abundance profile
    Output: normal / drought-stress (binary)
    Metrics: Accuracy, F1
    Data: Naylor et al. (623 samples)
    """
    logger.info("=== Task 4: Drought Stress Detection ===")

    X_train, X_test, y_train, y_test = _load_abundance_and_labels(
        abundance_path, metadata_path, "drought_stress"
    )

    baseline_results = run_classification_baselines(
        X_train, y_train, X_test, y_test
    )

    results = []
    for name, res in baseline_results.items():
        metrics = classification_metrics(y_test, res["y_pred"], res["y_proba"])
        results.append(
            BenchmarkResult(
                task_name="drought_detection",
                model_name=name,
                metrics=metrics,
            )
        )

    return results


def task5_abundance_reconstruction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[BenchmarkResult]:
    """
    Task 5: Abundance Reconstruction (pre-training quality).

    Input: 20%/30%/50% masked profiles
    Output: reconstructed profiles
    Metrics: Cosine similarity, MAE
    """
    logger.info("=== Task 5: Abundance Reconstruction ===")

    reconstruction_results = evaluate_reconstruction(
        model, dataloader, device, mask_ratios=[0.2, 0.3, 0.5]
    )

    results = []
    for mask_key, accuracy in reconstruction_results.items():
        results.append(
            BenchmarkResult(
                task_name="abundance_reconstruction",
                model_name=f"Gaia ({mask_key})",
                metrics={"reconstruction_accuracy": accuracy},
            )
        )

    return results
