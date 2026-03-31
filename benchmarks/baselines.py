"""
Baseline Models for Gaia Benchmarks.

Traditional ML baselines: Random Forest, XGBoost, SVM.
These serve as comparison points for the foundation model.
"""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

logger = logging.getLogger(__name__)


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier, XGBRegressor
    except ImportError:
        logger.warning("xgboost not installed, skipping XGBoost baseline")
        return None, None


def run_classification_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
) -> dict[str, dict]:
    """
    Run all classification baselines and return results.

    Returns:
        Dict mapping model_name -> {predictions, metrics}
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "SVM_L2": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    }

    XGBClassifier, _ = _try_import_xgboost()
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

    results = {}
    for name, model in models.items():
        logger.info(f"Training baseline: {name}")

        if name == "SVM_L2":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

        results[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "model": model,
        }

    return results


def run_regression_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """
    Run all regression baselines and return results.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "SVM_L2": SVR(kernel="rbf", C=1.0),
    }

    _, XGBRegressor = _try_import_xgboost()
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    results = {}
    for name, model in models.items():
        logger.info(f"Training baseline: {name}")

        if name == "SVM_L2":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results[name] = {
            "y_pred": y_pred,
            "model": model,
        }

    return results
