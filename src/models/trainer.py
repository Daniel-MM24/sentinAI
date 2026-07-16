"""Shared training orchestration for the sentinAI AML model suite."""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _cast_feature_cols(df: pl.DataFrame, cols: List[str]) -> np.ndarray:
    """Cast a Polars DataFrame to a clean float numpy array.

    Handles datetime → epoch seconds, Boolean → int, null → 0.0, and
    drops any remaining non-numeric columns.
    """
    numeric_cols = []
    expressions = []
    for c in cols:
        dtype = df[c].dtype
        if dtype == pl.Datetime:
            expressions.append(pl.col(c).dt.epoch("s").cast(pl.Float64).fill_null(0.0).alias(c))
            numeric_cols.append(c)
        elif dtype in (pl.Boolean,):
            expressions.append(pl.col(c).cast(pl.Int8).cast(pl.Float64).fill_null(0.0).alias(c))
            numeric_cols.append(c)
        elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                       pl.Float32, pl.Float64):
            expressions.append(pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c))
            numeric_cols.append(c)
        elif dtype == pl.String:
            # Skip string columns — encode manually or drop
            logger.debug("Dropping string feature column: %s", c)
            continue
        elif dtype == pl.Null:
            logger.debug("Dropping null-only column: %s", c)
            continue
        else:
            logger.debug("Dropping unsupported column '%s' (type=%s)", c, dtype)
            continue

    if not expressions:
        raise ValueError("No usable numeric feature columns found after type casting")

    result = df.with_columns(expressions).select(numeric_cols).to_numpy()
    return np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)


def prepare_training_data(
    df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Split a Gold feature DataFrame into train/test sets.

    Steps:
        1. Drop identifier and label columns from features (X).
        2. Isolate ``anomaly_flag`` as the binary target (y).
        3. Cast every column to a clean numeric representation.
        4. Stratified 80/20 split.

    Parameters
    ----------
    df : pl.DataFrame
        Gold feature store DataFrame.
    test_size : float
        Fraction of data to hold out for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names : List[str]
        Column names corresponding to feature dimensions.
    """
    # Identify columns to drop from features
    drop_cols = {"anomaly_flag", "anomaly_type", "anomaly_case_id", "transaction_id",
                 "customer_id", "counterparty_id", "partition_date", "data_provenance_hash",
                 "regulatory_report_status", "counterparty_risk_tier",
                 "timestamp", "ingestion_timestamp", "account_first_seen", "account_last_seen"}

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = _cast_feature_cols(df, feature_cols)

    # Derive feature names from what _cast_feature_cols actually kept
    caster = _cast_feature_cols(df, feature_cols + ["anomaly_flag"])  # probe call
    # Simpler: collect column names from _cast_feature_cols internals
    numeric_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                      pl.Float32, pl.Float64, pl.Boolean, pl.Datetime)
    effective_cols = [c for c in feature_cols
                      if c in df.columns
                      and df[c].dtype in numeric_dtypes]
    # Stretch to match X shape (Datetime was accepted)
    if len(effective_cols) != X.shape[1]:
        effective_cols = [c for c in feature_cols
                          if c in df.columns
                          and (df[c].dtype in numeric_dtypes
                               or df[c].dtype == pl.Datetime)]
    # If still wrong, use the _cast_feature_cols logic directly
    if len(effective_cols) != X.shape[1]:
        # Infer from first row non-null detection
        numeric_indices = set()
        for i, c in enumerate(feature_cols):
            if c in df.columns:
                dtype = df[c].dtype
                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                             pl.Float32, pl.Float64, pl.Boolean, pl.Datetime, pl.Null):
                    numeric_indices.add(i)
        effective_cols = [c for idx, c in enumerate(feature_cols) if idx in numeric_indices]
        effective_cols = effective_cols[:X.shape[1]]
    y = df["anomaly_flag"].cast(pl.Int8).to_numpy().ravel()

    # Handle any remaining NaN/Inf in X
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    logger.info("Feature matrix shape: %s  |  Positive ratio: %.4f",
                X.shape, y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_train, X_test, y_train, y_test, effective_cols


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute ``scale_pos_weight`` as ratio of negatives to positives."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return neg / pos if pos > 0 else 1.0


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train a LightGBM classifier.

    Requires ``lightgbm`` to be installed.
    """
    import lightgbm as lgb

    if config is None:
        from src.models.config import LightGBMConfig
        config = LightGBMConfig()

    params = {
        "objective": config.objective,
        "metric": config.metric,
        "boosting_type": "gbdt",
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_data_in_leaf": config.min_data_in_leaf,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "min_gain_to_split": config.min_gain_to_split,
        "verbosity": config.verbosity,
        "seed": config.random_state,
        "n_jobs": config.n_jobs,
    }
    if config.scale_pos_weight is None:
        params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=config.n_estimators,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds),
                   lgb.log_evaluation(0)],
    )
    logger.info("LightGBM trained — best iteration %d, best score %.6f",
                model.best_iteration, model.best_score["valid_0"][config.metric])
    return {"model": model, "config": config}


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train an XGBoost classifier."""
    import xgboost as xgb

    if config is None:
        from src.models.config import XGBoostConfig
        config = XGBoostConfig()

    params = {
        "objective": config.objective,
        "eval_metric": config.eval_metric,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "gamma": config.gamma,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "seed": config.random_state,
        "verbosity": config.verbosity,
        "n_jobs": config.n_jobs,
    }
    if config.scale_pos_weight is None:
        params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(**params, n_estimators=config.n_estimators)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    logger.info("XGBoost trained — best iteration %d", model.best_iteration
                if hasattr(model, "best_iteration") else config.n_estimators)
    return {"model": model, "config": config}


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train a Random Forest baseline classifier."""
    if config is None:
        from src.models.config import RandomForestConfig
        config = RandomForestConfig()

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest trained — %d estimators", config.n_estimators)
    return {"model": model, "config": config}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Find the decision threshold that maximises the F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_type: str = "lightgbm",
) -> Dict[str, Any]:
    """Unified model evaluation.

    Returns a dictionary of metrics plus predicted labels at the optimal
    threshold.
    """
    # Predict probabilities
    if model_type == "lightgbm":
        y_score = model.predict(X_test, num_iteration=model.best_iteration)
    elif model_type == "xgboost":
        y_score = model.predict_proba(X_test)[:, 1]
    elif model_type == "random_forest":
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.predict(X_test)

    # Optimal threshold from F1
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_score)
    y_pred = (y_score >= best_threshold).astype(int)

    # Core metrics
    auc_roc = roc_auc_score(y_test, y_score)
    auc_pr = None
    try:
        from sklearn.metrics import average_precision_score
        auc_pr = average_precision_score(y_test, y_score)
    except Exception:
        pass

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4) if auc_pr else None,
        "f1": round(best_f1, 4),
        "precision": round(report_dict["1"]["precision"], 4),
        "recall": round(report_dict["1"]["recall"], 4),
        "specificity": round(specificity, 4),
        "mcc": round(mcc, 4),
        "optimal_threshold": round(float(best_threshold), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "positive_ratio": round(float(y_test.mean()), 4),
        "classification_report": report_dict,
    }
    return metrics


def generate_shap(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    model_type: str = "lightgbm",
) -> Dict[str, Any]:
    """Compute SHAP values for a sample of the test set.

    Returns a dict with ``shap_values`` (list), ``mean_abs_shap`` (dict),
    and ``feature_names``.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis")
        return {"shap_values": None, "mean_abs_shap": {}, "feature_names": feature_names}

    # Subsample if large
    sample_size = min(X_sample.shape[0], 500)

    if model_type == "lightgbm":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample[:sample_size])
        # LightGBM returns list of arrays for multi-class; take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif model_type == "xgboost":
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample[:sample_size])
        except Exception as e:
            logger.warning("SHAP TreeExplainer failed for %s: %s", model_type, e)
            return {"shap_values": None, "mean_abs_shap": {}, "feature_names": feature_names}
    elif model_type == "random_forest":
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample[:sample_size])
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        except Exception as e:
            logger.warning("SHAP TreeExplainer failed for %s: %s", model_type, e)
            return {"shap_values": None, "mean_abs_shap": {}, "feature_names": feature_names}
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
        shap_values = explainer.shap_values(X_sample[:sample_size])
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    mean_abs = dict(zip(feature_names, np.abs(shap_values).mean(axis=0)))

    return {
        "shap_values": shap_values.tolist() if shap_values is not None else None,
        "mean_abs_shap": {k: float(v) for k, v in sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)},
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_model_artifacts(
    model: Any,
    metrics: Dict[str, Any],
    shap_result: Dict[str, Any],
    output_dir: str,
    model_name: str,
) -> None:
    """Serialise model, metrics, and SHAP data to ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = out / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", model_path)

    # Metrics
    metrics_path = out / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # SHAP
    shap_path = out / f"{model_name}_shap.json"
    with open(shap_path, "w") as f:
        json.dump(shap_result, f, indent=2)
    logger.info("SHAP summary saved to %s", shap_path)


def save_shap_summary_plot(
    shap_result: Dict[str, Any],
    output_dir: str,
    model_name: str,
    max_features: int = 15,
) -> None:
    """Generate and save a SHAP summary bar plot.

    Falls back to a matplotlib bar chart when ``shap`` is not available for
    plotting.
    """
    mean_abs = shap_result.get("mean_abs_shap", {})
    if not mean_abs:
        return

    names = list(mean_abs.keys())[:max_features]
    values = list(mean_abs.values())[:max_features]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(names)), values[::-1])
        plt.yticks(range(len(names)), names[::-1])
        plt.xlabel("Mean |SHAP|")
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        path = Path(output_dir) / f"{model_name}_shap_summary.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("SHAP summary plot saved to %s", path)
    except Exception as e:
        logger.warning("Failed to generate SHAP plot: %s", e)


TRAIN_REGISTRY = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "random_forest": train_random_forest,
}
