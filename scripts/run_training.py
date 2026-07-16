"""CLI entry point for AML model training and evaluation.

Trains one or more models on the Gold feature store, evaluates performance,
generates SHAP explanations, and saves artifacts to ``--output-dir``.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from src.models.trainer import (
    TRAIN_REGISTRY,
    evaluate_model,
    generate_shap,
    prepare_training_data,
    save_model_artifacts,
    save_shap_summary_plot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AML detection models on the Gold feature store.")
    parser.add_argument(
        "--data-dir",
        default="data/gold/features/vv1.0",
        help="Path to Gold feature store directory (default: data/gold/features/v1.0)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained models and artifacts (default: models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "xgboost", "random_forest"],
        choices=list(TRAIN_REGISTRY.keys()),
        help="Models to train (default: all)",
    )
    return parser.parse_args()


def load_gold_data(data_dir: str) -> pl.DataFrame:
    """Load Gold feature-store data, preferring the consolidated file."""
    consolidated = Path(data_dir) / "gold_features_consolidated.parquet"
    if consolidated.exists():
        logger.info("Loading consolidated data from %s", consolidated)
        return pl.read_parquet(consolidated)

    # Fall back to globbing partitioned files
    logger.info("Consolidated file not found — globbing partitioned parquet under %s", data_dir)
    files = list(Path(data_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return pl.concat(pl.read_parquet(f) for f in files)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    df = load_gold_data(args.data_dir)
    logger.info("Loaded Gold data: %d rows x %d cols", df.height, df.width)

    # Quick summary of anomaly distribution
    if "anomaly_flag" in df.columns:
        dist = df["anomaly_flag"].value_counts().sort("anomaly_flag")
        logger.info("Anomaly distribution:\n%s", dist)

    # ── Prepare train / test -------------------------------------------
    X_train, X_test, y_train, y_test, feature_names = prepare_training_data(df)

    # ── Train each requested model --------------------------------------
    all_metrics = {}
    all_shap = {}

    for model_name in args.models:
        logger.info("=" * 60)
        logger.info("Training: %s", model_name)
        logger.info("=" * 60)

        train_fn = TRAIN_REGISTRY[model_name]
        t0 = time.time()

        result = train_fn(
            X_train, y_train, X_test, y_test,
            feature_names=feature_names,
        )

        elapsed = time.time() - t0
        logger.info("Training complete in %.1f s", elapsed)

        model = result["model"]
        config = result["config"]
        model_type = getattr(config, "model_type", model_name)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, feature_names, model_type)
        metrics["training_time_s"] = round(elapsed, 2)
        all_metrics[model_name] = metrics

        logger.info("AUC-ROC: %.4f | AUC-PR: %s | F1: %.4f | MCC: %.4f",
                    metrics["auc_roc"],
                    metrics["auc_pr"] if metrics["auc_pr"] else "N/A",
                    metrics["f1"],
                    metrics["mcc"])
        logger.info("Confusion matrix: TN=%(tn)d FP=%(fp)d FN=%(fn)d TP=%(tp)d",
                    metrics["confusion_matrix"])

        # SHAP
        shap_result = generate_shap(model, X_test, feature_names, model_type)
        all_shap[model_name] = shap_result

        # Persist
        save_model_artifacts(model, metrics, shap_result, str(out_dir), model_name)
        save_shap_summary_plot(shap_result, str(out_dir), model_name)

    # ── Write combined performance report -------------------------------
    report = {
        "dataset": {
            "path": str(Path(args.data_dir).resolve()),
            "rows": df.height,
            "features": len(feature_names),
            "positive_ratio": float(y_train.mean()),
        },
        "models": all_metrics,
        "top_features": _build_top_features(all_shap, top_n=10),
    }
    report_path = out_dir / "model_performance.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Performance report saved to %s", report_path)


def _build_top_features(
    all_shap: dict, top_n: int = 10,
) -> dict:
    """Extract top-N features by mean |SHAP| for each model."""
    result = {}
    for model_name, shap_res in all_shap.items():
        mean_abs = shap_res.get("mean_abs_shap", {})
        result[model_name] = list(mean_abs.items())[:top_n]
    return result


if __name__ == "__main__":
    main()
