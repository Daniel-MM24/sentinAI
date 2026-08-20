"""Compare TVAE Hybrid vs Pure Monte Carlo Gold Datasets.

This script performs comprehensive comparative analysis between:
- Pure Monte Carlo gold: data/gold/features/v*/customer_features_{partition}.parquet
- TVAE hybrid gold: data/gold/tvae_hybrid_gold_{partition}.parquet

Comparisons include:
- Distribution comparison for each feature (KS tests, histograms)
- Correlation matrix comparison
- Temporal pattern comparison
- Tier compliance comparison
- Anomaly pattern comparison

Outputs:
- Comparison report with visualizations saved to reports/tvae_comparison_{timestamp}/
- Statistical distance metrics
- Feature coverage analysis
- Regulatory constraint compliance
- Computational efficiency metrics
- Recommendations for when to use TVAE vs Monte Carlo
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time

import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Target 21-feature gold schema
GOLD_SCHEMA_COLUMNS = [
    "customer_id",
    "tier",
    "archetype",
    "transaction_type",
    "amount",
    "timestamp",
    "direction",
    "balance",
    "tx_count_7d",
    "volume_7d",
    "night_tx_ratio",
    "rapid_tx_ratio",
    "volume_7d_vs_30d_ratio",
    "is_international",
    "distinct_counterparties_7d",
    "fan_in_fan_out_ratio",
    "close_to_limit_ratio",
    "balance_retention_ratio",
    "amount_roundness",
    "is_launderer",
    "aml_scenario",
]

# Numerical features for statistical comparison
NUMERICAL_FEATURES = [
    "amount",
    "balance",
    "tx_count_7d",
    "volume_7d",
    "night_tx_ratio",
    "rapid_tx_ratio",
    "volume_7d_vs_30d_ratio",
    "distinct_counterparties_7d",
    "fan_in_fan_out_ratio",
    "close_to_limit_ratio",
    "balance_retention_ratio",
    "amount_roundness",
]

# Categorical features for distribution comparison
CATEGORICAL_FEATURES = [
    "tier",
    "archetype",
    "transaction_type",
    "direction",
    "is_international",
    "is_launderer",
    "aml_scenario",
]

# Tier limits for compliance checking
TIER_LIMITS = {
    1: 50000.0,
    2: 200000.0,
    3: 1000000.0,
    4: 5000000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TVAE hybrid vs pure Monte Carlo gold datasets"
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="Partition key for data versioning (e.g., 2026-08-05)",
    )
    parser.add_argument(
        "--monte-carlo-dir",
        type=str,
        default="data/gold/features",
        help="Directory for Monte Carlo gold data (default: data/gold/features)",
    )
    parser.add_argument(
        "--tvae-dir",
        type=str,
        default="data/gold",
        help="Directory for TVAE hybrid gold data (default: data/gold)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory for comparison reports (default: reports)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Version string for Monte Carlo data (default: v1.0)",
    )
    return parser.parse_args()


class DatasetComparator:
    """Performs comprehensive comparison between TVAE and Monte Carlo datasets."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.report: Dict[str, Any] = {
            "partition": args.partition,
            "timestamp": datetime.now().isoformat(),
            "configuration": vars(args),
        }
        self.monte_carlo_df: Optional[pl.DataFrame] = None
        self.tvae_df: Optional[pl.DataFrame] = None
        self.output_dir: Optional[Path] = None

    def load_datasets(self) -> None:
        """Load both Monte Carlo and TVAE hybrid datasets."""
        logger.info("Loading datasets...")

        # Load Monte Carlo gold data
        monte_carlo_path = (
            Path(self.args.monte_carlo_dir) / 
            self.args.version / 
            f"customer_features_{self.args.partition}.parquet"
        )

        if monte_carlo_path.exists():
            logger.info(f"Loading Monte Carlo data from {monte_carlo_path}")
            self.monte_carlo_df = pl.read_parquet(monte_carlo_path)
            logger.info(f"Loaded {len(self.monte_carlo_df)} rows, {len(self.monte_carlo_df.columns)} columns")
        else:
            logger.warning(f"Monte Carlo data not found at {monte_carlo_path}")
            self.report["monte_carlo_data"] = {"status": "not_found", "path": str(monte_carlo_path)}

        # Load TVAE hybrid gold data
        tvae_path = Path(self.args.tvae_dir) / f"tvae_hybrid_gold_{self.args.partition}.parquet"

        if tvae_path.exists():
            logger.info(f"Loading TVAE hybrid data from {tvae_path}")
            self.tvae_df = pl.read_parquet(tvae_path)
            logger.info(f"Loaded {len(self.tvae_df)} rows, {len(self.tvae_df.columns)} columns")
        else:
            logger.warning(f"TVAE hybrid data not found at {tvae_path}")
            self.report["tvae_hybrid_data"] = {"status": "not_found", "path": str(tvae_path)}

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.args.report_dir) / f"tvae_comparison_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def compare_basic_statistics(self) -> Dict[str, Any]:
        """Compare basic dataset statistics."""
        logger.info("Comparing basic statistics...")

        stats = {
            "monte_carlo": {},
            "tvae_hybrid": {},
        }

        if self.monte_carlo_df is not None:
            stats["monte_carlo"] = {
                "total_rows": len(self.monte_carlo_df),
                "total_columns": len(self.monte_carlo_df.columns),
                "unique_customers": self.monte_carlo_df["customer_id"].n_unique() if "customer_id" in self.monte_carlo_df.columns else 0,
                "columns": list(self.monte_carlo_df.columns),
            }

        if self.tvae_df is not None:
            stats["tvae_hybrid"] = {
                "total_rows": len(self.tvae_df),
                "total_columns": len(self.tvae_df.columns),
                "unique_customers": self.tvae_df["customer_id"].n_unique() if "customer_id" in self.tvae_df.columns else 0,
                "columns": list(self.tvae_df.columns),
            }

            # Add anomaly statistics if available
            if "is_launderer" in self.tvae_df.columns:
                stats["tvae_hybrid"]["launderer_count"] = self.tvae_df.filter(pl.col("is_launderer")).height
                stats["tvae_hybrid"]["launderer_ratio"] = float(
                    self.tvae_df.filter(pl.col("is_launderer")).height / len(self.tvae_df)
                )

            if "aml_scenario" in self.tvae_df.columns:
                stats["tvae_hybrid"]["scenario_distribution"] = (
                    self.tvae_df.group_by("aml_scenario").len().to_dict(as_series=False)
                )

        self.report["basic_statistics"] = stats
        return stats

    def compare_distributions(self) -> Dict[str, Any]:
        """Compare feature distributions using KS tests and visualizations."""
        logger.info("Comparing feature distributions...")

        if self.monte_carlo_df is None or self.tvae_df is None:
            logger.warning("Skipping distribution comparison: one or both datasets not loaded")
            self.report["distribution_comparison"] = {"status": "skipped", "reason": "missing datasets"}
            return {}

        results = {}

        # Compare numerical features
        for feature in NUMERICAL_FEATURES:
            if feature not in self.monte_carlo_df.columns or feature not in self.tvae_df.columns:
                continue

            mc_data = self.monte_carlo_df[feature].drop_nulls().to_numpy()
            tvae_data = self.tvae_df[feature].drop_nulls().to_numpy()

            if len(mc_data) == 0 or len(tvae_data) == 0:
                continue

            # KS test
            ks_statistic, ks_pvalue = stats.ks_2samp(mc_data, tvae_data)

            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(mc_data, tvae_data)

            # Basic statistics
            feature_stats = {
                "monte_carlo": {
                    "mean": float(np.mean(mc_data)),
                    "std": float(np.std(mc_data)),
                    "min": float(np.min(mc_data)),
                    "max": float(np.max(mc_data)),
                    "median": float(np.median(mc_data)),
                },
                "tvae_hybrid": {
                    "mean": float(np.mean(tvae_data)),
                    "std": float(np.std(tvae_data)),
                    "min": float(np.min(tvae_data)),
                    "max": float(np.max(tvae_data)),
                    "median": float(np.median(tvae_data)),
                },
                "ks_test": {
                    "statistic": float(ks_statistic),
                    "p_value": float(ks_pvalue),
                    "significant": ks_pvalue < 0.05,
                },
                "wasserstein_distance": float(wasserstein_dist),
            }

            results[feature] = feature_stats

        # Compare categorical features
        for feature in CATEGORICAL_FEATURES:
            if feature not in self.monte_carlo_df.columns or feature not in self.tvae_df.columns:
                continue

            mc_dist = self.monte_carlo_df.group_by(feature).len().sort("len", descending=True)
            tvae_dist = self.tvae_df.group_by(feature).len().sort("len", descending=True)

            feature_stats = {
                "monte_carlo": mc_dist.to_dict(as_series=False),
                "tvae_hybrid": tvae_dist.to_dict(as_series=False),
            }

            # Chi-square test if categories match
            mc_dict = dict(zip(mc_dist[feature].to_list(), mc_dist["len"].to_list()))
            tvae_dict = dict(zip(tvae_dist[feature].to_list(), tvae_dist["len"].to_list()))

            all_categories = set(mc_dict.keys()) | set(tvae_dict.keys())
            mc_counts = [mc_dict.get(cat, 0) for cat in all_categories]
            tvae_counts = [tvae_dict.get(cat, 0) for cat in all_categories]

            if sum(mc_counts) > 0 and sum(tvae_counts) > 0:
                chi2, chi2_p = stats.chisquare(tvae_counts, f_exp=mc_counts)
                feature_stats["chi_square_test"] = {
                    "statistic": float(chi2),
                    "p_value": float(chi2_p),
                    "significant": chi2_p < 0.05,
                }

            results[feature] = feature_stats

        self.report["distribution_comparison"] = results
        return results

    def plot_distribution_comparison(self) -> None:
        """Generate histogram plots for numerical features."""
        logger.info("Generating distribution comparison plots...")

        if self.monte_carlo_df is None or self.tvae_df is None:
            logger.warning("Skipping plots: one or both datasets not loaded")
            return

        # Convert to pandas for plotting
        mc_pd = self.monte_carlo_df.to_pandas()
        tvae_pd = self.tvae_df.to_pandas()

        # Plot numerical features
        n_features = len(NUMERICAL_FEATURES)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        for idx, feature in enumerate(NUMERICAL_FEATURES):
            if feature not in mc_pd.columns or feature not in tvae_pd.columns:
                continue

            ax = axes[idx]
            mc_data = mc_pd[feature].dropna()
            tvae_data = tvae_pd[feature].dropna()

            ax.hist(mc_data, bins=50, alpha=0.5, label="Monte Carlo", density=True)
            ax.hist(tvae_data, bins=50, alpha=0.5, label="TVAE Hybrid", density=True)
            ax.set_xlabel(feature)
            ax.set_ylabel("Density")
            ax.set_title(f"Distribution: {feature}")
            ax.legend()

        # Hide unused subplots
        for idx in range(len(NUMERICAL_FEATURES), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plot_path = self.output_dir / "distribution_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved distribution comparison plot to {plot_path}")

    def compare_correlation_matrices(self) -> Dict[str, Any]:
        """Compare correlation matrices between datasets."""
        logger.info("Comparing correlation matrices...")

        if self.monte_carlo_df is None or self.tvae_df is None:
            logger.warning("Skipping correlation comparison: one or both datasets not loaded")
            self.report["correlation_comparison"] = {"status": "skipped", "reason": "missing datasets"}
            return {}

        # Get numerical columns present in both datasets
        mc_num_cols = [col for col in NUMERICAL_FEATURES if col in self.monte_carlo_df.columns]
        tvae_num_cols = [col for col in NUMERICAL_FEATURES if col in self.tvae_df.columns]
        common_cols = list(set(mc_num_cols) & set(tvae_num_cols))

        if len(common_cols) < 2:
            logger.warning("Not enough common numerical columns for correlation analysis")
            self.report["correlation_comparison"] = {"status": "skipped", "reason": "insufficient columns"}
            return {}

        # Convert to pandas and compute correlations
        mc_pd = self.monte_carlo_df.select(common_cols).to_pandas()
        tvae_pd = self.tvae_df.select(common_cols).to_pandas()

        mc_corr = mc_pd.corr()
        tvae_corr = tvae_pd.corr()

        # Compute correlation difference
        corr_diff = (mc_corr - tvae_corr).abs()
        mean_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()

        results = {
            "mean_absolute_correlation_difference": float(mean_corr_diff),
            "max_correlation_difference": float(corr_diff.max().max()),
            "monte_carlo_correlation_matrix": mc_corr.to_dict(),
            "tvae_hybrid_correlation_matrix": tvae_corr.to_dict(),
        }

        self.report["correlation_comparison"] = results

        # Plot correlation matrices
        self.plot_correlation_matrices(mc_corr, tvae_corr, corr_diff, common_cols)

        return results

    def plot_correlation_matrices(
        self,
        mc_corr: pd.DataFrame,
        tvae_corr: pd.DataFrame,
        corr_diff: pd.DataFrame,
        columns: List[str],
    ) -> None:
        """Generate correlation matrix comparison plots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Monte Carlo correlation
        sns.heatmap(mc_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
        axes[0].set_title("Monte Carlo Correlation Matrix")

        # TVAE correlation
        sns.heatmap(tvae_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1])
        axes[1].set_title("TVAE Hybrid Correlation Matrix")

        # Correlation difference
        sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap="Reds", ax=axes[2])
        axes[2].set_title("Absolute Correlation Difference")

        plt.tight_layout()
        plot_path = self.output_dir / "correlation_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved correlation comparison plot to {plot_path}")

    def compare_temporal_patterns(self) -> Dict[str, Any]:
        """Compare temporal patterns in transaction data."""
        logger.info("Comparing temporal patterns...")

        if self.monte_carlo_df is None or self.tvae_df is None:
            logger.warning("Skipping temporal comparison: one or both datasets not loaded")
            self.report["temporal_comparison"] = {"status": "skipped", "reason": "missing datasets"}
            return {}

        results = {}

        # Ensure timestamp is datetime
        for df_name, df in [("monte_carlo", self.monte_carlo_df), ("tvae_hybrid", self.tvae_df)]:
            if "timestamp" not in df.columns:
                continue

            df_with_ts = df.with_columns(
                pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False).alias("timestamp")
            )

            # Time range
            time_range = {
                "start": str(df_with_ts["timestamp"].min()),
                "end": str(df_with_ts["timestamp"].max()),
                "duration_days": float(
                    (df_with_ts["timestamp"].max() - df_with_ts["timestamp"].min()).total_seconds() / 86400
                ),
            }

            # Hour of day distribution
            hour_dist = (
                df_with_ts.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
                .group_by("hour")
                .len()
                .sort("hour")
                .to_dict(as_series=False)
            )

            # Day of week distribution
            day_dist = (
                df_with_ts.with_columns(pl.col("timestamp").dt.weekday().alias("day"))
                .group_by("day")
                .len()
                .sort("day")
                .to_dict(as_series=False)
            )

            results[df_name] = {
                "time_range": time_range,
                "hour_distribution": hour_dist,
                "day_distribution": day_dist,
            }

        self.report["temporal_comparison"] = results

        # Plot temporal patterns
        self.plot_temporal_patterns(results)

        return results

    def plot_temporal_patterns(self, results: Dict[str, Any]) -> None:
        """Generate temporal pattern comparison plots."""
        if "monte_carlo" not in results or "tvae_hybrid" not in results:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Hour distribution
        mc_hours = results["monte_carlo"]["hour_distribution"]
        tvae_hours = results["tvae_hybrid"]["hour_distribution"]

        hours_mc = mc_hours["hour"]
        counts_mc = mc_hours["len"]
        hours_tvae = tvae_hours["hour"]
        counts_tvae = tvae_hours["len"]

        axes[0].bar(hours_mc, counts_mc, alpha=0.5, label="Monte Carlo")
        axes[0].bar(hours_tvae, counts_tvae, alpha=0.5, label="TVAE Hybrid")
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel("Transaction Count")
        axes[0].set_title("Hourly Transaction Distribution")
        axes[0].legend()

        # Day distribution
        mc_days = results["monte_carlo"]["day_distribution"]
        tvae_days = results["tvae_hybrid"]["day_distribution"]

        days_mc = mc_days["day"]
        counts_mc = mc_days["len"]
        days_tvae = tvae_days["day"]
        counts_tvae = tvae_days["len"]

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        axes[1].bar(days_mc, counts_mc, alpha=0.5, label="Monte Carlo")
        axes[1].bar(days_tvae, counts_tvae, alpha=0.5, label="TVAE Hybrid")
        axes[1].set_xlabel("Day of Week")
        axes[1].set_ylabel("Transaction Count")
        axes[1].set_title("Day of Week Transaction Distribution")
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(day_labels)
        axes[1].legend()

        plt.tight_layout()
        plot_path = self.output_dir / "temporal_patterns.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved temporal patterns plot to {plot_path}")

    def compare_tier_compliance(self) -> Dict[str, Any]:
        """Compare tier compliance between datasets."""
        logger.info("Comparing tier compliance...")

        results = {}

        for df_name, df in [("monte_carlo", self.monte_carlo_df), ("tvae_hybrid", self.tvae_df)]:
            if df is None or "tier" not in df.columns or "balance" not in df.columns:
                continue

            # Check tier violations
            tier_limits_expr = pl.when(pl.col("tier") == 1).then(50000.0) \
                                .when(pl.col("tier") == 2).then(200000.0) \
                                .when(pl.col("tier") == 3).then(1000000.0) \
                                .otherwise(5000000.0)

            violations = df.with_columns(tier_limits_expr.alias("tier_limit")).filter(
                pl.col("balance") > pl.col("tier_limit")
            )

            tier_dist = df.group_by("tier").len().sort("tier").to_dict(as_series=False)

            results[df_name] = {
                "tier_distribution": tier_dist,
                "tier_violations": len(violations),
                "tier_violation_rate": float(len(violations) / len(df)) if len(df) > 0 else 0.0,
            }

        self.report["tier_compliance"] = results
        return results

    def compare_anomaly_patterns(self) -> Dict[str, Any]:
        """Compare anomaly patterns between datasets."""
        logger.info("Comparing anomaly patterns...")

        if self.tvae_df is None:
            logger.warning("Skipping anomaly comparison: TVAE data not loaded")
            self.report["anomaly_comparison"] = {"status": "skipped", "reason": "missing tvae data"}
            return {}

        results = {}

        if "is_launderer" in self.tvae_df.columns:
            launderer_count = self.tvae_df.filter(pl.col("is_launderer")).height
            results["launderer_count"] = launderer_count
            results["launderer_ratio"] = float(launderer_count / len(self.tvae_df))

        if "aml_scenario" in self.tvae_df.columns:
            scenario_dist = self.tvae_df.group_by("aml_scenario").len().sort("len", descending=True)
            results["scenario_distribution"] = scenario_dist.to_dict(as_series=False)

        # Compare feature distributions for launderers vs non-launderers
        if "is_launderer" in self.tvae_df.columns:
            launderers = self.tvae_df.filter(pl.col("is_launderer"))
            non_launderers = self.tvae_df.filter(~pl.col("is_launderer"))

            results["launderer_feature_stats"] = {}
            for feature in NUMERICAL_FEATURES:
                if feature not in self.tvae_df.columns:
                    continue

                l_data = launderers[feature].drop_nulls()
                nl_data = non_launderers[feature].drop_nulls()

                if len(l_data) > 0 and len(nl_data) > 0:
                    ks_stat, ks_p = stats.ks_2samp(l_data.to_numpy(), nl_data.to_numpy())
                    results["launderer_feature_stats"][feature] = {
                        "launderer_mean": float(l_data.mean()),
                        "non_launderer_mean": float(nl_data.mean()),
                        "ks_statistic": float(ks_stat),
                        "ks_p_value": float(ks_p),
                        "significant": ks_p < 0.05,
                    }

        self.report["anomaly_comparison"] = results
        return results

    def compute_feature_coverage(self) -> Dict[str, Any]:
        """Compute feature coverage and schema compliance."""
        logger.info("Computing feature coverage...")

        results = {}

        for df_name, df in [("monte_carlo", self.monte_carlo_df), ("tvae_hybrid", self.tvae_df)]:
            if df is None:
                continue

            actual_columns = set(df.columns)
            expected_columns = set(GOLD_SCHEMA_COLUMNS)

            missing = expected_columns - actual_columns
            extra = actual_columns - expected_columns

            # Null coverage
            null_coverage = {}
            for col in df.columns:
                null_count = df[col].null_count()
                null_coverage[col] = {
                    "null_count": null_count,
                    "null_ratio": float(null_count / len(df)) if len(df) > 0 else 0.0,
                }

            results[df_name] = {
                "total_columns": len(df.columns),
                "expected_columns": len(expected_columns),
                "missing_columns": list(missing),
                "extra_columns": list(extra),
                "schema_compliant": len(missing) == 0,
                "null_coverage": null_coverage,
            }

        self.report["feature_coverage"] = results
        return results

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for when to use TVAE vs Monte Carlo."""
        logger.info("Generating recommendations...")

        recommendations = {
            "use_tvae_when": [],
            "use_monte_carlo_when": [],
            "key_findings": [],
        }

        # Analyze distribution similarity
        if "distribution_comparison" in self.report:
            dist_comp = self.report["distribution_comparison"]
            significant_diffs = []

            for feature, stats in dist_comp.items():
                if isinstance(stats, dict) and "ks_test" in stats:
                    if stats["ks_test"]["significant"]:
                        significant_diffs.append(feature)

            if len(significant_diffs) > len(NUMERICAL_FEATURES) / 2:
                recommendations["key_findings"].append(
                    f"TVAE shows significant distribution differences in {len(significant_diffs)}/{len(NUMERICAL_FEATURES)} features"
                )
                recommendations["use_monte_carlo_when"].append(
                    "When strict distribution fidelity to Monte Carlo baseline is required"
                )
            else:
                recommendations["key_findings"].append(
                    f"TVAE maintains similar distributions to Monte Carlo baseline ({len(significant_diffs)} significant differences)"
                )
                recommendations["use_tvae_when"].append(
                    "When generating large-scale synthetic data with realistic distributions"
                )

        # Analyze computational efficiency
        if "basic_statistics" in self.report:
            mc_stats = self.report["basic_statistics"].get("monte_carlo", {})
            tvae_stats = self.report["basic_statistics"].get("tvae_hybrid", {})

            if mc_stats.get("total_rows", 0) > 0 and tvae_stats.get("total_rows", 0) > 0:
                recommendations["key_findings"].append(
                    f"Both methods generated comparable datasets: "
                    f"MC={mc_stats['total_rows']:,} rows, TVAE={tvae_stats['total_rows']:,} rows"
                )

        # Analyze anomaly injection
        if "anomaly_comparison" in self.report:
            anomaly_comp = self.report["anomaly_comparison"]
            if "launderer_ratio" in anomaly_comp:
                ratio = anomaly_comp["launderer_ratio"]
                recommendations["key_findings"].append(
                    f"TVAE hybrid successfully injected {ratio:.2%} labeled anomalies"
                )
                recommendations["use_tvae_when"].append(
                    "When labeled anomaly data is required for ML training"
                )

        # Analyze tier compliance
        if "tier_compliance" in self.report:
            tier_comp = self.report["tier_compliance"]
            for df_name, stats in tier_comp.items():
                if stats.get("tier_violation_rate", 0) > 0:
                    recommendations["key_findings"].append(
                        f"{df_name}: {stats['tier_violation_rate']:.2%} tier violation rate"
                    )

        # General recommendations
        recommendations["use_tvae_when"].extend([
            "When scaling to very large dataset sizes (TVAE generation is faster after training)",
            "When capturing complex multi-modal behavioral patterns",
            "When adaptability to new data distributions is needed",
        ])

        recommendations["use_monte_carlo_when"].extend([
            "When complete transparency and interpretability of generation process is required",
            "When computational resources for model training are limited",
            "When deterministic reproducibility is critical",
        ])

        self.report["recommendations"] = recommendations
        return recommendations

    def save_report(self) -> None:
        """Save comprehensive comparison report."""
        report_path = self.output_dir / "comparison_report.json"

        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        logger.info(f"Saved comparison report to {report_path}")

        # Also save a human-readable summary
        summary_path = self.output_dir / "summary.md"
        self._generate_markdown_summary(summary_path)

    def _generate_markdown_summary(self, output_path: Path) -> None:
        """Generate human-readable markdown summary."""
        with open(output_path, "w") as f:
            f.write("# TVAE Hybrid vs Monte Carlo Comparison Report\n\n")
            f.write(f"**Partition**: {self.args.partition}\n")
            f.write(f"**Generated**: {self.report['timestamp']}\n\n")

            # Basic Statistics
            f.write("## Basic Statistics\n\n")
            if "basic_statistics" in self.report:
                for dataset, stats in self.report["basic_statistics"].items():
                    if isinstance(stats, dict) and "total_rows" in stats:
                        f.write(f"### {dataset.replace('_', ' ').title()}\n\n")
                        f.write(f"- **Rows**: {stats['total_rows']:,}\n")
                        f.write(f"- **Columns**: {stats['total_columns']}\n")
                        f.write(f"- **Unique Customers**: {stats['unique_customers']:,}\n\n")

            # Distribution Comparison
            f.write("## Distribution Comparison\n\n")
            if "distribution_comparison" in self.report:
                dist_comp = self.report["distribution_comparison"]
                if isinstance(dist_comp, dict):
                    for feature, stats in list(dist_comp.items())[:5]:  # Show first 5
                        if isinstance(stats, dict) and "ks_test" in stats:
                            f.write(f"### {feature}\n\n")
                            f.write(f"- **KS Statistic**: {stats['ks_test']['statistic']:.4f}\n")
                            f.write(f"- **P-value**: {stats['ks_test']['p_value']:.4f}\n")
                            f.write(f"- **Significant**: {stats['ks_test']['significant']}\n\n")

            # Correlation Comparison
            f.write("## Correlation Comparison\n\n")
            if "correlation_comparison" in self.report:
                corr_comp = self.report["correlation_comparison"]
                if isinstance(corr_comp, dict) and "mean_absolute_correlation_difference" in corr_comp:
                    f.write(f"- **Mean Absolute Correlation Difference**: {corr_comp['mean_absolute_correlation_difference']:.4f}\n")
                    f.write(f"- **Max Correlation Difference**: {corr_comp['max_correlation_difference']:.4f}\n\n")

            # Tier Compliance
            f.write("## Tier Compliance\n\n")
            if "tier_compliance" in self.report:
                tier_comp = self.report["tier_compliance"]
                for dataset, stats in tier_comp.items():
                    if isinstance(stats, dict):
                        f.write(f"### {dataset.replace('_', ' ').title()}\n\n")
                        f.write(f"- **Tier Violations**: {stats['tier_violations']:,}\n")
                        f.write(f"- **Violation Rate**: {stats['tier_violation_rate']:.2%}\n\n")

            # Anomaly Comparison
            f.write("## Anomaly Comparison\n\n")
            if "anomaly_comparison" in self.report:
                anomaly_comp = self.report["anomaly_comparison"]
                if isinstance(anomaly_comp, dict) and "launderer_ratio" in anomaly_comp:
                    f.write(f"- **Launderer Ratio**: {anomaly_comp['launderer_ratio']:.2%}\n")
                    f.write(f"- **Launderer Count**: {anomaly_comp['launderer_count']:,}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if "recommendations" in self.report:
                recs = self.report["recommendations"]
                f.write("### Use TVAE When:\n\n")
                for rec in recs.get("use_tvae_when", []):
                    f.write(f"- {rec}\n")
                f.write("\n### Use Monte Carlo When:\n\n")
                for rec in recs.get("use_monte_carlo_when", []):
                    f.write(f"- {rec}\n")
                f.write("\n### Key Findings:\n\n")
                for finding in recs.get("key_findings", []):
                    f.write(f"- {finding}\n")

        logger.info(f"Saved markdown summary to {output_path}")

    def run(self) -> None:
        """Run complete comparison analysis."""
        logger.info("=" * 60)
        logger.info("Starting TVAE vs Monte Carlo Comparison")
        logger.info(f"Partition: {self.args.partition}")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            self.load_datasets()
            self.compare_basic_statistics()
            self.compare_distributions()
            self.plot_distribution_comparison()
            self.compare_correlation_matrices()
            self.compare_temporal_patterns()
            self.compare_tier_compliance()
            self.compare_anomaly_patterns()
            self.compute_feature_coverage()
            self.generate_recommendations()
            self.save_report()

            duration = time.time() - start_time
            self.report["total_duration_seconds"] = duration

            logger.info("=" * 60)
            logger.info(f"Comparison completed in {duration:.2f}s")
            logger.info(f"Report saved to {self.output_dir}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise


def main() -> None:
    args = parse_args()
    comparator = DatasetComparator(args)
    comparator.run()


if __name__ == "__main__":
    main()
