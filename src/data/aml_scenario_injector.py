"""
AML Scenario Injector — Behavioral Anomaly Injection for Ground-Truth Labeling.

Injects four known money-laundering typologies into a subset of customers (2% of
population) by overlaying scenario-specific transaction patterns on top of clean
M-PESA transaction histories.  Outputs an ``aml_ground_truth.csv`` with per-customer
ground-truth labels used for supervised model training and validation.

Scenario populations (as a share of total launderers):
    - Smurfing / Structuring    40 %
    - Layering                  30 %
    - Mule Account              20 %
    - Circular Trading          10 %

Reference: obsidian_notes/technical/AML_SCENARIOS.md
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSACTION_PATH = PROJECT_ROOT / "data" / "detailed_transactions.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "aml_ground_truth.csv"
CUSTOMER_PROFILES_PATH = (
    PROJECT_ROOT / "data" / "bronze" / "customers" / "customer_profiles_complete.csv"
)

# CBK / regulatory thresholds
STRUCTURING_THRESHOLD_KES = 100_000.0  # Transactions below this are "small"
CTR_THRESHOLD_KES = 1_000_000.0
KADOGO_THRESHOLD_KES = 100.0

# Scenario population shares (must sum to 1.0)
SCENARIO_SHARES: dict[str, float] = {
    "smurfing": 0.40,
    "layering": 0.30,
    "mule_account": 0.20,
    "circular_trading": 0.10,
}

# ---------------------------------------------------------------------------
# Scenario enums & configuration
# ---------------------------------------------------------------------------


class AMLScenario(str, Enum):
    SMURFING = "smurfing"
    LAYERING = "layering"
    MULE_ACCOUNT = "mule_account"
    CIRCULAR_TRADING = "circular_trading"
    NONE = "none"


@dataclass
class ScenarioInjectorConfig:
    """Configuration for the AML scenario injector."""

    launderer_fraction: float = 0.02  # 2 % of population
    seed: int = 42

    # Smurfing
    smurfing_max_amount: float = STRUCTURING_THRESHOLD_KES
    smurfing_min_tx_per_launderer: int = 30
    smurfing_target_counterparties_range: tuple[int, int] = (5, 15)

    # Layering
    layering_min_layers: int = 4
    layering_window_hours: int = 24
    layering_tx_per_cycle: tuple[int, int] = (2, 5)

    # Mule account
    mule_withdraw_pct: float = 0.80  # 80 %+ withdrawn within 24 h
    mule_min_receive_tx: int = 5

    # Circular trading
    circular_min_accounts: int = 3
    circular_max_accounts: int = 5
    circular_min_iterations: int = 5
    circular_max_iterations: int = 10

    # Output
    output_path: Path = OUTPUT_PATH


# ---------------------------------------------------------------------------
# Core injector
# ---------------------------------------------------------------------------


class AMLScenarioInjector:
    """Injects AML scenario patterns into clean M-PESA transaction data.

    The injector operates in two phases:

    Phase 1 — Selection
        Randomly selects 2 % of customers as launderers, then assigns each
        a scenario according to the configured scenario shares.

    Phase 2 — Injection
        For each launderer, generates additional scenario-specific transactions
        that overlay the clean transaction history.  Non-launderers are left
        untouched.

    The output is a customer-level ground-truth CSV with columns:
        user_id, is_launderer, aml_scenario
    """

    def __init__(self, config: Optional[ScenarioInjectorConfig] = None):
        self.config = config or ScenarioInjectorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._launderer_map: dict[str, str] = {}  # customer_id -> scenario

    # ------------------------------------------------------------------
    # Phase 1: selection
    # ------------------------------------------------------------------

    def _select_launderers(self, customer_ids: list[str]) -> dict[str, str]:
        """Select 2 % of customers and assign AML scenarios.

        Returns
            Dict mapping customer_id -> scenario name.
        """
        total = len(customer_ids)
        n_launderers = max(1, round(total * self.config.launderer_fraction))
        selected = self._rng.choice(customer_ids, size=n_launderers, replace=False).tolist()

        # Assign scenarios according to configured shares
        scenarios = list(SCENARIO_SHARES.keys())
        weights = [SCENARIO_SHARES[s] for s in scenarios]
        # Ensure exact count per scenario
        n_per = (np.array(weights) * n_launderers).astype(int)
        diff = n_launderers - n_per.sum()
        if diff > 0:
            n_per[:diff] += 1  # give extras to first scenarios

        assignments: list[str] = []
        for sc, cnt in zip(scenarios, n_per):
            assignments.extend([sc] * cnt)
        # Trim in case rounding overshoots
        assignments = assignments[:n_launderers]
        self._rng.shuffle(assignments)

        mapping: dict[str, str] = {}
        for cid, sc in zip(selected, assignments):
            mapping[cid] = sc

        logger.info(
            "Selected %d launderers (%.1f%% of %d): %s",
            n_launderers,
            self.config.launderer_fraction * 100,
            total,
            {s: sum(1 for v in mapping.values() if v == s) for s in scenarios},
        )
        return mapping

    # ------------------------------------------------------------------
    # Phase 2: pattern generators
    # ------------------------------------------------------------------

    def _inject_smurfing(
        self,
        customer_id: str,
        clean_txs: pl.DataFrame,
        all_customers: list[str],
        FY25_START: datetime,
        FY25_END: datetime,
    ) -> pl.DataFrame:
        """Generate many small Send Money transactions to many counterparties.

        Each transaction is kept under KES 100,000 (the CBK reporting
        threshold), funds are split across multiple counterparties, and
        values are varied to avoid obvious pattern detection.
        """
        n_tx = self.config.smurfing_min_tx_per_launderer + int(
            self._rng.exponential(20)
        )
        n_cp = self._rng.integers(*self.config.smurfing_target_counterparties_range)
        counterparties = self._rng.choice(
            [c for c in all_customers if c != customer_id],
            size=min(n_cp, len(all_customers) - 1),
            replace=False,
        ).tolist()

        rows: list[dict[str, Any]] = []
        # Spread transactions across time window of clean data
        clean_start = clean_txs["timestamp"].min()
        clean_end = clean_txs["timestamp"].max()
        span = (clean_end - clean_start).total_seconds()

        for i in range(n_tx):
            amount = self._rng.uniform(KADOGO_THRESHOLD_KES, self.config.smurfing_max_amount * 0.95)
            cp = counterparties[i % len(counterparties)]
            # Jitter timestamp across the customer's active period
            offset = self._rng.uniform(0, span)
            ts = clean_start + timedelta(seconds=offset)
            rows.append(
                self._make_tx_row(
                    customer_id=customer_id,
                    counterparty=cp,
                    transaction_type="Send Money",
                    amount=round(amount, 2),
                    direction="outflow",
                    timestamp=ts,
                )
            )

        logger.debug("Smurfing: %s → %d tx to %d counterparties", customer_id, n_tx, len(counterparties))
        return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    def _inject_layering(
        self,
        customer_id: str,
        clean_txs: pl.DataFrame,
        all_customers: list[str],
        FY25_START: datetime,
        FY25_END: datetime,
    ) -> pl.DataFrame:
        """Create rapid transaction chains through 4+ accounts within 24 h windows.

        Money flows: launderer -> layer_1 -> layer_2 -> ... -> layer_N -> sink.
        Each hop happens within 24 hours of the previous one.
        """
        n_layers = max(self.config.layering_min_layers, int(self._rng.exponential(2)) + 4)
        available = [c for c in all_customers if c != customer_id]
        layers = self._rng.choice(available, size=min(n_layers, len(available)), replace=False).tolist()
        # One more as sink (could re-use the last layer as sink)
        sink = self._rng.choice([c for c in available if c not in layers])
        chain = [customer_id] + layers + [sink]

        # Determine number of cycles
        n_cycles = self._rng.integers(*self.config.layering_tx_per_cycle)
        rows: list[dict[str, Any]] = []

        # Base time: pick a window from the customer's clean data
        clean_end = clean_txs["timestamp"].max()
        window_start = clean_txs["timestamp"].min()

        for cycle in range(n_cycles):
            # Each cycle is a 24 h burst
            burst_start = window_start + timedelta(
                hours=self._rng.uniform(0, (clean_end - window_start).total_seconds() / 3600)
            )
            for hop in range(len(chain) - 1):
                sender = chain[hop]
                receiver = chain[hop + 1]
                amount = self._rng.lognormal(mean=8.0, sigma=1.0)  # KES ~3k-30k
                ts = burst_start + timedelta(hours=hop * 2)  # 2 h between hops
                rows.append(
                    self._make_tx_row(
                        customer_id=sender,
                        counterparty=receiver,
                        transaction_type="Send Money",
                        amount=round(amount, 2),
                        direction="outflow",
                        timestamp=ts,
                    )
                )
                rows.append(
                    self._make_tx_row(
                        customer_id=receiver,
                        counterparty=sender,
                        transaction_type="Received Money",
                        amount=round(amount, 2),
                        direction="inflow",
                        timestamp=ts + timedelta(minutes=1),
                    )
                )

        logger.debug("Layering: %s → %d layers, %d cycles", customer_id, n_layers, n_cycles)
        return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    def _inject_mule_account(
        self,
        customer_id: str,
        clean_txs: pl.DataFrame,
        all_customers: list[str],
        FY25_START: datetime,
        FY25_END: datetime,
    ) -> pl.DataFrame:
        """High Received Money, immediate Agent Withdrawal of 80 %+ within 24 h.

        Each receive-withdraw pair forms a single cycle.
        """
        n_cycles = self.config.mule_min_receive_tx + int(self._rng.exponential(5))
        senders = self._rng.choice(
            [c for c in all_customers if c != customer_id],
            size=min(n_cycles, len(all_customers) - 1),
            replace=False,
        ).tolist()

        clean_end = clean_txs["timestamp"].max()
        clean_start = clean_txs["timestamp"].min()
        span = (clean_end - clean_start).total_seconds()
        rows: list[dict[str, Any]] = []

        for i in range(n_cycles):
            receive_amount = self._rng.lognormal(mean=9.0, sigma=0.8)  # KES ~8k-40k
            receive_ts = clean_start + timedelta(seconds=self._rng.uniform(0, span))

            # Withdraw 80-100 % within minutes to 24 h
            withdraw_pct = self._rng.uniform(self.config.mule_withdraw_pct, 1.0)
            withdraw_amount = round(receive_amount * withdraw_pct, 2)
            # Small delay (minutes to a few hours)
            delay_hours = self._rng.exponential(4)
            withdraw_ts = receive_ts + timedelta(hours=min(delay_hours, 24))

            rows.append(
                self._make_tx_row(
                    customer_id=customer_id,
                    counterparty=senders[i],
                    transaction_type="Received Money",
                    amount=round(receive_amount, 2),
                    direction="inflow",
                    timestamp=receive_ts,
                )
            )
            rows.append(
                self._make_tx_row(
                    customer_id=customer_id,
                    counterparty="Agent-Withdrawal",
                    transaction_type="Agent Withdrawal",
                    amount=withdraw_amount,
                    direction="outflow",
                    timestamp=withdraw_ts,
                )
            )

        logger.debug("Mule account: %s → %d receive-withdraw cycles", customer_id, n_cycles)
        return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    def _inject_circular_trading(
        self,
        customer_id: str,
        clean_txs: pl.DataFrame,
        all_customers: list[str],
        FY25_START: datetime,
        FY25_END: datetime,
    ) -> pl.DataFrame:
        """Money circulates among 3–5 accounts in 5–10 iterations.

        Cycle: A -> B -> C -> A.  Amounts are varied to avoid detection.
        """
        n_accounts = self._rng.integers(
            self.config.circular_min_accounts, self.config.circular_max_accounts + 1
        )
        n_iterations = self._rng.integers(
            self.config.circular_min_iterations, self.config.circular_max_iterations + 1
        )

        available = [c for c in all_customers if c != customer_id]
        accounts = [customer_id] + self._rng.choice(
            available, size=min(n_accounts - 1, len(available)), replace=False
        ).tolist()
        # Ensure at least 3 accounts in the cycle
        if len(accounts) < 3:
            accounts = [customer_id] + self._rng.choice(
                available, size=2, replace=False
            ).tolist()

        clean_end = clean_txs["timestamp"].max()
        clean_start = clean_txs["timestamp"].min()
        span = (clean_end - clean_start).total_seconds()
        rows: list[dict[str, Any]] = []

        for iteration in range(n_iterations):
            # Each iteration routes money around the cycle
            base_ts = clean_start + timedelta(
                seconds=self._rng.uniform(0, span)
            )
            amount = self._rng.lognormal(mean=8.5, sigma=0.9)  # Vary amounts
            for idx in range(len(accounts)):
                sender = accounts[idx]
                receiver = accounts[(idx + 1) % len(accounts)]
                ts = base_ts + timedelta(hours=idx)
                rows.append(
                    self._make_tx_row(
                        customer_id=sender,
                        counterparty=receiver,
                        transaction_type="Send Money",
                        amount=round(amount * self._rng.uniform(0.8, 1.2), 2),
                        direction="outflow",
                        timestamp=ts,
                    )
                )
                rows.append(
                    self._make_tx_row(
                        customer_id=receiver,
                        counterparty=sender,
                        transaction_type="Received Money",
                        amount=round(amount * self._rng.uniform(0.8, 1.2), 2),
                        direction="inflow",
                        timestamp=ts + timedelta(minutes=1),
                    )
                )

        logger.debug(
            "Circular trading: %s → %d accounts, %d iterations",
            customer_id,
            len(accounts),
            n_iterations,
        )
        return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tx_row(
        customer_id: str,
        counterparty: str,
        transaction_type: str,
        amount: float,
        direction: str,
        timestamp: datetime,
    ) -> dict[str, Any]:
        """Build a single transaction row dict matching the CSV schema."""
        paid_in = amount if direction == "inflow" else 0.0
        paid_out = amount if direction == "outflow" else 0.0
        return {
            "customer_id": customer_id,
            "counterparty": counterparty,
            "transaction_type": transaction_type,
            "amount": amount,
            "direction": direction,
            "timestamp": timestamp,
            "paid_in": paid_in,
            "paid_out": paid_out,
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def inject(
        self, transactions_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Run the full AML scenario injection pipeline.

        Parameters
        ----------
        transactions_df : pl.DataFrame, optional
            Clean transaction data.  If None, loads from the default CSV path.

        Returns
        -------
        pl.DataFrame
            Customer-level ground-truth labels with columns:
            user_id, is_launderer, aml_scenario
        """
        # --- load data ---
        if transactions_df is None:
            transactions_df = pl.read_csv(TRANSACTION_PATH, try_parse_dates=True)

        customer_ids = transactions_df["customer_id"].unique().to_list()
        all_customers = customer_ids  # used for counterparty selection

        # --- Phase 1: select launderers ---
        self._launderer_map = self._select_launderers(customer_ids)
        launderer_ids = set(self._launderer_map.keys())

        FY25_START = datetime(2024, 7, 1, tzinfo=timezone.utc)
        FY25_END = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

        # --- Phase 2: inject per scenario ---
        injected_txs: list[pl.DataFrame] = []
        scenario_injectors = {
            "smurfing": self._inject_smurfing,
            "layering": self._inject_layering,
            "mule_account": self._inject_mule_account,
            "circular_trading": self._inject_circular_trading,
        }

        for cid, scenario in self._launderer_map.items():
            customer_txs = transactions_df.filter(pl.col("customer_id") == cid)
            injector_fn = scenario_injectors.get(scenario)
            if injector_fn is None:
                logger.warning("Unknown scenario '%s' for customer %s", scenario, cid)
                continue
            new_txs = injector_fn(
                cid, customer_txs, all_customers, FY25_START, FY25_END
            )
            injected_txs.append(new_txs)

        # --- build ground-truth records ---
        records: list[dict[str, Any]] = []
        for cid in customer_ids:
            scenario = self._launderer_map.get(cid, "none")
            records.append(
                {
                    "user_id": cid,
                    "is_launderer": cid in launderer_ids,
                    "aml_scenario": scenario,
                }
            )

        gt_df = pl.DataFrame(records).with_columns(
            pl.col("is_launderer").cast(pl.Boolean),
            pl.col("aml_scenario").cast(pl.String),
        )

        # --- log summary ---
        summary = (
            gt_df.group_by("aml_scenario")
            .agg(pl.len().alias("count"))
            .sort("aml_scenario")
        )
        logger.info("Ground-truth label distribution:\n%s", summary)

        return gt_df

    def save_ground_truth(
        self, gt_df: pl.DataFrame, output_path: Optional[Path] = None
    ) -> Path:
        """Persist ground-truth labels to CSV."""
        path = output_path or self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        gt_df.write_csv(path)
        logger.info("Ground truth saved to %s", path)
        return path

    def get_launderer_map(self) -> dict[str, str]:
        """Return the customer_id -> scenario mapping for inspection."""
        return dict(self._launderer_map)

    def generate_summary(self, gt_df: pl.DataFrame) -> dict[str, Any]:
        """Generate summary statistics for the ground-truth dataset."""
        total = len(gt_df)
        launderers = gt_df.filter(pl.col("is_launderer"))
        scenario_counts = (
            gt_df.filter(pl.col("is_launderer"))
            .group_by("aml_scenario")
            .agg(pl.len().alias("count"))
            .sort("aml_scenario")
        )
        return {
            "total_customers": total,
            "total_launderers": len(launderers),
            "launderer_pct": round(len(launderers) / total * 100, 2),
            "scenario_counts": {
                row["aml_scenario"]: row["count"]
                for row in scenario_counts.iter_rows(named=True)
            },
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_injection(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the AML scenario injection pipeline and persist results.

    Parameters
    ----------
    input_path : Path, optional
        Path to clean transaction CSV.  Defaults to ``data/detailed_transactions.csv``.
    output_path : Path, optional
        Output path for ground-truth CSV.  Defaults to ``data/aml_ground_truth.csv``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``ground_truth``, ``summary``, ``output_path``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=== AML Scenario Injection ===")

    config = ScenarioInjectorConfig(seed=seed)
    if output_path is not None:
        config.output_path = output_path

    injector = AMLScenarioInjector(config)

    tx_path = input_path or TRANSACTION_PATH
    logger.info("Loading transactions from %s", tx_path)
    txs = pl.read_csv(tx_path, try_parse_dates=True)
    logger.info("Loaded %d transactions for %d customers", len(txs), txs["customer_id"].n_unique())

    gt_df = injector.inject(txs)
    out = injector.save_ground_truth(gt_df, output_path)
    summary = injector.generate_summary(gt_df)

    logger.info("=== Summary ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    return {"ground_truth": gt_df, "summary": summary, "output_path": out}


def main():
    """Entry point when run as a script."""
    result = run_injection()
    print("\n=== AML Scenario Injection Complete ===")
    print(f"Output: {result['output_path']}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
