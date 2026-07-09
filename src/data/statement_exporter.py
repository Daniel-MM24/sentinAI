"""
Dual-Layer Statement Export Engine

Transforms behavioral transaction data into M-PESA-style statement exports:
- detailed_ledgers.csv: time-series ledger with receipt numbers and running balances
- summary_statements.csv: per-user aggregated totals by transaction type
"""

from __future__ import annotations

import logging
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TRANSACTION_TYPES = [
    "Send Money",
    "Received Money",
    "Agent Deposit",
    "Agent Withdrawal",
    "Lipa Na M-PESA (Paybill)",
    "Lipa Na M-PESA (Buy Goods)",
    "Others",
]

INFLOW_TYPES = {"Received Money", "Agent Deposit", "Others"}
OUTFLOW_TYPES = {
    "Send Money",
    "Agent Withdrawal",
    "Lipa Na M-PESA (Paybill)",
    "Lipa Na M-PESA (Buy Goods)",
    "Others",
}

PAYBILL_CONTEXTS = [
    "School Fees",
    "Rent",
    "Electricity",
    "Water Bill",
    "Insurance",
    "TV Subscription",
    "Loan Repayment",
]


class StatementExportConfig(BaseModel):
    seed: int = Field(default=42, description="Random seed for receipt numbers and status injection")
    error_injection_rate: float = Field(
        default=0.005, description="Fraction of transactions marked Failed/Pending"
    )
    input_path: str = Field(default="data/detailed_transactions.csv")
    detailed_output_path: str = Field(default="data/detailed_ledgers.csv")
    summary_output_path: str = Field(default="data/summary_statements.csv")


class StatementExporter:
    """Exports dual-layer M-PESA statement files from behavioral transaction data."""

    def __init__(self, config: Optional[StatementExportConfig] = None):
        self.config = config or StatementExportConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._used_receipts: set[str] = set()

    def _generate_receipt_number(self) -> str:
        letters = string.ascii_uppercase
        while True:
            letter = str(self.rng.choice(list(letters)))
            digits = f"{self.rng.integers(0, 1_000_000):06d}"
            receipt = f"RX{letter}{digits}"
            if receipt not in self._used_receipts:
                self._used_receipts.add(receipt)
                return receipt

    def _inject_status(self, n: int) -> List[str]:
        statuses = ["Completed"] * n
        error_count = int(n * self.config.error_injection_rate)
        if error_count > 0:
            indices = self.rng.choice(n, size=error_count, replace=False)
            error_statuses = self.rng.choice(
                ["Failed", "Pending", "Reversed"],
                size=error_count,
                p=[0.5, 0.35, 0.15],
            )
            for idx, status in zip(indices, error_statuses):
                statuses[int(idx)] = str(status)
        return statuses

    def _format_counterparty_name(self, counterparty: str) -> str:
        if counterparty.startswith("User-"):
            return f"John {counterparty.split('-', 1)[1][-4:]}"
        if counterparty.startswith("Merchant-"):
            return f"Shop {counterparty.split('-', 1)[1]}"
        if counterparty.startswith("Paybill-"):
            return counterparty
        if counterparty.startswith("Agent-"):
            return f"Agent {counterparty.split('-', 1)[1]}"
        return counterparty

    def _build_details(
        self,
        transaction_type: str,
        counterparty: str,
        direction: str,
        is_betting: bool,
        is_international: bool,
        completion_time: datetime,
    ) -> str:
        name = self._format_counterparty_name(counterparty)

        if is_betting:
            return f"Send Money to {name} - Betting"
        if is_international:
            return f"International Transfer via {name}"

        if transaction_type == "Send Money":
            return f"Send Money to {name}"
        if transaction_type == "Received Money":
            return f"Received Money from {name}"
        if transaction_type == "Agent Deposit":
            return f"Agent Deposit from {name}"
        if transaction_type == "Agent Withdrawal":
            return f"Agent Withdrawal at {name}"
        if transaction_type == "Lipa Na M-PESA (Paybill)":
            context = str(self.rng.choice(PAYBILL_CONTEXTS))
            if completion_time.month in {1, 5, 9}:
                context = "School Fees"
            return f"Lipa Paybill - {context}"
        if transaction_type == "Lipa Na M-PESA (Buy Goods)":
            return f"Lipa Buy Goods - {name}"
        if transaction_type == "Others":
            if direction == "inflow":
                return f"{counterparty} - Credit"
            return f"{counterparty} - Debit"
        return f"{transaction_type} - {name}"

    def _resolve_paid_columns(
        self,
        transaction_type: str,
        direction: str,
        amount: float,
    ) -> Tuple[float, float]:
        if direction == "inflow":
            return amount, 0.0
        if direction == "outflow":
            return 0.0, amount
        if transaction_type in INFLOW_TYPES and transaction_type not in OUTFLOW_TYPES:
            return amount, 0.0
        if transaction_type in OUTFLOW_TYPES and transaction_type not in INFLOW_TYPES:
            return 0.0, amount
        if self.rng.random() < 0.5:
            return amount, 0.0
        return 0.0, amount

    def _compute_velocities(
        self,
        user_ids: List[str],
        timestamps: List[datetime],
    ) -> Tuple[List[int], List[int]]:
        velocity_1hr: List[int] = []
        velocity_24hr: List[int] = []
        by_user: Dict[str, List[datetime]] = {}
        for user_id, ts in zip(user_ids, timestamps):
            by_user.setdefault(user_id, []).append(ts)

        user_indices: Dict[str, int] = {uid: 0 for uid in by_user}
        user_times_sorted = {uid: sorted(times) for uid, times in by_user.items()}

        for user_id, ts in zip(user_ids, timestamps):
            times = user_times_sorted[user_id]
            idx = user_indices[user_id]
            window_start_1hr = ts - timedelta(hours=1)
            window_start_24hr = ts - timedelta(hours=24)
            count_1hr = sum(1 for t in times[: idx + 1] if window_start_1hr < t <= ts)
            count_24hr = sum(1 for t in times[: idx + 1] if window_start_24hr < t <= ts)
            velocity_1hr.append(count_1hr)
            velocity_24hr.append(count_24hr)
            user_indices[user_id] += 1

        return velocity_1hr, velocity_24hr

    def _reconcile_balances(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure per-user running balance continuity in ledger order."""
        reconciled: List[Dict] = []
        for user_id, group in df.sort("completion_time").group_by("user_id"):
            running = None
            for row in group.iter_rows(named=True):
                paid_in = float(row["paid_in"])
                paid_out = float(row["paid_out"])
                if running is None:
                    running = float(row["balance"]) - paid_in + paid_out
                running = round(running + paid_in - paid_out, 2)
                row["balance"] = running
                reconciled.append(row)
        return pl.DataFrame(reconciled).sort("completion_time")

    def export_from_dataframe(self, transactions: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        required = {
            "transaction_id",
            "customer_id",
            "counterparty",
            "transaction_type",
            "amount",
            "direction",
            "timestamp",
            "balance_after",
            "is_betting",
            "is_international",
            "is_kadogo",
        }
        missing = required - set(transactions.columns)
        if missing:
            raise ValueError(f"Input transactions missing columns: {sorted(missing)}")

        tx = transactions.with_columns(
            pl.col("timestamp")
            .str.to_datetime(time_zone="UTC")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .alias("completion_time_raw")
        ).sort("timestamp")

        n = len(tx)
        receipt_numbers = [self._generate_receipt_number() for _ in range(n)]
        statuses = self._inject_status(n)

        ledger_rows: List[Dict] = []
        completion_times: List[datetime] = []
        user_ids: List[str] = []

        for i, row in enumerate(tx.iter_rows(named=True)):
            amount = float(row["amount"])
            tx_type = str(row["transaction_type"])
            direction = str(row["direction"])
            paid_in, paid_out = self._resolve_paid_columns(tx_type, direction, amount)
            completion_dt = datetime.fromisoformat(
                str(row["timestamp"]).replace("Z", "+00:00")
            )
            completion_str = completion_dt.strftime("%Y-%m-%d %H:%M:%S")
            completion_times.append(completion_dt)
            user_id = str(row["customer_id"])
            user_ids.append(user_id)

            ledger_rows.append(
                {
                    "statement_id": f"STMT_{user_id}",
                    "user_id": user_id,
                    "transaction_id": row["transaction_id"],
                    "receipt_number": receipt_numbers[i],
                    "completion_time": completion_str,
                    "details": self._build_details(
                        tx_type,
                        str(row["counterparty"]),
                        direction,
                        bool(row["is_betting"]),
                        bool(row["is_international"]),
                        completion_dt,
                    ),
                    "transaction_type": tx_type,
                    "transaction_status": statuses[i],
                    "paid_in": round(paid_in, 2),
                    "paid_out": round(paid_out, 2),
                    "balance": float(row["balance_after"]),
                    "is_betting": bool(row["is_betting"]),
                    "is_international": bool(row["is_international"]),
                    "is_kadogo": bool(row["is_kadogo"]),
                }
            )

        velocity_1hr, velocity_24hr = self._compute_velocities(user_ids, completion_times)
        for idx, row in enumerate(ledger_rows):
            row["velocity_1hr"] = velocity_1hr[idx]
            row["velocity_24hr"] = velocity_24hr[idx]

        detailed = self._reconcile_balances(pl.DataFrame(ledger_rows))
        summary = self._build_summary(detailed)
        return detailed, summary

    def _type_column_prefix(self, tx_type: str) -> str:
        return (
            tx_type.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "")
            .replace("__", "_")
        )

    def _build_summary(self, detailed: pl.DataFrame) -> pl.DataFrame:
        summaries: List[Dict] = []

        for (user_id, statement_id), group in detailed.group_by(["user_id", "statement_id"]):
            row: Dict = {
                "statement_id": statement_id,
                "user_id": user_id,
                "total_transaction_count": len(group),
                "total_paid_in": round(float(group["paid_in"].sum()), 2),
                "total_paid_out": round(float(group["paid_out"].sum()), 2),
                "closing_balance": round(float(group.sort("completion_time")["balance"][-1]), 2),
            }

            for tx_type in TRANSACTION_TYPES:
                prefix = self._type_column_prefix(tx_type)
                subset = group.filter(pl.col("transaction_type") == tx_type)
                row[f"{prefix}_count"] = len(subset)
                row[f"{prefix}_paid_in"] = round(float(subset["paid_in"].sum()), 2)
                row[f"{prefix}_paid_out"] = round(float(subset["paid_out"].sum()), 2)

            summaries.append(row)

        summary_df = pl.DataFrame(summaries).sort("user_id")

        type_cols: List[str] = []
        for tx_type in TRANSACTION_TYPES:
            prefix = self._type_column_prefix(tx_type)
            type_cols.extend([f"{prefix}_count", f"{prefix}_paid_in", f"{prefix}_paid_out"])

        ordered = [
            "statement_id",
            "user_id",
            "total_transaction_count",
            "total_paid_in",
            "total_paid_out",
            "closing_balance",
            *type_cols,
        ]
        return summary_df.select([c for c in ordered if c in summary_df.columns])

    def export(
        self,
        input_path: Optional[str] = None,
        detailed_output_path: Optional[str] = None,
        summary_output_path: Optional[str] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        input_path = input_path or self.config.input_path
        detailed_output_path = detailed_output_path or self.config.detailed_output_path
        summary_output_path = summary_output_path or self.config.summary_output_path

        logger.info("Loading transactions from %s", input_path)
        transactions = pl.read_csv(input_path)

        detailed, summary = self.export_from_dataframe(transactions)

        Path(detailed_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_output_path).parent.mkdir(parents=True, exist_ok=True)
        detailed.write_csv(detailed_output_path)
        summary.write_csv(summary_output_path)

        logger.info("Exported %s ledger rows to %s", len(detailed), detailed_output_path)
        logger.info("Exported %s summary statements to %s", len(summary), summary_output_path)
        self._log_export_stats(detailed, summary)
        return detailed, summary

    def _log_export_stats(self, detailed: pl.DataFrame, summary: pl.DataFrame) -> None:
        status_counts = detailed.group_by("transaction_status").len()
        logger.info("Transaction status distribution:")
        for status, count in status_counts.iter_rows():
            logger.info("  %s: %s", status, count)

        logger.info(
            "Monitoring flags — betting: %s, international: %s, kadogo: %s",
            detailed.filter(pl.col("is_betting")).height,
            detailed.filter(pl.col("is_international")).height,
            detailed.filter(pl.col("is_kadogo")).height,
        )
        logger.info(
            "Summary totals — users: %s, avg tx/user: %.1f",
            len(summary),
            summary["total_transaction_count"].mean() if len(summary) else 0,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    exporter = StatementExporter()
    detailed, summary = exporter.export()
    print(f"Detailed ledger: {len(detailed)} rows")
    print(detailed.head(5))
    print(f"\nSummary statements: {len(summary)} rows")
    print(summary.head(3))


if __name__ == "__main__":
    main()
