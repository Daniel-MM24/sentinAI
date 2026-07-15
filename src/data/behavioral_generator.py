"""
Behavioral Transaction Generation Engine

Generates M-PESA-like transaction histories with:
- Transaction type distribution matching actual statement structure
- High-risk entity flagging (betting, international transfers)
- Per-archetype log-normal value generation with Kadogo thresholds
- Temporal cyclicality via inhomogeneous Poisson processes (168h intensity)
- Balance constraint enforcement with tier-specific velocity caps
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from src.data.temporal_model import (
    FY25_START,
    FY25_END,
    compute_intensity,
    TRUE_MAX_MULTIPLIER,
    MEAN_INTENSITY_ESTIMATE,
)


class TransactionType(str, Enum):
    SEND_MONEY = "Send Money"
    RECEIVED_MONEY = "Received Money"
    AGENT_DEPOSIT = "Agent Deposit"
    AGENT_WITHDRAWAL = "Agent Withdrawal"
    PAYBILL = "Lipa Na M-PESA (Paybill)"
    BUY_GOODS = "Lipa Na M-PESA (Buy Goods)"
    OTHERS = "Others"


class FlowDirection(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    BIDIRECTIONAL = "bidirectional"


@dataclass(frozen=True)
class TransactionTypeConfig:
    probability: float
    direction: FlowDirection
    description: str


@dataclass
class CustomerState:
    customer_id: str
    tier: int
    tier_cap: float
    balance_cap: float
    balance: float
    betting_flag: bool
    international_flag: bool
    archetype: str = "retail_standard"
    daily_outflow_total: float = 0.0
    last_daily_reset: Optional[datetime] = None
    opening_balance: float = 0.0
    balance_max: float = 0.0
    balance_min: float = 0.0
    balance_sum: float = 0.0
    balance_count: int = 0


# ---------------------------------------------------------------------------
# Per-archetype log-normal params for transaction values
# ---------------------------------------------------------------------------
# Tier-specific daily velocity caps (KES) — CBK PG/43 cumulative daily limits per tier
_DAILY_VELOCITY_CAPS: dict[int, float] = {
    1: 25_000.0,       # CBK Tier 1: KES 25,000 daily max
    2: 100_000.0,      # CBK Tier 2: KES 100,000 daily max
    3: 500_000.0,      # CBK Tier 3: KES 500,000 daily max
    4: 10_000_000.0,   # CBK Tier 4 (EDD): no regulatory cap — set high as soft boundary
}

# Per-archetype log-normal params
_ARCHETYPE_AMOUNT_PARAMS: dict[str, dict[str, float]] = {
    "retail_heavy":     {"mean": 7.2, "sigma": 1.3},
    "retail_standard":  {"mean": 6.5, "sigma": 1.2},
    "micro_merchant":   {"mean": 8.0, "sigma": 1.4},
    "corporate":        {"mean": 9.5, "sigma": 1.6},
}


class BehavioralGeneratorConfig(BaseModel):
    """Configuration for the behavioral transaction generator.

    CBK-aligned tier limits (PG/43 guidelines):
      Tier 1 (Basic):       tx ≤ KES 10K,   daily ≤ KES 25K,   balance ≤ KES 50K
      Tier 2 (Interim):     tx ≤ KES 50K,   daily ≤ KES 100K,  balance ≤ KES 200K
      Tier 3 (Full KYC):    tx ≤ KES 150K,  daily ≤ KES 500K,  balance ≤ KES 1M
      Tier 4 (EDD):         tx ≤ KES 500K,  daily ≤ KES 10M*,  balance ≤ KES 5M*
      * Tier 4 (EDD) has no regulatory caps — these are soft boundaries.
    """
    transaction_type_probs: Dict[str, float] = Field(
        default={
            TransactionType.SEND_MONEY.value: 0.25,
            TransactionType.RECEIVED_MONEY.value: 0.20,
            TransactionType.AGENT_DEPOSIT.value: 0.10,
            TransactionType.AGENT_WITHDRAWAL.value: 0.10,
            TransactionType.PAYBILL.value: 0.15,
            TransactionType.BUY_GOODS.value: 0.12,
            TransactionType.OTHERS.value: 0.08,
        }
    )
    betting_probability: float = 0.03
    international_probability: float = 0.02
    # Global fallback log-normal params (overridden per-archetype when profile available)
    amount_mean: float = 6.5
    amount_std: float = 1.2
    kadogo_p2p_threshold: float = 100.0
    kadogo_merchant_threshold: float = 200.0
    tier_caps: Dict[int, float] = Field(
        default={
            1: 10_000.0,       # CBK Tier 1: max single tx KES 10,000
            2: 50_000.0,       # CBK Tier 2: max single tx KES 50,000
            3: 150_000.0,      # CBK Tier 3: max single tx KES 150,000
            4: 500_000.0,      # CBK Tier 4 (EDD): max single tx KES 500,000
        }
    )
    daily_velocity_caps: Dict[int, float] = Field(default_factory=lambda: dict(_DAILY_VELOCITY_CAPS))
    balance_caps: Dict[int, float] = Field(
        default={
            1: 50_000.0,
            2: 200_000.0,
            3: 1_000_000.0,
            4: 5_000_000.0,
        }
    )
    max_rejection_attempts: int = 100
    look_ahead_window: int = 5
    seed: int = 42
    num_customers: int = 1000
    initial_balance_range: Tuple[float, float] = (500.0, 5000.0)
    customer_profiles_path: str = "data/bronze/customers/customer_profiles_complete.csv"


class BehavioralTransactionGenerator:
    """Generates behavioral transaction data with realistic M-PESA patterns."""

    TX_TYPE_CONFIGS: Dict[TransactionType, TransactionTypeConfig] = {
        TransactionType.SEND_MONEY: TransactionTypeConfig(0.25, FlowDirection.OUTFLOW, "P2P transfers"),
        TransactionType.RECEIVED_MONEY: TransactionTypeConfig(0.20, FlowDirection.INFLOW, "P2P receipts"),
        TransactionType.AGENT_DEPOSIT: TransactionTypeConfig(0.10, FlowDirection.INFLOW, "Cash deposits"),
        TransactionType.AGENT_WITHDRAWAL: TransactionTypeConfig(0.10, FlowDirection.OUTFLOW, "Cash withdrawals"),
        TransactionType.PAYBILL: TransactionTypeConfig(0.15, FlowDirection.OUTFLOW, "Bill payments"),
        TransactionType.BUY_GOODS: TransactionTypeConfig(0.12, FlowDirection.OUTFLOW, "Merchant payments"),
        TransactionType.OTHERS: TransactionTypeConfig(
            0.08, FlowDirection.BIDIRECTIONAL, "Fuliza, M-Shwari, Reversals"
        ),
    }

    BETTING_PLATFORMS = [
        "SportPesa", "Betika", "Betway", "1xBet", "22Bet",
        "BetLion", "Mcheza", "Elitebet", "Odibets",
    ]

    INTERNATIONAL_INDICATORS = [
        "Western Union", "MoneyGram", "WorldRemit", "Remitly",
        "Sendwave", "Wise", "PayPal", "Skrill",
    ]

    TIER_NAME_TO_INT = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4}

    # ------------------------------------------------------------------ #
    #  INIT
    # ------------------------------------------------------------------ #
    def __init__(self, config: Optional[BehavioralGeneratorConfig] = None):
        self.config = config or BehavioralGeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.customers: Dict[str, CustomerState] = {}
        self._lambda_max: float = 0.0  # computed in _load_customers
        logger.info("Initialized BehavioralTransactionGenerator with seed %s", self.config.seed)

    # ------------------------------------------------------------------ #
    #  CUSTOMER LOADING
    # ------------------------------------------------------------------ #
    def _load_customers(self) -> None:
        profiles_path = Path(self.config.customer_profiles_path)
        if profiles_path.exists():
            df = pl.read_csv(profiles_path)
            for row in df.iter_rows(named=True):
                tier_name = str(row["kyc_tier"])
                tier = self.TIER_NAME_TO_INT.get(tier_name, 1)
                archetype = str(row.get("archetype", "retail_standard"))
                opening = float(row.get("initial_balance_kes", row.get("opening_balance", 1_000.0)))
                self.customers[row["customer_id"]] = CustomerState(
                    customer_id=row["customer_id"],
                    tier=tier,
                    tier_cap=float(row["max_transaction_limit_kes"]),
                    balance_cap=float(row.get("max_balance_limit_kes", self.config.balance_caps.get(tier, self.config.balance_caps[1]))),
                    balance=opening,
                    betting_flag=bool(row["betting_platform_flag"]),
                    international_flag=bool(row["international_transaction_flag"]),
                    archetype=archetype,
                    opening_balance=opening,
                    balance_max=opening,
                    balance_min=opening,
                )
            logger.info("Loaded %s customers from %s", len(self.customers), profiles_path)
        else:
            customer_ids = [f"CUST_{i:06d}" for i in range(self.config.num_customers)]
            low, high = self.config.initial_balance_range
            initial_balances = self.rng.uniform(low, high, size=self.config.num_customers)
            tiers = self.rng.choice([1, 2, 3, 4], size=self.config.num_customers, p=[0.60, 0.20, 0.15, 0.05])
            archetypes_pool = ["retail_standard", "retail_heavy", "micro_merchant", "corporate"]
            archetype_weights = [0.70, 0.15, 0.12, 0.03]
            chosen = self.rng.choice(archetypes_pool, size=self.config.num_customers, p=archetype_weights)
            for idx, customer_id in enumerate(customer_ids):
                tier = int(tiers[idx])
                opening = float(initial_balances[idx])
                self.customers[customer_id] = CustomerState(
                    customer_id=customer_id,
                    tier=tier,
                    tier_cap=self.config.tier_caps.get(tier, self.config.tier_caps[1]),
                    balance_cap=self.config.balance_caps.get(tier, self.config.balance_caps[1]),
                    balance=opening,
                    betting_flag=False,
                    international_flag=False,
                    archetype=str(chosen[idx]),
                    opening_balance=opening,
                    balance_max=opening,
                    balance_min=opening,
                )

        # True maximum possible intensity (used for thinning acceptance denominator)
        self._TRUE_LAMBDA_MULTIPLIER = TRUE_MAX_MULTIPLIER

        total_customers = max(len(self.customers), 1)
        fy25_seconds = (FY25_END - FY25_START).total_seconds()
        # Per-customer proposal rate computed so that expected accepted tx = num_customers
        # E[accepted] = base_rate_per_customer * (mean_intensity / true_max) * n_customers * fy25_seconds
        mean_intensity = MEAN_INTENSITY_ESTIMATE
        tx_per_customer = 10  # average tx per customer
        self._base_rate = (
            tx_per_customer * self._TRUE_LAMBDA_MULTIPLIER
            / (mean_intensity * fy25_seconds)
        )
        # Global max proposal rate = per_customer_rate * true_max * n_customers
        self._lambda_max = self._base_rate * self._TRUE_LAMBDA_MULTIPLIER * total_customers

    # ------------------------------------------------------------------ #
    #  TRANSACTION TYPE / DIRECTION
    # ------------------------------------------------------------------ #
    def _sample_transaction_type(self) -> TransactionType:
        types = list(self.config.transaction_type_probs.keys())
        probs = list(self.config.transaction_type_probs.values())
        sampled = self.rng.choice(types, p=probs)
        return TransactionType(str(sampled))

    def _is_kadogo_type(self, tx_type: TransactionType) -> bool:
        return tx_type in {
            TransactionType.SEND_MONEY,
            TransactionType.RECEIVED_MONEY,
            TransactionType.PAYBILL,
            TransactionType.BUY_GOODS,
        }

    def _kadogo_threshold(self, tx_type: TransactionType) -> Optional[float]:
        if tx_type in {TransactionType.SEND_MONEY, TransactionType.RECEIVED_MONEY}:
            return self.config.kadogo_p2p_threshold
        if tx_type in {TransactionType.PAYBILL, TransactionType.BUY_GOODS}:
            return self.config.kadogo_merchant_threshold
        return None

    # ------------------------------------------------------------------ #
    #  VALUE GENERATION  (per-archetype log-normal)
    # ------------------------------------------------------------------ #
    def _sample_base_amount(self, tx_type: TransactionType, archetype: str = "retail_standard") -> float:
        params = _ARCHETYPE_AMOUNT_PARAMS.get(archetype, _ARCHETYPE_AMOUNT_PARAMS["retail_standard"])
        amount = float(self.rng.lognormal(mean=params["mean"], sigma=params["sigma"]))
        threshold = self._kadogo_threshold(tx_type)
        if threshold is not None and amount < threshold:
            low = 10.0 if tx_type in {TransactionType.SEND_MONEY, TransactionType.RECEIVED_MONEY} else 20.0
            amount = float(self.rng.uniform(low, threshold))
        return amount

    def _constrained_random_walk_amount(
        self,
        tx_type: TransactionType,
        tier_cap: float,
        current_balance: float,
        direction: FlowDirection,
        archetype: str = "retail_standard",
    ) -> float:
        """Reduce sampled amount via constrained random walk until tier/balance limits hold."""
        amount = self._sample_base_amount(tx_type, archetype)
        upper_bound = tier_cap
        if direction == FlowDirection.OUTFLOW:
            upper_bound = min(tier_cap, current_balance)

        attempts = 0
        while amount > upper_bound and attempts < self.config.max_rejection_attempts:
            step = float(self.rng.uniform(0.05, 0.25))
            amount *= 1.0 - step
            attempts += 1

        if upper_bound <= 0:
            return 0.0
        if amount > upper_bound and upper_bound >= 10.0:
            amount = float(self.rng.uniform(10.0, upper_bound))
        elif amount > upper_bound:
            # upper_bound < 10 but positive — clamp directly
            amount = upper_bound
        return round(max(min(amount, upper_bound), 1.0), 2)

    def _resolve_direction(
        self,
        configured_direction: FlowDirection,
        amount: float,
        current_balance: float,
    ) -> FlowDirection:
        if configured_direction != FlowDirection.BIDIRECTIONAL:
            return configured_direction
        if current_balance >= amount and self.rng.random() < 0.5:
            return FlowDirection.OUTFLOW
        return FlowDirection.INFLOW

    # ------------------------------------------------------------------ #
    #  TEMPORAL MODEL  (Inhomogeneous Poisson via thinning)
    #  Uses compute_intensity() from temporal_model module
    # ------------------------------------------------------------------ #

    def _generate_timestamp(self, start: datetime) -> datetime:
        """Generate next transaction timestamp using the thinning algorithm.

        Proposes from a homogeneous Poisson with rate λ_max and accepts
        with probability λ(t) / λ_max. The candidate time accumulates
        monotonically across all proposals (accepted and rejected).
        """
        t = start
        while True:
            dt = self.rng.exponential(scale=1.0 / self._lambda_max)
            t = t + timedelta(seconds=float(dt))
            if t > FY25_END:
                t = FY25_START + timedelta(
                    seconds=float(self.rng.uniform(0, (FY25_END - FY25_START).total_seconds()))
                )
            intensity = compute_intensity(t)
            if self.rng.random() < intensity / self._TRUE_LAMBDA_MULTIPLIER:
                return t

    # ------------------------------------------------------------------ #
    #  RISK FLAGGING
    # ------------------------------------------------------------------ #
    def _flag_high_risk_entities(self, customer: CustomerState) -> Tuple[bool, bool]:
        """Betting and international flags — per-transaction random assignment
        at the configured global rates (3% / 2%)."""
        return (
            self.rng.random() < self.config.betting_probability,
            self.rng.random() < self.config.international_probability,
        )

    def _generate_counterparty(
        self,
        tx_type: TransactionType,
        is_betting: bool,
        is_international: bool,
    ) -> str:
        if is_betting:
            return str(self.rng.choice(self.BETTING_PLATFORMS))
        if is_international:
            return str(self.rng.choice(self.INTERNATIONAL_INDICATORS))
        if tx_type == TransactionType.PAYBILL:
            return f"Paybill-{self.rng.integers(100000, 999999)}"
        if tx_type == TransactionType.BUY_GOODS:
            return f"Merchant-{self.rng.integers(1000, 9999)}"
        if tx_type in {TransactionType.AGENT_DEPOSIT, TransactionType.AGENT_WITHDRAWAL}:
            return f"Agent-{self.rng.integers(100, 999)}"
        if tx_type == TransactionType.OTHERS:
            others = ["Fuliza", "M-Shwari", "Reversal", "Okoa Jahazi", "Bonga Points"]
            return str(self.rng.choice(others))
        return f"User-{self.rng.integers(10000, 99999)}"

    # ------------------------------------------------------------------ #
    #  BALANCE CONSTRAINT ENFORCEMENT
    # ------------------------------------------------------------------ #
    def _apply_balance_change(
        self,
        amount: float,
        direction: FlowDirection,
        current_balance: float,
        tier_cap: float,
        balance_cap: float,
    ) -> Tuple[bool, float]:
        if amount > tier_cap:
            return False, current_balance

        if direction == FlowDirection.OUTFLOW:
            if current_balance < amount:
                return False, current_balance
            new_balance = current_balance - amount
        else:
            new_balance = current_balance + amount

        if new_balance < 0 or new_balance > balance_cap:
            return False, current_balance
        return True, round(new_balance, 2)

    def _plan_transaction(
        self,
        customer: CustomerState,
        tx_type: Optional[TransactionType] = None,
    ) -> Optional[Dict[str, Any]]:
        tx_type = tx_type or self._sample_transaction_type()
        tx_config = self.TX_TYPE_CONFIGS[tx_type]
        direction = tx_config.direction
        amount = 0.0
        for _ in range(self.config.max_rejection_attempts):
            direction = self._resolve_direction(tx_config.direction, amount, customer.balance)
            amount = self._constrained_random_walk_amount(
                tx_type, customer.tier_cap, customer.balance, direction,
                archetype=customer.archetype,
            )
            is_valid, _ = self._apply_balance_change(
                amount, direction, customer.balance, customer.tier_cap, customer.balance_cap
            )
            if is_valid:
                break
        else:
            return None

        # Check daily velocity cap for outflows
        if direction == FlowDirection.OUTFLOW:
            daily_cap = self.config.daily_velocity_caps.get(customer.tier, 1_000_000.0)
            if customer.daily_outflow_total + amount > daily_cap:
                return None

        is_valid, new_balance = self._apply_balance_change(
            amount, direction, customer.balance, customer.tier_cap, customer.balance_cap
        )
        if not is_valid:
            return None

        is_betting, is_international = self._flag_high_risk_entities(customer)
        threshold = self._kadogo_threshold(tx_type)
        is_kadogo = threshold is not None and amount < threshold

        return {
            "tx_type": tx_type,
            "amount": amount,
            "direction": direction,
            "new_balance": new_balance,
            "is_betting": is_betting,
            "is_international": is_international,
            "is_kadogo": is_kadogo,
            "counterparty": self._generate_counterparty(tx_type, is_betting, is_international),
        }

    def _reset_daily_velocity(self, customer: CustomerState, current_dt: datetime) -> None:
        """Reset daily outflow counter if we've moved to a new UTC day."""
        if customer.last_daily_reset is None:
            customer.last_daily_reset = current_dt
            return
        if current_dt.date() != customer.last_daily_reset.date():
            customer.daily_outflow_total = 0.0
            customer.last_daily_reset = current_dt

    def _look_ahead_validation(self, customer: CustomerState, first_plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and adjust the planned transaction to ensure a viable future sequence.

        Attempts to fit the plan into a look-ahead window of future transactions.
        If the original amount would violate balance constraints, tries reduced
        amounts (geometric decay 0.9, 0.8, ..., 0.1) to find one that allows
        forward progress.  Returns the adjusted plan or None if no viable
        amount is found.
        """
        temp_balance = customer.balance
        temp_daily_outflow = customer.daily_outflow_total

        for scale in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            adjusted = dict(first_plan)
            adjusted["amount"] = round(first_plan["amount"] * scale, 2)

            is_valid, new_balance = self._apply_balance_change(
                adjusted["amount"], adjusted["direction"],
                temp_balance, customer.tier_cap, customer.balance_cap,
            )
            if not is_valid:
                continue

            ft_daily = temp_daily_outflow
            if adjusted["direction"] == FlowDirection.OUTFLOW:
                daily_cap = self.config.daily_velocity_caps.get(customer.tier, 1_000_000.0)
                if ft_daily + adjusted["amount"] > daily_cap:
                    continue
                ft_daily += adjusted["amount"]

            ok = True
            for _ in range(self.config.look_ahead_window - 1):
                plan = self._plan_transaction(
                    CustomerState(
                        customer_id=customer.customer_id,
                        tier=customer.tier,
                        tier_cap=customer.tier_cap,
                        balance_cap=customer.balance_cap,
                        balance=new_balance,
                        betting_flag=customer.betting_flag,
                        international_flag=customer.international_flag,
                        archetype=customer.archetype,
                        daily_outflow_total=ft_daily,
                        last_daily_reset=customer.last_daily_reset,
                    )
                )
                if plan is None:
                    ok = False
                    break
                is_valid, new_balance = self._apply_balance_change(
                    plan["amount"], plan["direction"],
                    new_balance, customer.tier_cap, customer.balance_cap,
                )
                if not is_valid:
                    ok = False
                    break
                if plan["direction"] == FlowDirection.OUTFLOW:
                    daily_cap = self.config.daily_velocity_caps.get(customer.tier, 1_000_000.0)
                    if ft_daily + plan["amount"] > daily_cap:
                        ok = False
                        break
                    ft_daily += plan["amount"]
            if ok:
                return adjusted

        return None

    # ------------------------------------------------------------------ #
    #  MAIN GENERATION LOOP
    # ------------------------------------------------------------------ #
    def generate_transactions(
        self,
        n_transactions: int,
        start_date: Optional[datetime] = None,
        output_path: str = "data/detailed_transactions.csv",
    ) -> pl.DataFrame:
        if start_date is None:
            start_date = FY25_START

        self._load_customers()
        customer_ids = list(self.customers.keys())
        logger.info("Generating %s behavioral transactions", n_transactions)

        transactions: List[Dict[str, Any]] = []
        current_timestamp = self._generate_timestamp(start_date)
        generated = 0
        skipped = 0
        attempt = 0
        max_total_attempts = n_transactions * 5

        # Per-customer exhaustion tracking (max 100 consecutive failures = end sequence)
        customer_failure_count: Dict[str, int] = defaultdict(int)
        MAX_CONSECUTIVE_FAILURES = 100

        while generated < n_transactions and attempt < max_total_attempts:
            attempt += 1
            customer_id = str(self.rng.choice(customer_ids))
            customer = self.customers[customer_id]

            # End this customer's sequence early if they fail too many times in a row
            if customer_failure_count[customer_id] >= MAX_CONSECUTIVE_FAILURES:
                # Pick another customer that still has capacity
                active = [c for c in customer_ids if customer_failure_count[c] < MAX_CONSECUTIVE_FAILURES]
                if not active:
                    logger.warning("All customers exhausted after %s consecutive failures", MAX_CONSECUTIVE_FAILURES)
                    break
                customer_id = str(self.rng.choice(active))
                customer = self.customers[customer_id]

            # Reset daily velocity if new day
            self._reset_daily_velocity(customer, current_timestamp)

            plan = None
            for _ in range(self.config.max_rejection_attempts):
                candidate = self._plan_transaction(customer)
                if candidate is None:
                    continue
                adjusted = self._look_ahead_validation(customer, candidate)
                if adjusted is not None:
                    plan = adjusted
                    break

            if plan is None:
                customer_failure_count[customer_id] += 1
                skipped += 1
                continue

            # Reset failure count on success
            customer_failure_count[customer_id] = 0

            # Apply balance change to real state
            _, customer.balance = self._apply_balance_change(
                plan["amount"], plan["direction"],
                customer.balance, customer.tier_cap, customer.balance_cap,
            )
            if plan["direction"] == FlowDirection.OUTFLOW:
                customer.daily_outflow_total += plan["amount"]

            # Track balance statistics
            customer.balance_max = max(customer.balance_max, customer.balance)
            customer.balance_min = min(customer.balance_min, customer.balance)
            customer.balance_sum += customer.balance
            customer.balance_count += 1

            # Generate next timestamp using thinning
            current_timestamp = self._generate_timestamp(current_timestamp)

            paid_in = plan["amount"] if plan["direction"] == FlowDirection.INFLOW else 0.0
            paid_out = plan["amount"] if plan["direction"] == FlowDirection.OUTFLOW else 0.0

            transactions.append(
                {
                    "transaction_id": f"TXN_{generated:010d}",
                    "customer_id": customer_id,
                    "counterparty": plan["counterparty"],
                    "transaction_type": plan["tx_type"].value,
                    "amount": plan["amount"],
                    "direction": plan["direction"].value,
                    "timestamp": current_timestamp.isoformat(),
                    "paid_in": paid_in,
                    "paid_out": paid_out,
                    "balance": customer.balance,
                    "tier": customer.tier,
                    "hour": current_timestamp.hour,
                    "day_of_week": current_timestamp.weekday(),
                    "month": current_timestamp.month,
                    "is_weekend": current_timestamp.weekday() >= 5,
                    "is_night": current_timestamp.hour < 6 or current_timestamp.hour >= 22,
                    "is_betting": plan["is_betting"],
                    "is_international": plan["is_international"],
                    "is_kadogo": plan["is_kadogo"],
                }
            )
            generated += 1

        if generated < n_transactions:
            logger.warning(
                "Generated %s/%s transactions after %s attempts (%s skipped)",
                generated,
                n_transactions,
                attempt,
                skipped,
            )

        df = pl.DataFrame(transactions).sort("timestamp")
        # Add balance_after alias for downstream exporter compatibility
        df = df.with_columns(pl.col("balance").alias("balance_after"))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output)
        logger.info("Generated %s transactions saved to %s", len(df), output_path)
        self._log_generation_stats(df)
        self._log_balance_stats(df)
        return df

    # ------------------------------------------------------------------ #
    #  STATS
    # ------------------------------------------------------------------ #
    def _log_generation_stats(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            logger.warning("No transactions generated")
            return

        total = len(df)
        logger.info("Transaction Generation Statistics:")
        logger.info("  Total transactions: %s", total)
        logger.info("  Transaction type distribution:")
        for tx_type, count in df.group_by("transaction_type").len().sort("transaction_type").iter_rows():
            logger.info("    %s: %s (%.1f%%)", tx_type, count, count / total * 100)

        betting_count = df.filter(pl.col("is_betting")).height
        intl_count = df.filter(pl.col("is_international")).height
        logger.info("  High-risk flags:")
        logger.info("    Betting transactions: %s (%.1f%%)", betting_count, betting_count / total * 100)
        logger.info("    International transactions: %s (%.1f%%)", intl_count, intl_count / total * 100)
        logger.info("  Kadogo transactions: %s", df.filter(pl.col("is_kadogo")).height)
        logger.info(
            "  Amount stats — mean: %.2f, median: %.2f, min: %.2f, max: %.2f",
            df["amount"].mean(),
            df["amount"].median(),
            df["amount"].min(),
            df["amount"].max(),
        )
        logger.info("  Temporal distribution:")
        logger.info("    Weekend: %.1f%%", df.filter(pl.col("is_weekend")).height / total * 100)
        logger.info("    Night: %.1f%%", df.filter(pl.col("is_night")).height / total * 100)
        logger.info("    Hour distribution:")
        for hour, count in df.group_by("hour").len().sort("hour").iter_rows():
            logger.info("      h%02d: %s (%.1f%%)", hour, count, count / total * 100)

    def _log_balance_stats(self, df: pl.DataFrame) -> None:
        """Log balance integrity and tracking statistics."""
        if df.is_empty():
            return

        logger.info("Balance Constraint Enforcement:")
        neg_balance = df.filter(pl.col("balance") < 0).height
        logger.info("  Negative balances: %s (should be 0)", neg_balance)

        # Tier-specific limit violations (balance_cap stored per customer state)
        tier_violations = 0
        for cid, state in self.customers.items():
            cdf = df.filter(pl.col("customer_id") == cid)
            if cdf.height == 0:
                continue
            violations = cdf.filter(pl.col("balance") > state.balance_cap).height
            tier_violations += violations
        logger.info("  Tier limit violations: %s (should be 0)", tier_violations)

        logger.info("  Ledger continuity (per-customer):")
        continuity_ok = 0
        continuity_fail = 0
        for cid, state in self.customers.items():
            cdf = df.filter(pl.col("customer_id") == cid).sort("timestamp")
            if cdf.height == 0:
                continue
            prev_balance = state.opening_balance
            ledger_match = True
            for row in cdf.iter_rows(named=True):
                expected = prev_balance + row["paid_in"] - row["paid_out"]
                if abs(expected - row["balance"]) > 1.0:
                    ledger_match = False
                    break
                prev_balance = row["balance"]
            if ledger_match:
                continuity_ok += 1
            else:
                continuity_fail += 1
        logger.info("    Continuity OK: %s, Failures: %s", continuity_ok, continuity_fail)

        logger.info("  Customer balance summary (opening / max / min / avg):")
        samples = list(self.customers.items())[:5]
        for cid, state in samples:
            avg_b = state.balance_sum / max(state.balance_count, 1)
            logger.info(
                "    %s: opening=%.2f max=%.2f min=%.2f avg=%.2f",
                cid, state.opening_balance, state.balance_max, state.balance_min, avg_b,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    config = BehavioralGeneratorConfig(seed=42, num_customers=1000)
    generator = BehavioralTransactionGenerator(config)
    df = generator.generate_transactions(
        n_transactions=10_000,
        output_path="data/detailed_transactions.csv",
    )
    print(f"Generated {len(df)} transactions")
    print(df.head(10))


if __name__ == "__main__":
    main()
