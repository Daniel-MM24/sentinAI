"""
Behavioral Transaction Generation Engine

Generates M-PESA-like transaction histories with:
- Transaction type distribution matching actual statement structure
- High-risk entity flagging (betting, international transfers)
- Log-normal value generation with Kadogo thresholds
- Temporal cyclicality via inhomogeneous Poisson processes
- Balance constraint enforcement (rejection sampling + look-ahead validation)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FY25_START = datetime(2024, 7, 1, tzinfo=timezone.utc)
FY25_END = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)


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


class BehavioralGeneratorConfig(BaseModel):
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
  amount_mean: float = 6.5
  amount_std: float = 1.2
  kadogo_p2p_threshold: float = 100.0
  kadogo_merchant_threshold: float = 200.0
  tier_caps: Dict[int, float] = Field(
    default={
      1: 70_000.0,
      2: 150_000.0,
      3: 250_000.0,
      4: 1_000_000.0,
    }
  )
  diurnal_patterns: Dict[str, float] = Field(
    default={"morning": 0.35, "lunch": 0.25, "evening": 0.40}
  )
  school_fees_months: List[int] = Field(default=[1, 5, 9])
  holiday_month: int = 12
  weekend_multipliers: Dict[int, float] = Field(default={5: 0.6, 6: 0.5})
  base_inter_arrival_hours: float = 2.0
  max_rejection_attempts: int = 15
  look_ahead_window: int = 5
  seed: int = 42
  num_customers: int = 1000
  initial_balance_mean: float = 8.0
  initial_balance_std: float = 1.5
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

  def __init__(self, config: Optional[BehavioralGeneratorConfig] = None):
    self.config = config or BehavioralGeneratorConfig()
    self.rng = np.random.default_rng(self.config.seed)
    self.customers: Dict[str, CustomerState] = {}
    logger.info("Initialized BehavioralTransactionGenerator with seed %s", self.config.seed)

  def _load_customers(self) -> None:
    profiles_path = Path(self.config.customer_profiles_path)
    if profiles_path.exists():
      df = pl.read_csv(profiles_path)
      for row in df.iter_rows(named=True):
        tier_name = str(row["kyc_tier"])
        tier = self.TIER_NAME_TO_INT.get(tier_name, 1)
        self.customers[row["customer_id"]] = CustomerState(
          customer_id=row["customer_id"],
          tier=tier,
          tier_cap=float(row["max_transaction_limit_kes"]),
          balance_cap=float(row["max_balance_limit_kes"]),
          balance=float(row["initial_balance_kes"]),
          betting_flag=bool(row["betting_platform_flag"]),
          international_flag=bool(row["international_transaction_flag"]),
        )
      logger.info("Loaded %s customers from %s", len(self.customers), profiles_path)
      return

    customer_ids = [f"CUST_{i:06d}" for i in range(self.config.num_customers)]
    initial_balances = self.rng.lognormal(
      mean=self.config.initial_balance_mean,
      sigma=self.config.initial_balance_std,
      size=self.config.num_customers,
    )
    tiers = self.rng.integers(1, 4, size=self.config.num_customers)
    for idx, customer_id in enumerate(customer_ids):
      tier = int(tiers[idx])
      self.customers[customer_id] = CustomerState(
        customer_id=customer_id,
        tier=tier,
        tier_cap=self.config.tier_caps.get(tier, self.config.tier_caps[1]),
        balance_cap=self.config.tier_caps.get(tier, self.config.tier_caps[1]) * 5,
        balance=float(initial_balances[idx]),
        betting_flag=False,
        international_flag=False,
      )

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

  def _sample_base_amount(self, tx_type: TransactionType) -> float:
    amount = float(self.rng.lognormal(mean=self.config.amount_mean, sigma=self.config.amount_std))
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
  ) -> float:
    """Reduce sampled amount via constrained random walk until tier/balance limits hold."""
    amount = self._sample_base_amount(tx_type)
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
    if amount > upper_bound:
      amount = float(self.rng.uniform(10.0, upper_bound))
    return round(max(amount, 1.0), 2)

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

  def _temporal_intensity(self, dt: datetime) -> float:
    intensity = 1.0
    if dt.month in self.config.school_fees_months:
      intensity *= 3.0
    elif dt.month == self.config.holiday_month:
      intensity *= 1.5
    intensity *= self.config.weekend_multipliers.get(dt.weekday(), 1.0)
    return intensity

  def _sample_diurnal_time(self, base_date: datetime) -> datetime:
    windows = list(self.config.diurnal_patterns.keys())
    probs = list(self.config.diurnal_patterns.values())
    window = str(self.rng.choice(windows, p=probs))

    if window == "morning":
      hour = int(self.rng.integers(8, 10))
    elif window == "lunch":
      hour = int(self.rng.integers(12, 14))
    else:
      hour = int(self.rng.integers(17, 20))

    minute = int(self.rng.integers(0, 60))
    second = int(self.rng.integers(0, 60))
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=0)

  def _advance_timestamp(self, current: datetime) -> datetime:
    hours = self.rng.exponential(scale=self.config.base_inter_arrival_hours)
    hours /= max(self._temporal_intensity(current), 0.1)
    next_dt = current + timedelta(hours=float(hours))

    if next_dt > FY25_END:
      next_dt = FY25_START + timedelta(
        seconds=float(self.rng.uniform(0, (FY25_END - FY25_START).total_seconds()))
      )
    return self._sample_diurnal_time(next_dt)

  def _flag_high_risk_entities(self, _customer: CustomerState) -> Tuple[bool, bool]:
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
        tx_type, customer.tier_cap, customer.balance, direction
      )
      is_valid, _ = self._apply_balance_change(
        amount, direction, customer.balance, customer.tier_cap, customer.balance_cap
      )
      if is_valid:
        break
    else:
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

  def _look_ahead_validation(self, customer: CustomerState, first_plan: Dict[str, Any]) -> bool:
    temp_balance = customer.balance

    is_valid, temp_balance = self._apply_balance_change(
      first_plan["amount"],
      first_plan["direction"],
      temp_balance,
      customer.tier_cap,
      customer.balance_cap,
    )
    if not is_valid:
      return False

    for _ in range(self.config.look_ahead_window - 1):
      plan = self._plan_transaction(
        CustomerState(
          customer_id=customer.customer_id,
          tier=customer.tier,
          tier_cap=customer.tier_cap,
          balance_cap=customer.balance_cap,
          balance=temp_balance,
          betting_flag=customer.betting_flag,
          international_flag=customer.international_flag,
        )
      )
      if plan is None:
        return False
      is_valid, temp_balance = self._apply_balance_change(
        plan["amount"],
        plan["direction"],
        temp_balance,
        customer.tier_cap,
        customer.balance_cap,
      )
      if not is_valid:
        return False
    return True

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
    current_timestamp = self._sample_diurnal_time(start_date)
    generated = 0
    skipped = 0
    attempt = 0
    max_total_attempts = n_transactions * 5

    while generated < n_transactions and attempt < max_total_attempts:
      attempt += 1
      customer_id = str(self.rng.choice(customer_ids))
      customer = self.customers[customer_id]

      plan = None
      for _ in range(self.config.max_rejection_attempts):
        candidate = self._plan_transaction(customer)
        if candidate is None:
          continue
        if self._look_ahead_validation(customer, candidate):
          plan = candidate
          break

      if plan is None:
        skipped += 1
        continue

      customer.balance = plan["new_balance"]
      current_timestamp = self._advance_timestamp(current_timestamp)

      transactions.append(
        {
          "transaction_id": f"TXN_{generated:010d}",
          "customer_id": customer_id,
          "counterparty": plan["counterparty"],
          "transaction_type": plan["tx_type"].value,
          "amount": plan["amount"],
          "direction": plan["direction"].value,
          "timestamp": current_timestamp.isoformat(),
          "balance_after": plan["new_balance"],
          "tier": customer.tier,
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output)
    logger.info("Generated %s transactions saved to %s", len(df), output_path)
    self._log_generation_stats(df)
    return df

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


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
  config = BehavioralGeneratorConfig(seed=42, num_customers=1000)
  generator = BehavioralTransactionGenerator(config)
  df = generator.generate_transactions(
    n_transactions=10_000,
    start_date=FY25_START,
    output_path="data/detailed_transactions.csv",
  )
  print(f"Generated {len(df)} transactions")
  print(df.head(10))


if __name__ == "__main__":
  main()
