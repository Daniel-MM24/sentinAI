import numpy as np
import polars as pl
import scipy.stats as stats
from typing import List, Tuple, Dict, Any
from uuid import uuid4
from datetime import datetime, timedelta

from src.data.generator_engine import TransactionRecord, SyntheticStateRegistry

class DistributionEngine:
    """
    Synthetic Distribution Engine for generating mobile money transactions.
    Mimics behavioral velocity, network dynamics, credit product reliance,
    and regulatory tariff bands.
    """
    def __init__(self, registry: SyntheticStateRegistry):
        self.registry = registry

    def sample_amounts(self, n: int) -> np.ndarray:
        """
        Samples transaction amounts from a Bounded Finite Mixture Model.
        - Kadogo (w=0.45): Bounded Beta mapped to [1.0, 100.0]
        - Standard (w=0.50): Truncated Log-Normal mapped to [101.0, 150000.0]
        - Premium (w=0.05): Pareto Type I mapped to [150001.0, 250000.0]
        """
        amounts = np.zeros(n)
        
        # Multinomial selection based on weights
        segments = np.random.choice(
            ['kadogo', 'standard', 'premium'], 
            size=n, 
            p=[0.45, 0.50, 0.05]
        )
        
        # Segment 1: Kadogo
        kadogo_mask = segments == 'kadogo'
        n_kadogo = np.sum(kadogo_mask)
        if n_kadogo > 0:
            # Beta(2, 5) scaled to [1.0, 100.0]
            kadogo_raw = stats.beta.rvs(a=2, b=5, size=n_kadogo)
            amounts[kadogo_mask] = 1.0 + kadogo_raw * (100.0 - 1.0)
            
        # Segment 2: Standard
        standard_mask = segments == 'standard'
        n_standard = np.sum(standard_mask)
        if n_standard > 0:
            # Lognormal, then reject/clip to fit bounds or scale
            # We use a lognormal shape and shift/scale to roughly fit.
            # Truncated Log-Normal mapped to [101.0, 150000.0]
            s = 1.0 # shape parameter
            std_raw = stats.lognorm.rvs(s=s, size=n_standard)
            # Normalize to 0-1 based on a practical 99th percentile, then scale
            p99 = stats.lognorm.ppf(0.99, s=s)
            std_scaled = np.clip(std_raw / p99, 0, 1)
            amounts[standard_mask] = 101.0 + std_scaled * (150000.0 - 101.0)
            
        # Segment 3: Premium
        premium_mask = segments == 'premium'
        n_premium = np.sum(premium_mask)
        if n_premium > 0:
            # Pareto Type I (b=2), bounded [150001.0, 250000.0]
            b = 2.0
            prem_raw = stats.pareto.rvs(b, size=n_premium)
            # Clip to a reasonable range [1, 10] then map
            prem_scaled = np.clip((prem_raw - 1) / 9.0, 0, 1)
            amounts[premium_mask] = 150001.0 + prem_scaled * (250000.0 - 150001.0)
            
        return amounts

    def simulate_timestamps(self, start_time: datetime, n: int, days_duration: int = 7) -> List[str]:
        """
        Simulates timestamps using a Non-Homogeneous Poisson Process (NHPP) approximation.
        Captures diurnal business spikes (08:00, 17:00) using sinusoidal intensity λ(t).
        """
        timestamps = []
        total_seconds = days_duration * 24 * 3600
        
        # Generate uniform random times, then accept/reject based on intensity
        # Intensity function λ(t) with peaks around 8 AM (28800s) and 5 PM (61200s)
        # Using a simple sinusoidal combination.
        while len(timestamps) < n:
            batch_size = (n - len(timestamps)) * 2
            t_candidates = np.random.uniform(0, total_seconds, size=batch_size)
            
            # Map candidate to time of day in seconds (0 to 86400)
            tod = t_candidates % 86400
            
            # Peak 1: 8 AM (28800s), Peak 2: 5 PM (61200s)
            intensity = 0.2 + 0.8 * (
                np.exp(-((tod - 28800) ** 2) / (2 * 7200 ** 2)) + 
                np.exp(-((tod - 61200) ** 2) / (2 * 7200 ** 2))
            )
            
            # Weekend reduction factor (optional enhancement)
            # Assume start_time is a Monday for simplicity, just illustrative
            
            u = np.random.uniform(0, 1, size=batch_size)
            accepted_t = t_candidates[u < intensity]
            
            for t in accepted_t:
                if len(timestamps) < n:
                    dt = start_time + timedelta(seconds=float(t))
                    timestamps.append(dt.isoformat())
                    
        # Sort chronologically as it's a ledger
        timestamps.sort()
        return timestamps

    def generate_credit_triggers(self, amounts: np.ndarray, latent_balances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates credit product triggers via Bounded Conditional Bernoulli Trials.
        Deficit = Amount - LatentBalance
        Minor deficits prioritize Fuliza; major deficits prioritize M-Shwari.
        """
        deficits = amounts - latent_balances
        
        is_fuliza = np.zeros(len(amounts), dtype=bool)
        is_mshwari = np.zeros(len(amounts), dtype=bool)
        
        for i, deficit in enumerate(deficits):
            if deficit > 0:
                # Conditional Bernoulli Trials
                # Let's say a deficit up to 2000 is "minor", prioritizing Fuliza
                if deficit <= 2000.0:
                    prob_fuliza = min(0.9, deficit / 2000.0)
                    is_fuliza[i] = np.random.binomial(1, prob_fuliza)
                    if not is_fuliza[i]:
                        prob_mshwari = min(0.3, deficit / 2000.0)
                        is_mshwari[i] = np.random.binomial(1, prob_mshwari)
                else:
                    # Major deficit prioritizes M-Shwari
                    prob_mshwari = min(0.8, deficit / 150000.0)
                    is_mshwari[i] = np.random.binomial(1, prob_mshwari)
                    if not is_mshwari[i]:
                        prob_fuliza = 0.1
                        is_fuliza[i] = np.random.binomial(1, prob_fuliza)
                        
        return is_fuliza, is_mshwari

    def generate_ledger(self, num_transactions: int, start_time: datetime, days_duration: int = 7) -> List[TransactionRecord]:
        """
        Orchestrates the complete synthetic ledger generation.
        Assumes registry is pre-populated with users.
        """
        if self.registry.total_customers < 2:
            raise ValueError("Registry needs at least 2 customers to generate a network.")
            
        users = list(self.registry._registered_customers.keys())
        
        # 1. Sample sender/recipients
        senders = np.random.choice(users, size=num_transactions)
        recipients = np.random.choice(users, size=num_transactions)
        # Avoid self-loops (simple fix)
        self_loop_mask = senders == recipients
        for i in np.where(self_loop_mask)[0]:
            candidates = [u for u in users if u != senders[i]]
            recipients[i] = np.random.choice(candidates)
            
        # 2. Sample amounts
        amounts = self.sample_amounts(num_transactions)
        
        # 3. Simulate timestamps
        timestamps = self.simulate_timestamps(start_time, num_transactions, days_duration)
        
        # 4. Latent balances (mock random for credit triggers)
        # In a real dynamic state, we'd track actual balances over time.
        # For statistical simulation, we sample latent balances from a lognormal.
        latent_balances = stats.lognorm.rvs(s=1.0, scale=1000.0, size=num_transactions)
        
        # 5. Credit triggers
        is_fuliza, is_mshwari = self.generate_credit_triggers(amounts, latent_balances)
        
        # 6. Channel types
        channels = np.random.choice(["USSD", "App", "STK_Push"], size=num_transactions, p=[0.5, 0.4, 0.1])
        
        records = []
        for i in range(num_transactions):
            record = TransactionRecord(
                transaction_id=uuid4(),
                sender_id=senders[i],
                recipient_id=recipients[i],
                transaction_amount=float(amounts[i]),
                timestamp=timestamps[i],
                channel_type=channels[i],
                is_fuliza=bool(is_fuliza[i]),
                is_mshwari=bool(is_mshwari[i])
            )
            records.append(record)
            
        return records
