import numpy as np
import polars as pl
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Data Contracts for MRM Traceability
class CorporateIdentity(BaseModel):
    tax_id: str = Field(..., description="Unique deterministic Tax ID")
    email: str = Field(..., description="Unique deterministic email")
    registration_date: datetime = Field(default_factory=datetime.utcnow, description="Date of identity registration")

    @validator('tax_id')
    def validate_tax_id(cls, v):
        if not v.startswith('TAX-'):
            raise ValueError("Tax ID must start with 'TAX-'")
        return v

class TransactionRecord(BaseModel):
    transaction_id: str
    sender_id: str
    transaction_amount: float
    timestamp: datetime
    channel_type: str
    tax_id: str
    email: str
    receiver_id: str
    sender_county: str
    receiver_county: str
    device_age_days: int
    sim_match_status: bool
    wallet_tier_encoded: int
    kyc_level_encoded: int
    prev_fraud_flag_count_90d: int
    anomaly_flag: bool = False
    anomaly_type: Optional[str] = None

class MpesaDistributionSampler:
    """
    Core Statistical Engine for M-Pesa Synthetic Data Generation.
    Strictly conforms to defined distributions for MRM compliance:
    - Amounts: Log-Normal(μ=5.95, σ=1.25) capped at KES 250,000.
    - Inter-arrival times: Exponential(λ=320.0).
    - Anomaly: 0.15% exact contamination rate mapped to Safaricom FMS vectors.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.AMOUNT_MEAN = 5.95
        self.AMOUNT_STD = 1.25
        self.AMOUNT_CAP = 250000.0
        
        self.VELOCITY_LAMBDA = 320.0
        self.CONTAMINATION_RATE = 0.0015  # 0.15%

        self.FMS_VECTORS = [
            "Structuring/Smurfing", 
            "Agent Fraud", 
            "Digital Lending Misuse", 
            "Trade Based Money Laundering", 
            "PEP Transaction", 
            "Terrorist Financing", 
            "Shell Company Layering", 
            "Real Estate Money Laundering", 
            "Cryptocurrency Conversion"
        ]

    def sample_amounts(self, n: int) -> np.ndarray:
        """Sample transaction amounts strictly."""
        amounts = self.rng.lognormal(mean=self.AMOUNT_MEAN, sigma=self.AMOUNT_STD, size=n)
        return np.clip(amounts, a_min=1.0, a_max=self.AMOUNT_CAP)

    def sample_inter_arrival_times(self, n: int) -> np.ndarray:
        """Sample inter-arrival times strictly."""
        return self.rng.exponential(scale=self.VELOCITY_LAMBDA, size=n)

    def generate_anomalies(self, n_total: int) -> np.ndarray:
        """
        Generate exact 0.15% contamination rate flags and corresponding types.
        """
        n_anomalies = int(n_total * self.CONTAMINATION_RATE)
        
        flags = np.zeros(n_total, dtype=bool)
        flags[:n_anomalies] = True
        self.rng.shuffle(flags)
        
        types = np.array([None]*n_total, dtype=object)
        anomaly_indices = np.where(flags)[0]
        
        anomaly_types = self.rng.choice(self.FMS_VECTORS, size=n_anomalies)
        types[anomaly_indices] = anomaly_types
        
        return flags, types
