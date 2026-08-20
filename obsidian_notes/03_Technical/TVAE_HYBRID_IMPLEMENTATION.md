# TVAE Hybrid Implementation

## Overview

The SentinAI synthetic data generator has been transitioned from a deterministic/stochastic Monte Carlo simulation engine to a deep generative model using a **Tabular VAE (TVAE) Hybrid Pipeline**. This approach leverages industry best practices for financial data generation while maintaining strict regulatory compliance.

## Architecture Philosophy

### Hybrid Approach Rationale

Pure GANs struggle with exact mathematical constraints over long sequences (e.g., $Balance_t = Balance_{t-1} + Amount$). The industry standard for financial ledgers where continuity is strictly audited (like M-PESA) is a **Hybrid Approach**:

- **Generative Layer**: The TVAE generates core event metadata (`customer_id`, `amount`, `timestamp`, `transaction_type`, `direction`, `is_international`)
- **Deterministic Layer**: Downstream pipelines recalculate strict mathematical dependencies (running `balance`, rolling aggregations, network features)

This guarantees:
- 100% ledger continuity
- Zero tier violations
- Realistic distributions learned by the TVAE

### Anomaly Injection Strategy

Generative models are susceptible to *mode collapse* when dealing with rare events (1-2% AML typologies). The implementation uses **Deterministic Post-Generation Injection**:

1. Train TVAE only on "clean" data to generate realistic baseline behavior
2. Use `FinancialAnomalyInjector` to graft precise AML scenarios onto synthetic baseline
3. Preserves absolute control over `is_launderer` and `aml_scenario` labels

## Implementation Components

### 1. TVAE Core Generator (`tvae_generator.py`)

**Location**: `src/data/tvae_generator.py`

**Features**:
- Uses `CTGAN` library (TVAE architecture) optimized for tabular data
- Handles core 8 event columns natively:
  - `customer_id`
  - `tier`
  - `archetype`
  - `transaction_type`
  - `amount`
  - `timestamp`
  - `direction`
  - `is_international`
- Preprocesses timestamps by converting to numerical epochs for latent space
- Reconstructs timestamps to datetime objects during sampling
- Ensures all generated amounts are positive real numbers (clipping + rounding)

**Pre-processing**:
- Applies log-transforms to `amount` using priors from `stat_distribution.md`
- Maps to Gaussian latent space using Mode-Specific Normalization

### 2. Deterministic Balance Reconstructor (`hybrid_reconstructor.py`)

**Location**: `src/data/hybrid_reconstructor.py`

**Purpose**: Post-processing layer to handle mathematical constraints that TVAE cannot learn

**Process**:
1. Takes TVAE-generated isolated events
2. Groups by `customer_id`
3. Sorts strictly by time
4. Recalculates exact `balance` dynamically based on direction (inflow/outflow)
5. **Enforces regulatory tier caps** by hard-rejecting transactions that would cause limit violations

**Tier Limits**:
- Tier 1: KES 50,000
- Tier 2: KES 300,000
- Tier 3: KES 1,000,000

### 3. Enriched Feature Engineering (`feature_engineering.py`)

**Location**: `src/data/feature_engineering.py`

**Updates**: Rewrote `CustomerFeatureEngineer.compute_features` to compute exact 10 downstream features for 21-feature schema

**Velocity/Temporal Features**:
- `tx_count_7d`: Transaction count in last 7 days
- `volume_7d`: Transaction volume in last 7 days
- `night_tx_ratio`: Ratio of night transactions
- `rapid_tx_ratio`: Ratio of rapid transactions
- `volume_7d_vs_30d_ratio`: Burst Ratio (7d volume vs 30d volume)

**Network Features**:
- `distinct_counterparties_7d`: Number of unique counterparties in 7 days
- `fan_in_fan_out_ratio`: Ratio of incoming to outgoing transactions

**Structuring/Mule Features**:
- `close_to_limit_ratio`: Threshold evasion behavior
- `amount_roundness`: Synthetic bot behavior detection
- `balance_retention_ratio`: Pass-through account detection

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: TVAE Training & Generation                         │
├─────────────────────────────────────────────────────────────┤
│ Input: Clean baseline data                                  │
│ Process: TVAE.fit() → TVAE.sample()                         │
│ Output: Raw events (8 core columns)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Deterministic Balance Reconstruction               │
├─────────────────────────────────────────────────────────────┤
│ Input: TVAE raw events                                      │
│ Process: Sort by customer_id + timestamp                    │
│          → Recalculate balance                              │
│          → Enforce tier caps                                │
│ Output: Balance-corrected events (9 columns)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Feature Engineering                                │
├─────────────────────────────────────────────────────────────┤
│ Input: Balance-corrected events                             │
│ Process: CustomerFeatureEngineer.compute_features()         │
│ Output: Enriched dataset (19 columns)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Anomaly Injection                                  │
├─────────────────────────────────────────────────────────────┤
│ Input: Clean enriched dataset                               │
│ Process: FinancialAnomalyInjector                           │
│ Output: Final Gold dataset (21 columns + labels)            │
└─────────────────────────────────────────────────────────────┘
```

## Target Schema (21 Gold Features)

### Core Identity & Event (8)
1. `customer_id`
2. `tier`
3. `archetype`
4. `transaction_type`
5. `amount`
6. `timestamp`
7. `direction`
8. `balance`

### Temporal & Velocity (5)
9. `tx_count_7d`
10. `volume_7d`
11. `night_tx_ratio`
12. `rapid_tx_ratio`
13. `volume_7d_vs_30d_ratio` (Burst Ratio)

### Network & Structuring (6)
14. `is_international`
15. `distinct_counterparties_7d`
16. `fan_in_fan_out_ratio`
17. `close_to_limit_ratio`
18. `balance_retention_ratio`
19. `amount_roundness`

### Labels (2)
20. `is_launderer`
21. `aml_scenario`

## CPU-Friendly Design

**Why TVAE over GAN?**
- TVAEs are generally more stable and faster to train on CPU
- Suitable for local testing without Kaggle/Colab
- Better mode coverage for tabular data
- Less prone to mode collapse

**Training Considerations**:
- Pre-processing uses known statistical priors from `stat_distribution.md`
- Mode-Specific Normalization maps features to Gaussian latent space
- Log-transforms applied to heavy-tailed distributions (amount)

## Usage

### Model Training
```python
from src.data.tvae_generator import TVAEGenerator

# Initialize generator
tvae = TVAEGenerator(
    core_columns=['customer_id', 'tier', 'archetype', 'transaction_type', 
                  'amount', 'timestamp', 'direction', 'is_international']
)

# Fit on clean baseline data
tvae.fit(clean_baseline_data)

# Sample synthetic events
synthetic_events = tvae.sample(n_samples=10000)
```

### Pipeline Integration
```python
from src.data.hybrid_reconstructor import BalanceReconstructor
from src.data.feature_engineering import CustomerFeatureEngineer
from src.data.anomaly_injector import FinancialAnomalyInjector

# Step 1: Generate events with TVAE
raw_events = tvae.sample(n_samples=10000)

# Step 2: Reconstruct balances
reconstructor = BalanceReconstructor()
balance_corrected = reconstructor.reconstruct(raw_events)

# Step 3: Engineer features
engineer = CustomerFeatureEngineer()
enriched_data = engineer.compute_features(balance_corrected)

# Step 4: Inject anomalies
injector = FinancialAnomalyInjector()
final_gold_dataset = injector.inject(enriched_data)
```

## Advantages Over Monte Carlo

1. **Realistic Distributions**: Learns actual patterns from real data instead of parametric assumptions
2. **Mode Coverage**: Captures complex multi-modal distributions in financial behavior
3. **Scalability**: Once trained, generation is fast and scalable
4. **Adaptability**: Can be retrained on new data distributions
5. **Hybrid Reliability**: Combines generative realism with deterministic correctness

## References

- TimeGAN: Yoon, J., Jarrett, D., & van der Schaar, M. (2019). "Time-series Generative Adversarial Networks"
- DoppelGANger: Mottini, D., et al. (2022). "DoppelGANger: Semi-Supervised Generation of Time-Series"
- CTGAN: Xu, L., et al. (2019). "Modeling Tabular Data using Conditional GAN"
