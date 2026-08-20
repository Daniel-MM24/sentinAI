# Phase 1: Synthetic Data Generation

## Objective

Generate statistically realistic M-PESA transactions across synthetic customer profiles, covering a fiscal year (FY25: 2024-07-01 to 2025-06-30). The generated data must preserve the statistical properties of real M-PESA transaction flows while guaranteeing zero privacy risk — no real customer data ever touches the system.

## Architecture

```
simulation_profiles.yaml     behavioral_generator.py      stratified_profiles.py
         │                            │                           │
         ▼                            ▼                           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      GENERATOR ENGINE                               │
    │                                                                     │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
    │  │ Log-Normal  │  │ Constrained │  │ Inhomo.     │  │ Archetype │  │
    │  │ Sampler     │─►│ Random Walk │─►│ Poisson     │─►│ Profiles  │  │
    │  │ (Values)    │  │ (Balances)  │  │ (Thinning)  │  │ (Behav.)  │  │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         OUTPUTS                                     │
    │  ┌──────────────────────┐  ┌──────────────────┐                     │
    │  │ customer_profiles    │  │ detailed_tx       │                    │
    │  │ (1,000 profiles)     │  │ (10,000 rows)     │                    │
    │  └──────────────────────┘  └──────────────────┘                     │
    └─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Stratified Customer Profiles (`stratified_profiles.py`)

Generates 1,000 customer records with demographic and behavioral attributes:

- **KYC Tiers**: Tier 1 (60%), Tier 2 (30%), Tier 3 (10%)
- **Archetypes**: Retail Heavy (15%), Retail Standard (70%), Micro-Merchant (12%), Corporate (3%)
- **Demographics**: Age (18–80, mean 32), Gender (49% M / 51% F), County (47 counties, population-weighted)
- **Income**: Low (45%), Medium (40%), High (15%)
- **Tier caps**: Tier 1 (50K balance, 100K daily velocity), Tier 2 (500K balance, 1M velocity), Tier 3 (5M balance, 10M velocity)
- **Output path**: `data/bronze/customers/customer_profiles.csv`

### 2. Behavioral Transaction Generation (`behavioral_generator.py`)

Core transaction generation engine that produces realistic M-PESA transaction histories:

- **Transaction types**: 7 types with calibrated probabilities
  - Send Money (25%), Received Money (20%), Agent Deposit (10%)
  - Agent Withdrawal (10%), PayBill (15%), Buy Goods (12%), Others (8%)
- **Value generation**: Per-archetype LogNormal distribution
  - Retail Standard: μ=6.5, σ=1.2 → median ~KES 665
  - Retail Heavy: μ=7.2, σ=1.3 → median ~KES 1,340
  - Micro-Merchant: μ=8.0, σ=1.4 → median ~KES 2,980
  - Corporate: μ=9.5, σ=1.6 → median ~KES 13,360
- **Kadogo thresholds**: <KES 100 for P2P, <KES 200 for merchant — auto-categorized
- **Tier-specific caps**: Tier 1 (KES 50K), Tier 2 (KES 500K), Tier 3 (KES 5M)
- **Constrained random walk**: 100-attempt rejection sampling with look-ahead validation (5 tx window) and geometric amount scaling (1.0→0.1) when balance is tight
- **Balance initialization**: Opening balance sampled uniformly from KES 500–5,000
- **High-risk entity flagging**:
  - Betting platform transactions (~3%): SportPesa, Betika, Betway, 1xBet, 22Bet, etc.
  - International transfers (~2%): Western Union, MoneyGram, WorldRemit, Remitly, Wise, PayPal, etc.
  - Flags attached per-transaction (`is_betting`, `is_international`)
- **Balance enforcement**:
  - Rejection sampling: no transaction can cause balance to go negative
  - Look-ahead validation: next 5 transactions are simulated before accepting current one
  - Amount adjustment: if original amount fails look-ahead, tries geometric scaling (0.9, 0.8, ..., 0.1)
  - Per-customer exhaustion: 100 consecutive failures ends the customer's sequence
  - Daily velocity caps per tier prevent excessive daily outflow accumulation
  - Running balance tracked after each transaction; balance_max/min/sum tracked per customer
- **Temporal model**: Inhomogeneous Poisson with thinning algorithm (see Temporal Pattern Model)
- **Output**: `data/detailed_transactions.csv` (~10K rows)

### 3. Statistical Distribution Configuration (`synthetic_distributions.py`)

Centralized distribution parameterization:

```python
# Value distributions per archetype
_ARCHETYPE_AMOUNT_PARAMS = {
    "retail_heavy":     {"mean": 7.2, "sigma": 1.3},
    "retail_standard":  {"mean": 6.5, "sigma": 1.2},
    "micro_merchant":   {"mean": 8.0, "sigma": 1.4},
    "corporate":        {"mean": 9.5, "sigma": 1.6},
}
```

### 4. Temporal Pattern Model (`temporal_model.py`)

Inhomogeneous Poisson process for realistic transaction timing — generates timestamps via thinning:

- **168-hour weekly intensity vector**: Hourly patterns indexed as `(dow × 24 + hour)`
- **Diurnal patterns**: Morning ramp (06:00–09:00), peak (10:00–11:00 / 16:00–19:00), overnight lull (23:00–05:00)
- **Day-of-week effects**: Weekday intensity up to 1.0, Saturday max 0.55, Sunday max 0.35
- **Monthly seasonality**: 12 factors (0.85–1.20), salary-effect surge around 25th of each month (2x)
- **School fees waves**: January, May, September (3x multiplier)
- **Holiday surge**: December (1.5x multiplier)
- **Weekend reduction**: Saturday ~0.6x, Sunday ~0.5x (from 168h vector)

### 5. Generator Engine (`behavioral_generator.py`)

Orchestration layer within BehavioralTransactionGenerator:

- Manages per-customer state (balance, daily outflow, tier, balance stats)
- Coordinates LogNormal value sampling with constrained random walk
- Generates timestamps via thinning algorithm on 168h intensity vector
- Detects and prevents balance constraint violations with look-ahead (5 tx window) and amount scaling
- Tracks per-customer balance stats: opening_balance, balance_max, balance_min, balance_sum, balance_count
- Implements early sequence termination when a customer exhausts 100 consecutive rejection attempts
- Logs generation progress with statistics on distribution, risk flags, temporal spread
- Validates ledger continuity post-generation

### 6. AML Scenario Injection (`aml_scenario_injector.py`)

Injects known money laundering patterns into 2% of customers:

- **Scenario split**: Smurfing (40%), Layering (30%), Mule Account (20%), Circular Trading (10%)
- **Smurfing**: 30+ Send Money tx under KES 100K to 5-15 counterparties
- **Layering**: 4+ account chains with transfers within 24-hour windows, 2-5 cycles
- **Mule Account**: 5+ receive-withdraw cycles, 80%+ withdrawal within 24h
- **Circular Trading**: 3-5 account loops with 5-10 iterations, ±20% amount jitter
- **Output**: `data/aml_ground_truth.csv` with `user_id`, `is_launderer`, `aml_scenario`

### 7. Temporal Feature Extraction (`temporal_features.py`)

Stateful temporal pattern tracking with rolling window aggregators:

- **RollingBuffer**: Per-customer 30-day sliding window with auto-eviction
- **Daily patterns**: Dominant hour, morning/lunch/evening window %, hourly entropy
- **Weekly patterns**: Weekend ratio, weekend anomaly detection
- **Monthly patterns**: Month-end ratio, salary receipt ratio, school fee flag
- **Rolling aggregators**: 1h/24h count, 24h volume, 7d avg value, 30d volume/count
- **Output**: `data/temporal_features.csv` (1,000 rows × 24 columns)

## Monte Carlo Simulation Details

### Transaction Value Sampling

Values are drawn from archetype-specific LogNormal distributions:
```
value ~ exp(μ + σ × Z)  where Z ~ N(0, 1)
amount = constrained_random_walk(value, tier_cap, balance, direction)
```

- Retail Standard: μ=6.5, σ=1.2 → median ~KES 665
- Corporate: μ=9.5, σ=1.6 → median ~KES 13,360
- Kadogo P2P: <KES 100; Kadogo merchant: <KES 200

### Transaction Count Sampling

~10 transactions per customer on average, configurable via `n_transactions` parameter.

### Inter-Arrival Time Generation (Thinning Algorithm)

```
1. Draw Δt ∼ Exp(λ_max)              # homogeneous proposal
2. t = t + Δt                         # accumulate monotonically
3. Compute λ(t) = intensity(t)         # from 168h × monthly × seasonal × eom
4. Accept with P = λ(t) / λ_true_max  # rejection sampling (thinning)
5. If rejected, goto 1                # t already advanced
```

### Diurnal Pattern Validation

| Time Range | Expected from 168h Vector | Generator Output (10K tx) |
| :--- | :---: | :---: |
| 00:00–05:59 | Low (0.01–0.02) | ~0.9% |
| 06:00–08:59 | Ramp (0.05–0.40) | ~8.4% |
| 09:00–11:59 | Peak (0.30–1.00) | ~20.7% |
| 12:00–14:59 | Mid (0.30–0.85) | ~19.2% |
| 15:00–17:59 | Peak (0.65–1.00) | ~20.5% |
| 18:00–20:59 | High (0.15–0.90) | ~18.6% |
| 21:00–23:59 | Low (0.15–0.60) | ~11.7% |

## Output Summary

| Output | Rows | Columns |
| :--- | :---: | :---: |
| customer_profiles.csv | 1,000 | 3 |
| detailed_transactions.csv | 10,000 | 20 |
| aml_ground_truth.csv | 1,000 | 3 |
| temporal_features.csv | 1,000 | 24 |

### Output Columns (detailed_transactions.csv)

| Column | Type | Description |
| :--- | :--- | :--- |
| transaction_id | str | Unique TXN_XXXXXXXXXX identifier |
| customer_id | str | Customer reference |
| counterparty | str | Counterparty name |
| transaction_type | str | One of 7 M-PESA transaction types |
| amount | float64 | Transaction value in KES |
| direction | str | "inflow" or "outflow" |
| timestamp | str | ISO 8601 timestamp with timezone |
| paid_in | float64 | amount if inflow, 0 if outflow |
| paid_out | float64 | amount if outflow, 0 if inflow |
| balance | float64 | Running balance after transaction |
| balance_after | float64 | Alias of balance (downstream compatibility) |
| tier | int64 | Customer KYC tier (1-3) |
| hour | int64 | Timestamp hour (0-23) |
| day_of_week | int64 | 0=Monday, 6=Sunday |
| month | int64 | Calendar month (1-12) |
| is_weekend | bool | Saturday or Sunday |
| is_night | bool | 22:00-05:59 |
| is_betting | bool | Betting platform transaction (~3%) |
| is_international | bool | International transfer (~2%) |
| is_kadogo | bool | Below Kadogo threshold |

## Configuration Reference

Key parameters in `BehavioralGeneratorConfig` (dataclass):

| Parameter | Default | Description |
| :--- | :---: | :--- |
| transaction_type_probs | 7-type distribution | Type probability breakdown |
| betting_probability | 0.03 | Betting flag rate |
| international_probability | 0.02 | International flag rate |
| amount_mean | 6.5 | Global fallback log-normal μ |
| amount_std | 1.2 | Global fallback log-normal σ |
| kadogo_p2p_threshold | 100.0 | P2P Kadogo threshold (KES) |
| kadogo_merchant_threshold | 200.0 | Merchant Kadogo threshold (KES) |
| daily_velocity_caps | Tier-dependent | Tier 1: 100K, Tier 2: 1M, Tier 3: 10M |
| max_rejection_attempts | 100 | Max retries per transaction |
| look_ahead_window | 5 | Look-ahead validation depth |
| initial_balance_range | (500.0, 5000.0) | Fallback opening balance range (KES) |
| seed | 42 | RNG seed for reproducibility |

## Validation

- **Balance integrity**: No negative balances; ledger continuity verified per customer (Balance_t = Balance_{t-1} + PaidIn - PaidOut)
- **Tier compliance**: No transaction exceeds KYC tier balance cap; daily velocity cap enforced
- **Temporal validation**: All timestamps within FY25 window; thinning algorithm produces correct diurnal/weekend/monthly distribution
- **Type distribution**: Generated transaction types within expected range of configured probabilities
- **Risk flagging**: Betting (~3%) and international (~2%) flags at configured rates
- **Archetype compliance**: Log-normal value parameters produce expected median/max per archetype
- **Schema compliance**: 20 columns with correct types; zero nulls in output
