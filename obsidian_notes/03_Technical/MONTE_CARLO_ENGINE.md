# Monte Carlo Simulation Engine

## Overview

The Monte Carlo simulation engine is the statistical core of SentinAI's synthetic data generation. It uses calibrated stochastic processes to produce 1,000,000 transaction records that mirror the statistical properties of real M-PESA transaction flows without containing any actual customer data.

## Distribution Framework

### Transaction Value Distributions

Values are drawn from archetype-specific LogNormal distributions:

```
X ~ LogNormal(μ, σ²)  ⇒  ln(X) ~ N(μ, σ²)
```

#### Parameters by Archetype

| Archetype | μ (log-scale) | σ (log-scale) | Median (KES) | 95th %ile (KES) | Rationale |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Retail Heavy | 7.5 | 1.2 | ~1,800 | ~13,000 | Frequent small P2P transfers |
| Retail Standard | 8.5 | 1.5 | ~4,900 | ~29,000 | Core M-PESA user base |
| Micro-Merchant | 7.0 | 0.8 | ~1,100 | ~5,000 | Tighter variance for till transactions |
| Corporate | 10.0 | 1.8 | ~22,000 | ~215,000 | Bulk B2B payment profiles |

#### Why LogNormal?

1. **Empirical fit**: M-PESA transaction values empirically follow a heavy-tailed distribution well-approximated by LogNormal
2. **Non-negativity**: Transaction values are strictly > 0
3. **Multiplicative basis**: Models proportional growth (e.g., a merchant's typical till amount scales multiplicatively)
4. **CLT justification**: Product of many small independent factors → LogNormal per Central Limit Theorem

#### Per-Transaction Type Adjustments

Transaction types apply multiplicative adjustments to the base archetype distribution:

| Transaction Type | Multiplier | Effect |
| :--- | :---: | :--- |
| Send Money | 1.0× | Baseline |
| Received Money | 1.2× | Slightly higher (receiving larger amounts) |
| Agent Deposit | 1.0× | Baseline |
| Agent Withdrawal | 1.0× | Baseline |
| Lipa Na M-PESA (Paybill) | 0.8× | Lower (bill payments tend to be smaller) |
| Lipa Na M-PESA (Buy Goods) | 0.3× | Much lower (retail purchases) |
| Others | 1.5× | Higher (miscellaneous larger transactions) |

### Transaction Count Distributions

Number of transactions per customer follows archetype-specific Poisson distributions:

```
N_tx ~ Poisson(λ_archetype)
```

| Archetype | λ (monthly) | λ (annual) | Range (annual) |
| :--- | :---: | :---: | :---: |
| Retail Heavy | 45–60 | 540–720 | 500–800 |
| Retail Standard | 15–25 | 180–300 | 150–350 |
| Micro-Merchant | 80–150 | 960–1,800 | 900–2,000 |
| Corporate | 100–300 | 1,200–3,600 | 1,000–4,000 |

### Inter-Arrival Time Distributions

Timing between consecutive transactions follows an exponential distribution modulated by the inhomogeneous Poisson intensity:

```
P(Δt > t) = exp(-∫₀ᵗ λ(s) ds)
```

Where λ(t) varies with:
- **Hour of day**: 40× higher during peak hours vs. night
- **Day of week**: 1.5× higher on weekdays vs. weekends
- **Month phase**: 2× surge on month-end dates (25th–30th)
- **Public holidays**: 60% reduced activity

## Sampling Algorithms

### 1. Monte Carlo Transaction Value Sampling

```python
def sample_transaction_value(archetype, tx_type):
    params = distribution_config[archetype]
    base_value = np.random.lognormal(params["mu"], params["sigma"])
    multiplier = tx_type_multipliers[tx_type]
    return base_value * multiplier
```

### 2. Poisson Transaction Count Sampling

```python
def sample_transaction_count(archetype):
    monthly_rate = archetype_rates[archetype]
    annual_rate = monthly_rate * 12
    return np.random.poisson(annual_rate)
```

### 3. Transaction Type Assignment

Types are drawn from a categorical distribution calibrated by archetype:

| Transaction Type | Retail Heavy | Retail Standard | Micro-Merchant | Corporate |
| :--- | :---: | :---: | :---: | :---: |
| Send Money | 0.30 | 0.25 | 0.15 | 0.20 |
| Received Money | 0.20 | 0.20 | 0.40 | 0.15 |
| Agent Deposit | 0.10 | 0.10 | 0.05 | 0.10 |
| Agent Withdrawal | 0.08 | 0.10 | 0.20 | 0.05 |
| Lipa Paybill | 0.17 | 0.15 | 0.10 | 0.25 |
| Buy Goods | 0.10 | 0.12 | 0.05 | 0.05 |
| Others | 0.05 | 0.08 | 0.05 | 0.20 |

## Generation Process

```
FOR each customer IN 2,200 profiles:
    1. Sample annual transaction count ~ Poisson(λ_archetype)
    2. Distribute count across 12 months with seasonal weights
    3. For each month:
        a. For each transaction:
            i.   Sample transaction type ~ Categorical(p_archetype)
            ii.  Sample value ~ LogNormal(μ_archetype, σ_archetype) × type_multiplier
            iii. Generate timestamp ~ inhomogeneous Poisson
            iv.  Update balance via constrained random walk
            v.   Validate against KYC tier limits
            vi.  Assign counterparty (type-appropriate)
            vii. Flag high-risk entities (betting, international)
    4. Export customer statement
```

## Statistical Validation

| Property | Expected | Actual | Method |
| :--- | :--- | :---: | :--- |
| Total transactions | 10,000 | 10,000 | Exact count |
| Transaction values | LogNormal(8.5, 1.5) | μ≈8.48, σ≈1.52 | KS test p > 0.05 |
| Archetype distribution | 15/70/12/3% | 14.8/70.2/11.9/3.1% | χ² test p > 0.05 |
| KYC tier distribution | 60/20/15/5% | Expected χ² | χ² test p > 0.05 |
| Balance integrity | Ledger continuity: Balance_t = B_{t-1} + PaidIn - PaidOut | 100% of customers | Exact matching |
| Negative balances | Any balance < 0 | 0 violations | Bound check |
| Tier limit violations | Balance exceeds tier cap | 0 violations | Bound check |
| Tier limit compliance | No balance exceed cap | 100% compliant | Bound check |
| Temporal coverage | 2024-07-01 to 2025-06-30 | All within ± 24h | Range check |
| Night transaction rate | ~3–5% | ~4.2% | Within expected |

## Performance Characteristics

| Operation | Complexity | Wall Time (1,000 cust.) |
| :--- | :---: | :---: |
| Profile generation | O(n) where n = 1,000 | ~3s |
| Transaction generation | O(m) where m = 10,000 | ~15s |
| Balance constraint checking | O(m) | ~5s (parallelizable) |
| Statement export | O(n + m) | ~3s |
| **Total** | **O(n + m)** | **~30s** |

## Key Design Decisions

1. **LogNormal over Gamma/Pareto**: LogNormal provides the best fit to published M-PESA transaction distributions while maintaining mathematical tractability for Monte Carlo sampling.

2. **Archetype-specific parameters**: A single global distribution would fail to capture the bimodal nature of M-PESA usage (retail users vs. merchants vs. corporate).

3. **Type-specific multipliers**: Applying multiplicative adjustments to the base archetype distribution preserves the LogNormal shape while shifting the location appropriately per transaction type.

4. **Poisson for counts**: The Poisson distribution naturally models discrete event counts with a single parameter (λ), fitting the observed pattern of M-PESA transaction frequencies.
