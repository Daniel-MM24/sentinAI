# AML Scenario Injection Logic

## Overview

SentinAI injects four distinct AML typologies into the synthetic transaction data at a controlled ~2% prevalence rate. Each scenario is modeled as a behavioral overlay that generates additional scenario-specific transactions alongside the clean transaction history, creating realistic ground-truth labels for supervised model training.

**Module:** `src/data/aml_scenario_injector.py`
**Output:** `data/aml_ground_truth.csv` — per-customer labels (1,000 rows, 3 columns)

## Injection Strategy

```
Clean Data (10,000 transactions, 1,000 customers)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│             AML SCENARIO INJECTION                           │
│                                                             │
│  Phase 1: Select launderers (~20 customers = 2%)           │
│    ├── Smurfing (40%)          → 8  customers              │
│    ├── Layering (30%)          → 6  customers              │
│    ├── Mule Accounts (20%)     → 4  customers              │
│    └── Circular Trading (10%)  → 2  customers              │
│                                                             │
│  Phase 2: Generate scenario-specific transaction patterns  │
│    ├── Creates new transaction sequences per launderer      │
│    ├── Non-launderer transactions left untouched            │
│    └── Patterns respect CBK regulatory thresholds           │
│                                                             │
│  Phase 3: Build ground-truth labels                        │
│    ├── Output: data/aml_ground_truth.csv                   │
│    ├── user_id: customer identifier                         │
│    ├── is_launderer: True/False                             │
│    └── aml_scenario: smurfing/layering/mule_account/        │
│                      circular_trading/none                  │
└─────────────────────────────────────────────────────────────┘
```

## Scenario 1: Smurfing / Structuring (40% of launderers)

### Description
Breaking large amounts into many small transactions below KES 100,000 (the CBK reporting threshold for mobile money). Funds are split across multiple counterparties with varied amounts to avoid pattern detection.

### Injection Implementation

The injector generates 30+ Send Money transactions per smurfing customer. Key parameters:
- **Amounts:** Uniform(KES 100, KES 95,000) — always under the structuring threshold
- **Counterparties:** 5–15 unique counterparties per launderer
- **Timestamps:** Spread uniformly across the customer's active period
- **Direction:** 100% outflow (Send Money)

```python
def _inject_smurfing(self, customer_id, clean_txs, all_customers):
    n_tx = 30 + exponential(20)
    n_cp = uniform_int(5, 15)
    counterparties = sample(all_customers, n_cp)
    
    for i in range(n_tx):
        amount = uniform(100.0, 95000.0)
        cp = counterparties[i % len(counterparties)]
        timestamp = jitter(clean_start, clean_end)
        # Produces a Send Money outflow to cp
```

### Detection Signature

| Indicator | Expected Value |
| :--- | :---: |
| `send_receive_ratio` | Very high (pure outflows) |
| `avg_tx_value` | < KES 50,000 |
| `tx_count` | 30+ per launderer |
| `unique_counterparties` | 5–15 |
| `amount_variance` | High (uniform spread) |

## Scenario 2: Layering (30% of launderers)

### Description
Moving funds through a chain of intermediate accounts to obscure the audit trail. The injector creates transaction chains of 4+ layers where money flows from launderer → layer_1 → layer_2 → ... → layer_N → sink within 24-hour windows.

### Injection Implementation

```python
def _inject_layering(self, customer_id, clean_txs, all_customers):
    n_layers = max(4, exponential(2) + 4)
    chain = [launderer, layer_1, ..., layer_N, sink]
    
    for cycle in range(2, 5):  # 2-5 cycles
        burst_start = uniform over customer's active period
        for hop in range(len(chain) - 1):
            amount = lognormal(mean=8.0, sigma=1.0)  # KES ~3k-30k
            ts = burst_start + hop * 2h  # 2h between hops
            # Send Money from sender→receiver
            # Received Money at receiver←sender
```

Key parameters:
- **Minimum layers:** 4 (launderer + intermediaries + sink)
- **Time-decay:** Each hop within ~2 hours; full cycle within 24 hours
- **Cycles:** 2–5 per launderer
- **Amounts:** Log-normal (KES ~3,000–30,000) — varied per hop

## Scenario 3: Mule Account (20% of launderers)

### Description
Accounts that receive consolidated funds and immediately cash out via Agent Withdrawal. The injector generates receive-withdraw pairs where 80%+ of received funds are withdrawn within 24 hours.

### Injection Implementation

```python
def _inject_mule_account(self, customer_id, clean_txs, all_customers):
    n_cycles = 5 + exponential(5)  # 5+ cycles
    
    for i in range(n_cycles):
        receive_amount = lognormal(9.0, 0.8)  # KES ~8k-40k
        withdraw_pct = uniform(0.80, 1.00)    # 80-100% withdrawn
        withdraw_amount = receive_amount * withdraw_pct
        delay_hours = exponential(4)          # minutes to ~24h
        # Row 1: Received Money (inflow)
        # Row 2: Agent Withdrawal (outflow)
```

Key parameters:
- **Receive amounts:** Log-normal (KES ~8,000–40,000)
- **Withdrawal ratio:** 80–100% of received amount
- **Withdrawal delay:** Minutes to ~24 hours (exponential with λ=4h)
- **Min cycles:** 5 receive-withdraw pairs

## Scenario 4: Circular Trading (10% of launderers)

### Description
Self-referential transaction loops where money circulates among 3–5 accounts in 5–10 iterations. Amounts are varied ±20% per leg to avoid pattern detection.

### Injection Implementation

```python
def _inject_circular_trading(self, customer_id, clean_txs, all_customers):
    n_accounts = uniform_int(3, 5)
    n_iterations = uniform_int(5, 10)
    accounts = [launderer, account_1, ..., account_N]
    
    for iteration in range(n_iterations):
        amount = lognormal(8.5, 0.9)  # KES ~5k-15k
        for idx in range(len(accounts)):
            sender = accounts[idx]
            receiver = accounts[(idx + 1) % len(accounts)]
            leg_amount = amount * uniform(0.8, 1.2)  # ±20% jitter
            # Send Money sender→receiver
            # Received Money receiver←sender
```

Key parameters:
- **Cycle size:** 3–5 accounts
- **Iterations:** 5–10 rounds
- **Amount jitter:** ±20% per leg (avoids uniform amounts)
- **Amounts:** Log-normal (KES ~5,000–15,000)

## Anomaly Ratio Calibration

| Scenario | % of Launderers | % of Total Customers |
| :--- | :---: | :---: |
| Smurfing | 40% | ~0.8% |
| Layering | 30% | ~0.6% |
| Mule Accounts | 20% | ~0.4% |
| Circular Trading | 10% | ~0.2% |
| **Total** | **100%** | **~2%** |

## Output Schema

### `aml_ground_truth.csv`

| Column | Type | Description |
| :--- | :---: | :--- |
| `user_id` | string | Customer identifier (e.g., `CUST_000000`) |
| `is_launderer` | boolean | Whether this customer is a launderer |
| `aml_scenario` | string | One of: `smurfing`, `layering`, `mule_account`, `circular_trading`, `none` |

The ground-truth labels are used as the target variable for supervised AML model training and evaluation. Non-launderers have `aml_scenario = 'none'`.
