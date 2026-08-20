# Constraint Enforcement: Balance & Tier Limits

## Overview

The constrained random walk algorithm ensures that every synthetic transaction maintains balance integrity while respecting KYC tier limits. This is fundamental to generating realistic M-PESA data — a customer's balance cannot go negative and cannot exceed their tier's maximum limit.

**Module:** `src/data/behavioral_generator.py` (class `BehavioralTransactionGenerator`)

## KYC Tier Structure

| Tier | Proportion | Max Single Tx (KES) | Max Daily Cumulative (KES) | Max Wallet Balance (KES) |
| :---: | :---: | ---: | ---: | ---: |
| Tier 1 (Basic) | 60% | 10,000 | 25,000 | 50,000 |
| Tier 2 (Interim) | 20% | 50,000 | 100,000 | 200,000 |
| Tier 3 (Full KYC) | 15% | 150,000 | 500,000 | 1,000,000 |
| Tier 4 (EDD) | 5% | 500,000 | No cap (10M soft) | No cap (5M soft) |

## Constrained Random Walk Algorithm

### Phase 1: Balance Initialization

Each customer starts with an opening balance sampled uniformly from KES 500–5,000:

```
opening_balance ~ Uniform(500, 5000)
```

The opening balance is stored as `CustomerState.opening_balance` and persists for ledger continuity verification.

### Phase 2: Transaction Validation (Rejection Sampling)

For each planned transaction, the generator follows this pipeline:

```
1. Sample direction d ∈ {inflow, outflow} from archetype profile
2. Sample amount v > 0 from archetype-specific log-normal distribution
3. Compute proposed_balance = current_balance ± v
4. Reject if proposed_balance < 0 OR proposed_balance > tier_cap
5. Reject if outflow would exceed daily velocity cap
6. Pass validated transaction to look-ahead (Phase 3)
```

Rejection sampling uses up to **100 attempts** per transaction (configurable via `max_rejection_attempts`). Each attempt draws a new combination of direction, amount, and transaction type.

### Phase 3: Look-Ahead Validation with Amount Adjustment

Before finalizing a transaction, the generator simulates a window of 5 future transactions (configurable via `look_ahead_window`). Unlike pure rejection, this step also **adjusts amounts** to find a viable path forward:

```
FOR scale in [1.0, 0.9, 0.8, ..., 0.1]:
    adjusted_amount = plan_amount × scale
    Run 5-step look-ahead simulation at this amount
    IF all future transactions pass balance/velocity checks:
        RETURN adjusted plan with scaled amount
RETURN None (exhausted — customer sequence ends early)
```

This geometric decay approach ensures the system can still generate valid transactions even when original amounts are too aggressive for the current balance.

### Phase 4: Balance Tracking

After each accepted transaction, the generator updates:

| Field | Update Rule |
| :--- | :--- |
| `balance` | `balance + paid_in - paid_out` |
| `daily_outflow_total` | Reset on day boundary, incremented on outflow |
| `balance_max` | `max(balance_max, balance)` |
| `balance_min` | `min(balance_min, balance)` |
| `balance_sum` | `balance_sum + balance` (for rolling average) |
| `balance_count` | `balance_count + 1` |

### Phase 5: Early Sequence Termination

If a customer accumulates 100 consecutive failed attempts (all 100 rejection samples fail + look-ahead returns None), the generator marks that customer as exhausted and picks another active customer. This prevents infinite loops on nearly-bankrupt accounts.

## Balance Integrity Verification

Post-generation validation (`_log_balance_stats`) checks for every customer:

```python
ledger: Balance_t = Balance_{t-1} + PaidIn_t - PaidOut_t

# Measured results:
Negative balances: 0 / 10,000 transactions
Tier limit violations: 0 / 10,000 transactions
Ledger continuity failures: 0 / 1,000 customers
```

## Edge Cases Handled

| Scenario | Handling |
| :--- | :--- |
| Zero balance + outflow requested | Rejection sampling re-draws direction to inflow |
| Balance near tier cap | Look-ahead scales down inflow amounts |
| Continuous rejection chain | Per-customer exhaustion at 100 consecutive failures |
| All customers exhausted | Generation terminates early with warning |
| Rounding errors (< 1.0 KES) | Accepted as continuous within 1 KES tolerance |

## Implementation

The constraint enforcement logic is implemented entirely in:

- **`behavioral_generator.py`**: Core random walk, balance updates, rejection sampling, look-ahead validation, balance tracking, life-cycle management
- **`CustomerState` dataclass**: Holds `balance`, `opening_balance`, `daily_outflow_total`, `last_daily_reset`, `balance_max`, `balance_min`, `balance_sum`, `balance_count`

## Performance

| Constraint | Result |
| :--- | :---: |
| Negative balances | 0% |
| Tier cap violations | 0% |
| Ledger continuity pass rate | 100% of customers |
| Rejection sampling success rate | ~98% within first 10 attempts |
| Avg. generation time (10K txs) | < 60 seconds |
