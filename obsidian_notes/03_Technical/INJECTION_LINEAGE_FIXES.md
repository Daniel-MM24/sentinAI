---
created: 2026-07-16
aliases: [Injector Migration, OpenLineage Fallback, Gold Column Mapping]
tags: [Technical, BugFix, Architecture, OpenLineage]
---

# Anomaly Injection & OpenLineage Fixes

## Change 1: Anomaly Injection Moved into `generate_normalized()`

### Problem

`FinancialAnomalyInjector.inject()` crashed with `ColumnNotFoundError` because the injector references 26 INJECTABLE_FEATURES, but `AMLGenerator.generate_normalized()` stripped aggregate columns from the transactions DataFrame *before* the injector ran. 6 of 8 injection methods failed:

| Method | Missing Columns |
|---|---|
| `_inject_velocity_funnel` | `tx_count_1h`, `tx_count_24h`, `burst_ratio`, `funnel_score` |
| `_inject_mule_activity` | `pass_through_ratio`, `zero_balance_frequency`, `balance_retention_ratio`, `balance_depletion_rate` |
| `_inject_layering` | `degree_centrality`, `reciprocity_ratio`, `new_relationships_7d`, `behavioral_shift_score` |
| `_inject_circular_trading` | `community_id`, `degree_centrality` |
| `_inject_temporal_anomaly` | `hour_of_day`, `device_changes_7d`, `location_entropy` |
| `_inject_ceiling_violation` | `wallet_tier_encoded` (customer static, stripped separately), `current_balance` |

### Solution

Moved anomaly injection **inside** `AMLGenerator.generate_normalized()` so it operates on the combined DataFrame (before aggregate columns are stripped):

- **New signature:** `generate_normalized(anomaly_ratio=None, anomaly_seed=None)`
- When `anomaly_ratio > 0`, creates a `FinancialAnomalyInjector` internally, calls `injector.inject(combined_df)` on the full combined DataFrame, **then** strips customer static + aggregate columns for the normalized return values
- Aggregate columns modified by the injector (e.g., `current_balance`, `community_id`, `hour_of_day`) are correctly stripped after injection — they were never intended to survive to Silver/Gold, as the Gold layer recomputes all rolling features from raw Silver data
- `anomaly_flag` (the training label) survives because it's not in the `aggregate_cols` or `customer_static_cols` sets

### Files Changed

- `src/data/synthetic_generator.py` — `generate_normalized()` signature + injection logic
- `src/data/medallion_stages.py` — `run_bronze_stage()` simplified: removed `FinancialAnomalyInjector` instantiation, now passes `anomaly_ratio`/`anomaly_seed` to `generate_normalized()`

### Architecture

```
Before:
  generator.generate_normalized() → stripped transactions
  injector.inject(stripped transactions) → CRASH (missing columns)

After:
  generator.generate_normalized(anomaly_ratio=0.015)
    ├── generate() → combined df (all columns)
    ├── injector.inject(combined df) → modifications to aggregates + anomaly_flag
    └── strip aggregates → clean normalized output
```

## Change 2: Gold Layer Silver Column Mapping

### Problem

`silver_to_transaction_features()` crashed with `ColumnNotFoundError` for `amount` because the Silver pipeline renames columns during `_normalize_bronze()`:
- `amount` → `transaction_amount`
- `post_tx_balance` → `account_balance_after`
- `current_balance` → `account_balance_before`

The existing rename only handled `entity_id` → `customer_id`.

### Solution

Extended the column rename block in `silver_to_transaction_features()` to handle all Silver-level renames as a single dictionary:

```python
renames = {}
if "transaction_amount" in transactions.columns and "amount" not in transactions.columns:
    renames["transaction_amount"] = "amount"
if "account_balance_after" in transactions.columns and "post_tx_balance" not in transactions.columns:
    renames["account_balance_after"] = "post_tx_balance"
if "account_balance_before" in transactions.columns and "current_balance" not in transactions.columns:
    renames["account_balance_before"] = "current_balance"
```

### Files Changed

- `src/datasets/gold.py` — `silver_to_transaction_features()` column rename block

## Change 3: OpenLineage Graceful Fallback

### Problem

`_create_openlineage_client()` unconditionally used `HttpTransport` with `http://localhost:5000` (the Pydantic `Settings` default), causing 5 retry attempts × 5s timeout = ~25s delay per lineage event. Each pipeline stage emits 3 events (START, COMPLETE, FAIL), resulting in ~75s of blocking retries per stage and log spam.

### Solution

Added `_probe_openlineage_backend()` — a lightweight health-check (HEAD request, 2s timeout) that runs once during client creation:

- **Reachable** → `HttpTransport` as before
- **Unreachable** → logs one `WARNING`, falls back to console transport for the entire session
- Zero retries, zero delay, zero log spam

```python
def _probe_openlineage_backend(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/v1/lineage", method="HEAD")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False
```

### Files Changed

- `src/data/lineage_decorator.py` — `_create_openlineage_client()` + new `_probe_openlineage_backend()` function

## Related

- [[03_Technical/SCHEMA_COMPATIBILITY_FIXES]] — Previous column mapping fixes
- [[01_Project/CHANGELOG]] — Version history
- [[02_Data/FEATURE_DICTIONARY]] — Feature definitions
