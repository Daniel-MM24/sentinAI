---
tags:
  - gold
  - feature-store
  - columns
  - modeling
  - AML
---

# Gold Feature Store — Column Reference

The Gold layer joins Silver transaction facts with customer dimensions and applies feature engineering. The output is a denormalized, feature-rich dataset at `data/gold/features/v{version}/`.

Output includes both partitioned Parquet (by `anomaly_case_id`) and a consolidated file (`gold_features_consolidated.parquet`).

## Column Reference

### 1. Identity & Keys

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `customer_id` | str (non-null) | Primary entity key | Entity join key, per-customer aggregations, train/test split boundary |
| `transaction_id` | str | Individual transaction ID | Prevents data leakage, MRM traceability |
| `anomaly_case_id` | str | Anomaly type label (SMURFING, LAYERING, NONE, etc.) | Multi-class classification target, stratified sampling |
| `partition_date` | str | Date extracted from timestamp | Physical partition key, temporal filtering |

### 2. Base Model Card Features

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `amount` | float | Transaction value (KES) | Primary anomaly signal — threshold proximity, structuring |
| `timestamp` | datetime (UTC) | Event time | Temporal ordering, rolling windows, lag features |
| `log_amount` | float | `log(amount + 1)` | Normalized version for linear model stability |
| `hour_of_day` | int (0–23) | Hour extracted from timestamp | Off-hours anomaly detection (3 a.m. structuring) |
| `day_of_week` | int (0–6) | Day of week | Weekend vs weekday prior shift |
| `is_weekend` | bool | Sat/Sun flag | Simplification of day_of_week for rule-based filters |
| `device_age_days` | int | Device tenure (days) | Account takeover signal — new device + anomaly |
| `sim_match_status` | bool | SIM registration matches KYC | Direct CBK regulatory flag |
| `wallet_tier_encoded` | int | M-PESA wallet tier (ordinal) | Higher tier = higher limits = different risk profile |
| `kyc_level_encoded` | int | KYC tier at transaction time | Low KYC + high amount = risk signal |
| `prev_fraud_flag_count_90d` | int | Historical fraud activity (90d window) | Recidivism — strongest individual AML predictor |

### 3. Feature-Engineered (Gold Layer)

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `transaction_velocity` | float | Total transactions per customer | Core velocity metric. CBK threshold monitoring |
| `mean_transaction_amount` | float | Per-customer average spend | Behavioral baseline for deviation scoring |
| `z_score_deviation` | float | `(amount - global_mean) / global_std` | Standardized outlier score, model-agnostic |
| `amount_near_threshold` | int (0/1) | Amount is KES 140K–149,999 | Regulator-driven — structuring below CBK reporting limit |
| `is_round_number_100k` | int (0/1) | Amount is exact multiple of KES 100K | Smurfing pattern — structured rounds |
| `is_stk_push` | int (0/1) | Transaction is C2B (STK Push) | Channel-based smurfing indicator |
| `is_b2c` | int (0/1) | Transaction is B2C (disbursement) | B2C reversal fraud pattern |
| `clv` | float | Customer Lifetime Value (sum of all amounts) | Affluent baseline — changes anomaly expectation |
| `high_risk_amount` | int (0/1) | Amount > KES 10,000 | Simple rule-based risk flag |

### 4. AML Tier 1 — Real-Time Velocity

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `tx_count_1h` | int | Transactions in last hour | Burst detection — rapid-fire structuring |
| `tx_count_24h` | int | Transactions in last 24 hours | Daily velocity ceiling — CBK compliance |
| `amount_sum_24h` | float | Total value in 24h window | Volume-based anomaly scoring |
| `amount_vs_profile_avg` | float | Deviation from customer's normal | Behavior shift detection |
| `time_since_last_tx` | float | Seconds since previous transaction | Bot-like regularity detection |

### 5. AML Tier 1 — Balance Patterns

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `avg_balance_30d` | float | 30-day average balance | Baseline for depletion detection |
| `balance_volatility_30d` | float | Balance variance over 30 days | Smurfing volatility signature |
| `current_balance` | float | Post-tx balance | Immediate state for pass-through detection |
| `min_balance_30d` | float | 30-day minimum | Floor detection — near-zero mule wallets |
| `max_balance_30d` | float | 30-day maximum | Peak accumulation for layering detection |
| `balance_retention_ratio` | float | How long money stays in wallet | Pass-through smurfing signature |
| `zero_balance_frequency` | float | How often wallet hits zero | Mule account indicator |

### 6. AML Tier 1 — Amount Patterns

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `amount_roundness` | float | How round the amount is | Structuring — round splits |
| `amount_just_below_threshold` | float | Distance below KES 140K | Deliberate threshold avoidance |
| `similar_amount_count_24h` | int | Near-identical amounts in 24h | Structured transfer pattern |
| `identical_amount_count_24h` | int | Exact duplicate amounts | Smurfing — same amount, many sends |
| `structuring_amount_entropy` | float | Entropy of amount distribution | Low entropy = structured splits |

### 7. AML Tier 1 — Network Features

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `pass_through_ratio` | float | Throughput / balance ratio | Money passes without lingering |
| `degree_centrality` | float | Normalized node centrality | Hub detection — central mule |
| `in_degree` | int | Unique senders to this node | Fan-in — consolidation in layering |
| `out_degree` | int | Unique receivers from this node | Fan-out — distribution in smurfing |
| `funnel_score` | float | In/out asymmetry score | One-directional flow detection |
| `reciprocity_ratio` | float | How often sends back to same peer | Normal vs. one-way mule patterns |

### 8. AML Tier 2 — Temporal & Rolling

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `burst_ratio` | float | Peak / normal rate ratio | Activity burst detection |
| `velocity_change_pct` | float | % change in activity rate | Behavior shift — mule activation |
| `is_anomalous_hour` | float | Transaction at unusual hour | 2–5 a.m. structuring flag |
| `device_changes_7d` | int | Device changes in 7 days | Account takeover / SIM swap |
| `device_change_flag` | float | Any device change (bool) | Binary simplification |
| `location_entropy` | float | Geographic dispersion | Multiple jurisdictions = risk |
| `rolling_avg_tx_amount_30d` | float | 30-day moving average amount | Trend detection — escalating amounts |
| `rolling_net_flow_7d` | float | 7-day net (inflow - outflow) | Accumulation vs disbursement cycle |
| `new_relationships_7d` | int | New transaction peers in 7 days | Network expansion — mule recruitment |

### 9. AML Tier 3 — Advanced Analytics

| Column | Type | Description | Modeling Role |
|---|---|---|---|
| `community_id` | float | Graph community cluster | Fraud ring membership |
| `behavioral_shift_score` | float | Composite behavior change score | Aggregate anomaly score |

### 10. Regulatory / Compliance

| Column | Type | Description | Role |
|---|---|---|---|
| `counterparty_id` | str | Transaction counterparty | Audit trail (not a model feature) |
| `counterparty_risk_tier` | str | CBK risk tier (LOW/MEDIUM/HIGH/CRITICAL) | Regulatory classification |
| `regulatory_report_status` | str | CBK threshold reporting status | Compliance flag |
| `data_provenance_hash` | str | SHA-256 lineage fingerprint | MRM audit trail |
| `post_tx_balance` | float | Balance after transaction | Audit trail / balance pattern input |
| `account_balance_before` | float | Pre-transaction balance | Audit trail |
| `account_balance_after` | float | Post-transaction balance | Audit trail |

## Important

### TVAE Hybrid Implementation (v2.0)

As of version 2.0, the SentinAI data generation pipeline has been transitioned from Monte Carlo simulation to a **TVAE Hybrid Pipeline**. The Gold layer now targets a streamlined **21-feature schema** optimized for tree-based AML models like XGBoost.

**Current Implementation**: The CustomerFeatureEngineer in `src/data/feature_engineering.py` computes exactly 10 downstream features for the 21-feature schema:

**Velocity/Temporal Features**:
- `tx_count_7d`: Rolling 7-day transaction count
- `volume_7d`: Rolling 7-day transaction volume
- `night_tx_ratio`: Ratio of transactions at night (22:00-06:00)
- `rapid_tx_ratio`: Ratio of transactions with ≤5min gap to previous transaction
- `volume_7d_vs_30d_ratio`: Burst Ratio (7d volume vs 30d volume)

**Network Features**:
- `distinct_counterparties_7d`: Number of unique counterparties in 7 days
- `fan_in_fan_out_ratio`: Ratio of incoming to outgoing transactions

**Structuring/Mule Features**:
- `close_to_limit_ratio`: Threshold evasion behavior
- `amount_roundness`: Synthetic bot behavior detection
- `balance_retention_ratio`: Pass-through account detection

**Full 21-Feature Schema**:
1. `customer_id` (Core)
2. `tier` (Core)
3. `archetype` (Core)
4. `transaction_type` (Core)
5. `amount` (Core)
6. `timestamp` (Core)
7. `direction` (Core)
8. `balance` (Core)
9. `tx_count_7d` (Temporal)
10. `volume_7d` (Temporal)
11. `night_tx_ratio` (Temporal)
12. `rapid_tx_ratio` (Temporal)
13. `volume_7d_vs_30d_ratio` (Temporal)
14. `is_international` (Network)
15. `distinct_counterparties_7d` (Network)
16. `fan_in_fan_out_ratio` (Network)
17. `close_to_limit_ratio` (Structuring)
18. `balance_retention_ratio` (Structuring)
19. `amount_roundness` (Structuring)
20. `is_launderer` (Label)
21. `aml_scenario` (Label)

See [[03_Technical/TVAE_HYBRID_IMPLEMENTATION]] for complete implementation details.

### Legacy Schema (Pre-v2.0)

The schema sections 4–9 above (AML Tier 1–3 features) represent the legacy 50+ column schema. These remain documented for reference but are **not used** in the current TVAE hybrid pipeline.
