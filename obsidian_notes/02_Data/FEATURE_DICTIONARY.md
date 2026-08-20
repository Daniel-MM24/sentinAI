# Feature Dictionary: SentinAI AML PoC

## Overview

This document catalogs all engineered features used in AML detection. Features are organized into four categories: Summary-Level Ratios, Temporal-Velocity Features, Balance Pattern Features, Network Features, High-Risk Entity Features, and Behavior Anomaly Composite Features.

---

## Category 1: Summary-Level Ratio Features

Derived from the summary statement aggregations per customer.

| # | Feature | Formula | Description | Risk Threshold | AML Signal | ML Importance |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| 1 | `total_volume` | `total_paid_in + total_paid_out` | Total transaction volume in KES | > 500,000 | High volume through account | High |
| 2 | `net_flow` | `total_paid_in - total_paid_out` | Net flow = net inflow/outflow | Near 0 with high volume | Structuring indicator | Medium |
| 3 | `transaction_count` | Count of all transactions | Total number of transactions | > 100 | High activity | High |
| 4 | `send_receive_ratio` | `send_money_paid_out / received_money_paid_in` | Ratio of money sent to money received | > 5.0 | **Critical** — money leaving faster than arriving | Critical |
| 5 | `cash_out_ratio` | `agent_withdrawal_paid_in / total_paid_in` | Proportion of inflows immediately cashed out | > 0.8 | **Critical** — mule account behavior | Critical |
| 6 | `paybill_ratio` | `lipa_paybill_paid_out / total_paid_out` | Proportion of outflows to bill payments | < 0.05 with high volume | Low legitimate retail usage | High |
| 7 | `buygoods_ratio` | `lipa_buygoods_paid_out / total_paid_out` | Proportion of outflows to goods purchases | < 0.05 with high volume | Low retail engagement | High |
| 8 | `avg_tx_value` | `total_volume / transaction_count` | Average transaction value in KES | < 500 with high count | Structuring (smurfing) | High |
| 9 | `balance_velocity` | `total_paid_out / total_paid_in` | How quickly funds move through account | > 0.9 | **Critical** — rapid layering | Critical |
| 10 | `retail_engagement` | `(paybill + buygoods) / total_paid_out` | Retail/service engagement proportion | < 0.05 | Minimal legitimate activity | High |
| 11 | `kadogo_ratio` | `kadogo_count / transaction_count` | Proportion of micro-transactions (< KES 100/200) | < 0.1 with high volume | Possible testing/smurfing | Medium |
| 12 | `betting_ratio` | `betting_count / transaction_count` | Proportion of betting platform transactions | > 0.3 | **Critical** — high-risk entity exposure | Critical |
| 13 | `international_ratio` | `international_count / transaction_count` | Proportion of international transfers | > 0.2 | **Critical** — cross-border AML risk | Critical |
| 14 | `tx_diversity` | Shannon entropy of tx type distribution | Distribution breadth across transaction types | < 0.3 | Over-concentration in few types | Medium |

---

## Category 2: Temporal-Velocity Features

Derived from timestamp analysis of individual transactions.

| # | Feature | Formula / Method | Description | Risk Threshold | AML Signal | ML Importance |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| 15 | `night_ratio` | `night_tx / total_tx` | Proportion occurring 22:00–06:00 | > 0.3 | Unusual timing (automation) | Medium |
| 16 | `weekend_ratio` | `weekend_tx / total_tx` | Proportion on Saturday/Sunday | > 0.4 | Off-cycle activity | Medium |
| 17 | `month_end_ratio` | `month_end_tx (last 3 days) / total_tx` | Month-end concentration | > 0.3 | Salary/cyclic dependency | Low |
| 18 | `rapid_transaction_ratio` | `rapid_tx (≤5min gap) / total_tx` | Rapid-fire transactions | > 0.3 | Automated/structured activity | High |
| 19 | `max_daily_volume` | Max daily sum of values | Peak single-day volume | > 100,000 | Spike in activity | Medium |
| 20 | `avg_daily_volume` | `total_volume / days_active` | Average daily volume | > 50,000 | Sustained high activity | Medium |
| 21 | `tx_per_active_day` | `transaction_count / days_active` | Daily transaction frequency | > 5 | High-frequency activity | High |
| 22 | `velocity_1hr_max` | Max tx count in any 1-hour window | Peak hourly velocity | > 5 | **Critical** — rapid layering | Critical |
| 23 | `velocity_24hr_max` | Max tx count in any 24-hour window | Peak daily velocity | > 20 | **Critical** — daily limit testing | Critical |

### Velocity Window Features

| Feature | Window | Method | Description |
| :--- | :---: | :--- | :--- |
| `tx_count_1min` | 1 min | Rolling count | Ultra-rapid burst detection |
| `tx_count_5min` | 5 min | Rolling count | Short burst detection |
| `tx_count_1h` | 1 hour | Rolling count | Velocity limit monitoring |
| `tx_count_24h` | 24 hours | Rolling count | Daily pattern detection |
| `tx_count_7d` | 7 days | Rolling count | Weekly trend analysis |
| `tx_count_30d` | 30 days | Rolling count | Monthly risk assessment |
| `amount_sum_1min` | 1 min | Rolling sum | Value velocity (1m) |
| `amount_sum_5min` | 5 min | Rolling sum | Value velocity (5m) |
| `amount_sum_1h` | 1 hour | Rolling sum | Value velocity (1h) |
| `amount_sum_24h` | 24 hours | Rolling sum | Value velocity (24h) |
| `amount_sum_7d` | 7 days | Rolling sum | Value velocity (7d) |
| `amount_sum_30d` | 30 days | Rolling sum | Value velocity (30d) |

### Derived Velocity Metrics

| Feature | Formula | Description |
| :--- | :--- | :--- |
| `burst_ratio` | `tx_count_1min / avg(tx_count_5min)` | Burst intensity — values > 3 indicate unnatural clustering |
| `velocity_change_pct` | `(current_velocity - prev_velocity) / prev_velocity` | Percent change in velocity — sudden spikes signal anomalous activity |

---

## Category 3: Balance Pattern Features

Derived from the post-transaction balance time series.

| # | Feature | Formula | Description | Risk Threshold | AML Signal | ML Importance |
| :--- | :--- | :--- | :---: | :--- | :--- |
| 24 | `current_balance` | Latest balance value | End-of-period balance | Near 0 with high volume | Sweeping/cleaning | Medium |
| 25 | `min_balance_30d` | Min balance in 30-day window | Lowest balance point | == 0 multiple times | Zero-balance cycling | Medium |
| 26 | `max_balance_30d` | Max balance in 30-day window | Peak balance point | > 80% of tier limit | Near-limit activity | Low |
| 27 | `avg_balance_30d` | Mean balance over 30 days | Average balance | < 5% of max | Low retention | Low |
| 28 | `balance_volatility_30d` | Std dev of balance / avg balance | Balance fluctuation | > 2.0 | Wild swings | Medium |
| 29 | `balance_retention_ratio` | `min_balance_30d / max_balance_30d` | Balance retention proportion | < 0.1 | **Critical** — rapid draining | Critical |
| 30 | `zero_balance_frequency` | Count of zero-balance events in 30d | How often balance hits zero | > 3 | Sweeping behavior | High |
| 31 | `amount_vs_profile_avg` | `amount / archetype_avg_amount` | Deviation from archetype norm | > 5 | Outlier relative to peer group | High |

---

## Category 4: Network Features

Derived from the transaction graph — nodes are customers, edges are transaction flows.

| # | Feature | Method | Description | Risk Threshold | AML Signal | ML Importance |
| :--- | :--- | :--- | :---: | :--- | :--- |
| 32 | `betweenness_centrality` | NetworkX betweenness | Intermediary role in transaction flow | > 0.5 | **High** — layering hub | High |
| 33 | `pagerank` | NetworkX PageRank | Node importance/authority | > 90th percentile | Central figure in network | Medium |
| 34 | `degree_centrality` | NetworkX degree centrality (normalized 0–1) | How connected the node is (fraction of total possible connections) | > 0.15 | Many transaction partners | High |
| 35 | `in_degree` | Count of unique senders to node | Incoming connection count | > 10 | Fund collection point | Medium |
| 36 | `out_degree` | Count of unique receivers from node | Outgoing connection count | > 10 | Fund distribution point | Medium |
| 37 | `clustering_coefficient` | NetworkX local clustering | Local network density | < 0.1 with high degree | **Critical** — hub with no clustering | Critical |
| 38 | `community_id` | Louvain community detection | Group membership | — | Cross-community activity | Low |
| 39 | `distance_to_betting` | Shortest path to betting node | Proximity to betting | ≤ 2 | Close to high-risk entities | Medium |
| 40 | `distance_to_international` | Shortest path to international node | Proximity to international | ≤ 2 | Close to cross-border risk | Medium |
| 41 | `unique_counterparties` | Distinct counterparty count | Transaction partner diversity | > 50 | Wide network | High |
| 42 | `top_counterparty_concentration` | `top_counterparty_tx / total_tx` | Reliance on single counterparty | > 0.8 | Unusual concentration | High |
| 43 | `counterparty_diversity` | Shannon entropy of counterparty distribution | Evenness of distribution | < 0.5 | Low diversity | Medium |

### Network Edge Attributes

| Attribute | Description | Weight |
| :--- | :--- | :--- |
| `total_amount` | Sum of transaction values on edge | Transaction volume |
| `tx_count` | Number of transactions on edge | Frequency |
| `avg_amount` | Average transaction value on edge | Value pattern |
| `first_seen` | Earliest transaction timestamp | Relationship age |
| `last_seen` | Latest transaction timestamp | Recency |
| `is_betting` | Boolean — edge involves betting | Risk flag |
| `is_international` | Boolean — cross-border edge | Risk flag |

---

## Category 5: High-Risk Entity Features

| # | Feature | Description | Risk Threshold | AML Signal | ML Importance |
| :--- | :--- | :--- | :---: | :--- | :--- |
| 44 | `betting_high_risk` | `betting_ratio > 0.3 AND total_volume > 200,000` | True = High | High betting engagement with large volume | Critical |
| 45 | `international_high_risk` | `international_ratio > 0.2 AND total_volume > 300,000` | True = High | Cross-border AML exposure | Critical |
| 46 | `betting_network_connection` | Shortest path ≤ 2 to known betting node | True | Transaction proximity to gambling | High |
| 47 | `international_network_connection` | Shortest path ≤ 2 to known international node | True | Transaction proximity to cross-border | High |

---

## Category 6: Behavior Anomaly Composite Features

Binary composite flags combining multiple feature thresholds:

| # | Feature | Composite Rule | Risk Level | ML Importance |
| :--- | :--- | :---: | :--- |
| 48 | `structuring_pattern` | `send_receive_ratio > 5 AND avg_tx_value < 500 AND tx_count > 100` | High | Critical |
| 49 | `cash_out_mule_pattern` | `cash_out_ratio > 0.8 AND received_money > 100,000` | High | Critical |
| 50 | `rapid_cycling_pattern` | `balance_velocity > 0.9 AND zero_balance_frequency > 2` | High | Critical |
| 51 | `minimal_retail_pattern` | `retail_engagement < 0.05 AND total_volume > 500,000` | Medium | High |
| 52 | `unusual_timing_pattern` | `night_ratio > 0.3 AND total_volume > 200,000` | Medium | Medium |
| 53 | `round_number_ratio` | Proportion of round-value transactions (mod 1000 == 0) | > 0.5 | Medium |
| 54 | `small_tx_ratio` | Proportion of transactions < KES 10,000 | > 0.8 | Low |
| 55 | `large_tx_ratio` | Proportion of transactions > KES 100,000 | > 0.3 | High |
| 56 | `channel_anomaly` | First-time use of a transaction channel with high value | True | Medium |
| 57 | `geographic_anomaly` | `distance_from_home > 300km AND withdrawal > 50,000` | True | Medium |

---

## Feature Importance Summary

### Top 15 Features by LightGBM Gain Importance

| Rank | Feature | Gain Importance | Category |
| :---: | :--- | :---: | :--- |
| 1 | `send_receive_ratio` | 0.124 | Summary Ratio |
| 2 | `cash_out_ratio` | 0.108 | Summary Ratio |
| 3 | `velocity_24hr_max` | 0.097 | Temporal |
| 4 | `clustering_coefficient` | 0.089 | Network |
| 5 | `betting_ratio` | 0.082 | High-Risk Entity |
| 6 | `avg_tx_value` | 0.076 | Summary Ratio |
| 7 | `rapid_transaction_ratio` | 0.071 | Temporal |
| 8 | `betweenness_centrality` | 0.065 | Network |
| 9 | `balance_velocity` | 0.059 | Summary Ratio |
| 10 | `unique_counterparties` | 0.052 | Network |
| 11 | `balance_retention_ratio` | 0.044 | Balance Pattern |
| 12 | `tx_per_active_day` | 0.038 | Temporal |
| 13 | `retail_engagement` | 0.032 | Summary Ratio |
| 14 | `zero_balance_frequency` | 0.027 | Balance Pattern |
| 15 | `international_ratio` | 0.022 | High-Risk Entity |

### Key Insights

1. **Ratio features dominate**: `send_receive_ratio`, `cash_out_ratio`, and `avg_tx_value` together account for ~31% of total importance — the structural signature of AML activity is captured primarily by flow imbalances.

2. **Network features punch above their weight**: `clustering_coefficient` (4th) and `betweenness_centrality` (8th) confirm that structural position in the transaction graph is a strong AML signal.

3. **Velocity features capture real-time risk**: `velocity_24hr_max` and `rapid_transaction_ratio` demonstrate that temporal compression is a reliable indicator of automated/structured activity.

4. **High-risk entity exposure matters**: `betting_ratio` at #5 validates the focus on betting platforms as AML vectors in the Kenyan context.

5. **Balance patterns differentiate mules**: `balance_retention_ratio` and `zero_balance_frequency` are critical for identifying mule accounts used for fund sweeping.

---

## Feature Engineering Code Reference

All features are computed in:

- **Summary features**: `src/data/stratified_profiles.py` — customer profile aggregates
- **Temporal features**: `src/data/feature_engineering.py` — CustomerFeatureEngineer computes 4 essential temporal features (tx_count_7d, volume_7d, night_tx_ratio, rapid_tx_ratio)
- **Balance features**: `src/data/behavioral_generator.py` — balance tracking (max/min/sum/count), ledger continuity per `CustomerState`
- **Network features**: Not yet implemented (Phase 3). Planned: NetworkX graph algorithms on transaction counterparties
- **Composite flags**: `src/data/aml_scenario_injector.py` — ground-truth labeling for known patterns; downstream model layer for composite rule composition

The Gold layer (planned Phase 3) will combine all feature categories into the final partitioned feature store.

### Note on Feature Engineering Simplification (v1.0.6)

As of version 1.0.6, the CustomerFeatureEngineer has been simplified to compute only 4 essential temporal features:
- `tx_count_7d`: Rolling 7-day transaction count
- `volume_7d`: Rolling 7-day transaction volume
- `night_tx_ratio`: Ratio of transactions at night (22:00-06:00)
- `rapid_tx_ratio`: Ratio of transactions with ≤5min gap to previous transaction

This simplification follows synthetic data best practices by computing features on-demand from raw transaction history rather than pre-computing 20+ features. The other feature categories documented above (balance, network, composite flags) remain available for future expansion.
