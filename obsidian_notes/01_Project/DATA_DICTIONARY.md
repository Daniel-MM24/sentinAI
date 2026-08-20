# Data Dictionary: SentinAI AML PoC

## Customer Profiles (`customer_profiles.csv`)

| Field | Type | Description | Values / Range | Source |
| :--- | :--- | :--- | :--- | :--- |
| `customer_id` | String | Unique customer identifier | `CUST_000000` to `CUST_NNNNNN` | Generated |
| `tier` | String | KYC tier level | tier_1 (60%), tier_2 (20%), tier_3 (15%), tier_4 (5%) | Generated |
| `archetype` | String | User behavioral category | retail_heavy (15%), retail_standard (70%), micro_merchant (12%), corporate_sme (3%) | Generated |

**Note:** Following synthetic data best practices (IBM/NVIDIA), customer profiles now contain only essential identifiers and behavioral archetypes. Transaction limits, balance limits, account age, and other dynamic attributes are handled in downstream transaction generation rather than static customer profiles.

### Customer Archetype Profiles

| Archetype | % of Pop. | Avg. Tx/Month | Avg. Tx Value | Key Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **Retail Heavy** | 15% | 45–60 | KES 2,000–8,000 | Frequent small-value P2P, high PayBill/BuyGoods engagement |
| **Retail Standard** | 70% | 15–25 | KES 3,000–15,000 | Moderate usage, balanced inflows/outflows, regular retail purchases |
| **Micro-Merchant** | 12% | 80–150 | KES 500–3,000 | High transaction volume, predominantly receiving payments, agent cash-outs |
| **Corporate** | 3% | 100–300 | KES 10,000–100,000 | Bulk payments, large-value B2B transfers, multiple counterparties |

## Detailed Transactions (`detailed_transactions.csv`)

| Field | Type | Description | Values / Range | Source |
| :--- | :--- | :--- | :--- | :--- |
| `tx_id` | String | Unique transaction identifier | `TX_000000001` to `TX_001000000` | Generated |
| `user_id` | String | Foreign key → `customers_metadata.user_id` | `C_000001` to `C_002200` | Generated |
| `receipt_number` | String | M-PESA receipt format | `RX[A-Z][0-9]{6}` (e.g. `RXA123456`) | Generated |
| `completion_time` | DateTime | Transaction timestamp (FY25) | 2024-07-01 to 2025-06-30 | Generated |
| `details` | String | M-PESA-style transaction description | 'Send Money to John', 'Lipa Paybill - KPLC' | Generated |
| `transaction_status` | String | M-PESA transaction status | Completed (99%), Pending (1%), Failed (<1%) | Generated |
| `paid_in` | Float | Amount received by customer (KES) | ≥ 0 | Generated |
| `paid_out` | Float | Amount sent/withdrawn by customer (KES) | ≥ 0 | Generated |
| `balance` | Float | Running balance after transaction (KES) | 0 to tier limit (50K/500K/5M) | Calculated |
| `transaction_type` | String | M-PESA transaction category | send_money, received_money, agent_deposit, agent_withdrawal, lipa_paybill, lipa_buygoods, others | Generated |
| `value` | Float | Absolute transaction value = max(paid_in, paid_out) | > 0 | Derived |
| `counterparty` | String | Counterparty name/merchant | 'John', 'KPLC', 'Agent Mary', 'Betika' | Generated |
| `is_betting` | Boolean | Betting platform transaction | True (if counterparty matched to betting), False | Flagged |
| `is_international` | Boolean | International transfer | True, False | Flagged |
| `is_kadogo` | Boolean | Micro-transaction below KES thresholds | True (< KES 100 P2P, < KES 200 Merchant) | Calculated |
| `hour` | Integer | Hour of day extracted from completion_time | 0–23 | Derived |
| `day_of_week` | Integer | Day of week | 0=Monday, 6=Sunday | Derived |
| `month` | Integer | Month of transaction | 1–12 | Derived |
| `is_weekend` | Boolean | Saturday or Sunday | True, False | Derived |
| `is_night` | Boolean | Night transaction (22:00–06:00) | True, False | Derived |

### Transaction Type Distribution

| Transaction Type | Flow Direction | Probability | Typical Value Range |
| :--- | :--- | :---: | :--- |
| Send Money | Outflow | 25% | KES 500–50,000 |
| Received Money | Inflow | 20% | KES 1,000–100,000 |
| Agent Deposit | Outflow | 10% | KES 500–50,000 |
| Agent Withdrawal | Inflow | 10% | KES 500–50,000 |
| Lipa Na M-PESA (Paybill) | Outflow | 15% | KES 100–50,000 |
| Lipa Na M-PESA (Buy Goods) | Outflow | 12% | KES 50–10,000 |
| Others | Bidirectional | 8% | KES 100–100,000 |

### Transaction Value Distribution Parameters

| Archetype | Distribution | μ (log) | σ (log) | Median Value |
| :--- | :--- | :---: | :---: | :---: |
| Retail Heavy | LogNormal | 7.5 | 1.2 | ~KES 1,800 |
| Retail Standard | LogNormal | 8.5 | 1.5 | ~KES 4,900 |
| Micro-Merchant | LogNormal | 7.0 | 0.8 | ~KES 1,100 |
| Corporate | LogNormal | 10.0 | 1.8 | ~KES 22,000 |

## Summary Statements (`summary_statements.csv`)

| Field | Type | Description | Values / Range | Source |
| :--- | :--- | :--- | :--- | :--- |
| `user_id` | String | Foreign key → customers | `C_000001` to `C_002200` | Generated |
| `send_money_paid_out` | Float | Total Send Money outflow (KES) | ≥ 0 | Aggregated |
| `received_money_paid_in` | Float | Total Received Money inflow (KES) | ≥ 0 | Aggregated |
| `agent_deposit_paid_out` | Float | Total Agent Deposit outflow (KES) | ≥ 0 | Aggregated |
| `agent_withdrawal_paid_in` | Float | Total Agent Withdrawal inflow (KES) | ≥ 0 | Aggregated |
| `lipa_paybill_paid_out` | Float | Total PayBill outflow (KES) | ≥ 0 | Aggregated |
| `lipa_buygoods_paid_out` | Float | Total Buy Goods outflow (KES) | ≥ 0 | Aggregated |
| `others_paid_in` | Float | Total Other category inflow (KES) | ≥ 0 | Aggregated |
| `others_paid_out` | Float | Total Other category outflow (KES) | ≥ 0 | Aggregated |
| `total_paid_in` | Float | Sum of all inflows (KES) | ≥ 0 | Calculated |
| `total_paid_out` | Float | Sum of all outflows (KES) | ≥ 0 | Calculated |
| `transaction_count` | Integer | Total transaction count | 0–N | Aggregated |
| `betting_ratio` | Float | Proportion of transactions to betting platforms | 0.0–1.0 | Calculated |
| `international_ratio` | Float | Proportion of international transfers | 0.0–1.0 | Calculated |

## Bronze Layer Schema (`bronze_transactions.parquet`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `customer_id` | String | Customer identifier |
| `customer_name` | String | Customer name (synthetic) |
| `email` | String | Email address (synthetic) |
| `tax_id` | String | Tax identifier (synthetic; `TAX_C_XXXXX`) |
| `currency` | String | Transaction currency |
| `amount` | Float64 | Transaction amount |
| `timestamp` | String | ISO 8601 timestamp |
| `source_table` | String | Origin table name |
| `ingestion_date` | String | Bronze ingestion timestamp |
| `source_type` | String | Data source: `postgresql`, `synthetic`, `csv` |
| `synthetic_flag` | Boolean | True if synthetically generated |

## Silver Layer Schema (`SilverRecordSchema` — Pandera)

| Field | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `customer_id` | String | Not null | Customer identifier |
| `transaction_id` | String | Not null, unique | Transaction identifier |
| `counterparty_id` | String | Not null | Counterparty identifier |
| `timestamp` | DateTime | Not null, tz-aware | Transaction timestamp |
| `currency` | String | In {KES, USD} | Transaction currency |
| `amount` | Float64 | gt=0 | Transaction amount |
| `post_tx_balance` | Float64 | ge=0 | Balance after transaction |
| `device_age_days` | Int64 | ge=0 | Device age in days |
| `sim_match_status` | Boolean | Not null | SIM registration match |
| `wallet_tier_encoded` | Int64 | In {1,2,3} | Wallet tier code |
| `kyc_level_encoded` | Int64 | In {1,2,3} | KYC level code |
| `prev_fraud_flag_count_90d` | Int64 | ge=0 | Prior fraud flags (90d) |
| `receiver_id` | String | Not null | Receiver identifier |
| `sender_county` | String | Not null | Sender county |
| `receiver_county` | String | Not null | Receiver county |
| `anomaly_flag` | Int32 | In {0,1} | Anomaly indicator |
| `anomaly_type` | String | Nullable | Anomaly type description |

## Gold Layer Feature Store Schema (`gold/features/vv1.0/`)

Full schema in `manifest.json`. Includes all Silver layer fields plus engineered features:

**Temporal-Velocity Features** (12 fields):
- `tx_count_{1min,5min,1h,24h,7d,30d}` — Transaction counts per time window
- `amount_sum_{1min,5min,1h,24h,7d,30d}` — Amount sums per time window
- `burst_ratio` — Transaction burst intensity score
- `velocity_change_pct` — Percent change in transaction velocity

**Balance Pattern Features** (8 fields):
- `current_balance`, `min_balance_30d`, `max_balance_30d`, `avg_balance_30d`
- `balance_volatility_30d`, `balance_retention_ratio`, `zero_balance_frequency`
- `amount_vs_profile_avg` — Deviation from archetype average amount

**Risk Indicator Features** (varies by construction):
- Feature vectors derived from SilverRecordSchema + temporal + balance + network features

See [FEATURE_DICTIONARY.md](./FEATURE_DICTIONARY.md) for complete feature formulas and descriptions.

## Transaction Velocity Windows

| Window | Description | Risk Use Case |
| :--- | :--- | :--- |
| 1-minute | Number of tx in 60 seconds | Real-time smurfing detection |
| 5-minute | Number of tx in 300 seconds | Micro-burst detection |
| 1-hour | Number of tx in 60 minutes | Velocity threshold enforcement |
| 24-hour | Number of tx in 1 day | Daily limit monitoring (CBK) |
| 7-day | Number of tx in 1 week | Weekly pattern analysis |
| 30-day | Number of tx in 1 month | Monthly trend detection |

## Regulatory Reference: CBK Transaction Limits

| Tier | Max Balance (KES) | Daily Velocity (KES) | Monthly Aggregate (KES) |
| :--- | :---: | :---: | :---: |
| Tier 1 (60% of customers) | 50,000 | 100,000 | 300,000 |
| Tier 2 (30% of customers) | 500,000 | 1,000,000 | 3,000,000 |
| Tier 3 (10% of customers) | 5,000,000 | 10,000,000 | 30,000,000 |

*Vendor/Merchant accounts operate under Tier 3-equivalent limits with additional transaction-type restrictions.*
