# Stateful Temporal Pattern Tracking

## Overview

The temporal pattern tracker extracts behavioral features from M-PESA transaction data across daily, weekly, and monthly time horizons. It uses a stateful `RollingBuffer` per customer that maintains a sliding window of recent transactions with automatic eviction, enabling both batch feature extraction and real-time monitoring.

**Module:** `src/data/temporal_features.py`
**Output:** `data/temporal_features.csv` — 1,000 rows × 24 feature columns

## Rolling Buffer Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 RollingBuffer (per customer)              │
│                                                          │
│  Window: sliding 30-day buffer with auto-eviction        │
│                                                          │
│  Records: TransactionRecord(timestamp, amount,            │
│                             transaction_type, direction)  │
│                                                          │
│  Queries:                                                 │
│    count_1h(ts)    — Transaction count in last 1 hour    │
│    count_24h(ts)   — Transaction count in last 24 hours  │
│    volume_24h(ts)  — Total amount in last 24 hours       │
│    avg_7d(ts)      — Average value over last 7 days      │
│    volume_30d(ts)  — Total volume over last 30 days      │
└──────────────────────────────────────────────────────────┘
```

## Feature Set

### Daily Pattern Detection (24 columns total)

| Feature | Type | Description |
| :--- | :---: | :--- |
| `dominant_hour` | int | Hour with most transactions (0-23) |
| `dominant_hour_pct` | float | % of all tx at dominant hour |
| `has_consistent_pattern` | bool | >70% of tx at same hour |
| `morning_pct` | float | % of tx in 6-9 AM window |
| `lunch_pct` | float | % of tx in 12-2 PM window |
| `evening_pct` | float | % of tx in 5-8 PM window |
| `hourly_entropy` | float | Shannon entropy of hourly distribution |

### Weekly Pattern Detection

| Feature | Type | Description |
| :--- | :---: | :--- |
| `weekend_ratio` | float | Fraction of tx on Sat/Sun |
| `weekend_tx_count` | int | Total weekend transactions |
| `weekday_tx_count` | int | Total weekday transactions |
| `weekend_anomaly` | bool | weekend_ratio > 0.5 AND weekday_ratio < 0.3 |

**Anomaly logic:** Flags users who spike to >50% weekend activity despite normally being weekday-inactive. This pattern is common with mule accounts that activate only on weekends when monitoring is lighter.

### Monthly Pattern Detection

| Feature | Type | Description |
| :--- | :---: | :--- |
| `month_end_tx_ratio` | float | % of tx on days 25-31 of month |
| `salary_receipt_ratio` | float | % of tx on days 25-28 classified as B2C/salary |
| `school_fee_total` | float | Total amount sent via PayBill/Send Money in Jan/May/Sep |
| `school_fee_flag` | bool | Any school-fee-related outflow detected |

**Salary detection logic:** A B2C salary receipt is identified as a `Received Money` or `Lipa Na M-PESA (Paybill)` transaction occurring between the 25th and 28th of the month — Kenya's common salary disbursement window.

### Rolling Aggregators (last-snapshot state)

| Feature | Type | Window | Description |
| :--- | :---: | :---: | :--- |
| `roll_1h_count` | int | 1 hour | Real-time velocity count |
| `roll_24h_count` | int | 24 hours | Daily transaction frequency |
| `roll_24h_volume` | float | 24 hours | Daily transaction volume (KES) |
| `roll_7d_avg_value` | float | 7 days | Rolling average transaction value |
| `roll_30d_volume` | float | 30 days | Monthly total volume (KES) |
| `roll_30d_count` | int | 30 days | Monthly total tx count |

## Real-Time Detection Usage

The `TemporalPatternTracker` maintains per-customer `RollingBuffer` instances that support incremental updates:

```python
tracker = TemporalPatternTracker()
features = tracker.compute_features(batch_df)  # batch pass

# Real-time: feed transactions as they arrive
buffer = tracker.get_buffer_state("CUST_000000")
buffer.push(TransactionRecord(ts, amount, tx_type, direction))

count_1h = buffer.count_1h(now())  # current velocity
volume_24h = buffer.volume_24h(now())
```

This enables real-time monitoring rules such as:
- **Velocity anomaly:** >10 transactions in 1 hour for a low-activity user
- **Volume spike:** 24h volume > 3σ above 7-day rolling average
- **Monthly threshold breach:** 30d volume exceeding CBK tier limit
