# Phase 2: Feature Engineering

## Objective

Transform raw transaction data into high-signal AML detection features. The feature engineering pipeline derives 57 features across 6 categories, capturing structural patterns that distinguish money laundering from legitimate M-PESA usage.

## Pipeline Architecture

```
detailed_transactions.csv    summary_statements.csv    customer_metadata.csv
         │                           │                         │
         ▼                           ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING PIPELINE                        │
│                                                                         │
│  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │ Summary-Level  │  │ Temporal-     │  │ Network       │              │
│  │ Ratio Features │  │ Velocity      │  │ Features      │              │
│  │ (14 features)  │  │ Features      │  │ (12 features) │              │
│  │                │  │ (16 features) │  │               │              │
│  └────────────────┘  └───────────────┘  └───────────────┘              │
│                                                                         │
│  ┌────────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │ Balance        │  │ High-Risk     │  │ Behavior      │              │
│  │ Pattern        │  │ Entity        │  │ Anomaly       │              │
│  │ Features       │  │ Features      │  │ Composite     │              │
│  │ (8 features)   │  │ (4 features)  │  │ (10 features) │              │
│  └────────────────┘  └───────────────┘  └───────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                feature_set_complete.csv
                (2,200 rows × 60+ columns)
```

## Feature Categories

### 1. Summary-Level Ratio Features (14 features)

Computed from per-customer aggregated statistics in `summary_statements.csv`:

| Feature | Formula | AML Signal |
| :--- | :--- | :--- |
| `total_volume` | `paid_in + paid_out` | Overall activity level |
| `send_receive_ratio` | `send_money_out / received_money_in` | Flow imbalance — critical for smurfing |
| `cash_out_ratio` | `agent_withdrawal / total_in` | Mule indicator — immediate cash-out |
| `balance_velocity` | `total_out / total_in` | Fund pass-through rate |
| `avg_tx_value` | `total_volume / tx_count` | Structuring indicator — low avg with high count |
| `retail_engagement` | `(paybill + buygoods) / total_out` | Legitimate retail usage |
| `kadogo_ratio` | `kadogo_count / tx_count` | Micro-transaction prevalence |
| `betting_ratio` | `betting_tx / total_tx` | Betting platform engagement |
| `international_ratio` | `international_tx / total_tx` | Cross-border exposure |

### 2. Temporal-Velocity Features (16 features)

Computed from timestamp-level analysis of individual transactions:

- **Cyclical ratios**: `night_ratio`, `weekend_ratio`, `month_end_ratio`
- **Velocity metrics**: `velocity_1hr_max`, `velocity_24hr_max`, `rapid_transaction_ratio`
- **Rolling windows**: Transaction counts and amount sums over 1-min, 5-min, 1-hr, 24-hr, 7-day, 30-day windows
- **Derived metrics**: `burst_ratio`, `velocity_change_pct`

**Rolling window features** (computed via streaming window functions):
```
tx_count_1min = COUNT(tx) OVER (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN '1m' PRECEDING AND CURRENT ROW)
amount_sum_1h = SUM(amount) OVER (PARTITION BY customer_id ORDER BY ts RANGE BETWEEN '1h' PRECEDING AND CURRENT ROW)
```

### 3. Balance Pattern Features (8 features)

Computed from the post-transaction balance time series:

| Feature | Description |
| :--- | :--- |
| `balance_volatility_30d` | Std dev of balance / mean balance |
| `balance_retention_ratio` | Min balance / Max balance (30-day) |
| `zero_balance_frequency` | Count of zero-balance events |
| `current_balance` | End-of-period balance |
| `amount_vs_profile_avg` | Amount deviation from archetype average |

### 4. Network Features (12 features)

Built from a directed transaction graph where nodes are customers and edges represent transaction flows:

- **Centrality metrics**: `betweenness_centrality`, `degree_centrality`, `pagerank`
- **Connection metrics**: `in_degree`, `out_degree`, `unique_counterparties`
- **Structural metrics**: `clustering_coefficient`, `community_id`
- **Proximity metrics**: `distance_to_betting`, `distance_to_international`
- **Concentration metrics**: `top_counterparty_concentration`, `counterparty_diversity`

Graph construction properties:
```
- Directed weighted graph (direction = fund flow)
- Edge weight = total transaction volume
- Node attributes: anomaly_flag, archetype, is_betting, is_international
```

### 5. High-Risk Entity Features (4 features)

Composite flags combining entity type with volume thresholds:

| Feature | Rule |
| :--- | :--- |
| `betting_high_risk` | `betting_ratio > 0.3 AND total_volume > 200,000` |
| `international_high_risk` | `international_ratio > 0.2 AND total_volume > 300,000` |

### 6. Behavior Anomaly Composite Features (10 features)

Binary flags that encode known AML typologies as feature combinations:

| Feature | Rule | Typology |
| :--- | :--- | :--- |
| `structuring_pattern` | `send_receive_ratio > 5 AND avg_tx_value < 500 AND tx_count > 100` | Smurfing |
| `cash_out_mule_pattern` | `cash_out_ratio > 0.8 AND received_money > 100,000` | Mule Account |
| `rapid_cycling_pattern` | `balance_velocity > 0.9 AND zero_balance_frequency > 2` | Layering |
| `minimal_retail_pattern` | `retail_engagement < 0.05 AND total_volume > 500,000` | General Suspicious |

## Feature Engineering Code (`src/data/features.py`)

The `FeatureEngineer` class (scikit-learn `BaseEstimator`/`TransformerMixin` compatible):

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_metadata = {}

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self.create_temporal_features(df)     # Cyclical features
        df = self.create_risk_indicators(df)       # Risk flags + velocity
        return df

    def create_temporal_features(self, df):
        # Extracts day_of_month, day_of_week, hour_of_day
        ...

    def create_risk_indicators(self, df):
        # High-risk amount flags, rolling velocity, z-scores, CLV
        ...
```

## Output

| Output | Rows | Columns | Format |
| :--- | :---: | :---: | :---: |
| Gold Feature Store | 5,000 (partitioned) | 57 features + metadata | Parquet (partitioned) |
| Feature Manifest | 1 | Schema + version | JSON |
| SHAP Explanations | Per prediction | Feature importance | CSV |

## Feature Validation

- **Great Expectations**: Automated validation of feature value ranges, null rates, and distributions
- **Pandera schema**: Silver layer schema enforcement before feature derivation
- **SHAP analysis**: Post-hoc feature importance validation
- **Cross-field consistency**: Logical relationships enforced (e.g., `paid_in` and `paid_out` should not both be > 0)

## Key Design Decisions

1. **Ratio features over absolute values**: Ratios normalize for account scale, making them comparable across vastly different customer sizes (retail vs. corporate)
2. **Multiple time windows**: 1-min through 30-day windows capture both real-time bursts and long-term pattern shifts
3. **Network proximity scoring**: Graph distance to known high-risk entities captures indirect exposure
4. **Composite flags encode domain knowledge**: AML typology rules expressed as feature combinations for model consumption
5. **Stateful stream processing**: Velocity window features are computed as streaming aggregations, matching real-time scoring requirements
