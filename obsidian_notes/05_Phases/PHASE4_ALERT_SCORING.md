# Phase 4: Alert Generation & Risk Scoring

## Objective

Generate actionable AML alerts by combining three independent detection methods — rule-based scoring, ML probability calibration, and graph-based anomaly detection — into a unified risk score with investigation-ready case management.

## Alert Engine Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HYBRID ALERT ENGINE                                 │
│                                                                         │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │   RULE ENGINE    │   │   ML SCORER      │   │  GRAPH SCORER    │    │
│  │                  │   │                  │   │                  │    │
│  │ • 25+ CBK Rules  │   │ • LightGBM prob  │   │ • Betweenness    │    │
│  │ • Weighted sum   │   │ • Platt scaling  │   │ • Clustering     │    │
│  │ • Thresholds     │   │ • Calibration    │   │ • Anomaly score  │    │
│  │ • Regulatory     │   │ • SHAP expl.     │   │ • Proximity      │    │
│  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘    │
│           │                      │                      │              │
│           ▼                      ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SCORE FUSION LAYER                            │   │
│  │  hybrid_score = w1 * rule_score + w2 * ml_score + w3 * graph_   │   │
│  │  (w1=0.35, w2=0.45, w3=0.20 — tuned via grid search)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CASE MANAGEMENT                               │   │
│  │  Priority: CRITICAL (≥0.90) → Immediate SAR                      │   │
│  │             HIGH    (≥0.75) → 24hr Investigation                  │   │
│  │             MEDIUM  (≥0.50) → 7-day Review                        │   │
│  │             LOW     (<0.50) → Periodic Review                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Rule-Based Scoring

### Rule Categories (25+ rules from `config/regulatory.yaml`)

| Category | Rules | Max Weight | Regulatory Basis |
| :--- | :--- | :---: | :--- |
| **Ratio Anomalies** | send_receive_ratio > 5, cash_out_ratio > 0.8 | 0.20 | CBK Guideline 5.2 |
| **Velocity Alerts** | velocity_1hr > 5, velocity_24hr > 20 | 0.18 | CBK Guideline 5.3 |
| **Pattern Detection** | structuring, smurfing, layering patterns | 0.22 | CBK Guideline 7.3 |
| **High-Risk Entities** | betting_ratio > 0.3, international_ratio > 0.2 | 0.15 | CBK Guideline 4.4 |
| **Balance Anomalies** | balance_retention < 0.1, zero_balance_freq > 3 | 0.12 | CBK Guideline 5.4 |
| **Network Anomalies** | low clustering + high degree, high betweenness | 0.13 | FATF Recommendation 16 |

### Scoring Formula

```
rule_score = Σ(rule_weight_i * alert_i) / Σ(rule_weight_i)
```

Where `alert_i ∈ {0, 1}` based on threshold exceedance and `rule_weight_i` reflects regulatory severity.

## 2. ML Risk Scoring

### Calibration

Supervised model probability outputs are calibrated via **Platt scaling**:

```
P(y=1|x) = 1 / (1 + exp(A * f(x) + B))
```

Where `f(x)` is the raw model output and `A, B` are fitted via logistic regression on a held-out validation set.

### Score Distribution

| Score Range | % Predictions | Interpretation |
| :--- | :---: | :--- |
| 0.00–0.25 | 94.2% | Low risk — normal transaction behavior |
| 0.25–0.50 | 3.1% | Elevated — minor pattern deviations |
| 0.50–0.75 | 1.8% | Medium — warrants review |
| 0.75–0.90 | 0.6% | High — likely suspicious |
| 0.90–1.00 | 0.3% | Critical — immediate action |

## 3. Graph Anomaly Scoring

### Score Components

| Component | Weight | Description |
| :--- | :---: | :--- |
| Structural Anomaly Score | 0.40 | Composite of low clustering + high betweenness |
| Network Distance Score | 0.30 | Proximity to known high-risk nodes |
| Community Isolation Score | 0.20 | Degree of community isolation vs. expected |
| Temporal Graph Drift | 0.10 | Change in graph position over time |

### Graph Score Formula

```
graph_anomaly_score = 0.40 * SAS + 0.30 * NDS + 0.20 * CIS + 0.10 * TGD
```

Where each sub-score is min-max normalized to [0, 1].

## 4. Hybrid Score Fusion

### Weight Optimization

Weights tuned via grid search on validation set (metric: F1 at detection threshold):

| Model Component | Weight | Rationale |
| :--- | :---: | :--- |
| ML Score (w1) | 0.45 | Highest standalone performance (AUC 0.973) |
| Rule Score (w2) | 0.35 | Regulatory compliance, domain knowledge |
| Graph Score (w3) | 0.20 | Independent signal, catches structural anomalies |

### Fusion Formula

```
hybrid_score = 0.45 * ml_score_calibrated + 0.35 * rule_score_normalized + 0.20 * graph_score_normalized
```

All sub-scores scaled to [0, 1] before fusion.

## 5. Alert Priority Matrix

| Risk Level | Score Range | TTF* | Action Required |
| :--- | :---: | :--- | :--- |
| **CRITICAL** | 0.90–1.00 | < 1 hour | Immediate SAR filing, account freeze, FIU notification |
| **HIGH** | 0.75–0.89 | < 24 hours | Investigation required, enhanced due diligence |
| **MEDIUM** | 0.50–0.74 | < 7 days | Review queue, pattern monitoring |
| **LOW** | 0.00–0.49 | > 7 days | Periodic review, statistical tracking |

*TTF = Time to File (SAR filing deadline guidance)

## 6. Alert Quality Metrics

| Metric | Value | Target |
| :--- | :---: | :---: |
| **Detection Rate** | 94% | > 90% |
| **False Positive Rate** | 3.2% | < 5% |
| **Precision** | 0.91 | > 0.85 |
| **SAR Conversion Rate** | 12% | 5-15% |
| **Time to Detection (avg)** | 2.3 days | < 3 days |
| **Alert Volume (total)** | ~30,000 | Manageable for ~6 analysts |

### By Scenario

| Scenario | Detection Rate | FPR | Avg. Time to Detect |
| :--- | :---: | :---: | :---: |
| Smurfing | 96.2% | 2.8% | 4.7 days |
| Layering | 93.1% | 3.1% | 0.8 days |
| Mule Accounts | 95.4% | 2.5% | 1.2 days |
| Circular Trading | 91.3% | 4.2% | 2.5 days |

## 7. Case Management Output

Each alert generates a structured case record:

| Field | Description | Example |
| :--- | :--- | :--- |
| `case_id` | Unique case identifier | `CASE_000042` |
| `customer_id` | Subject customer | `C_001234` |
| `alert_timestamp` | When alert was generated | `2025-06-15 14:30:00 UTC` |
| `hybrid_score` | Combined risk score | 0.87 |
| `risk_level` | Priority level | `HIGH` |
| `triggering_rules` | Rules that fired | `cash_out_ratio > 0.8, velocity_24hr > 20` |
| `ml_probability` | ML model output | 0.91 |
| `graph_anomaly_score` | Graph detection score | 0.76 |
| `shap_top_features` | Top 3 SHAP features | `[send_receive_ratio: 0.21, cash_out_ratio: 0.18]` |
| `primary_typology` | Most likely AML typology | `Mule Account` |
| `investigation_status` | Current status | `OPEN` |
| `assigned_to` | Assigned analyst | — |
| `sar_filed` | SAR filing status | `False` |

## 8. SAR Workflow

```
Alert Generated (hybrid_score ≥ 0.50)
      │
      ▼
┌─────────────────┐     NO     ┌─────────────────┐
│ Score ≥ 0.90?   │───────────►│ Score ≥ 0.75?   │
└────────┬────────┘            └────────┬────────┘
         │ YES                          │ YES
         ▼                              ▼
┌─────────────────┐           ┌─────────────────┐
│ AUTO SAR        │           │ 24hr            │
│ IMMEDIATE       │           │ Investigation   │
│ Account Freeze  │           │ Queue           │
└─────────────────┘           └────────┬────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ Evidence        │
                              │ Package         │
                              │ Compiled        │
                              └────────┬────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ SAR Filed       │
                              │ (via FIU API)   │
                              └─────────────────┘
```

## 9. Performance Metrics

| Component | Processing Rate | Latency (per tx) |
| :--- | :---: | :---: |
| Rule Scoring | 50,000 tx/s | 0.02 ms |
| ML Scoring | 10,000 tx/s | 0.10 ms |
| Graph Scoring | 1,000 tx/s | 1.00 ms |
| **Hybrid Fusion** | **~800 tx/s** | **1.25 ms** |

*Benchmark: Intel i7-12700H, 32 GB RAM, single-threaded. Real-time scoring with FastAPI expected to handle >100 tx/s per instance with <100ms p99 latency.*
