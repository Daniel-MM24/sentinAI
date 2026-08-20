# M-Pesa Fraud Detection — XGBoost Model Card

## 1. Model Identity

| Field | Value |
|-------|-------|
| Model ID | sentinai_aml_xgb_001 |
| Model Name | M-Pesa Transaction Fraud Classifier |
| Model Version | 2.3.1 |
| Model Type | Gradient Boosted Decision Tree (XGBoost) |
| Training Framework | XGBoost 2.0.3 + scikit-learn 1.4.0 |
| Owner | SentinAI AML Engineering |
| Steward | CBK Compliance Team |
| Last Updated | 2026-04-15 |

## 2. Model Description

The M-Pesa Fraud XGBoost model is a supervised binary classifier that scores each mobile money transaction as either legitimate (0) or suspicious/fraudulent (1). It is the primary real-time scoring engine in the SentinAI transaction monitoring pipeline, processing C2B STK Push, B2C disbursements, P2P transfers, and agent float transactions.

The model outputs a fraud probability score (0.0 to 1.0), which is then thresholded into risk tiers:
- 0.00 – 0.30: Low risk — auto-approve (subject to regulatory limit checks)
- 0.31 – 0.65: Medium risk — queue for manual review within 4 hours
- 0.66 – 0.85: High risk — queue for manual review within 1 hour, block outbound >KES 10,000
- 0.86 – 1.00: Critical risk — block transaction immediately, freeze wallet, escalate to AML officer

## 3. Training Dataset

| Split | Records | Positive Class Ratio |
|-------|---------|---------------------|
| Training | 4,200,000 | 3.2% |
| Validation | 900,000 | 3.1% |
| Holdout Test | 900,000 | 3.3% |
| **Total** | **6,000,000** | **3.2%** |

Source: Sampled from M-Pesa transaction logs (Jan–Dec 2025), oversampled for fraud cases using SMOTE-NC to achieve 1:4 positive:negative in training. All PII (MSISDN, names) stripped before ingestion.

## 4. Feature Engineering

### 4.1 Transaction Features (38 features)

| Feature Group | Features | Count |
|--------------|----------|-------|
| Amount Statistics | txn_amount, log_amount, amount_percentile_7d, amount_zscore_30d | 4 |
| Temporal | hour_of_day, day_of_week, is_weekend, days_since_last_txn, txn_frequency_1h | 5 |
| Velocity | txn_count_1h_sender, txn_count_24h_sender, sum_amount_1h_sender, sum_amount_24h_sender | 4 |
| Velocity (receiver side) | txn_count_1h_receiver, txn_count_24h_receiver, sum_amount_1h_receiver | 3 |
| Agent Features | agent_txn_count_1h, agent_float_growth_3h, agent_cash_in_out_ratio_6h | 3 |
| Geographic | sender_counties_24h, receiver_counties_24h, sender_receiver_distance_km, is_same_county | 4 |
| Device/Channel | is_stk_push, is_b2c, is_p2p, is_agent_float, device_age_days, sim_match_status | 6 |
| MSISDN Features | account_age_days, wallet_tier_encoded, kyc_level_encoded, prev_fraud_flag_count_90d | 4 |
| CBK Compliance | amount_near_threshold (140k–150k), is_round_number_100k, daily_cumulative_remaining_pct | 3 |
| Interaction | amount_x_velocity, amount_x_account_age, threshold_proximity_x_velocity | 3 |

### 4.2 Key Engineered Features

**amount_near_threshold:** Binary indicator (1 if amount is KES 140,000–149,999). Captures CBK structuring typology (STRUC-001 through STRUC-005), which targets the 90–98% range of the KES 150,000 per-transaction ceiling.

**daily_cumulative_remaining_pct:** Float 0.0–1.0 indicating what fraction of the KES 500,000 daily cap remains. Structuring patterns often show a rapid depletion of this headroom within short windows. Values approaching 0.0 with new large transactions are high-risk.

**agg_velocity_12h:** Rolling count of all transactions for the same sender MSISDN in the trailing 12-hour window, including an exponentially weighted variant that decays older counts.

**agent_cash_in_out_ratio_6h:** Ratio of cash-in to cash-out for agent float transactions across a 6-hour window. Values exceeding 4:1 are flagged as Cuckoo Smurfing Risk per CBK typology.

**threshold_proximity_x_velocity:** Interaction term: `(150000 - amount) * log(velocity_1h + 1)`. Produces high scores when amount is near the ceiling AND velocity is high — the strongest single predictor of structuring behavior.

## 5. Model Architecture

```
XGBoost Classifier
├── Base learner: gradient boosted decision tree
├── Booster: gbtree
├── Objective: binary:logistic
├── Evaluation metric: AUC-PR (area under precision-recall curve)
├── Number of estimators: 450
├── Max depth: 8
├── Learning rate (eta): 0.05
├── Subsample ratio: 0.8
├── Column subsample by tree: 0.7
├── Min child weight: 5
├── Gamma: 0.1
├── Lambda (L2): 1.5
├── Alpha (L1): 0.5
├── Scale pos weight: 12.5 (1:3.125 negative:positive after SMOTE adjustment)
├── Early stopping rounds: 25 (on validation AUC-PR)
└── Seed: 42
```

## 6. Performance Metrics

| Metric | Training Set | Validation Set | Holdout Test Set |
|--------|-------------|---------------|------------------|
| AUC-ROC | 0.983 | 0.971 | 0.968 |
| AUC-PR | 0.921 | 0.894 | 0.887 |
| Precision at 0.66 threshold | 0.812 | 0.783 | 0.775 |
| Recall at 0.66 threshold | 0.745 | 0.721 | 0.714 |
| F1 Score at 0.66 threshold | 0.777 | 0.751 | 0.743 |
| Precision at 0.86 threshold | 0.941 | 0.922 | 0.918 |
| Recall at 0.86 threshold | 0.412 | 0.389 | 0.381 |
| False Positive Rate at 0.66 threshold | 0.008 | 0.011 | 0.012 |
| False Negative Rate at 0.66 threshold | 0.255 | 0.279 | 0.286 |
| Log Loss | 0.072 | 0.089 | 0.093 |

## 7. Feature Importance (Top 15 by Gain)

| Rank | Feature | Gain | Cover | Frequency |
|------|---------|------|-------|-----------|
| 1 | threshold_proximity_x_velocity | 0.142 | 0.118 | 0.095 |
| 2 | amount_near_threshold | 0.118 | 0.097 | 0.112 |
| 3 | txn_frequency_1h | 0.105 | 0.089 | 0.081 |
| 4 | log_amount | 0.089 | 0.102 | 0.073 |
| 5 | daily_cumulative_remaining_pct | 0.072 | 0.065 | 0.068 |
| 6 | agent_cash_in_out_ratio_6h | 0.067 | 0.058 | 0.052 |
| 7 | txn_count_1h_sender | 0.061 | 0.072 | 0.059 |
| 8 | account_age_days | 0.055 | 0.048 | 0.061 |
| 9 | amount_x_velocity | 0.048 | 0.041 | 0.044 |
| 10 | agg_velocity_12h | 0.041 | 0.036 | 0.039 |
| 11 | sender_receiver_distance_km | 0.035 | 0.029 | 0.031 |
| 12 | amount_percentile_7d | 0.031 | 0.034 | 0.028 |
| 13 | wallet_tier_encoded | 0.026 | 0.031 | 0.024 |
| 14 | prev_fraud_flag_count_90d | 0.022 | 0.019 | 0.017 |
| 15 | is_round_number_100k | 0.018 | 0.015 | 0.022 |

## 8. Threshold Calibration

Thresholds are calibrated on the validation set using cost-sensitive optimization:

| Risk Tier | Score Range | Transaction Volume | Capture Rate (actual fraud) | Action |
|-----------|------------|-------------------|---------------------------|--------|
| Critical | ≥0.86 | 0.7% | 38.1% | Block + freeze + AML escalation |
| High | 0.66–0.85 | 2.1% | 33.3% | Manual review in 1h + partial block |
| Medium | 0.31–0.65 | 5.8% | 18.9% | Manual review in 4h |
| Low | 0.00–0.30 | 91.4% | 9.7% | Auto-approve |

**Total fraud capture rate at or above Medium: 90.3%** (with 8.6% false positive rate flagged for review).

## 9. Fairness and Bias Evaluation

| Protected Group | Sample Count | False Positive Rate | False Negative Rate | Disparate Impact Ratio |
|----------------|-------------|-------------------|-------------------|----------------------|
| Tier 1 (Basic KYC) | 1,200,000 | 0.019 | 0.302 | 1.58 |
| Tier 2 (Interim KYC) | 1,800,000 | 0.014 | 0.291 | 1.27 |
| Tier 3 (Full KYC) | 2,700,000 | 0.009 | 0.278 | 1.00 (ref) |
| Tier 4 (EDD) | 300,000 | 0.007 | 0.265 | 0.82 |
| Urban (Nairobi/Mombasa) | 2,900,000 | 0.010 | 0.282 | 1.00 (ref) |
| Rural (all other counties) | 3,100,000 | 0.013 | 0.291 | 1.18 |

**Mitigation:** Tier 1 users show elevated FNR (0.302 vs 0.278 baseline). A per-tier score calibration adjustment has been applied (additive offset of +0.03 for Tier 1 predictions) to reduce systematic disparity. Retargeted for EDD uplift campaign: 92,000 Tier 1 accounts migrated to Tier 2 in Q1 2026.

## 10. Model Monitoring and Retraining

| Signal | Metric | Alert Threshold | Current Value |
|--------|--------|----------------|---------------|
| Data drift | PSI (Population Stability Index) | >0.15 | 0.072 (stable) |
| Concept drift | AUC-ROC drop vs. baseline | >0.03 | 0.011 (stable) |
| Prediction drift | Mean score shift vs. baseline | >0.05 | 0.018 (stable) |
| Calibration | Expected Calibration Error | >0.04 | 0.021 (stable) |
| Volume anomaly | Transaction volume % change | >±20% | +3.2% (stable) |

**Retraining schedule:** Monthly full retrain (every 1st of month); triggered hot retrain if any alert threshold breached.

## 11. Regulatory Compliance Mapping

| CBK Requirement | Model Implementation | Audit Evidence |
|----------------|---------------------|---------------|
| PG/43 threshold breach detection | Hard rule override: amount>150k = auto-block (precedes ML score) | Rule log line 1-1 |
| Velocity warning ≥KES 100,000 daily | Feature daily_cumulative_remaining_pct triggers score uplift | Feature importance rank 5 |
| Structuring detection (KES 140k–150k) | Feature amount_near_threshold + threshold_proximity_x_velocity | Feature importance rank 1, 2 |
| Cuckoo smurfing (agent float anomaly) | Feature agent_cash_in_out_ratio_6h | Feature importance rank 6 |
| New account rapid escalation (7d, KES 500k) | Feature account_age_days + prev_fraud_flag_count_90d | Rule override synergy |
| EDD trigger at 20+ transactions/day | Feature txn_count_24h_sender at review threshold | Automated queue routing |

---

*Model card maintained by SentinAI AML Engineering. Version 2.3.1. Last reviewed: 2026-05-10 by CBK Compliance Audit Team.*
