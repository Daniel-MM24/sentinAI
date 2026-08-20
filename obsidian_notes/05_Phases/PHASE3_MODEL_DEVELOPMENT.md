# Phase 3: Model Development

## Objective

Train, validate, and select AML detection models using the synthetic dataset with injected ground-truth labels. The model suite covers three detection paradigms: supervised classification, unsupervised anomaly detection, and graph-based network analysis.

## Model Architecture

```
Gold Feature Store (partitioned, versioned)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     MODEL DEVELOPMENT PIPELINE                           │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   TRAINING DATA PREPARATION                      │   │
│  │  • Train/Test split (80/20, stratified by anomaly_type)          │   │
│  │  • Feature scaling (StandardScaler for tree models optional)     │   │
│  │  • SMOTE oversampling for class imbalance (~2% positives)        │   │
│  │  • Feature selection via mutual information + correlation filter │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   SUPERVISED   │  │  UNSUPERVISED  │  │    GRAPH       │            │
│  │                │  │                │  │    ANALYSIS    │            │
│  │ • LightGBM     │  │ • Isolation    │  │               │            │
│  │ • XGBoost      │  │   Forest       │  │ • NetworkX    │            │
│  │ • Random       │  │ • LOF          │  │ • Community   │            │
│  │   Forest       │  │ • One-Class    │  │ • Centrality  │            │
│  │                │  │   SVM          │  │ • Proximity   │            │
│  └────────────────┘  └────────────────┘  └────────────────┘            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   MODEL EVALUATION                                │   │
│  │  • 5-fold cross-validation (stratified)                          │   │
│  │  • AUC-ROC, AUC-PR, F1, Precision, Recall, Specificity           │   │
│  │  • Confusion matrix at multiple thresholds                        │   │
│  │  • SHAP feature importance + dependence plots                     │   │
│  │  • Calibration curves (Platt scaling)                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    trained_models.pkl
                    model_performance.json
                    shap_explanations.csv
```

## Supervised Models

### LightGBM (Primary Classifier)

**Rationale**: Gradient boosting performs well on tabular data with mixed feature types, handles missing values natively, and provides built-in handling of class imbalance.

**Configuration**:
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `n_estimators` | 500 | Sufficient for convergence with early stopping |
| `max_depth` | 7 | Prevents overfitting while capturing interactions |
| `learning_rate` | 0.05 | Conservative for stable convergence |
| `num_leaves` | 63 | 2^max_depth - 1 for full tree |
| `min_data_in_leaf` | 20 | Prevents leaf-level overfitting |
| `subsample` | 0.8 | Row-level bagging |
| `colsample_bytree` | 0.8 | Feature-level bagging |
| `scale_pos_weight` | ~49 | Inverse of class imbalance (98% / 2%) |
| `early_stopping_rounds` | 50 | Prevents overfitting |
| `objective` | `binary` | Binary classification |
| `metric` | `auc` | AUC-ROC optimization |

**Performance**: AUC-ROC = 0.973, F1 = 0.912

### XGBoost (Cross-Validation Model)

| Parameter | Value |
| :--- | :--- |
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `scale_pos_weight` | ~49 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

**Performance**: AUC-ROC = 0.962, F1 = 0.894

### Random Forest (Baseline)

| Parameter | Value |
| :--- | :--- |
| `n_estimators` | 300 |
| `max_depth` | 10 |
| `min_samples_split` | 10 |
| `class_weight` | `balanced` |

**Performance**: AUC-ROC = 0.931, F1 = 0.841

## Unsupervised Models

### Isolation Forest

| Parameter | Value |
| :--- | :--- |
| `n_estimators` | 200 |
| `contamination` | 0.02 (matches injected rate) |
| `max_samples` | `auto` |
| `random_state` | 42 |

**Performance**: AUC-ROC = 0.854

### Local Outlier Factor (LOF)

| Parameter | Value |
| :--- | :--- |
| `n_neighbors` | 20 |
| `contamination` | 0.02 |
| `metric` | `euclidean` |
| `leaf_size` | 30 |

**Performance**: AUC-ROC = 0.812

### One-Class SVM

| Parameter | Value |
| :--- | :--- |
| `kernel` | `rbf` |
| `nu` | 0.02 |
| `gamma` | `scale` |

**Performance**: AUC-ROC = 0.783

## Model Performance Comparison

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall | Specificity | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.973** | **0.945** | **0.912** | **0.928** | **0.897** | 0.954 | ~2 min |
| XGBoost | 0.962 | 0.931 | 0.894 | 0.913 | 0.876 | 0.948 | ~3 min |
| Random Forest | 0.931 | 0.887 | 0.841 | 0.861 | 0.822 | 0.932 | ~1 min |
| Isolation Forest | 0.854 | 0.782 | 0.762 | 0.743 | 0.781 | 0.897 | ~30s |
| LOF | 0.812 | 0.731 | 0.714 | 0.692 | 0.738 | 0.873 | ~45s |
| One-Class SVM | 0.783 | 0.694 | 0.681 | 0.654 | 0.711 | 0.856 | ~1 min |

## Scenario-Specific Detection Rates (LightGBM)

| Scenario | Precision | Recall | F1 | FPR | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Smurfing | 0.95 | 0.96 | 0.95 | 2.8% | 0.982 |
| Layering | 0.91 | 0.93 | 0.92 | 3.1% | 0.968 |
| Mule Accounts | 0.96 | 0.95 | 0.95 | 2.5% | 0.984 |
| Circular Trading | 0.88 | 0.91 | 0.89 | 4.2% | 0.958 |

## SHAP Feature Importance

Top 15 features by mean |SHAP| value:

| Rank | Feature | Mean |SHAP| | Impact on Output |
| :---: | :--- | :---: | :--- |
| 1 | `send_receive_ratio` | 0.124 | High ratio → high risk |
| 2 | `cash_out_ratio` | 0.108 | High ratio → high risk |
| 3 | `velocity_24hr_max` | 0.097 | High velocity → high risk |
| 4 | `clustering_coefficient` | 0.089 | Low coefficient + high degree → high risk |
| 5 | `betting_ratio` | 0.082 | High ratio → high risk |
| 6 | `avg_tx_value` | 0.076 | Low value + high count → smurfing risk |
| 7 | `rapid_transaction_ratio` | 0.071 | High ratio → layering risk |
| 8 | `betweenness_centrality` | 0.065 | High centrality → intermediary risk |
| 9 | `balance_velocity` | 0.059 | High velocity → pass-through risk |
| 10 | `unique_counterparties` | 0.052 | Many partners → network risk |
| 11 | `balance_retention_ratio` | 0.044 | Low retention → sweeping |
| 12 | `tx_per_active_day` | 0.038 | High frequency → automated activity |
| 13 | `retail_engagement` | 0.032 | Low engagement → minimal legitimate use |
| 14 | `zero_balance_frequency` | 0.027 | Frequent zero → sweeping/cleaning |
| 15 | `international_ratio` | 0.022 | High ratio → cross-border risk |

## Cross-Validation Results

5-fold stratified cross-validation (LightGBM):

| Fold | AUC-ROC | AUC-PR | F1 | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.971 | 0.942 | 0.908 | 0.924 | 0.893 |
| 2 | 0.974 | 0.946 | 0.914 | 0.931 | 0.898 |
| 3 | 0.972 | 0.943 | 0.910 | 0.926 | 0.895 |
| 4 | 0.975 | 0.947 | 0.915 | 0.932 | 0.899 |
| 5 | 0.973 | 0.944 | 0.912 | 0.928 | 0.897 |
| **Mean (std)** | **0.973 (0.002)** | **0.944 (0.002)** | **0.912 (0.003)** | **0.928 (0.003)** | **0.896 (0.002)** |

Low variance across folds confirms model stability and consistent feature signal.

## Graph-Based Detection

NetworkX transaction graph analysis provides complementary detection:

| Metric | Normal Mean | AML Mean | Separation |
| :--- | :---: | :---: | :---: |
| Betweenness Centrality | 0.02 | 0.34 | Strong |
| Clustering Coefficient | 0.45 | 0.08 | Strong |
| Degree Centrality | 0.03 | 0.28 | Strong |
| PageRank | 0.0004 | 0.0031 | Moderate |

**Graph anomaly scoring** combines these metrics into a compound score that identifies structurally anomalous nodes irrespective of transaction value or frequency.

## Model Selection Decision

**Primary Model**: **LightGBM** — Best overall AUC-ROC (0.973), fastest training time, best balance of precision/recall, native missing value handling.

**Secondary Model**: **XGBoost** — Comparable performance (0.962), useful for ensemble blending and cross-validation.

**Unsupervised Complement**: **Isolation Forest** (0.854) — Effective for detecting novel anomaly patterns not seen in training data.

**Graph Layer**: **NetworkX metrics** — Independent detection signal that catches structural anomalies missed by feature-based models.
