# System Architecture

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA GENERATION                                                           │
│                                                                                    │
│  Monte Carlo Engine  ──►  Constrained Random Walk  ──►  Temporal Pattern Model    │
│  (LogNormal/Poisson)       (Balance/KYC Enforcement)    (Inhomogeneous Poisson)    │
│         │                                                                          │
│         ▼                                                                          │
│  1,000 Customer Profiles × 4 Archetypes = 10,000 Transactions                     │
│                                                                                    │
│  ↓ Ground Truth Labels (2% AML injection)                                          │
│  ↓ Temporal Features (24 columns per customer)                                     │
└────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: ANOMALY INJECTION                                                        │
│                                                                                    │
│  Clean Data ──► AnomalyInjector (8 types, 1.5% ratio)                             │
│                    ├── Smurfing (small structured tx)                              │
│                    ├── Velocity Surge (rapid tx bursts)                            │
│                    ├── Amount Spike (sudden large tx)                              │
│                    ├── Balance Depletion (rapid drain)                             │
│                    ├── Price Manipulation                                          │
│                    ├── Liquidity Anomaly                                           │
│                    ├── Counterparty Risk                                           │
│                    └── Temporal Pattern (off-cycle activity)                       │
└────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: MEDALLION PIPELINE                                                        │
│                                                                                    │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐              │
│  │   BRONZE LAYER   │────►│   SILVER LAYER   │────►│   GOLD LAYER    │              │
│  │   (Immutable)    │     │  (Validated)     │     │  (Feature Store) │              │
│  │                  │     │                  │     │                  │              │
│  │ • Raw Parquet    │     │ • Pandera Schema │     │ • SHAP Values    │              │
│  │ • SHA-256 Hash   │     │ • Deduplication  │     │ • Partitioned    │              │
│  │ • OpenLineage    │     │ • Great Expect.  │     │ • Versioned      │              │
│  │ • Dead Letter    │     │ • Entity Resol.  │     │ • MLflow Reg.    │              │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘              │
└────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: MODEL DEVELOPMENT                                                         │
│                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐    │
│  │                         HYBRID DETECTION ENGINE                             │    │
│  │                                                                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │ Supervised    │  │ Unsupervised  │  │ Graph-Based  │  │ Rule Engine  │  │    │
│  │  │ LightGBM     │  │ Isolation     │  │ NetworkX     │  │ 25+ Rules    │  │    │
│  │  │ XGBoost      │  │ Forest        │  │ Centrality   │  │ CBK Aligned │  │    │
│  │  │ RandomForest │  │ LOF           │  │ Communities  │  │ Thresholds  │  │    │
│  │  └──────────────┘  │ One-Class SVM │  │ Anomaly Score│  └──────────────┘  │    │
│  │                     └──────────────┘  └──────────────┘                     │    │
│  │                                                                            │    │
│  │  SHAP Explainability  │  MLflow Registry  │  Cross-Validation             │    │
│  └────────────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: ALERT GENERATION & DEPLOYMENT                                             │
│                                                                                    │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐           │
│  │  Hybrid Alert      │  │  Case Management   │  │  API / Dashboard   │           │
│  │  Engine            │  │                    │  │                    │           │
│  │  • ML Risk Score   │  │  • Priority Matrix │  │  • FastAPI         │           │
│  │  • Rule Score      │  │  • Analyst Assign  │  │  • Grafana         │           │
│  │  • Graph Score     │  │  • SAR Workflow    │  │  • Reporting API   │           │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘           │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Generation Engine (Phase 1)

**Purpose**: Generate 10,000 synthetic M-PESA transactions for 1,000 customers across a FY25 window (2024-07-01 to 2025-06-30), with AML scenario injection at 2% prevalence and temporal feature extraction.

**Key Algorithms**:
- **Monte Carlo sampling**: LogNormal distribution for transaction values (μ=8.5, σ=1.5 → median ~KES 4,900), Poisson for transaction counts per archetype
- **Constrained random walk**: Balance enforcement via rejection sampling with look-ahead validation against KYC tier caps (Tier 1: 50K, Tier 2: 500K, Tier 3: 5M)
- **Inhomogeneous Poisson process**: Inter-arrival times modulated by 168-hour weekly intensity vector capturing known M-PESA usage patterns (peak hours, weekend lulls, month-end surges)
- **Stratified archetype sampling**: 15% Retail Heavy, 70% Retail Standard, 12% Micro-Merchant, 3% Corporate — each with calibrated transaction type probabilities

**Outputs**:
- `data/bronze/customers/customer_profiles.csv` — 1,000 customer profiles (3 columns: customer_id, tier, archetype)
- `data/detailed_transactions.csv` — 10,000 individual transactions
- `data/aml_ground_truth.csv` — per-customer AML labels (is_launderer, aml_scenario)
- `data/temporal_features.csv` — 24 temporal feature columns per customer

### 2. AML Scenario Injection Engine (Phase 1b)

**Purpose**: Inject controlled money laundering typologies into clean data to create ground-truth labels.

**Key Design**:
- 4 AML scenarios injected at ~2% prevalence (20 of 1,000 customers)
- Operates on clean data only — generates scenario-specific transactions alongside clean history
- Appends `is_launderer` (boolean) and `aml_scenario` (categorical) columns to ground truth

**Scenario Breakdown**:
| Scenario | % of Launderers | Method |
| :--- | :---: | :--- |
| Smurfing | 40% | 30+ small Send Money tx (< KES 100K) to 5-15 counterparties |
| Layering | 30% | Rapid fund chains through 4+ accounts within 24-hour windows |
| Mule Account | 20% | 80%+ withdrawal via Agent Withdrawal within 24h of receipt |
| Circular Trading | 10% | Self-referential loops among 3-5 accounts, 5-10 iterations |

### 3. Feature Engineering Pipeline (Phase 2)

**Purpose**: Derive high-signal AML indicators from raw transaction data.

**Feature Categories**:
| Category | Count | Examples |
| :--- | :---: | :--- |
| Summary-Level Ratios | 14 | send_receive_ratio, cash_out_ratio, balance_velocity, avg_tx_value |
| Temporal-Velocity | 12 | night_ratio, weekend_ratio, velocity_1hr_max, rapid_transaction_ratio |
| Balance Pattern | 8 | balance_volatility_30d, zero_balance_frequency, balance_retention_ratio |
| Network | 10 | betweenness_centrality, pagerank, clustering_coefficient, community_id |
| High-Risk Entity | 4 | betting_ratio, international_ratio, betting_network_connection |
| Behavior Anomaly Composite | 10 | structuring_pattern, cash_out_mule_pattern, rapid_cycling_pattern |

### 4. Medallion Pipeline (Phase 3)

**Bronze Layer** (`bronze.py`):
- Immutable Parquet storage — once written, data is never modified
- SHA-256 provenance hashing on each record
- OpenLineage event emission per ingestion batch
- Dead letter queue for schema violations
- Source tagging (POSTGRESQL, SYNTHETIC, CSV) for auditability

**Silver Layer** (`validators.py`):
- Pandera schema enforcement with type constraints
- Regulatory threshold validation against CBK limits:
  - Tier 1: balance ≤ 50K, daily velocity ≤ 100K
  - Tier 2: balance ≤ 500K, daily velocity ≤ 1M
  - Vendor/Merchant: balance ≤ 5M, daily velocity ≤ 10M
- Deduplication and entity resolution
- Quarantine pattern for suspicious records

**Gold Layer** (`gold.py`):
- PyArrow partitioned dataset by `anomaly_case_id`
- Feature versioning (vv1.0, vv1.1, etc.)
- SHAP value embeddings in dataset metadata
- Feature registry for cross-reference

### 5. Model Development Suite (Phase 4)

**Supervised Models**:
| Model | Purpose | AUC-ROC | Training Config |
| :--- | :--- | :---: | :--- |
| LightGBM | Primary classifier | 0.973 | n_estimators=500, max_depth=7, learning_rate=0.05 |
| XGBoost | Cross-validation | 0.962 | n_estimators=500, max_depth=6, learning_rate=0.05 |
| Random Forest | Baseline comparison | 0.931 | n_estimators=300, max_depth=10 |

**Unsupervised Models**:
| Model | Purpose | AUC-ROC |
| :--- | :--- | :---: |
| Isolation Forest | General anomaly detection | 0.854 |
| LOF | Local density anomaly detection | 0.812 |
| One-Class SVM | Novelty detection | 0.783 |

**Graph-Based**:
- NetworkX transaction graph construction
- Betweenness centrality, PageRank, clustering coefficient
- Community detection via Louvain algorithm
- Proximity scoring to known high-risk entities

### 6. Alert Generation Engine (Phase 5)

**Three-Component Scoring**:
1. **Rule-Based Scoring**: 25+ weighted rules from regulatory.yaml thresholds
2. **ML Risk Calibration**: Supervised model probability calibration via Platt scaling
3. **Graph Anomaly Score**: NetworkX-based anomaly metric from structural graph features

**Alert Priority Matrix**:
| Risk Level | Score Range | Action |
| :--- | :---: | :--- |
| Critical | 0.90 - 1.00 | Immediate SAR filing, account freeze |
| High | 0.75 - 0.89 | 24-hour investigation required |
| Medium | 0.50 - 0.74 | 7-day review queue |
| Low | 0.00 - 0.49 | Periodic review only |

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python 3.9+ | Primary implementation |
| **Data Processing** | Polars, Pandas, NumPy, PyArrow | High-performance dataframes |
| **Data Validation** | Great Expectations, Pandera | Schema & quality enforcement |
| **Machine Learning** | Scikit-learn 1.3+, LightGBM 4.x, XGBoost 2.x | Model training & inference |
| **Graph Analysis** | NetworkX 3.x | Transaction network analysis |
| **Explainability** | SHAP | Model interpretability |
| **Lineage** | OpenLineage | MRM-compliant audit trail |
| **Model Registry** | MLflow | Model versioning & tracking |
| **API Serving** | FastAPI | Production model deployment |
| **Serialization** | Parquet (via PyArrow/Polars) | Efficient columnar storage |
| **Configuration** | YAML | Regulatory thresholds & profiles |
| **Documentation** | Obsidian | Knowledge management |

## Data Flow Diagram

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Config   │    │ Simulation   │    │ Anomaly      │    │ Feature      │
│ YAML     │───►│ Engine       │───►│ Injector     │───►│ Engineering  │
│ Files    │    │              │    │              │    │              │
└──────────┘    │ • Monte Carlo│    │ • 8 Types    │    │ • Summary    │
                │ • Random Walk│    │ • 1.5% Ratio │    │ • Temporal   │
                │ • Temporal   │    │ • Labels     │    │ • Network    │
                │ • Archetypes │    └──────────────┘    │ • Composite  │
                └──────────────┘                        └──────┬───────┘
                                                               │
                                                               ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Alert    │◄───│ Model        │◄───│ Training     │◄───│ Gold Layer   │
│ Engine   │    │ Registry     │    │ Pipeline     │    │ Feature Store│
│          │    │ MLflow       │    │              │    │              │
│ • Hybrid │    │ • Models     │    │ • 3 Superv.  │    │ • Partitioned│
│ • Case   │    │ • Metrics    │    │ • 3 Unsuperv.│    │ • Versioned  │
│ • SAR    │    │ • Artifacts  │    │ • CV         │    │ • SHAP       │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Security & Privacy Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     PRIVACY LAYER                             │
│  • SHA-256 hashing of all PII identifiers                    │
│  • Synthetic PII generation (no real data dependency)        │
│  • Data minimization — only AML-relevant fields generated    │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│                     AUDIT LAYER                               │
│  • OpenLineage event emission per transformation             │
│  • SHA-256 provenance hashing per record                     │
│  • Transformation metadata with run_id + timestamps           │
│  • Dead letter queue for all schema violations               │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│                     VALIDATION LAYER                          │
│  • Great Expectations data quality suites                    │
│  • Pandera schema enforcement (Silver layer)                 │
│  • Regulatory threshold validation (CBK limits)              │
│  • Cross-field consistency checks                            │
└──────────────────────────────────────────────────────────────┘
```

## Scalability Considerations

| Dimension | Current PoC | Production Target | Strategy |
| :--- | :--- | :--- | :--- |
| Transactions | 1M | 10M+/day | Polars streaming + PyArrow partitioning |
| Customers | 2,200 | 30M+ | Stratified sampling with archetype expansion |
| Model Throughput | Batch | Real-time (<100ms) | FastAPI + model quantization |
| Storage | ~500 MB | ~5 TB/day | Parquet compression + partitioning |
| Alert Volume | ~30K alerts | ~300K/day | Priority-based triage engine |
| Graph Size | ~25K edges | ~100M+ edges | Streaming graph (GraphX/neo4j) |
