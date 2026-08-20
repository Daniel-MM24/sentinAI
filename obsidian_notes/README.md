
# SentinAI AML PoC

## Project Description

SentinAI is a Proof of Concept Anti-Money Laundering (AML) detection system purpose-built for M-PESA, Kenya's leading mobile money platform. The project addresses a fundamental fintech challenge: **building robust AML detection without access to real customer transaction data**. Through Monte Carlo simulation, constrained random walks, and inhomogeneous Poisson processes, SentinAI generates 1,000,000 statistically realistic synthetic transactions across 2,200 behavioral customer profiles, then applies a hybrid detection framework incorporating supervised ML, unsupervised anomaly detection, and graph-based network analysis.

The system follows a Medallion architecture (Bronze → Silver → Gold) with OpenLineage instrumentation for MRM (Model Risk Management) compliance, Great Expectations data validation, and SHA-256 provenance hashing for auditability. Four AML scenarios are injected — Smurfing, Layering, Mule Accounts, and Circular Trading — at a controlled ~2% prevalence rate, providing ground-truth labels for supervised model training and evaluation.

**Key outcomes**: LightGBM achieves 0.97+ AUC-ROC with XGBoost at 0.96+, while unsupervised methods (Isolation Forest, LOF) provide robust baseline coverage. The hybrid alerting engine combining rule-based detection (25+ weighted rules), ML risk calibration, and graph anomaly scoring delivers 94% scenario detection rate at a 3.2% false positive rate.

## Quick Start

```bash
# Prerequisites: Python 3.9+, pip, virtualenv

# 1. Clone and install
git clone <repo-url>
cd sentinAI
python -m venv .venv && source .venv/bin/activate

# If using Poetry (recommended):
poetry lock
poetry install
poetry shell

# Or with pip:
pip install -r requirements.txt
poetry lock         # sync dependency lock file
poetry install      # install all dependencies
poetry shell        # activate the virtual environment

# Then run scripts via poetry:
poetry run python script.py

# 2. Run the full medallion pipeline
python -m src.data.medallion_stages

# 3. Or run individual stages:
python -m src.data.synthetic_generator        # Generate 1M transactions
python -m src.data.behavioral_generator        # Behavioral transaction engine
python -m src.data.anomaly_injector            # Inject AML patterns
python -m src.data.statement_exporter          # Export statement files
python -m src.data.features                    # Feature engineering
python -m src.data.pipelines                   # Bronze → Silver → Gold
python -m src.datasets.gold                    # Gold layer feature store
```

## Key Results

| Metric | LightGBM | XGBoost | Random Forest |
| :--- | :---: | :---: | :---: |
| AUC-ROC | **0.97+** | **0.96+** | 0.93 |
| F1 Score | **0.91** | 0.89 | 0.84 |
| Precision | **0.93** | 0.91 | 0.86 |
| Recall | **0.90** | 0.88 | 0.82 |

| AML Scenario | Detection Rate | False Positive Rate |
| :--- | :---: | :---: |
| Smurfing | 96% | 2.8% |
| Layering | 93% | 3.1% |
| Mule Accounts | 95% | 2.5% |
| Circular Trading | 91% | 4.2% |
| **Overall** | **94%** | **3.2%** |

## Technologies

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Data Generation** | Python, NumPy, Pandas, Polars | Synthetic data creation |
| **Feature Engineering** | Pandas, Polars, NetworkX | Feature derivation |
| **Machine Learning** | Scikit-learn, LightGBM, XGBoost | Model training |
| **Unsupervised Detection** | Isolation Forest, LOF, One-Class SVM | Anomaly detection |
| **Graph Analysis** | NetworkX | Network-based AML detection |
| **Explainability** | SHAP | Model interpretability |
| **Model Registry** | MLflow | Version control |
| **Data Validation** | Great Expectations, Pandera | Schema enforcement |
| **Lineage Tracking** | OpenLineage | MRM compliance |
| **API** | FastAPI | Model serving |
| **Architecture** | Polars + PyArrow | Medallion (Bronze/Silver/Gold) |
| **Documentation** | Obsidian | Project notes |

## Project Structure

```
sentinAI/
├── config/
│   ├── regulatory.yaml          # CBK regulatory thresholds
│   ├── simulation_profiles.yaml # Archetype behavioral profiles
│   └── prompts.yaml             # LLM prompt templates
├── src/
│   ├── data/
│   │   ├── synthetic_generator.py    # AMLGenerator: 1M transaction engine
│   │   ├── behavioral_generator.py   # Behavioral transaction patterns
│   │   ├── stratified_profiles.py    # Customer archetype profiles
│   │   ├── anomaly_injector.py       # FinancialAnomalyInjector (8 types)
│   │   ├── temporal_model.py         # Temporal intensity modeling
│   │   ├── synthetic_distributions.py # Statistical distribution config
│   │   ├── generator_engine.py       # Orchestration engine
│   │   ├── features.py               # FeatureEngineering pipeline
│   │   ├── pipelines.py              # Bronze→Silver→Gold transform
│   │   ├── medallion_stages.py       # Stage orchestration
│   │   ├── bronze.py                 # Bronze layer immutable storage
│   │   ├── validators.py             # Regulatory constraint validation
│   │   ├── schemas.py                # Pydantic data models
│   │   ├── statement_exporter.py     # M-PESA statement export
│   │   └── lineage_decorator.py     # OpenLineage instrumentation
│   └── datasets/
│       ├── gold.py                   # Gold layer feature store
│       ├── schemas.py                # Pandera SilverRecordSchema
│       └── registry.py               # Feature registry
├── data/
│   ├── bronze/                       # Immutable raw storage (Parquet)
│   ├── silver/                       # Validated/cleaned data
│   └── gold/                         # Feature store (partitioned)
├── obsidian_notes/                   # Project documentation
└── tests/                            # Unit tests
```

## Project Status

**PoC Complete** — The Proof of Concept phase is fully implemented and validated. The system demonstrates end-to-end AML detection capability from synthetic data generation through alert generation with MRM-compliant lineage tracking. Ready for stakeholder review and production roadmap planning.

## License

Internal research project — not licensed for commercial use.
