# Deployment Guide

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / macOS 12+ / Windows WSL2
- **Python**: 3.9 - 3.11 (3.10 recommended)
- **RAM**: 8 GB minimum (16 GB recommended for full pipeline)
- **Disk**: 5 GB free for data generation and model artifacts
- **CPU**: 4+ cores recommended for parallel processing

### Dependencies

```bash
# Core data processing
pip install numpy pandas polars pyarrow

# Machine Learning
pip install scikit-learn lightgbm xgboost

# Graph Analysis
pip install networkx

# Explainability
pip install shap

# Data Validation
pip install great-expectations pandera

# Lineage Tracking (optional)
pip install openlineage-python

# Model Registry (optional)
pip install mlflow

# API Server (optional)
pip install fastapi uvicorn

# Visualization (optional)
pip install matplotlib seaborn plotly

# Testing
pip install pytest pytest-cov
```

### Verified Versions (Tested Configuration)

```
numpy==1.24.3
pandas==2.0.3
polars==0.19.0
pyarrow==12.0.0
scikit-learn==1.3.0
lightgbm==4.1.0
xgboost==2.0.0
networkx==3.1
shap==0.42.1
great-expectations==0.17.0
pandera==0.17.0
mlflow==2.6.0
fastapi==0.104.0
pytest==7.4.0
```

## Installation

### 1. Clone Repository

```bash
git clone <repo-url>
cd sentinAI
```

### 2. Virtual Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import polars; import lightgbm; import shap; import networkx; print('All dependencies OK')"
```

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)

Run the complete Bronze → Silver → Gold Medallion pipeline end-to-end:

```bash
python -m src.data.medallion_stages
```

This executes:
1. Synthetic data generation (Phase 1)
2. Behavioral transaction generation (Phase 1b)
3. Anomaly injection (Phase 1c)
4. Bronze layer ingestion (immutable raw storage)
5. Silver layer transformation (validation + dedup)
6. Feature engineering (Phase 2)
7. Gold layer feature store (partitioned + versioned)
8. Model training (Phase 4)
9. Model evaluation and metrics export
10. Alert generation (Phase 5)

### Option 2: Individual Stages

```bash
# Step 1: Generate synthetic data
python -m src.data.synthetic_generator

# Step 2: Run behavioral generator
python -m src.data.behavioral_generator

# Step 3: Inject anomalies
python -m src.data.anomaly_injector

# Step 4: Ingest to Bronze layer
python -m src.data.bronze

# Step 5: Validate and transform to Silver
python -m src.data.validators

# Step 6: Engineer features
python -m src.data.features

# Step 7: Transform Silver → Gold
python -m src.data.pipelines

# Step 8: Build Gold feature store
python -m src.datasets.gold

# Step 9: Train models
python -m src.models.train

# Step 10: Generate alerts
python -m src.models.alert_generation
```

### Option 3: Configuration-Driven

Edit `config/simulation_profiles.yaml` to customize:
- Number of customers
- Archetype distributions
- Transaction type probabilities
- Balance tier limits
- Anomaly ratios and types

Then run the full pipeline via Option 1.

## Configuration Reference

### `config/simulation_profiles.yaml`

```yaml
customers:
  total: 2200
  archetypes:
    retail_heavy: 0.15
    retail_standard: 0.70
    micro_merchant: 0.12
    corporate: 0.03

kyc_tiers:
  tier_1: { limit: 50000, proportion: 0.60 }
  tier_2: { limit: 500000, proportion: 0.30 }
  tier_3: { limit: 5000000, proportion: 0.10 }

transactions:
  total: 1000000
  distributions:
    send_money: 0.25
    received_money: 0.20
    agent_deposit: 0.10
    agent_withdrawal: 0.10
    paybill: 0.15
    buy_goods: 0.12
    others: 0.08
```

### `config/regulatory.yaml`

```yaml
cbk:
  tier_limits:
    tier_1: { max_balance: 50000, daily_velocity: 100000 }
    tier_2: { max_balance: 500000, daily_velocity: 1000000 }
    tier_3: { max_balance: 5000000, daily_velocity: 10000000 }

  thresholds:
    cash_out_ratio: 0.8
    send_receive_ratio: 5.0
    kadogo_threshold_p2p: 100
    kadogo_threshold_merchant: 200
    structuring_threshold: 10000
    large_tx_threshold: 100000
    betting_ratio_threshold: 0.3
    international_ratio_threshold: 0.2
```

## Output Artifacts

### Data Outputs

| Artifact | Path | Format | Description |
| :--- | :--- | :--- | :--- |
| Customer Metadata | `data/bronze/customers/customer_profiles.csv` | CSV | 1,000 customer profiles (3 columns) |
| Detailed Transactions | `data/detailed_transactions.csv` | CSV | 10,000 individual transactions |
| Ground Truth Labels | `data/aml_ground_truth.csv` | CSV | Per-customer AML scenario labels |
| Temporal Features | `data/temporal_features.csv` | CSV | 24 temporal feature columns |
| Summary Statements | `output/summary_statements.csv` | CSV | Aggregated per-customer |
| Bronze Transactions | `data/bronze/transactions/` | Parquet | Immutable raw storage |
| Silver Transactions | `data/silver/transactions/` | Parquet | Validated, deduplicated |
| Gold Features | `data/gold/features/vv1.0/` | Parquet | Partitioned feature store |

### Model Artifacts

| Artifact | Path | Description |
| :--- | :--- | :--- |
| LightGBM Model | `output/models/lightgbm_model.pkl` | Primary classifier |
| XGBoost Model | `output/models/xgboost_model.pkl` | Cross-validation model |
| Random Forest Model | `output/models/rf_model.pkl` | Baseline model |
| Model Metrics | `output/models/model_performance.json` | All evaluation metrics |
| SHAP Explanations | `output/models/shap_explanations.csv` | Feature importance |

### Alert Outputs

| Artifact | Path | Description |
| :--- | :--- | :--- |
| Hybrid Alerts | `output/alerts/hybrid_alerts.csv` | Scored alert queue |
| Case Management | `output/alerts/case_management.csv` | Investigation cases |
| Alert Metrics | `output/alerts/alert_metrics.json` | Alert quality metrics |

### Metadata Outputs

| Artifact | Path | Description |
| :--- | :--- | :--- |
| Bronze Metadata | `data/bronze/metadata/` | JSON | Ingestion metadata |
| Gold Manifest | `data/gold/features/vv1.0/manifest.json` | JSON | Feature store manifest |
| Transformation Logs | `logs/` | Text | Pipeline run logs |

## Deployment Modes

### Development Mode

```bash
# Full pipeline with verbose logging
LOG_LEVEL=DEBUG python -m src.data.medallion_stages

# Jupyter notebook exploration
jupyter notebook notebooks/exploration.ipynb
```

### Production Mode (API Serving)

```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Score a transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C_000001", "amount": 50000, "type": "send_money"}'
```

### Batch Scoring

```bash
# Score entire dataset
python -m src.models.batch_score \
  --input data/gold/features/vv1.0/ \
  --output output/scored_transactions.csv \
  --model output/models/lightgbm_model.pkl
```

## Monitoring & Observability

### Logging
- Pipeline logs: `logs/pipeline_YYYYMMDD.log`
- Model training logs: `logs/training_YYYYMMDD.log`
- Alert generation logs: `logs/alerts_YYYYMMDD.log`

### Metrics Export
- Model metrics: JSON format at `output/models/model_performance.json`
- Alert metrics: JSON format at `output/alerts/alert_metrics.json`
- Data quality reports: Great Expectations HTML at `output/quality_reports/`

### Lineage Tracking
When OpenLineage is configured:
- Lineage events emitted to configurable HTTP endpoint
- Each transformation includes: `run_id`, `input_datasets`, `output_datasets`, `transformation_type`, `timestamp`
- View lineage via Marquez UI (default: `http://localhost:8001`)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html

# Run specific test suites
pytest tests/test_synthetic_generator.py -v
pytest tests/test_anomaly_injector.py -v
pytest tests/test_validators.py -v
pytest tests/test_features.py -v
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `MemoryError` during generation | 1M transactions exceed RAM | Reduce `simulation_profiles.yaml` → `transactions.total` or use Polars streaming |
| `Parquet schema mismatch` | Column type inconsistency | Clear `data/` directories and regenerate |
| `OpenLineage import error` | Optional dependency missing | `pip install openlineage-python` or set `OL_AVAILABLE = False` |
| `Great Expectations import error` | Optional dependency missing | `pip install great-expectations` or set `GX_AVAILABLE = False` |
| `Balance constraint violation` | Archetype/limit mismatch | Verify tier caps in `config/simulation_profiles.yaml` |
| Slow generation | Single-threaded processing | Set `OMP_NUM_THREADS=4` or adjust `n_jobs` in generator config |

### Clean Rebuild

```bash
# Remove all generated data and models
rm -rf data/bronze/ data/silver/ data/gold/ data/dead_letter/
rm -rf output/customers_metadata.csv output/detailed_transactions.csv
rm -rf output/models/ output/alerts/
rm -rf logs/

# Regenerate from scratch
python -m src.data.medallion_stages
```

## Performance Benchmarks

| Operation | Time | Memory | Notes |
| :--- | :---: | :---: | :--- |
| Operation | Time | Memory | Notes |
| :--- | :---: | :---: | :--- |
| Stratified Profile Generation | ~3s | 50 MB | 1,000 customers |
| Behavioral Transaction Gen | ~15s | 200 MB | 10K transactions |
| AML Scenario Injection | ~2s | 50 MB | 2% prevalence, 4 typologies |
| Temporal Feature Extraction | ~5s | 100 MB | 24 feature columns |
| Bronze Ingestion | ~10s | 200 MB | Parquet write |
| Silver Validation | ~20s | 300 MB | Schema + dedup |

*Benchmarks on: Intel i7-12700H, 32 GB RAM, NVMe SSD, Ubuntu 22.04*
