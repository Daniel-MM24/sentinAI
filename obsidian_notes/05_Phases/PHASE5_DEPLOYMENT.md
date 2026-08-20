# Phase 5: Production Deployment Readiness

## Objective

Architect and document the production deployment path for the SentinAI AML detection system, covering API serving, monitoring, model registry, and operational workflows.

## Deployment Architecture

```
                      ┌──────────────────────┐
                      │   Transactions        │
                      │   (Kafka/API Stream)  │
                      └──────────┬───────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME SCORING LAYER                            │
│                                                                         │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐             │
│  │  FastAPI      │   │  Feature      │   │  Model        │             │
│  │  Endpoint     │──►│  Pipeline     │──►│  Inference    │             │
│  │  POST /score  │   │  (real-time)  │   │  (LightGBM)   │             │
│  └───────────────┘   └───────────────┘   └───────┬───────┘             │
│                                                   │                     │
│                                                   ▼                     │
│                                          ┌───────────────┐             │
│                                          │  Alert Engine  │             │
│                                          │  (hybrid)      │             │
│                                          └───────┬───────┘             │
└──────────────────────────────────────────────────┼─────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      OPERATIONS LAYER                                   │
│                                                                         │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐          │
│  │  Model Registry│   │  Monitoring    │   │  Case          │          │
│  │  (MLflow)      │   │  (Grafana)     │   │  Management    │          │
│  │                │   │                │   │                │          │
│  │ • Model Store  │   │ • Alert Volume │   │ • Queue        │          │
│  │ • Metrics      │   │ • Latency p99  │   │ • Assignment   │          │
│  │ • Versioning   │   │ • False Pos.   │   │ • SAR Workflow │          │
│  └────────────────┘   └────────────────┘   └────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      REPORTING LAYER                                    │
│                                                                         │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐          │
│  │  Regulatory    │   │  Audit Reports │   │  Analytics     │          │
│  │  (SAR/STR)     │   │  (MRM)         │   │  Dashboard     │          │
│  └────────────────┘   └────────────────┘   └────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

## API Specification (FastAPI)

### Endpoints

| Endpoint | Method | Description | Request | Response |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | GET | Service health check | — | `{"status": "ok", "model": "lightgbm_v1", "uptime": 3600}` |
| `/score` | POST | Score a single transaction | Transaction JSON | `{risk_score, risk_level, shap_values}` |
| `/score/batch` | POST | Score batch transactions | Transaction list | `[{risk_score, ...}]` |
| `/alerts` | GET | Query alert queue | Query params | Alert list with pagination |
| `/alerts/{id}` | GET | Get alert details | Alert ID | Full alert with evidence |
| `/models` | GET | List available models | — | Model registry contents |
| `/models/activate` | POST | Activate model version | `{version}` | Activation confirmation |

### Scoring Request Schema

```json
{
  "customer_id": "C_001234",
  "transaction_id": "TX_000123456",
  "timestamp": "2025-06-15T14:30:00Z",
  "transaction_type": "send_money",
  "amount": 45000.00,
  "paid_in": 0.00,
  "paid_out": 45000.00,
  "balance": 5000.00,
  "counterparty": "John",
  "is_betting": false,
  "is_international": false,
  "is_kadogo": false
}
```

### Scoring Response Schema

```json
{
  "transaction_id": "TX_000123456",
  "risk_score": 0.87,
  "risk_level": "HIGH",
  "ml_probability": 0.91,
  "rule_score": 0.82,
  "graph_anomaly_score": 0.76,
  "top_risk_factors": [
    {"feature": "send_receive_ratio", "value": 12.5, "shap_contribution": 0.21},
    {"feature": "velocity_24hr", "value": 28, "shap_contribution": 0.15}
  ],
  "triggered_rules": ["cash_out_ratio > 0.8", "velocity_24hr > 20"],
  "primary_typology": "Layering",
  "model_version": "lightgbm_v1.0",
  "score_timestamp": "2025-06-15T14:30:00.123Z"
}
```

## Model Registry (MLflow)

### Registry Structure

```
models:/sentinai_aml_detection/
├── 1/  (lightgbm_v1.0) — Active
│   ├── model.pkl
│   ├── conda.yaml
│   ├── requirements.txt
│   └── artifacts/
│       ├── feature_importance.png
│       ├── confusion_matrix.png
│       ├── model_performance.json
│       └── shap_summary.png
├── 2/  (xgboost_v1.0) — Staging
│   └── ...
└── 3/  (ensemble_v1.0) — Development
    └── ...
```

### Model Lifecycle

| Stage | Description | Deployment |
| :--- | :--- | :--- |
| **Development** | Experimental model, testing | None |
| **Staging** | Validated, awaiting review | Shadow traffic |
| **Production** | Active scoring model | 100% traffic |
| **Archived** | Retired model | None |

### MLflow Tracking

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(lightgbm_params)
    mlflow.log_metrics({
        "auc_roc": 0.973,
        "f1_score": 0.912,
        "precision": 0.928,
        "recall": 0.897
    })
    mlflow.lightgbm.log_model(model, "model")
    mlflow.log_artifact("shap_summary.png")
    mlflow.log_artifact("feature_importance.csv")
```

## Monitoring & Observability

### Key Metrics (Grafana Dashboard)

| Category | Metric | Alert Threshold | Severity |
| :--- | :--- | :--- | :--- |
| **Volume** | Alert rate (alerts/min) | > 2σ from baseline | Warning |
| **Performance** | Scoring latency p99 | > 500ms | Critical |
| **Accuracy** | False positive rate (daily) | > 5% | Warning |
| **Accuracy** | Detection rate (daily) | < 85% | Critical |
| **Health** | Model prediction drift | PSI > 0.2 | Warning |
| **Health** | Data drift (feature distributions) | KS test p < 0.05 | Info |
| **Operations** | Queue depth (pending alerts) | > 10,000 | Warning |
| **Operations** | SAR filing backlog | > 48 hours | Critical |

### Logging Strategy

| Log Type | Content | Retention |
| :--- | :--- | :--- |
| Transaction Scoring | Request/response, score, latency | 90 days |
| Model Predictions | Features, prediction, SHAP values | 30 days |
| Alert Events | Alert creation, assignment, resolution | 7 years (regulatory) |
| System Health | CPU, memory, throughput | 30 days |
| Audit Logs | Model changes, config changes | 7 years |

## Batch Scoring Pipeline

For historical/backfill scoring:

```bash
python -m src.models.batch_score \
  --input data/gold/features/vv1.0/ \
  --output output/scored_transactions.csv \
  --model output/models/lightgbm_model.pkl \
  --batch-size 10000 \
  --parallel 4
```

## Performance Targets

| Metric | PoC | Target (Production) |
| :--- | :---: | :---: |
| Throughput (single instance) | ~800 tx/s | > 5,000 tx/s |
| p50 Latency | 1.25 ms | < 10 ms |
| p99 Latency | 5 ms | < 100 ms |
| Model Load Time | 2s | < 500ms |
| Feature Pipeline | 5 min (batch) | < 50ms (stream) |
| Alert Generation | 1 min (batch) | < 100ms (stream) |
| Uptime SLA | N/A | 99.9% |

## Scalability Strategy

| Constraint | Current Limit | Solution |
| :--- | :--- | :--- |
| Model Inference | ~10K tx/s single thread | Horizontal scaling via container orchestration |
| Feature Computation | ~50ms per transaction | Pre-computed features + incremental updates |
| Graph Analysis | ~1K tx/s | Subgraph sampling + incremental PageRank |
| Data Storage | ~500 MB (PoC) | Parquet partitioning + S3/GCS lifecycle policies |
| Alert Volume | ~30K alerts | Priority-based triage + auto-SAR for CRITICAL |

## Security Considerations

| Concern | Mitigation |
| :--- | :--- |
| Model poisoning | Input validation, anomaly detection on feature vectors |
| API abuse | Rate limiting, authentication (API keys/JWT) |
| Data leakage | Synthetic-only data in PoC, encrypted at rest in production |
| Model theft | Registry access controls, model encryption |
| Adversarial attacks | Feature clipping, ensemble voting, SHAP-based outlier detection |

## DR / Business Continuity

| Scenario | RTO | RPO | Recovery Strategy |
| :--- | :---: | :---: | :--- |
| Single instance failure | 5 min | 0 | Auto-restart via container orchestration |
| Model corruption | 15 min | 1 version | Rollback via MLflow registry |
| Data corruption | 1 hour | 1 hour | Restore from Bronze immutable layer |
| Full region failure | 4 hours | 1 hour | Active-passive multi-region deployment |
