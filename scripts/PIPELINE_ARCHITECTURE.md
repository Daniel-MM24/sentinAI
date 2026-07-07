# SentinAI Pipeline Architecture Documentation

## Overview

The SentinAI data pipeline follows a modular medallion architecture with three distinct data layers (Bronze, Silver, Gold) and an orchestration layer that coordinates the end-to-end pipeline.

## Architecture Components

### 1. Individual Layer Scripts

Each layer has its own dedicated script that can be run independently or called programmatically:

#### **run_bronze.py** - Bronze Layer Ingestion
- **Purpose**: Generates synthetic M-Pesa transaction data and ingests it into the Bronze layer
- **Key Function**: `run_bronze_ingestion()`
- **Inputs**: 
  - `n_records`: Number of synthetic records to generate
  - `num_users`: Number of unique users
  - `target_distribution_params`: M-Pesa distribution parameters
  - `partition_key`: Date partition for storage
- **Outputs**: 
  - `bronze_path`: Path to stored Bronze data
  - `synthetic_data`: Generated synthetic dataframe
  - `partition_key_used`: Partition key used
- **Storage**: `data/bronze/transactions/{partition_key}/`

#### **run_silver.py** - Silver Layer Transformation
- **Purpose**: Transforms Bronze data into clean, feature-ready Silver data
- **Key Function**: `run_silver_transformation()`
- **Inputs**:
  - `partition_key`: Partition key for Bronze data
  - `bronze_base_path`: Base path for Bronze layer
  - `silver_base_path`: Base path for Silver layer
- **Outputs**:
  - `transactions_path`: Path to Silver transactions
  - `customers_path`: Path to Silver customers
  - `transaction_fact_df`: Transaction fact dataframe
  - `customer_dimension_df`: Customer dimension dataframe
  - `partition_key_used`: Partition key used
- **Storage**: 
  - `data/silver/transactions/silver_transactions_{partition_key}.parquet`
  - `data/silver/customers/silver_customers_{partition_key}.parquet`

#### **run_gold.py** - Gold Layer Materialization
- **Purpose**: Creates ML-ready training data with optional anomaly injection
- **Key Functions**:
  - `run_gold_materialization()`: Main Gold layer processing
  - `run_anomaly_injection()`: Injects fraud anomalies for ML training
  - `finalize_gold_dataset()`: Finalizes Gold dataset with labels
- **Inputs**:
  - `partition_key`: Partition key for Silver data
  - `with_anomaly_injection`: Whether to perform anomaly injection
  - `baseline_df`: Baseline dataframe for anomaly injection
  - `contamination_rate`: Global contamination rate
  - `anomaly_distribution`: Distribution of anomaly types
  - Various other configuration parameters
- **Outputs**:
  - `gold_uri`: Path to Gold data
  - `partition_key_used`: Partition key used
  - `gold_df`: Combined Gold dataframe
  - `injection_summary`: Anomaly injection summary
- **Storage**: `data/gold/features/transaction_date=*/`

### 2. Orchestration Script

#### **run_audit_and_synth.py** - End-to-End Pipeline Orchestration
- **Purpose**: Orchestrates the complete pipeline from data generation to ML-ready Gold layer
- **Function**: Calls individual layer functions in sequence
- **Pipeline Flow**:
  1. Bronze layer ingestion (via `run_bronze_ingestion()`)
  2. Silver layer transformation (via `run_silver_transformation()`)
  3. Gold layer materialization with anomaly injection (via `run_gold_materialization()`)
- **Configuration**: Loads from `config/data/data_config.yaml`

## Usage Patterns

### Pattern 1: End-to-End Pipeline Execution

Run the complete pipeline using the orchestrator:

```bash
python scripts/run_audit_and_synth.py
```

This executes all layers sequentially with anomaly injection for ML-ready data.

### Pattern 2: Individual Layer Execution

Run each layer independently for testing or partial pipeline execution:

```bash
# Bronze layer only
python scripts/run_bronze.py

# Silver layer only (requires Bronze data)
python scripts/run_silver.py

# Gold layer only (requires Silver data)
python scripts/run_gold.py
```

### Pattern 3: Programmatic Layer Composition

Import and call layer functions in custom scripts:

```python
from scripts.run_bronze import run_bronze_ingestion
from scripts.run_silver import run_silver_transformation
from scripts.run_gold import run_gold_materialization

# Execute Bronze layer
bronze_path, synthetic_df, partition_key = run_bronze_ingestion(
    n_records=1000000,
    num_users=200000
)

# Execute Silver layer
transactions_path, customers_path, tx_df, cust_df, partition_key = run_silver_transformation(
    partition_key=partition_key
)

# Execute Gold layer with anomaly injection
gold_uri, partition_key, gold_df, injection_summary = run_gold_materialization(
    partition_key=partition_key,
    with_anomaly_injection=True,
    baseline_df=synthetic_df,
    contamination_rate=0.0015
)
```

## Data Flow

```
Synthetic Data Generation
    ↓
Bronze Layer (Raw, Immutable)
    ↓
Silver Layer (Clean, Feature-Ready)
    ↓
Gold Layer (ML-Ready with Labels)
```

## Key Features

### Modularity
- Each layer can be developed, tested, and executed independently
- Clear separation of concerns between layers
- Reusable functions for programmatic composition

### Reproducibility
- Individual scripts can reproduce the full pipeline when run sequentially
- Same configuration parameters across standalone and orchestrated execution
- Deterministic random seeds for consistent results

### Flexibility
- Support for both standard feature store creation and anomaly injection
- Configurable partition keys and storage paths
- Optional anomaly injection for ML training data

### Auditability
- OpenLineage integration for all layers
- Comprehensive logging and metadata
- Artifact generation for pipeline runs

## Configuration

Pipeline behavior is controlled through:

1. **Configuration Files**: `config/data/data_config.yaml`
2. **Function Parameters**: Direct parameter passing to layer functions
3. **Environment Variables**: For deployment-specific settings

## Anomaly Injection

The Gold layer supports injection of 7 types of financial fraud anomalies:
- Case 001: Structuring/Smurfing
- Case 002: Agent Fraud / Cash-Out Collusion
- Case 003: Digital Lending Misuse
- Case 006: Trade-Based Money Laundering
- Case 007: PEP Monitoring Breach
- Case 008: Terrorist Financing Patterns
- Case 009: Shell Company Layering

Anomaly injection is controlled via:
- `contamination_rate`: Global anomaly rate (default: 0.0015)
- `anomaly_distribution`: Distribution across anomaly types
- `random_seed`: For reproducibility

## Storage Structure

```
data/
├── bronze/
│   └── transactions/
│       └── {partition_key}/
│           └── bronze_synthetic_*.parquet
├── silver/
│   ├── transactions/
│   │   └── silver_transactions_{partition_key}.parquet
│   └── customers/
│       └── silver_customers_{partition_key}.parquet
├── gold/
│   └── features/
│       └── transaction_date={date}/
│           └── *.parquet
└── quarantine/
    └── quarantine_*.parquet
```

## Migration Notes

### Previous Architecture Issues
- Layer logic was duplicated in `run_audit_and_synth.py`
- Individual scripts had limited standalone functionality
- No clear separation between layer concerns

### Current Architecture Benefits
- Clear separation of concerns
- No code duplication
- Individual scripts are fully functional
- Orchestrator simply composes layer functions
- Easier testing and maintenance
- Better modularity and reusability

## Testing

To test the refactored pipeline:

1. Test individual layers:
   ```bash
   python scripts/run_bronze.py
   python scripts/run_silver.py
   python scripts/run_gold.py
   ```

2. Test end-to-end pipeline:
   ```bash
   python scripts/run_audit_and_synth.py
   ```

3. Verify data storage in correct locations:
   - Bronze: `data/bronze/transactions/{date}/`
   - Silver: `data/silver/transactions/` and `data/silver/customers/`
   - Gold: `data/gold/features/transaction_date={date}/`

## Troubleshooting

### Common Issues

**Issue**: Silver data stored in wrong location
- **Solution**: Ensure `run_silver.py` uses the updated version with proper subdirectory structure

**Issue**: Gold layer missing anomaly injection
- **Solution**: Set `with_anomaly_injection=True` and provide `baseline_df` parameter

**Issue**: Import errors when running orchestrator
- **Solution**: Ensure scripts directory is in Python path (handled in `run_audit_and_synth.py`)

## Future Enhancements

Potential improvements to consider:
- Add validation layer between Silver and Gold
- Implement incremental processing for large datasets
- Add data quality metrics collection
- Create pipeline monitoring dashboard
- Implement automated rollback mechanisms
