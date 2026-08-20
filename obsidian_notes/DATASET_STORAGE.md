# Dataset Storage Strategy

## Storage Conventions

### Monte Carlo Baseline
- **Path**: `data/bronze/monte_carlo_baseline_{partition}.parquet`
- **Description**: Clean training data for TVAE
- **Retention**: Retained for model retraining
- **Partitioning**: Partitioned by date

### TVAE Models
- **Path**: `models/tvae_model_{timestamp}.pkl`
- **Description**: Trained TVAE models
- **Retention**: Versioned by timestamp
- **Metadata**: Include metadata JSON with training parameters

### TVAE Raw Events
- **Path**: `data/tvae_raw_events_{partition}.parquet`
- **Description**: Intermediate output from TVAE sampling
- **Retention**: Can be deleted after balance reconstruction

### TVAE Balance Corrected
- **Path**: `data/tvae_balance_corrected_{partition}.parquet`
- **Description**: Intermediate output after balance reconstruction
- **Retention**: Can be deleted after feature engineering

### TVAE Enriched
- **Path**: `data/tvae_enriched_{partition}.parquet`
- **Description**: Intermediate output after feature engineering
- **Retention**: Can be deleted after anomaly injection

### TVAE Hybrid Gold
- **Path**: `data/gold/tvae_hybrid_gold_{partition}.parquet`
- **Description**: Final output with 21 features + labels
- **Retention**: Retained for model training/evaluation
- **Partitioning**: Partitioned by date

## Cleanup Strategy

Add script to clean intermediate files while preserving baseline and gold datasets.

### Files to Preserve
- Monte Carlo Baseline: `data/bronze/monte_carlo_baseline_{partition}.parquet`
- TVAE Hybrid Gold: `data/gold/tvae_hybrid_gold_{partition}.parquet`
- TVAE Models: `models/tvae_model_{timestamp}.pkl` (and associated metadata JSON)

### Files to Clean
- TVAE Raw Events: `data/tvae_raw_events_{partition}.parquet`
- TVAE Balance Corrected: `data/tvae_balance_corrected_{partition}.parquet`
- TVAE Enriched: `data/tvae_enriched_{partition}.parquet`
