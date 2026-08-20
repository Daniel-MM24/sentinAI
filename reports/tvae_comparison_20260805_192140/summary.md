# TVAE Hybrid vs Monte Carlo Comparison Report

**Partition**: 2026-08-05
**Generated**: 2026-08-05T19:21:40.585331

## Basic Statistics

### Monte Carlo

- **Rows**: 41
- **Columns**: 5
- **Unique Customers**: 41

## Distribution Comparison

## Correlation Comparison

## Tier Compliance

## Anomaly Comparison

## Recommendations

### Use TVAE When:

- When generating large-scale synthetic data with realistic distributions
- When scaling to very large dataset sizes (TVAE generation is faster after training)
- When capturing complex multi-modal behavioral patterns
- When adaptability to new data distributions is needed

### Use Monte Carlo When:

- When complete transparency and interpretability of generation process is required
- When computational resources for model training are limited
- When deterministic reproducibility is critical

### Key Findings:

- TVAE maintains similar distributions to Monte Carlo baseline (0 significant differences)
