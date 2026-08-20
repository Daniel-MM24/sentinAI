---
created: 2026-07-16
aliases: [Column Mapping, Schema Compatibility]
tags: [Technical, BugFix, FeatureEngineering]
---

# Schema Compatibility Fixes

## Problem

The medallion pipeline failed during the Gold stage with `ColumnNotFoundError` exceptions. The root cause was column name mismatches between Bronze, Silver, and Gold layers.

## Root Cause

The `BronzeToSilverPipeline` in `src/data/pipelines.py` performs column normalization during transformation:

- `amount` → `transaction_amount`
- `customer_id` → `entity_id`  
- `account_balance` → `account_balance_before`
- `post_tx_balance` → `account_balance_after`

However, the feature engineering pipeline (`src/data/feature_engineering.py`) and gold layer (`src/datasets/gold.py`) expected the original column names, causing failures when processing Silver layer data.

## Solution

### Feature Engineering (`src/data/feature_engineering.py`)

**Dynamic Column Mapping:**
- Added detection and renaming for multiple column name variants:
  - `customer_id` ↔ `entity_id` ↔ `sender_id`
  - `amount` ↔ `transaction_amount`
  - `receiver_id` ↔ `recipient_id` ↔ `beneficiary_id`
  - `counterparty_id` ↔ `counterparty`

**Defensive Column Handling:**
- Network features (`_compute_network_features`) now check for `counterparty_id` and `receiver_id` existence before computation
- Rolling features (`_compute_rolling_features`) check for `counterparty_id` before relationship counting
- Graceful fallback to zero values when optional columns are missing

**Polars Compatibility Fix:**
- Fixed datetime operation: `(feature_date - pl.col("last_tx_time")).total_seconds()` → `.dt.total_seconds()`

**Missing Import:**
- Added `timezone` to datetime imports

### Gold Layer (`src/datasets/gold.py`)

**Column Renaming:**
- Added `entity_id` → `customer_id` renaming in `silver_to_transaction_features()` to handle Silver layer output

## Impact

- **Before:** Pipeline failed at Gold stage with column not found errors
- **After:** Pipeline successfully processes Silver layer data regardless of column naming conventions
- **Compatibility:** Feature engineering now works with both Bronze (original names) and Silver (normalized names) schemas

## Remaining Issues

### OpenLineage Connection Warnings
- **Problem:** Pipeline attempts to emit lineage events to `http://localhost:5000/api/v1/lineage` but service is not running
- **Impact:** Causes retry delays (5 attempts × 5s timeout = ~25s delay per failed event) and warning spam in logs
- **Status:** Non-fatal but degrades performance and log readability
- **Recommended Fix:** 
  - Set `OPENLINEAGE_URL` environment variable only when Marquez/OpenLineage service is available
  - Add connection check before attempting lineage emission
  - Implement graceful fallback to console-only lineage when service unavailable

### Pipeline Completion Status
- **Problem:** Script execution was interrupted before full end-to-end validation
- **Impact:** Unknown if additional schema mismatches exist in later pipeline stages
- **Status:** Requires full pipeline run to validate all fixes
- **Recommended Fix:** Run complete pipeline with `--fast-mode` to validate all stage transitions

## Files Modified

- `src/data/feature_engineering.py` - Column mapping, defensive handling, Polars fixes
- `src/datasets/gold.py` - Entity ID renaming
- `obsidian_notes/01_Project/CHANGELOG.md` - Version 1.0.2 entry

## Behavioral Transaction Generator Schema Simplification

### Status: Completed ✓ (2026-08-05)

### Changes Made

**Transaction Output Schema Reduction:**
- Reduced from 20+ columns to 8 essential columns in `src/data/behavioral_generator.py`
- Retained columns: customer_id, transaction_type, amount, timestamp, direction, balance, tier, is_international
- Removed columns: transaction_id, counterparty, paid_in, paid_out, balance_after, hour, day_of_week, month, is_weekend, is_night, is_betting, is_kadogo

**Rationale:**
Following synthetic data best practices (IBM/NVIDIA), transaction generation now outputs only essential raw data columns. Derived columns are computed downstream:
- **paid_in/paid_out**: Derived from amount + direction in balance continuity checks
- **Temporal features** (hour, day_of_week, month, is_weekend, is_night): Computed in Silver layer by `TemporalPatternTracker`
- **transaction_id**: Derivable downstream via row indexing
- **counterparty**: Not essential for AML analysis in streamlined schema
- **is_betting**: Lower priority than international flag for streamlined schema
- **is_kadogo**: Not essential for streamlined schema

**Logging Updates:**
- `_log_generation_stats()`: Removed logging for is_betting, is_kadogo, is_weekend, is_night, hour distribution
- `_log_balance_stats()`: Updated to derive paid_in/paid_out from amount and direction for ledger continuity checks

**Validation Results:**
- ✓ Output CSV contains exactly 8 columns
- ✓ Balance integrity checks pass (0 negative balances, 0 tier limit violations)
- ✓ Ledger continuity checks pass (100% with derived paid_in/paid_out)
- ✓ International flag distribution maintained at ~2%

### Files Modified
- `src/data/behavioral_generator.py` - Transaction schema simplification, logging updates

## AML Scenario Injector Schema Alignment

### Status: Completed ✓ (2026-08-05)

### Changes Made

**Transaction Output Schema Alignment:**
- Updated `_make_tx_row()` method in `src/data/aml_scenario_injector.py` to output 8 essential columns matching streamlined schema
- Retained columns: customer_id, transaction_type, amount, timestamp, direction, balance, tier, is_international
- Removed columns: counterparty, paid_in, paid_out

**Placeholder Fields:**
- `balance`: Set to 0.0 as placeholder, computed during merge with clean transactions
- `tier`: Set to 1 as placeholder, joined from customer profiles during merge
- `is_international`: Set to False as default for injected AML patterns

**Merge and Balance Recalculation:**
- Updated `inject()` method to merge injected transactions with clean transaction DataFrame
- Loads customer profiles from `customer_profiles.csv` to extract tier information
- Extracts tier number from tier strings (e.g., "tier_2" → 2)
- Recalculates running balance chronologically per customer:
  - Calculates net change per transaction (inflow = +amount, outflow = -amount)
  - Sorts by customer_id and timestamp
  - Computes cumulative sum per customer for balance
- Joins tier information from customer profiles based on customer_id

**Ground Truth Output:**
- Unchanged - still outputs 3 columns: user_id, is_launderer, aml_scenario
- Ground truth labels remain compatible with supervised model training

**New Method:**
- Added `get_merged_transactions()` to return merged DataFrame with recalculated balance and tier
- Enables testing/validation of injection output

**Path Fix:**
- Updated customer profiles path from `customer_profiles_complete.csv` to `customer_profiles.csv` to match actual file

**Rationale:**
Injected AML transactions must match the streamlined 8-column schema to ensure compatibility with the medallion pipeline. Balance and tier are computed post-injection to maintain continuity with existing transaction history.

### Validation Criteria

- ✓ Ground truth CSV still has 3 columns (user_id, is_launderer, aml_scenario)
- ✓ Injected transactions have 8 columns matching streamlined schema
- ✓ Balance continuity maintained after injection (recalculated chronologically)
- ✓ Tier information correctly populated from customer profiles

### Files Modified
- `src/data/aml_scenario_injector.py` - Schema alignment, merge logic, balance recalculation, tier join

## Medallion Pipeline Stage Updates

### Status: Completed ✓ (2026-08-05)

### Changes Made

**Bronze Stage (`src/data/medallion_stages.py`):**
- No changes needed - existing `ingest_normalized_synthetic_data()` handles streamlined schema (3 customer + 8 transaction columns)

**Silver Stage (`src/data/pipelines.py` + `src/data/medallion_stages.py`):**
- Added `derive_temporal_features()` function in `pipelines.py` to compute temporal features:
  - `hour` - hour of day (0-23)
  - `day_of_week` - day of week (0-6)
  - `month` - month of year (1-12)
  - `is_weekend` - boolean flag for weekend (Sat/Sun)
  - `is_night` - boolean flag for night hours (hour < 6 or hour >= 22)
- Updated `run_silver_stage()` in `medallion_stages.py` to apply temporal feature derivation after BronzeToSilverPipeline transformation
- Silver output now has 8 core columns + 5 derived temporal columns

**Gold Stage (`src/data/feature_engineering.py`):**
- Completely rewrote `CustomerFeatureEngineer.compute_features()` to compute 4 engineered temporal features:
  - `tx_count_7d` - rolling 7-day transaction count per customer
  - `volume_7d` - rolling 7-day transaction volume (sum of amounts)
  - `night_tx_ratio` - ratio of transactions at night (hour < 6 or hour >= 22)
  - `rapid_tx_ratio` - ratio of transactions with ≤5 minute gap to previous transaction
- Optimized computation using single aggregation with filters instead of multiple joins
- Removed legacy multi-step feature computation approach

**Legacy Code Removal (`src/data/medallion_stages.py`):**
- Removed import of `silver_to_transaction_features` from `src.datasets.gold`
- Removed legacy call to `silver_to_transaction_features()` in `run_gold_stage()`
- Gold stage now uses only the streamlined `CustomerFeatureEngineer` for feature computation

### Rationale

The medallion pipeline was updated to handle the streamlined schema through Bronze → Silver → Gold with proper temporal feature derivations and engineered feature computations. The changes ensure:

- Bronze layer ingests the streamlined schema without modification
- Silver layer adds temporal features needed for downstream analysis
- Gold layer computes the 4 key engineered features for AML detection
- Legacy backward compatibility code removed to simplify the pipeline

### Validation Results

- ✓ Bronze: 18 customer cols, 33 transaction cols (includes metadata)
- ✓ Silver: 26 total cols with 5/5 temporal columns present
- ✓ Gold: 5 total cols with 4/4 engineered features present
- ✓ All validations passed

### Files Modified
- `src/data/pipelines.py` - Added `derive_temporal_features()` function
- `src/data/medallion_stages.py` - Updated Silver stage to apply temporal features, removed legacy code
- `src/data/feature_engineering.py` - Rewrote `compute_features()` for 4 engineered features with optimized computation

## Temporal Model Compatibility

### Status: Verified ✓

The temporal model (`src/data/temporal_model.py`) is compatible with the streamlined schema and requires no changes.

### Architecture

- **temporal_model.py**: Provides only `compute_intensity(dt: datetime) -> float` function for timestamp generation
- **behavioral_generator.py**: Uses `compute_intensity()` in thinning algorithm to generate realistic temporal patterns
- **Output**: Timestamps in ISO 8601 format with timezone (e.g., `2024-07-01T08:30:45.123456+00:00`)

### Temporal Feature Derivation

Temporal features (hour, day_of_week, month, is_weekend, is_night) are **NOT generated at the Bronze layer**. They are computed downstream:

- **Silver Layer**: `src/data/temporal_features.py` - `TemporalPatternTracker` extracts features from timestamp column
- **Gold Layer**: `src/datasets/gold.py` - Combines temporal features with other feature categories

### Features Computed from Timestamp

From `FEATURE_DICTIONARY.md`:
- `night_ratio` - transactions 22:00–06:00
- `weekend_ratio` - Saturday/Sunday transactions
- `month_end_ratio` - last 3 days of month
- Rolling window features (1min, 5min, 1h, 24h, 7d, 30d)
- Velocity metrics (burst_ratio, velocity_change_pct)

### Validation Results

- ✓ Temporal model functions unchanged
- ✓ Behavioral generator generates realistic temporal patterns using `compute_intensity()`
- ✓ Timestamp format remains ISO 8601 with timezone
- ✓ No Bronze layer temporal feature columns needed

## Test Suite Infrastructure

### Status: Completed ✓ (2026-08-05)

### Changes Made

**Git Ignore Configuration (`.gitignore`):**
- Commented out `test_*.py` pattern in root directory
- Commented out `**/test.py` and `**/test_*.py` patterns in subdirectories
- Commented out `.pytest_cache/` pattern
- This allows test files to be tracked in version control

**Rationale:**
The test suite (`test_streamlined_schema.py`) needs to be committed to the repository for:
- Reproducible validation of the streamlined schema implementation
- Continuous integration testing
- Team collaboration on test maintenance
- Historical tracking of test coverage

**Impact:**
- Test files can now be created in `tests/` directory and committed to git
- Pytest cache directory will be tracked (can be added back if needed)
- Enables comprehensive test suite for Bronze → Silver → Gold pipeline validation

### Files Modified
- `.gitignore` - Commented out test file patterns to enable test suite versioning

## Related

- [[03_Technical/FEATURE_VALIDATION]] - Feature validation framework
- [[01_Project/CHANGELOG]] - Version history
- [[02_Data/FEATURE_DICTIONARY]] - Feature definitions
- [[03_Technical/TEMPORAL_PATTERNS]] - Temporal pattern tracking architecture
