# Changelog

## [1.0.6] — 2026-08-05 — Feature Engineering Simplification

### Changed
- **CustomerFeatureEngineer.compute_features()** simplified from 24+ temporal features to 4 essential engineered features:
  - Retained: `tx_count_7d`, `volume_7d`, `night_tx_ratio`, `rapid_tx_ratio`
  - Removed: All other velocity, balance, amount, network, temporal, device, and rolling features
- **Implementation simplified** from multi-step computation to single `group_by` aggregation:
  - Replaced separate feature computation steps with unified aggregation
  - All 4 features now computed in one pass over the data
- **Polars API fix**: Changed `between()` to `is_between()` for hour-based filtering compatibility
- **Output schema** now contains exactly 5 columns: `customer_id`, `tx_count_7d`, `volume_7d`, `night_tx_ratio`, `rapid_tx_ratio`

### Rationale
Following synthetic data best practices, feature engineering now focuses on essential temporal features for the Gold layer. Derived features are computed downstream rather than stored at the Silver layer. This reduces computational overhead and aligns with the principle of computing features on-demand from raw transaction history rather than pre-computing 20+ features that may not all be used.

### Validation
- Output verified to contain exactly 5 columns
- All values are non-null (nulls filled with 0 for customers with no transactions in window)
- Ratios are between 0 and 1
- Rolling window calculations use correct 7-day window

## [1.0.5] — 2026-08-05 — Test File Tracking Enabled

### Changed
- **.gitignore updated** to enable tracking of test files:
  - Uncommented `test_*.py` pattern to track test files in root directory
  - Uncommented `**/test.py` pattern to track test files in subdirectories
  - Uncommented `**/test_*.py` pattern to track test files with test_ prefix in subdirectories
  - Uncommented `.pytest_cache/` pattern to track pytest cache directory

### Rationale
Test files are now tracked in version control to ensure reproducibility of test suites and enable collaboration on test development. This aligns with best practices for maintaining test coverage and test-driven development workflows.

## [1.0.4] — 2026-08-05 — Behavioral Transaction Generator Schema Simplification

### Changed
- **Transaction output reduced from 20+ columns to 8 essential columns** in `src/data/behavioral_generator.py`:
  - Retained: customer_id, transaction_type, amount, timestamp, direction, balance, tier, is_international
  - Removed: transaction_id, counterparty, paid_in, paid_out, balance_after, hour, day_of_week, month, is_weekend, is_night, is_betting, is_kadogo
- **Removed balance_after alias line** from generate_transactions() method
- **Updated _log_generation_stats()** to remove logging for deleted columns:
  - Removed is_betting, is_kadogo, is_weekend, is_night, hour distribution logging
  - Kept is_international logging for high-risk flag tracking
- **Updated _log_balance_stats()** to derive paid_in/paid_out from amount and direction for balance continuity checks

### Rationale
Following synthetic data best practices, transaction generation now outputs only essential columns. Derived columns (paid_in/paid_out, temporal features) are computed downstream in feature engineering rather than stored at the Bronze layer. This reduces storage overhead and aligns with the principle of storing raw data only.

### Validation
- Output CSV verified to contain exactly 8 columns
- Balance integrity checks pass (0 negative balances, 0 tier limit violations)
- Ledger continuity checks pass (100% continuity with derived paid_in/paid_out)
- International flag distribution maintained at ~2%

## [1.0.3] — 2026-08-05 — Stratified Profiles Schema Simplification

### Changed
- **CustomerProfile dataclass** reduced from 7 fields to 3 essential fields:
  - Retained: `customer_id`, `kyc_tier`, `archetype`
  - Removed: `max_transaction_limit`, `max_balance_limit`, `account_age_days`, `initial_balance`
- **profiles_to_dataframe()** method now outputs only 3 columns:
  - `customer_id` (String)
  - `tier` (String) - renamed from `kyc_tier`
  - `archetype` (String)
- **Schema casting** updated to match new 3-column structure
- **Output file path** updated from `customer_profiles_complete.csv` to `customer_profiles.csv`

### Removed
- `_generate_account_age()` method - no longer needed
- `_generate_initial_balance()` method - no longer needed
- Transaction and balance limit generation from `_assign_kyc_tier()` - now returns only KYC tiers

### Validation
- KYC tier distribution maintained: Tier 1 (60%), Tier 2 (20%), Tier 3 (15%), Tier 4 (5%)
- Archetype distribution maintained: Retail Heavy (15%), Retail Standard (70%), Micro-Merchant (12%), Corporate (3%)
- Output CSV verified to contain exactly 3 columns with correct String data types

### Rationale
Following synthetic data best practices (IBM/NVIDIA), customer profile generation now focuses on essential identifiers and behavioral archetypes only. Transaction limits, balance limits, and account age are better handled in downstream transaction generation rather than static customer profiles.

## [1.0.2] — 2026-07-16 — Feature Engineering Schema Compatibility Fixes

### Fixed
- **Column name compatibility** in `src/data/feature_engineering.py`:
  - Added dynamic column mapping for `amount` ↔ `transaction_amount` 
  - Added dynamic column mapping for `customer_id` ↔ `entity_id`
  - Added dynamic column mapping for `receiver_id` ↔ `recipient_id` ↔ `beneficiary_id`
  - Added dynamic column mapping for `counterparty_id` ↔ `counterparty`
- **Polars datetime operation** in `src/data/feature_engineering.py`:
  - Fixed `.total_seconds()` → `.dt.total_seconds()` for Polars expression compatibility
- **Missing import** in `src/data/feature_engineering.py`:
  - Added `timezone` to datetime imports
- **Column name compatibility** in `src/datasets/gold.py`:
  - Added `entity_id` → `customer_id` renaming for silver layer compatibility
- **Defensive column handling** in `src/data/feature_engineering.py`:
  - Network features now check for `counterparty_id` and `receiver_id` existence before computation
  - Rolling features check for `counterparty_id` existence before relationship counting
  - Graceful fallback to zero values when optional columns are missing

### Root Cause
Silver layer pipeline (`BronzeToSilverPipeline`) renames columns during normalization (e.g., `amount` → `transaction_amount`, `customer_id` → `entity_id`), but feature engineering and gold layer code expected original column names. This caused `ColumnNotFoundError` during medallion pipeline execution.

### Known Issues
- **OpenLineage connection warnings:** Pipeline attempts lineage emission to unavailable service, causing ~25s delays per failed event
- **Incomplete validation:** Script execution interrupted before full end-to-end pipeline validation

## [1.0.0] — 2026-07-10 — PoC Complete

### Added
- BehavioralGenerator: 1,000 synthetic customer profiles across 4 archetypes with KYC tier assignment
- Constrained random walk with balance constraint enforcement:
  - 100-attempt rejection sampling, look-ahead validation with geometric amount scaling
  - Zero negative balances, zero tier limit violations, 100% ledger continuity
  - Per-customer balance tracking: opening/max/min/avg, daily velocity caps
- Temporal pattern model with inhomogeneous Poisson process (168-hour weekly intensity vector)
- M-PESA transaction type distributions (7 types, calibrated probabilities per archetype)
- High-risk entity flagging: betting platforms (~3%), international transfers (~2%)
- Kadogo micro-transaction threshold enforcement (KES 100 P2P, KES 200 merchant)
- AML Scenario Injector: 4 typologies at 2% prevalence (20 of 1,000 customers)
  - Smurfing (40%), Layering (30%), Mule Account (20%), Circular Trading (10%)
  - Output: `data/aml_ground_truth.csv` with user_id, is_launderer, aml_scenario
- Temporal pattern tracking: 24 feature columns per customer via stateful RollingBuffer
  - Daily/weekly/monthly pattern detection, rolling aggregators (1h/24h/7d/30d)
  - Output: `data/temporal_features.csv`
- Config-driven architecture via behavioral_generator_config dataclass

### Documentation
- PROJECT_OVERVIEW.md: Executive summary for stakeholders
- ARCHITECTURE.md: System architecture and design decisions
- DATA_DICTIONARY.md: Complete field descriptions for all datasets
- FEATURE_DICTIONARY.md: Engineered features with formulas
- DEPLOYMENT_GUIDE.md: How to deploy and run the system
- Phase documentation: PHASE1 through PHASE5 in obsidian_notes/phase_docs/
- Technical documentation: MONTE_CARLO_ENGINE, CONSTRAINT_ENFORCEMENT, TEMPORAL_PATTERNS, AML_SCENARIOS, FEATURE_VALIDATION
- Reports: VALIDATION_REPORT, ODPC_COMPLIANCE

### Documentation
- PROJECT_OVERVIEW.md: Executive summary for stakeholders
- ARCHITECTURE.md: System architecture and design decisions
- DATA_DICTIONARY.md: Complete field descriptions for all datasets
- FEATURE_DICTIONARY.md: 57 engineered features with formulas
- DEPLOYMENT_GUIDE.md: How to deploy and run the system
- Phase documentation: PHASE1 through PHASE5 in obsidian_notes/phase_docs/
- Technical documentation: MONTE_CARLO_ENGINE, CONSTRAINT_ENFORCEMENT, TEMPORAL_PATTERNS, AML_SCENARIOS, FEATURE_VALIDATION
- Reports: MODEL_PERFORMANCE_REPORT, VALIDATION_REPORT, ODPC_COMPLIANCE

## [0.4.0] — 2026-07-08 — Gold Layer & Feature Engineering

### Added
- Gold layer feature store with manifest.json versioning
- Partitioned dataset output by anomaly_case_id
- SHAP value computation and metadata embedding
- Feature registry for cross-referencing feature definitions
- Temporal feature generation: 1-min/5-min/1-hr/24-hr/7-day/30-day velocity windows
- Balance pattern features: volatility, retention ratio, zero-balance frequency
- Burst ratio and velocity change percentage features

### Changed
- Migrated from Pandas to Polars for all Silver→Gold transformations
- Optimized PyArrow dataset partitioning for columnar access patterns

## [0.3.0] — 2026-07-07 — Bronze & Silver Layers

### Added
- Bronze layer immutable Parquet storage with SHA-256 provenance hashing
- Silver layer with Pandera schema enforcement (SilverRecordSchema)
- OpenLineage instrumentation for MRM compliance
- Transaction velocity window computation (1-min through 30-day)
- Regulatory threshold validation against CBK limits
- Dead letter queue for schema violations
- Great Expectations data quality validation suite
- Bronze metadata JSON files with run_id tracking

### Changed
- Restructured data pipeline from flat CSV to layered Medallion architecture
- Enhanced lineage decorator with run_id, timestamps, and row counts

## [0.2.0] — 2026-07-05 — Anomaly Injection

### Added
- FinancialAnomalyInjector with 8 anomaly types (1.5% ratio)
- Amount spike injection (5-20x multiplier)
- Velocity surge injection (90%+ inter-arrival compression)
- Balance depletion injection (95%+ drawdown)
- Price manipulation patterns
- Liquidity anomaly patterns
- Spread abnormality patterns
- Counterparty risk patterns
- Temporal pattern anomalies (off-cycle timing)
- Ground-truth label columns: anomaly_flag, anomaly_type

## [0.1.0] — 2026-07-01 — Initial Synthetic Data Generation

### Added
- AMLGenerator: 1M transaction generation engine
- Monte Carlo sampler with LogNormal, Poisson, Exponential distributions
- Constrained random walk for balance integrity
- KYC tier enforcement (Tier 1: 50K, Tier 2: 500K, Tier 3: 5M)
- Customer archetype profiling: Retail Heavy, Retail Standard, Micro-Merchant, Corporate
- M-PESA transaction type distribution (7 types with calibrated probabilities)
- Temporal pattern model with inhomogeneous Poisson process
- 168-hour weekly intensity vector for hourly patterns
- Monthly seasonal effects (mid-month salary, end-of-month cycles)
- High-risk entity flagging (betting, international transfers)
- Kadogo micro-transaction threshold enforcement
- Statement exporter: per-customer CSV statement generation
- Config-driven architecture via simulation_profiles.yaml

## [1.1.0] — 2026-07-16 — Data Model Redesign: Normalized Schema with Customer-Centric Primary Keys

### Changed
- **Implemented database normalization** with proper primary/foreign key relationships
- **New CustomerSchema** (customer_id as primary key) for static customer attributes
- **New TransactionSchema** (transaction_id as primary key, customer_id as foreign key) for raw transaction events
- **New CustomerFeaturesSchema** (customer_id + feature_date composite key) for time-varying customer features
- **Separation of concerns**: Raw transactions in silver, computed features in gold layer
- **Removed 30+ aggregate/rolling columns** from transaction generation (moved to feature engineering)

### Added
- **generate_normalized()** method in AMLGenerator to output separate customers_df and transactions_df
- **CustomerFeatureEngineer** class in src/data/feature_engineering.py for computing customer features from raw transactions
- **ingest_normalized_synthetic_data()** method in BronzeLayer to handle separate customer/transaction ingestion
- **read_normalized_bronze_partition()** method in BronzeLayer to read normalized bronze data
- Feature computation methods: velocity, balance, amount, network, temporal, device, and rolling features

### Updated
- **run_bronze_stage** to use generate_normalized() instead of generate()
- **run_silver_stage** to read normalized bronze data (separate customers and transactions)
- **run_gold_stage** to compute customer features using CustomerFeatureEngineer
- Bronze layer directory structure: separate customers/ and transactions/ subdirectories

### Architectural Benefits
- Customer-centric tracking: customer_id is now the proper primary key for customer data
- Foreign key relationships: transactions.customer_id → customers.customer_id
- Database normalization: Follows primary/foreign key conventions
- Scalability: Features computed on-demand from transaction history
- Proper separation: Raw transactions → feature engineering → customer_features table

## [1.0.1] — 2026-07-14 — Full-Mode Pipeline Fix & Schema Alignment

### Changed
- **`degree_centrality` normalized to 0–1 float** (was unnormalized integer count). The anomaly injector writes `0.35` as a ratio (fraction of total possible connections), aligning with NetworkX's `nx.degree_centrality()` semantics. Threshold updated from `> 20` to `> 0.15`.
- Full-mode pipeline now generates **61,703 transactions** across **365 days** with **6,285 customers** (~7m42s runtime)
- Fast-mode defaults: 75 customers, 3 days (dev/test)

### Fixed
- **Pandera schema type mismatches** — Anomaly injector uses np.float64 arrays for injected values, widening int columns to Float64. Changed affected schema fields in `src/datasets/schemas.py`:
  - `degree_centrality`: `int` → `float` (ratio 0-1 scale, injector writes `0.35`)
  - `community_id`: `int` → `float` (injector uses `np.full(..., dtype=np.float64)`)
  - Applied across `SilverRecordSchema`, `SilverCompactSchema`, and `GoldFeatureSchema`
- **GoldFeatureSchema restored** in `schemas.py` after accidental deletion during rewrite
- **schema section comment reorg** in `schemas.py`: `amount_roundness` moved from Tier 1 → Tier 2, `avg_balance_30d` / `balance_volatility_30d` reordered, `zero_balance_frequency` `float` → `int`

## [v1.1.1] — 2026-07-16 — Anomaly Injection Fix, Gold Schema, OpenLineage Fallback

### Fixed
- **Anomaly injection crash** in `src/data/synthetic_generator.py`:
  - Moved injection inside `generate_normalized(anomaly_ratio, anomaly_seed)` so the injector operates on the combined DataFrame where all 26 `INJECTABLE_FEATURES` exist
  - 6 of 8 injection methods were crashing with `ColumnNotFoundError` (velocity_funnel, mule_activity, layering, circular_trading, temporal_anomaly, ceiling_violation)
- **Gold layer column mapping** in `src/datasets/gold.py`:
  - Extended rename to cover Silver's `transaction_amount`, `account_balance_before`, `account_balance_after` conventions
- **OpenLineage connection retry storm** in `src/data/lineage_decorator.py`:
  - Added `_probe_openlineage_backend()` — single lightweight health-check (HEAD, 2s timeout), graceful fallback to console transport when no server is running
  - Eliminated ~25s blocking retry delays per lineage event

### Changed
- `src/data/medallion_stages.py` — `run_bronze_stage()` simplified: removed `FinancialAnomalyInjector` instantiation, passes params to `generate_normalized()`
- All three pipeline stages (bronze/silver/gold) now log zero `urllib3` retry noise

### Documentation
- Added `obsidian_notes/03_Technical/INJECTION_LINEAGE_FIXES.md`

### [1.1.0] — Planned — Production Hardening
- Real-time FastAPI scoring endpoint
- Grafana monitoring dashboard
- Batch scoring pipeline optimization
- Stratoflow integration for stream processing
- Additional AML typologies (TBML, crypto-related)

### [2.0.0] — Planned — Enterprise
- Multi-tenant support (Tigo, Airtel Money)
- Federated learning across institutions
- FIU regulatory filing API
- Real-time streaming graph analysis (GraphX/neo4j)
- 10M+ transactions per day throughput

---

## Versioning Convention

- **Major**: Enterprise-ready releases with multi-tenant support
- **Minor**: Production feature releases (real-time scoring, new models, new scenarios)
- **Patch**: Bug fixes, performance optimization, documentation updates
