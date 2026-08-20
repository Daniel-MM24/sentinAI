# TVAE Hybrid Pipeline Implementation Work Plan

## Current State Analysis

### Existing Pipeline Architecture
- **Entry Point**: `scripts/run_audit_and_synth.py` → `run_medallion_orchestrator()`
- **Bronze Stage**: Uses `AMLGenerator` (Monte Carlo simulation) in `src/data/synthetic_generator.py`
- **Silver Stage**: Transforms bronze to silver with regulatory validation
- **Gold Stage**: Materializes feature store using `CustomerFeatureEngineer`
- **TVAE Hybrid**: Partial implementation in `run_tvae_hybrid_stage()` (stub)

### Key Findings
1. **Monte Carlo Bottleneck**: `BehavioralTransactionGenerator` has complex look-ahead validation making it slow
2. **Existing Bronze Data**: Bronze data already exists at `data/bronze/transactions/2026-08-05/`
3. **TVAE Components**: TVAE generator, balance reconstructor, feature engineer, and anomaly injector exist
4. **Integration Gap**: `run_tvae_hybrid_stage()` is a simplified stub that doesn't use the full TVAE pipeline

## Implementation Work Plan

### Phase 1: Baseline Preparation (Skip Slow Monte Carlo)
**Objective**: Use existing bronze data as TVAE training baseline

**Tasks**:
1. ✅ Created `scripts/prepare_baseline_from_bronze.py` to convert existing bronze data to TVAE baseline format
2. Extract 8 core TVAE columns from existing bronze transactions:
   - customer_id, tier, archetype, transaction_type, amount, timestamp, direction, is_international
3. Remove anomaly flags and labels (clean baseline for TVAE training)
4. Output to `data/bronze/monte_carlo_baseline_{partition}.parquet`

**Status**: Script created, pending execution

### Phase 2: TVAE Training Integration
**Objective**: Integrate TVAE training into medallion stages

**Tasks**:
1. Review `scripts/train_tvae_on_baseline.py` implementation
2. Integrate TVAE training logic into `run_tvae_hybrid_stage()` in `medallion_stages.py`
3. Add checkpointing to skip training if model exists
4. Add TVAE model path to result object

**Dependencies**: Phase 1 completion

### Phase 3: TVAE Sampling Integration
**Objective**: Integrate TVAE sampling into medallion stages

**Tasks**:
1. Review `scripts/sample_tvae_events.py` implementation
2. Integrate sampling logic into `run_tvae_hybrid_stage()`
3. Handle timestamp reconstruction and amount clipping
4. Output raw events to intermediate location

**Dependencies**: Phase 2 completion

### Phase 4: Deterministic Post-Processing
**Objective**: Apply balance reconstruction and feature engineering

**Tasks**:
1. Integrate `BalanceReconstructor` from `src/data/hybrid_reconstructor.py`
2. Integrate `CustomerFeatureEngineer` from `src/data/feature_engineering.py`
3. Ensure tier cap enforcement and balance continuity
4. Compute 10 downstream features for 21-feature schema

**Dependencies**: Phase 3 completion

### Phase 5: Anomaly Injection
**Objective**: Inject AML scenarios with deterministic labels

**Tasks**:
1. Integrate `FinancialAnomalyInjector` with `TVAEInjectorConfig`
2. Apply 1.5% launderer ratio (configurable)
3. Ensure precise control over `is_launderer` and `aml_scenario` labels
4. Output final gold dataset with 21 columns

**Dependencies**: Phase 4 completion

### Phase 6: CLI Integration
**Objective**: Add TVAE hybrid flag to existing CLI

**Tasks**:
1. Add `--use-tvae-hybrid` flag to `scripts/run_audit_and_synth.py`
2. Add `--tvae-samples` flag for sample count
3. Add `--tvae-anomaly-ratio` flag for anomaly ratio
4. Update `run_medallion_orchestrator()` to pass flags to `run_tvae_hybrid_stage()`

**Dependencies**: Phase 5 completion

### Phase 7: Validation & Testing
**Objective**: Validate TVAE hybrid output matches schema

**Tasks**:
1. Validate 21-feature gold schema output
2. Compare distributions between Monte Carlo and TVAE hybrid
3. Ensure tier cap compliance
4. Verify anomaly injection accuracy
5. Generate comparison report

**Dependencies**: Phase 6 completion

## Implementation Priority

### High Priority (Core Functionality)
1. Phase 1: Baseline preparation (enables fast iteration)
2. Phase 4: Deterministic post-processing (ensures correctness)
3. Phase 5: Anomaly injection (provides labeled data)

### Medium Priority (Integration)
4. Phase 2: TVAE training integration
5. Phase 3: TVAE sampling integration
6. Phase 6: CLI integration

### Low Priority (Validation)
7. Phase 7: Validation & testing

## Technical Considerations

### Performance Optimizations
- **Baseline Reuse**: Skip Monte Carlo generation by using existing bronze data
- **Model Checkpointing**: Cache trained TVAE models to avoid retraining
- **Incremental Sampling**: Sample in batches if needed for memory constraints

### Data Quality Guarantees
- **Balance Continuity**: Deterministic reconstruction ensures mathematical correctness
- **Tier Compliance**: Hard enforcement of CBK tier caps
- **Schema Consistency**: Validation against 21-feature gold schema

### Regulatory Compliance
- **PII Hashing**: SHA-256 tokenization for customer IDs
- **Audit Trails**: Export customer metadata for MRM compliance
- **Reproducibility**: Seed-based generation for consistent results

## Success Criteria

1. ✅ TVAE hybrid pipeline runs in full mode without Monte Carlo bottleneck
2. ✅ Output matches 21-feature gold schema exactly
3. ✅ Tier caps are enforced (100% compliance)
4. ✅ Balance continuity is maintained (zero violations)
5. ✅ Anomaly injection produces accurate labels (1.5% launderer ratio)
6. ✅ CLI integration allows seamless switching between Monte Carlo and TVAE hybrid
7. ✅ Performance improvement: TVAE hybrid runs faster than pure Monte Carlo

## Next Steps

1. Execute Phase 1: Run baseline preparation script
2. Execute Phase 2-5: Integrate TVAE components into medallion stages
3. Execute Phase 6: Add CLI flags
4. Execute Phase 7: Validate and test
5. Generate comparison report between Monte Carlo and TVAE hybrid approaches
