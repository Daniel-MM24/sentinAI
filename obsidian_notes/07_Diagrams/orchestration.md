Single orchestrator

  scripts/run_audit_and_synth.py is the canonical entry point. It calls
  run_medallion_orchestrator() in the new library module with OpenLineage on every
  stage.

  Removed: scripts/run_medallion_pipeline.py, src/data/run_bronze.py, run_silver.py,
  run_gold.py, run_audit_and_synth.py

  Library vs CLI split

  ┌──────────────────┬─────────────────────────────────────────────┬──────────────────┐
  │ Layer            │ Library (src/data/)                         │ CLI (scripts/)   │
  ├──────────────────┼─────────────────────────────────────────────┼──────────────────┤
  │ Orchestrator     │ medallion_stages.run_medallion_orchestrator │ run_audit_and_sy │
  │                  │ ()                                          │ nth.py           │
  ├──────────────────┼─────────────────────────────────────────────┼──────────────────┤
  │ Bronze           │ run_bronze_stage() — AMLGenerator + anomaly │ run_bronze.py    │
  │                  │ injection                                   │                  │
  ├──────────────────┼─────────────────────────────────────────────┼──────────────────┤
  │ Silver           │ run_silver_stage() — BronzeToSilverPipeline │ run_silver.py    │
  │                  │ (POCAMLA AML)                               │                  │
  ├──────────────────┼─────────────────────────────────────────────┼──────────────────┤
  │ Gold             │ run_gold_stage() — GoldLayer feature store  │ run_gold.py      │
  ├──────────────────┼─────────────────────────────────────────────┼──────────────────┤
  │ Standalone synth │ synthetic_audit_pipeline.py                 │ (library only,   │
  │ audit            │                                             │ no CLI)          │
  └──────────────────┴─────────────────────────────────────────────┴──────────────────┘

  Silver path

  Silver now uses src/data/pipelines.py (BronzeToSilverPipeline) instead of SilverLayer
  entity resolution. SilverLayer remains for its own tests and future ER work.

  aml_silver_to_feature_store_inputs() overlays AML compliance fields onto full Bronze
  rows so Gold keeps all AML feature columns (tx_count_1h, etc.).

  Verified behavior

  • OpenLineage: START/COMPLETE/FAIL on orchestrator and each stage via lineage_trace
  • Synthetic data: AMLGenerator in bronze stage
  • Anomaly injection: FinancialAnomalyInjector at ~1.5% before bronze ingest
    (verified: 1.49% in bronze)
  • Medallion pipeline: fast-mode end-to-end run succeeded (5702 bronze → silver →
    gold)

  Commands

  # Full medallion pipeline (fast local run)
  poetry run python scripts/run_audit_and_synth.py --fast-mode --force-refresh
  # Individual stages
  poetry run python scripts/run_bronze.py --fast-mode
  poetry run python scripts/run_silver.py
  poetry run python scripts/run_gold.py
  # AML engine unit tests
  poetry run pytest tests/test_pipelines.py -v -m 'not slow'

  Fast mode is now 200 customers × 3 days (~15s) instead of 5000×30, which was
  impractical for local runs. Use --full-mode for production-scale generation.

  Finished Run full medallion pipeline in fast mode

  The initial fast-mode run was stopped because the old settings (5000 customers × 30
  days) were too heavy. That was tightened to 200 customers × 3 days, and the pipeline
  was re-run successfully.

  Result: Bronze → Silver → Gold completed in ~17s with OpenLineage events, ~1.5%
  anomaly injection in bronze, and gold output at data/gold/features/vv1.0/.

  Canonical command:

  poetry run python scripts/run_audit_and_synth.py --fast-mode --force-refresh
