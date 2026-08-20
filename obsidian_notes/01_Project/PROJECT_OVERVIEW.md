# SentinAI AML PoC: Executive Summary

## Problem Statement

Safaricom/M-PESA operates Kenya's dominant mobile money platform, processing billions of KES daily across 30+ million active customers. As a regulated financial institution, M-PESA must comply with:

- **Central Bank of Kenya (CBK) AML Guidelines** — requiring transaction monitoring, suspicious activity reporting (SARs), and customer due diligence (CDD)
- **Kenya Data Protection Act (ODPC)** — mandating strict privacy controls over personal financial data
- **Model Risk Management (MRM) Standards** — demanding transparent, auditable, and validated detection models

The core challenge: **developing and validating an AML detection system without access to real customer transaction data**, due to data protection regulations and commercial sensitivity. This necessitates a synthetic data generation approach that preserves the statistical properties of real M-PESA transaction flows while guaranteeing zero privacy risk.

## Solution Approach

SentinAI addresses this through a four-layer approach:

### 1. Privacy-Preserving Synthetic Data Generation
- **Monte Carlo simulation** using LogNormal, Poisson, and Exponential distributions calibrated to public M-PESA behavioral research
- **Constrained random walk** with look-ahead validation ensuring balance integrity within KYC tier limits (KES 50K/500K/5M), zero negative balances, and 100% ledger continuity
- **Inhomogeneous Poisson process** for realistic temporal patterns (hourly, weekly, seasonal cycles)
- **1,000 synthetic customer profiles** across 4 archetypes: Retail Heavy, Retail Standard, Micro-Merchant, Corporate
- **10,000 behavioral transactions** with KYC tier calibration (60% Tier 1, 30% Tier 2, 10% Tier 3)

### 2. Realistic AML Scenario Injection
- Four AML typologies injected at ~2% prevalence (20 of 1,000 customers):
  - **Smurfing (40%)** — Structured small transactions below KES 100,000 CBK reporting threshold, split across 5-15 counterparties
  - **Layering (30%)** — Rapid fund movement through 4+ account chains within 24-hour windows, 2-5 cycles per launderer
  - **Mule Accounts (20%)** — 80%+ of received funds withdrawn via Agent Withdrawal within 24 hours, 5+ receive-withdraw cycles
  - **Circular Trading (10%)** — Self-referential loops among 3-5 accounts with 5-10 iterations and ±20% amount jitter
- Ground-truth output: `data/aml_ground_truth.csv` with `user_id`, `is_launderer`, `aml_scenario`

### 3. Stateful Temporal Pattern Tracking
- **RollingBuffer** per customer with 30-day sliding window and automatic eviction
- **24 temporal features**: dominant hour, weekend ratio, month-end activity, hourly entropy
- **Rolling aggregators**: 1h/24h/7d/30d velocity counts and volume metrics
- **Real-time monitoring support**: buffers support incremental push for live scoring

### 4. Medallion Architecture with MRM Compliance
- **Bronze Layer**: Immutable raw transaction storage with provenance hashing
- **Silver Layer**: Validated, deduplicated data with schema enforcement
- **Gold Layer**: Feature store with partitioned, versioned feature vectors
- **OpenLineage** instrumentation for full lineage tracking
- **Great Expectations** data quality validation at each stage

## Key Innovations

1. **Privacy-by-design synthetic data**: M-PESA-specific distributions parameterized from public research — no real customer data ever touches the system
2. **Constrained random walk with look-ahead validation**: Maintains balance integrity across 10K transactions with geometric amount scaling (0.1-1.0) when balance is tight, ensuring zero violations
3. **Stateful temporal pattern tracking**: Rolling buffers with sub-hourly, daily, weekly, and 30-day aggregators supporting both batch and real-time modes
4. **AML scenario injection controlled at regulatory thresholds**: Smurfing stays under KES 100K CBK reporting threshold, layering respects 24-hour cycling windows, mule accounts model 80%+ cash-out behaviour
5. **MRM-compliant pipeline**: Every transformation emits lineage events, enabling full audit reconstruction

## Compliance

### ODPC Compliance (Kenya Data Protection Act)
- **Data minimization**: No real PII used — all customer data is procedurally generated
- **Purpose limitation**: Synthetic data used exclusively for AML model development
- **Privacy by design**: All identifiers procedurally generated, synthetic PII generation
- **Retention limitation**: Generated data is deletable on demand (no real data dependency)

### CBK AML Guideline Alignment
- **Guideline 4.1**: Customer due diligence reflected in KYC tier modeling
- **Guideline 5.2**: Transaction monitoring rules align with CBK thresholds (KES 100K structuring, KES 1M CTR)
- **Guideline 7.3**: Suspicious activity reporting workflow implemented
- **Guideline 9.1**: Record-keeping via immutable Bronze layer

### MRM Adherence
- Full lineage tracking via OpenLineage at every transformation stage
- Great Expectations data quality validation at Bronze → Silver → Gold transitions
- Versioned model registry (MLflow)
- Audit trail with checksums

## Next Steps

### Short-Term (Production Readiness)
1. **Real data calibration**: Calibrate synthetic distributions against de-identified M-PESA aggregates
2. **Threshold tuning**: Optimize alert thresholds on production data distributions
3. **SAR generation**: Implement structured STR/SAR report formatting per CBK template
4. **Scale testing**: Profile and optimize for 1M+ daily transaction throughput

### Medium-Term (Enhancement)
1. **Real-time scoring**: Deploy FastAPI-based real-time transaction scoring endpoint using `TemporalPatternTracker` rolling buffers
2. **Dashboard**: Build AML analyst dashboard with drill-down investigation workflow
3. **Additional scenarios**: Implement Trade-Based ML (TBML) and crypto-related typologies
4. **Graph analysis**: Integrate network-based anomaly detection (centrality, community detection)

### Long-Term (Enterprise)
1. **Multi-tenant**: Extend to support multiple mobile money operators (Tigo, Airtel Money)
2. **Federated learning**: Privacy-preserving cross-institution model training
3. **Regulatory filing API**: Automated STR filing to FIU via API integration
4. **Real-time network graphs**: Streaming graph analysis for live transaction networks

## Conclusion

SentinAI demonstrates that statistically realistic synthetic data, generated through Monte Carlo methods calibrated to public market research, can support the development of production-quality AML detection models. The constrained random walk ensures balance integrity with zero violations, while the four AML scenario typologies provide controlled ground-truth labels for model training and evaluation. The MRM-compliant Medallion architecture ensures auditability, reproducibility, and regulatory readiness.
