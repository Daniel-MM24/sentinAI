# SentinAI Overview

SentinAI is an AI-powered Anti-Money Laundering (AML) compliance system designed for Kenyan mobile money platforms, specifically M-Pesa. The system generates synthetic transaction data, injects known money laundering patterns, and produces ground-truth labels for training and validating AML detection models.

## System Architecture

```
Synthetic Data Generation
  └─ Behavioral Generator (10K transactions, 1K customers)
       ├─ Constrained Random Walk with balance integrity
       ├─ KYC Tier calibration (T1/T2/T3)
       └─ Archetype profiles (retail, merchant, corporate)

AML Scenario Injection  
  └─ 4 typologies at 2% prevalence
       ├─ Smurfing (40%) — small tx under KES 100K
       ├─ Layering (30%) — rapid 4+ account chains
       ├─ Mule Account (20%) — receive → withdraw cycles
       └─ Circular Trading (10%) — self-referential loops

Temporal Feature Extraction
  └─ Stateful rolling buffers
       ├─ Daily/weekly/monthly pattern detection
       ├─ 1h/24h/7d/30d rolling aggregators
       └─ Real-time monitoring support

Output
  ├─ data/detailed_transactions.csv    — Clean transaction history
  ├─ data/aml_ground_truth.csv         — Per-customer labels
  ├─ data/temporal_features.csv        — 24 temporal feature columns
  └─ data/bronze/customers/            — Customer profiles
```

## Key Capabilities
- Synthetic transaction generation with balance constraint enforcement
- AML scenario injection with controlled prevalence and regulatory threshold alignment
- Stateful temporal pattern tracking with rolling window aggregators
- Real-time monitoring support via per-customer `RollingBuffer` incremental updates

## Regulatory Context
- Central Bank of Kenya (CBK) Prudential Guidelines
- Anti-Money Laundering & Counter Financing of Terrorism (AML/CFT) requirements
- CBK reporting thresholds: KES 100K structuring, KES 1M CTR
- Suspicious Activity Report (SAR) filing mandates
- Kenya Data Protection Act (ODPC) compliance
