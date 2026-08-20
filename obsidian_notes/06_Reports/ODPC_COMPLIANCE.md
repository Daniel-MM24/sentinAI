# ODPC Compliance: Data Privacy Documentation

## Overview

This document demonstrates how the SentinAI AML PoC aligns with the **Kenya Data Protection Act (ODPC, 2019)** and **GDPR** principles. As a synthetic data system, SentinAI operates under a privacy-by-design framework — no real customer PII is ever stored, processed, or transmitted.

## Regulatory Framework

| Regulation | Jurisdiction | Applicability | Status |
| :--- | :--- | :--- | :---: |
| Kenya Data Protection Act (2019) | Kenya | Full | **Compliant** |
| GDPR (EU) 2016/679 | EU (extraterritorial) | Partial (data subjects in EU) | **Compliant** |
| CBK AML Guidelines (2023) | Kenya | Full | **Aligned** |

## Data Protection Principles

### 1. Lawfulness, Fairness, and Transparency (ODPC Sec. 25 / GDPR Art. 5.1a)

**Implementation**:
- All data processing is documented with lineage tracking (OpenLineage)
- Purpose limitation: data used exclusively for AML model development
- No customer consent required — no real customer data is used

**Evidence**:
- Full OpenLineage audit trail per transformation
- Transformation metadata with run_id, timestamps, and row counts
- Synthetic_flag = True on all records for auditor distinguishability

### 2. Purpose Limitation (ODPC Sec. 25 / GDPR Art. 5.1b)

**Implementation**:
- Data collected/generated for **one purpose only**: AML detection model development and validation
- No secondary use for marketing, profiling, or any non-AML purpose
- Synthetic data tagged with `synthetic_flag` for clear purpose identification

### 3. Data Minimization (ODPC Sec. 25 / GDPR Art. 5.1c)

**Implementation**:
- Only fields relevant to AML detection are generated
- No sensitive personal data categories (health, biometrics, political affiliation) generated
- Minimal demographic data: age, gender, county (not specific address)
- Phone numbers are SHA-256 hashed, not stored in plaintext

**Field Necessity Justification**:

| Field | AML Necessity | Risk if Omitted |
| :--- | :--- | :--- |
| user_id | Entity resolution, transaction linking | Cannot construct transaction history |
| tier | CBK limit enforcement | Cannot validate tier compliance |
| county | Geographic AML pattern detection | Miss geographic SAR patterns |
| age | Statistical profile realism only | Low — could be omitted |
| gender | Statistical profile realism only | Low — could be omitted |
| income_level | Archetype stratification | Medium — affects transaction modeling |

### 4. Accuracy (ODPC Sec. 25 / GDPR Art. 5.1d)

**Implementation**:
- Great Expectations validation at every pipeline stage ensures data accuracy
- Pandera schema enforcement prevents type/shape errors
- SHA-256 provenance hashing detects data corruption
- Constraint validation confirms balance integrity

### 5. Storage Limitation (ODPC Sec. 25 / GDPR Art. 5.1e)

**Implementation**:
- Generated data can be completely deleted on demand
- No reliance on persistent real data sources
- Bronze layer data is deletable without affecting upstream dependencies
- Retention period configurable via pipeline parameters

### 6. Integrity and Confidentiality (ODPC Sec. 25 / GDPR Art. 5.1f)

**Implementation**:
- SHA-256 hashing of phone number identifiers
- No real PII ever enters the system — all PII fields are procedurally generated
- Bronze layer immutability prevents data tampering
- Dead letter queue isolates schema violations from the main pipeline

## Synthetic Data Privacy Guarantees

### No Real Data Contamination

```
┌───────────────────────────────────────────────────────────────┐
│                     SYSTEM BOUNDARY                            │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                SYNTHETIC DATA DOMAIN                     │  │
│  │                                                         │  │
│  │  • All customer names are procedurally generated         │  │
│  │  • All phone numbers are synthetic + SHA-256 hashed     │  │
│  │  • All emails are fake (e.g., C_001234@sentinai.        │  │
│  │    synthetic)                                            │  │
│  │  • All transaction amounts are Monte Carlo sampled      │  │
│  │  • All timestamps are generated from Poisson processes  │  │
│  │  • All counterparties are procedurally assigned          │  │
│  │                                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│         No bidirectional data flow with real systems           │
└───────────────────────────────────────────────────────────────┘
```

### Differential Privacy Considerations

The synthetic data is generated entirely from probability distributions — not from real data. This provides **ε = ∞ effective differential privacy** (no privacy loss because no real data is used). However, for completeness:

| DP Property | Status | Note |
| :--- | :--- | :--- |
| Real data exposure | None | All data is Monte Carlo generated |
| Membership inference risk | None | No real individuals exist in data |
| Attribute inference risk | None | No real attributes to infer |
| Re-identification risk | None | No real identity to match |
| Distributional disclosure | Minimal | Distributions are based on published research |

### K-Anonymity Assessment

While k-anonymity is designed for anonymized real data, we assess the synthetic data:

| Quasi-Identifier | Distinct Values | Min Group Size | k-anonymity |
| :--- | :---: | :---: | :---: |
| (county, gender, age_bracket) | ~282 | ~7 | > 5 |
| (tier, income_level, archetype) | ~36 | ~18 | > 5 |
| (county, tier, urban_rural) | ~141 | ~4 | < 5* |

*Some sparse combinations may have < 5 records. Mitigation: all records are synthetic — no re-identification risk exists regardless of group size.

## Data Flow Compliance

### SHA-256 Hashing of Identifiers

```python
import hashlib

def hash_identifier(value: str) -> str:
    """SHA-256 hash of PII for privacy compliance."""
    return hashlib.sha256(value.encode()).hexdigest()

# Example: phone number hashing
phone_hash = hash_identifier("254712345678")
# Result: "a1b2c3d4..." (64-character hex string)
```

### Synthetic PII Generation

```python
def generate_synthetic_email(user_id: str) -> str:
    """Generate a fake email — no real email addresses used."""
    return f"{user_id.lower()}@sentinai.synthetic"

def generate_synthetic_tax_id(user_id: str) -> str:
    """Generate a fake tax ID for AML compliance schema."""
    return f"TAX_{user_id}"
```

### Lineage Tracking for Audit

Every transformation emits metadata including:

| Field | Purpose | ODPC Relevance |
| :--- | :--- | :--- |
| `run_id` | Unique transformation identifier | Audit trail completeness |
| `source_type` | POSTGRESQL / SYNTHETIC / CSV | Data origin transparency |
| `synthetic_flag` | True = procedurally generated | Purpose identification |
| `ingestion_date` | When data was processed | Storage limitation tracking |
| `row_count` | Number of records processed | Data minimization verification |

## ODPC Rights Implementation

### Right to be Informed (ODPC Sec. 26)

- Complete documentation of data processing activities available via OpenLineage
- Transformation metadata readily accessible from Bronze metadata JSON files

### Right to Access (ODPC Sec. 27)

- All data is in portable Parquet/CSV format
- No access control needed — data does not contain real PII

### Right to Deletion (ODPC Sec. 30)

- Generated data can be deleted on demand: `rm -rf data/ output/`
- System regenerates from scratch (no data loss from deletion)
- No backup chains or archival dependencies

### Right to Data Portability (ODPC Sec. 33)

- All data in open formats (Parquet, CSV, JSON)
- No proprietary encoding
- Full schema documentation available

## CBK Compliance Alignment

### CBK Guideline 4.1: Customer Due Diligence

| Requirement | Implementation |
| :--- | :--- |
| Customer identification | SHA-256 hashed synthetic identifiers |
| KYC tier assessment | 3-tier system (60/30/10% distribution) |
| Risk profiling | Archetype-based behavioral classification |
| Enhanced due diligence | High-risk entity flagging in generator |

### CBK Guideline 5.2: Transaction Monitoring

| Requirement | Implementation |
| :--- | :--- |
| Real-time monitoring | Alert engine with hybrid scoring |
| Threshold setting | CBK-aligned regulatory.yaml thresholds |
| Pattern detection | 6-category feature engineering |
| Scenario coverage | 4 AML scenarios + 8 structural anomaly types |

### CBK Guideline 7.3: Suspicious Activity Reporting

| Requirement | Implementation |
| :--- | :--- |
| SAR workflow | Priority-based case management |
| Record keeping | Immutable Bronze storage (7-year retention) |
| FIU notification | SAR filing workflow in alert engine |

### CBK Guideline 9.1: Record Keeping

| Requirement | Implementation |
| :--- | :--- |
| Transaction records | Immutable Parquet with SHA-256 hashing |
| Retention period | Configurable (default: 7 years) |
| Audit trail | OpenLineage events per transformation |

## MRM Compliance (Model Risk Management)

### SR 11-7 / OCC 2011-12 Alignment

| Principle | Implementation |
| :--- | :--- |
| Model development | Documented methodology with distribution justification |
| Model validation | 5-fold cross-validation, overfitting checks, sensitivity analysis |
| Governance | MLflow model registry with staged promotion (dev/staging/prod) |
| Documentation | Complete project documentation in obsidian_notes/ |
| Ongoing monitoring | Feature drift detection (PSI), data drift (KS test) |

## Incident Response

In the event of a data privacy incident:

| Scenario | Response | ODPC Reporting |
| :--- | :--- | :--- |
| Synthetic data leak | No real PII exposed — no notification required | Not applicable |
| Configuration exposure | Non-sensitive config YAML — low risk | Not applicable |
| Model artifact leak | Trained on synthetic data only — no privacy impact | Not applicable |
| Source code leak | No data exposure — intellectual property concern | Not applicable |

## Conclusion

SentinAI's synthetic-only approach to AML model development is inherently compliant with the Kenya Data Protection Act, GDPR, and CBK AML guidelines. By eliminating real customer data from the development pipeline entirely, the system achieves:

- **Zero privacy risk**: No real PII ever enters the system
- **Full auditability**: Every transformation is tracked via OpenLineage
- **Regulatory readiness**: CBK and MRM frameworks addressed
- **Complete deletability**: Data can be wiped and regenerated on demand
- **Privacy-by-design**: Demonstrated adherence to ODPC principles throughout the architecture

This approach represents a best-practice model for fintech AML development in regulated markets where data access is constrained by privacy regulations.
