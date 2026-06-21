# SentinAI

Autonomous, agentic AI platform for enterprise financial compliance.

## Overview

SentinAI is a comprehensive financial compliance platform that implements a Medallion architecture (Bronze-Silver-Gold) for data processing with built-in auditability, lineage tracking, and differential privacy. It serves as an autonomous, audit-ready multi-agent framework for financial compliance and operational forecasting, integrating RAG-based grounding with SHAP-explainable predictive models to ensure transparent, compliant, and data-driven business process automation.

## Architecture

- **Bronze Layer**: Raw data ingestion from PostgreSQL and S3 sources
- **Silver Layer**: Data cleaning, entity resolution, and quality validation
- **Gold Layer**: Feature engineering and feature store creation

## Key Features

- OpenLineage integration for complete audit trails
- Differential privacy for synthetic data generation
- Great Expectations for data quality validation
- Entity resolution using Jaro-Winkler similarity
- M-Pesa transaction pattern simulation
- Agentic RAG system for Kenyan financial crime compliance
- Hybrid search with BM25 and cross-encoder reranking
- Comprehensive evaluation metrics (Precision@k, MRR, Citation Fidelity)

## Installation

```bash
poetry install
```

## Usage

Run the orchestrator to initialize the pipeline:

```bash
python scripts/run_audit_and_synth.py
```


## Compliance

This platform follows MRM (Model Risk Management) compliance standards with immutable audit trails and version-controlled data transformations. It prioritizes Kenyan statutory law (POCAMLA, DPA 2019) as absolute authority for financial crime compliance decisions.
