# SentinAI

Autonomous, agentic AI platform for enterprise financial compliance.

## Overview

SentinAI is a comprehensive financial compliance platform that implements a Medallion architecture (Bronze-Silver-Gold) for data processing with built-in auditability, lineage tracking, and differential privacy.

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

## Installation

```bash
poetry install
```

## Usage

Run the orchestrator to initialize the pipeline:

```bash
python scripts/run_audit_and_synth.py
```

## Configuration

Copy `.env.example` to `.env` and configure your environment variables.

## Compliance

This platform follows MRM (Model Risk Management) compliance standards with immutable audit trails and version-controlled data transformations.
