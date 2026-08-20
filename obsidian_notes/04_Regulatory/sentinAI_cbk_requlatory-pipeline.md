---
name: sentinAI-cbk-regulatory-pipeline
description: Actively building CBK AML/CFT regulatory document ingestion for SentinAI — seed data and ChunkModel mapping complete
type: project
---

Active work stream: building the CBK regulatory seed data pipeline for SentinAI. As of 2026-06-25, the following is complete:
- `/home/dan/cbk_amln_tf_guidelines.md` — comprehensive CBK AML/CFT source document (transactional limits, KYC tiers, reporting thresholds, money laundering typologies, analytical action rules, penalties)
- `src/retrieval/seed_data.json` — 9 pre-chunked regulatory entries (doc_cbk_aml_001 through 009) in the SentinAI ingestion schema
- `src/retrieval/ingestion.py` — `load_markdown_seed()` method added to `DocumentIngestor` that maps seed JSON into `ChunkModel` with SHA-256 hashing

**Why:** Compliance requirements for Kenyan mobile money transaction monitoring under CBK PG/43 and POCAMLA.

**How to apply:** When working on SentinAI retrieval, the seed pipeline feeds regulatory context into the vector DB. The `load_markdown_seed()` path at `ingestion.py:102-145` is the bridge between the raw regulatory documents and the embedding pipeline.
