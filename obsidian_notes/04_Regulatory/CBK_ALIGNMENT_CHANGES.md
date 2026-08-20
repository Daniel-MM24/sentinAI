# CBK PG/43 Alignment: Discrepancies and Changes

## Overview

This document records all discrepancies found between the sentinAI implementation and the CBK PG/43 mobile money regulatory guidelines, and the changes made to align them.

**Alignment date:** 2026-07-14
**Previous config state:** Pre-alignment (codebase as of prior commits)
**Current config state:** Post-alignment to CBK PG/43 4-tier structure

---

## Identified Discrepancies

### 1. Tier Count: 3 vs 4

| Aspect | Before | CBK Requires |
| :--- | :---: | :---: |
| Number of tiers | 3 (Tier 1, 2, 3) | 4 (Tier 1 Basic, Tier 2 Interim, Tier 3 Full KYC, Tier 4 EDD) |
| Missing tier | — | Tier 4 (Enhanced Due Diligence for corporate/institutional) |

**Fix:** Added `KYCTier.TIER_4` / `WalletTier.TIER_4` to all enum classes, config objects, and distribution logic.

---

### 2. Tier Proportion Split

| Tier | CBK Expected | Before (Code) | Before (Docs) | After |
| :---: | :---: | :---: | :---: | :---: |
| Tier 1 | ~60% | 60% | 60% | **60%** |
| Tier 2 | ~20% | 30% | 30% | **20%** |
| Tier 3 | ~15% | 10% | 10% | **15%** |
| Tier 4 | ~5% | — | — | **5%** |

**Fix:** Adjusted `tier_2_pct` (0.30 → 0.20), `tier_3_pct` (0.10 → 0.15), added `tier_4_pct: 0.05` in `stratified_profiles.py`. Updated fallback tier sampling in `behavioral_generator.py` from `rng.integers(1, 4)` to `rng.choice([1,2,3,4], p=[0.60, 0.20, 0.15, 0.05])`.

---

### 3. Transaction Amount Limits (per single tx)

| Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 (EDD) |
| :--- | ---: | ---: | ---: | ---: |
| **CBK PG/43** | KES 10,000 | KES 50,000 | KES 150,000 | KES 500,000 |
| **stratified_profiles.py** (before) | KES 70,000 | KES 150,000 | KES 250,000 | — |
| **behavioral_generator.py** `tier_caps` (before) | KES 70,000 | KES 150,000 | KES 250,000 | KES 1,000,000 |
| **synthetic_generator.py** (before) | KES 70,000 | KES 150,000 | KES 250,000 | — |
| **After** | **KES 10,000** | **KES 50,000** | **KES 150,000** | **KES 500,000** |

**Impact:** Tier 1 max tx reduced 85% (70K → 10K). Tier 2 reduced 67% (150K → 50K). Tier 3 reduced 40% (250K → 150K). Tier 4 new cap (was absent or 1M in behavioral_generator, now 500K).

---

### 4. Wallet Balance Caps

| Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 (EDD) |
| :--- | ---: | ---: | ---: | ---: |
| **CBK PG/43** | KES 50,000 | KES 200,000 | KES 1,000,000 | No cap |
| **stratified_profiles.py** (before) | KES 300,000 | KES 500,000 | KES 500,000 | — |
| **synthetic_generator.py** (before) | KES 300,000 | KES 500,000 | KES 500,000 | — |
| **behavioral_generator.py** (before) | KES 350,000 | KES 750,000 | KES 1,250,000 | KES 5,000,000 |
| **behavioral_generator.py** *(calc: `tier_cap * 5`)* | KES 350,000 | KES 750,000 | KES 1,250,000 | KES 5,000,000 |
| **After** — stratified_profiles | **KES 50,000** | **KES 200,000** | **KES 1,000,000** | **KES 5,000,000** |
| **After** — behavioral_generator | **KES 50,000** | **KES 200,000** | **KES 1,000,000** | **KES 5,000,000** |

**Fix:** Changed `stratified_profiles.py` balance limits from `[300K, 500K, 500K]` → `[50K, 200K, 1M, 5M]`. Changed `behavioral_generator.py` to use explicit `balance_caps` dict instead of `tier_caps * 5`. Tier 4 set at KES 5M as soft limit (CBK specifies no cap).

---

### 5. Daily Velocity Caps

| Source | Tier 1 | Tier 2 | Tier 3 | Tier 4 (EDD) |
| :--- | ---: | ---: | ---: | ---: |
| **CBK PG/43** | KES 25,000 | KES 100,000 | KES 500,000 | No cap |
| **behavioral_generator.py** (before) | KES 70,000 | KES 150,000 | KES 250,000 | KES 1,000,000 |
| **config/regulatory.yaml** (before — `daily_velocity_cap_kes`) | KES 100,000 | KES 1,000,000 | KES 10,000,000 | — |
| **After** | **KES 25,000** | **KES 100,000** | **KES 500,000** | **KES 10,000,000** |

**Fix:** Updated `_DAILY_VELOCITY_CAPS` and `regulatory.yaml` to CBK values. Tier 4 set at KES 10M as soft limit.

---

### 6. Inconsistent Values Across Codebase

Before alignment, no two files agreed on the same tier limits:

| File | Tier 1 Limits | Tier 2 Limits | Tier 3 Limits | Tier 4 |
| :--- | :--- | :--- | :--- | :--- |
| `stratified_profiles.py` | Tx 70K, Bal 300K | Tx 150K, Bal 500K | Tx 250K, Bal 500K | — |
| `behavioral_generator.py` | Tx 70K, Vel 70K | Tx 150K, Vel 150K | Tx 250K, Vel 250K | Tx 1M, Vel 1M |
| `synthetic_generator.py` | Tx 70K, Bal 300K | Tx 150K, Bal 500K | Tx 250K, Bal 500K | — |
| `config/regulatory.yaml` | Bal 50K, Vel 100K | Bal 500K, Vel 1M | Bal 5M, Vel 10M | — |
| Obsidian notes | Bal 50K, Vel 100K | Bal 500K, Vel 1M | Bal 5M, Vel 10M | — |

**Fix:** All files now reference a single source of truth (CBK PG/43 limits). Values are consistent across code + docs.

---

### 7. Naming: VENDOR_MERCHANT vs TIER_3

| Aspect | Before | After |
| :--- | :--- | :--- |
| YAML key | `VENDOR_MERCHANT` | `TIER_3` |
| `regulatory.yaml` `allowed_values.kyc_tier_level` | `[TIER_1, TIER_2, VENDOR_MERCHANT]` | `[TIER_1, TIER_2, TIER_3, TIER_4]` |
| `validators.py` cap expressions | Referenced `VENDOR_MERCHANT` | Referenced `TIER_3` and `TIER_4` |
| `pipelines.py` cap expressions | Referenced `VENDOR_MERCHANT` | Referenced `TIER_3` and `TIER_4` |

**Fix:** Renamed all `VENDOR_MERCHANT` → `TIER_3` and added `TIER_4` handling in all mapping expressions.

---

### 8. YAML Structure: Before vs After

| Aspect | Before | After |
| :--- | :--- | :--- |
| Flat `kyc_tiers` keys | `TIER_1`, `TIER_2`, `VENDOR_MERCHANT` | `TIER_1`, `TIER_2`, `TIER_3`, `TIER_4` |
| Per-tier metadata | Only balance + velocity caps | Name, description, proportion, all 3 limits, reporting flags |
| Velocity caps matched CBK? | No (100K / 1M / 10M vs CBK 25K/100K/500K) | Yes |
| Balance caps matched CBK? | No (50K / 500K / 5M vs CBK 50K/200K/1M) | Yes |

---

### 9. synthetic_generator.py: Tier Probability Distribution

| Archetype | Before (T1/T2/T3) | After (T1/T2/T3/T4) |
| :--- | :---: | :---: |
| CORPORATE_SME | 0.10 / 0.30 / 0.60 | 0.05 / 0.15 / 0.50 / 0.30 |
| MICRO_MERCHANT | 0.20 / 0.60 / 0.20 | 0.10 / 0.55 / 0.30 / 0.05 |
| RETAIL_HEAVY | 0.40 / 0.50 / 0.10 | 0.35 / 0.50 / 0.12 / 0.03 |
| RETAIL_STANDARD | 0.70 / 0.25 / 0.05 | 0.70 / 0.20 / 0.08 / 0.02 |

---

### 10. Obsidian Docs: Before vs After

| Doc | Change |
| :--- | :--- |
| `CONSTRAINT_ENFORCEMENT.md` | Extended tier table from 3 rows → 4 rows. Added `Max Single Tx` and `Max Daily Cumulative` columns. Updated all values to CBK. |
| `MONTE_CARLO_ENGINE.md` | KYC tier distribution validation from `60/30/10%` → `60/20/15/5%` |

---

## Files Changed

### Code Changes

| File | What Changed |
| :--- | :--- |
| `src/data/stratified_profiles.py` | Added `TIER_4` to `KYCTier` enum; added `tier_4_pct` config field; updated `_assign_kyc_tier` to 4-tier choice with CBK limits; updated progress log |
| `src/data/behavioral_generator.py` | Updated `BehavioralGeneratorConfig` docstring with CBK table; updated `tier_caps`, `balance_caps`, `_DAILY_VELOCITY_CAPS` to CBK values; added `balance_caps` Field; changed fallback tier sampling from uniform 1-3 to weighted 1-4; replaced `tier_caps * 5` with `balance_caps` dict lookup |
| `src/data/synthetic_generator.py` | Added `TIER_4` to `WalletTier` enum; updated `WalletTier` docstrings; updated tier probability distributions for all archetypes; updated `_enforce_tier_limit` and `_enforce_balance_limit` dicts with CBK values |
| `config/regulatory.yaml` | Added `TIER_3` and `TIER_4` with CBK-aligned caps; renamed `VENDOR_MERCHANT` → `TIER_3`; added per-tier metadata (name, description, proportion, tx limit, flags); updated velocity caps to CBK values; updated allowed_values |
| `src/data/pipelines.py` | Replaced `VENDOR_MERCHANT` → `TIER_3` in both cap expressions; added `TIER_4` branch to both expressions |
| `src/data/validators.py` | Updated `tier_cap_expr` to include `TIER_3` and `TIER_4` branches; replaced `VENDOR_MERCHANT` → `TIER_3` |

### Documentation Changes

| File | What Changed |
| :--- | :--- |
| `obsidian_notes/regulatory/` | New directory with CBK regulatory docs copied from `src/retrieval/` |
| `obsidian_notes/regulatory/CBK_ALIGNMENT_CHANGES.md` | **This file** |
| `obsidian_notes/technical/CONSTRAINT_ENFORCEMENT.md` | Tier table expanded to 4 tiers with full CBK limits |
| `obsidian_notes/technical/MONTE_CARLO_ENGINE.md` | KYC distribution validation target updated |

### New Files

| File | Description |
| :--- | :--- |
| `obsidian_notes/regulatory/CBK_PG43_GUIDELINES.md` | CBK PG/43 regulatory framework guidelines |
| `obsidian_notes/regulatory/CBK_TIER_STRUCTURE.md` | CBK KYC tier structure with limits |
| `obsidian_notes/regulatory/CBK_CTR_STR_GUIDELINES.md` | CBK mandatory reporting thresholds (CTR/STR) |
| `obsidian_notes/regulatory/CBK_TRANSACTION_MONITORING.md` | CBK transaction monitoring & velocity rules |
| `obsidian_notes/regulatory/CBK_ENFORCEMENT.md` | CBK enforcement & penalty framework |
| `obsidian_notes/regulatory/CBK_AUDIT_REQUIREMENTS.md` | CBK audit & compliance requirements |
| `obsidian_notes/regulatory/POCAMLA_REGULATORY_FRAMEWORK.md` | POCAMLA regulatory framework overview |
