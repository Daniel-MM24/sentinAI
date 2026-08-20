---
created: 2026-07-14
modified: 2026-07-14
aliases: [Discrepancies]
tags: [Meta, Log]
cost_multiplier: 2x
---

# Contradictions Log

| Date | Contradiction | Resolution |
| :--- | :--- | :--- |
| 2026-07-14 | Code had 3-tier KYC; CBK mandates 4 tiers. Multiple files had conflicting limit values, none matching CBK PG/43. | Aligned all code + docs to 4-tier CBK structure (see [[04_Regulatory/CBK_ALIGNMENT_CHANGES\|CBK Alignment Doc]]) |
| 2026-07-14 | `config/regulatory.yaml` key `VENDOR_MERCHANT` didn't match `TIER_3` naming convention used elsewhere | Renamed to `TIER_3` across YAML, validators, and pipelines |
| 2026-07-14 | Stratified vs behavioral vs synthetic generators had different tier values and no single source of truth | All now reference CBK PG/43 limits (10K/50K/150K/500K) |
