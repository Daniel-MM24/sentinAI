---
created: 2026-07-14
modified: 2026-07-14
aliases: [Meta Index, Agent Memory]
tags: [Meta, Index]
---

# _Meta/ — Agent Memory System

System-critical files for agent identity, cost tracking, and fact management.

## Files

| Note | Purpose |
| :--- | :--- |
| [[_Meta/Source_of_Truth]] | Central index & vault map |
| [[_Meta/Identity]] | Agent persona & constraints |
| [[_Meta/User]] | User profile & current focus |
| [[_Meta/Contradictions]] | Discrepancies log |
| [[_Meta/Cost_Log]] | Session cost tracking |
| [[_Meta/Token_Budget_Guard]] | Token management, cache, compression |
| [[_Meta/Capabilities]] | Tools, archives, embeddings, web context |
| [[_Meta/Vault_Admin]] | Backups, batch ops, conventions |
| [[_Meta/Initialization_Flow]] | Session start Phases 1–4 |

## Rules

- **Single Source:** Check Source_of_Truth before any response
- **Preview-Then-Commit:** User approves before writes
- **Cost Awareness:** Peak = 2x (UTC 01:00-04:00, 06:00-10:00)
- **Token Budget:** Progressive read, LRU cache, T1/T2/T3 tiers (see Token_Budget_Guard)
- **Init Flow:** Execute Phases 1–4 on session start (see Initialization_Flow)
