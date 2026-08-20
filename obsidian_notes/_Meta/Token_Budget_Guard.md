---
created: 2026-07-14
tags: [Meta, TokenGuard]
---

# Token Budget Guard

## 8.1 Token Tiers

| Tier | Content | Retention | Access |
| :--- | :--- | :--- | :--- |
| **T1** | Identity, User, _Index, Source_of_Truth | Full session | Always loaded |
| **T2** | Cost_Log, Contradictions, active context | Per-session | Session start |
| **T3** | Archived, resolved, historical | Evictable | On-demand via index |

## 8.2 Progressive Read Protocol

- Read first ~200 characters of any unknown file before deciding to load more
- For files >50 lines: scan headings/structure first
- T3 files: headings-only unless explicitly needed

## 8.3 Token Cache

| Object | Policy | Limit |
| :--- | :--- | :--- |
| File paths + headings | Session | 500 entries |
| Recently read bodies | LRU | 20 entries |
| Grep results | TTL 5 min | 50 results |

## 8.4 Compression Triggers

Proactively compress or evict context when:

- Context approaching limits
- >3 consecutive file reads without output or completion
- User indicates "from memory" or "without reading"
- Broad grep returns >50 files (re-scope)

## 8.5 Fault Tolerance

| Situation | Response |
| :--- | :--- |
| File not found | Grep by alias / check Source_of_Truth |
| Grep timeout | Narrow query scope, add directory filter |
| Ambiguous reference | Check _Index of target section |
| Stale context | Re-read T1, invalidate T2/T3 cache |
