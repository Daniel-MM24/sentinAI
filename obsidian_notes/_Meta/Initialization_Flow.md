---
created: 2026-07-14
tags: [Meta, Protocol]
---

# Initialization Flow

On session start, execute the following phases sequentially.

## Phase 1 — Scan Mode

- Identify all `.md` files under `obsidian_notes/`
- If >30 files: headings-only scan for T3, full scan for T1/T2
- Detect vault structure: folders, index files, stubs

## Phase 2 — Digest Mode

- Load T1 files: Identity, User, _Index, Source_of_Truth
- Load T2 files: Contradictions, Cost_Log, Token_Budget_Guard, Capabilities, Vault_Admin
- Map current project state from 05_Phases and 01_Project indexes

## Phase 3 — State Alignment

- Cross-reference Contradictions with recent Changelog
- Flag if vault last modified >7 days ago
- Report: file count, T1 loaded, peak status, last modified

## Phase 4 — Gatekeeper Greeting

```
Phase: 🟢 Loaded
T1: N files, N KB
T2: N files, N KB
Peak: 🚨 2x (HH:MM-HH:MM UTC) | ✅ Off-Peak
Last Vault Write: YYYY-MM-DD
Vault: N .md files, N folders
```
