---
created: 2026-07-14
tags: [Meta, Vault Admin]
---

# Vault Administration

## 7.1 Backups

Before any destructive or large-scale vault operation:

- Full vault backup: `tar -czf backup/obsidian_vault_YYYY-MM-DD.tar.gz obsidian_notes/`
- Section backup: `cp -r target_section/ target_section.bak.YYYY-MM-DD/`

## 7.2 Batch File Tasks

Common batch operations:

- Rename files: `for f in *old_pattern*.md; do mv "$f" "${f//old_pattern/new_pattern}"; done`
- Bulk frontmatter updates: Use `sed` or a short Python script to update YAML frontmatter across multiple files
- Find broken wikilinks: `grep -roP '\[\[([^\]]+)\]\]' obsidian_notes/ \| sed 's/.*\[\[\|\]\]//g' \| sort -u \| while read link; do [ ! -f "obsidian_notes/${link}.md" ] && echo "BROKEN: $link"; done`
- Find orphaned files: Files in the vault that no other file links to

## 7.3 Renaming Convention

- **Folders**: `NN_DescriptiveName/` (zero-padded number, underscore, PascalCase)
- **Files**: `Snake_Case.md` with aliases in frontmatter for alternative names
- **Wikilinks**: Always `[[Target_File]]` or `[[Section/Target_File]]` from vault root
- **Screenshots**: `07_Diagrams/{topic}_{YYYY-MM-DD}.png`
