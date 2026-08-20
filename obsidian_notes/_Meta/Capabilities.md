---
created: 2026-07-14
tags: [Meta, Capabilities]
---

# Capabilities

## 6.1 Execution Tools

You have access to a set of tools you can use to answer the user's questions and complete tasks. These tools are invoked via tool call blocks. Always use the appropriate tool for the job — avoid shelling out to system commands when a dedicated tool exists.

Available tool categories:

- **File Operations**: Read, Write, Edit for file handling; Glob for pattern-based file discovery; Grep for content search
- **Shell**: Bash for system commands, scripts, and any operation not covered by dedicated tools
- **Web**: WebSearch for information retrieval; WebFetch for URL content fetching
- **Planning & Memory**: EnterPlanMode for complex tasks; TaskCreate/TaskUpdate for progress tracking; CronCreate for scheduling
- **Agent**: Agent for delegating sub-tasks to specialized sub-agents
- **Notebook**: NotebookEdit for Jupyter notebook manipulation

## 6.2 Archives

When handling archive files:

- **Create `.tar.gz`**: `tar -czf archive.tar.gz directory/`
- **Extract `.tar.gz`**: `tar -xzf archive.tar.gz`
- **Create `.zip`**: `zip -r archive.zip directory/`
- **Extract `.zip`**: `unzip archive.zip -d output_dir/`

## 6.3 Embeddings

When semantic understanding of the vault is required beyond keyword search:

1. **Script**: `scripts/build_vault_embeddings.py` (to be created if absent)
2. **Model**: SentenceTransformers `all-MiniLM-L6-v2`
3. **Index**: FAISS `IndexIDMap, IndexFlatIP` (inner product similarity)
4. **Storage**: `_Meta/.embeddings/faiss.index` + `_Meta/.embeddings/chunks.pkl`
5. **Query**: Return top-k=5 chunks with `(filename, heading, similarity_score)`
6. **Refresh**: Rebuild whenever vault structure changes meaningfully

## 6.4 Web Context

When retrieving external context:

1. Fetch via WebFetch
2. Extract relevant section(s)
3. Summarize concisely (3-5 bullets)
4. Cite source and retrieval timestamp
