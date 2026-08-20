# Dead Code and Unused Directory Analysis Prompt

**Objective**: Map the main production pipeline to all called modules, files, and directories to identify dead code and unused directories in the SentinAI repository.

## Context

SentinAI is an autonomous AI platform for enterprise financial compliance with a Medallion architecture (Bronze-Silver-Gold data processing). The project uses Python with Poetry for dependency management.

**Known Production Entry Points** (from README.md):
- `scripts/run_audit_and_synth.py` - Main orchestrator (Bronze → Silver → Gold with AML engine, anomaly injection, and OpenLineage)
- `scripts/run_bronze.py` - Bronze layer wrapper
- `scripts/run_silver.py` - Silver layer wrapper  
- `scripts/run_gold.py` - Gold layer wrapper
- `src/data/pipelines.py` - Core POCAMLA-compliant transformation engine
- `src/data/medallion_stages.py` - Medallion stage implementations

**Project Structure**:
- `src/` - Main source code (agents/, core/, data/, datasets/, models/, retrieval/)
- `scripts/` - Entry point scripts
- `tests/` - Test files
- `config/` - Configuration files
- `api/` - API endpoints
- `deployment/` - Deployment configurations
- `eval/` - Evaluation scripts

## Analysis Instructions

### Step 1: Identify All Production Entry Points
1. Examine all files in `scripts/` directory to identify production entry points
2. Check for any additional entry points in `api/` (FastAPI application)
3. Look for any CLI entry points defined in `pyproject.toml`
4. Identify any Docker entry points from `docker-compose.yml` and `Dockerfile`

### Step 2: Build Call Graph from Entry Points
For each production entry point:
1. Parse the file and extract all import statements (both absolute and relative)
2. Identify function calls and class instantiations within the entry point
3. Recursively trace all imported modules and their dependencies
4. Build a comprehensive call graph showing:
   - Which files are directly imported
   - Which functions/classes are called from each file
   - The depth of dependency chains

### Step 3: Classify Files by Usage Category
Create a mapping of all files in the repository into these categories:

**PRODUCTION CODE** (actively used in pipeline):
- Files directly imported by production entry points
- Files transitively imported through the call graph
- Configuration files referenced by production code (check for file I/O operations)

**TEST CODE** (only used for testing):
- Files in `tests/` directory
- Files with `test_` prefix
- Files only imported by test files

**DOCUMENTATION/ARTIFACTS** (non-code):
- Markdown files (README.md, docs/)
- Jupyter notebooks (.ipynb)
- Data files (.csv, .parquet, .duckdb)
- Log files and outputs

**POTENTIALLY DEAD CODE**:
- Files not imported by any production entry point
- Files not imported by any test file
- Files with no references in the codebase
- Directories with no active files

### Step 4: Directory-Level Analysis
For each directory in the repository:
1. Determine if any files in the directory are used in production
2. If a directory has no production files, check if it contains:
   - Test files only
   - Documentation only
   - Truly dead code (no imports, no references)

### Step 5: Generate Dependency Map
Create a visual/textual representation showing:
```
Entry Point → Direct Imports → Indirect Imports → Leaf Dependencies
```

### Step 6: Identify Dead Code Candidates
List files and directories that appear to be unused:
- Files with zero import references
- Directories with no production files
- Old/legacy files (check git history or timestamps)
- Duplicate functionality (multiple files doing similar things)

### Step 7: Validation Checks
Before marking code as dead, verify:
1. Check for dynamic imports (using `importlib`, `__import__`, or string-based imports)
2. Check for plugin/extension patterns where files are discovered at runtime
3. Check for configuration-driven loading (YAML/JSON config specifying modules)
4. Check for CLI subcommands that might load modules dynamically
5. Check for test-only utilities that might still be valuable

## Output Format

Provide the analysis in this structure:

### 1. Production Pipeline Call Graph
- Tree structure showing all active dependencies from each entry point
- Include depth levels and file paths

### 2. File Usage Classification
- **Production Files**: [List with usage context]
- **Test Files**: [List]
- **Documentation/Artifacts**: [List]
- **Unclassified**: [List requiring manual review]

### 3. Directory Analysis
For each directory:
- Usage status (Production/Test/Docs/Dead)
- Active file count
- Dead file count
- Recommendation (Keep/Review/Delete)

### 4. Dead Code Candidates
- **High Confidence Dead**: Files with absolutely no references
- **Medium Confidence**: Files with only indirect or unclear references
- **Low Confidence**: Files that might be loaded dynamically

### 5. Cleanup Recommendations
Prioritized list of files/directories safe to remove:
1. Safe to delete (high confidence dead code)
2. Requires review (check with team)
3. Keep (potential dynamic loading or future use)

## Tools to Use

- Use `grep` to search for import statements and file references
- Use `find` to catalog all Python files
- Parse Python AST to extract imports accurately
- Check configuration files for module references
- Use `git log` to identify recently touched files vs stale files

## Important Notes

- **DO NOT** automatically delete anything - this is an analysis only
- Be conservative with "dead code" classification - when in doubt, flag for review
- Pay special attention to:
  - `src/agents/` - May use dynamic agent loading
  - `src/retrieval/` - May have plugin architecture
  - `config/` - Check for module references in YAML files
  - `api/` - FastAPI might have route-based imports
- The project uses LangGraph and may have graph-based module loading
- Check for any `__init__.py` files that might expose modules dynamically

## Success Criteria

The analysis should:
1. Clearly identify the production pipeline's complete dependency tree
2. Distinguish between production, test, documentation, and dead code
3. Provide actionable cleanup recommendations with confidence levels
4. Flag any files that require manual review due to dynamic loading patterns
