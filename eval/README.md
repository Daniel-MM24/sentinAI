# Evaluation Module for SentinAI

This module provides comprehensive evaluation tools for the SentinAI agentic RAG system, specifically designed for Kenyan financial crime compliance.

## Overview

The evaluation module tests the Agent's ability to:
- Identify compliance violations from transaction patterns
- Retrieve correct statutory citations from the Kenyan regulatory corpus
- Justify audit decisions with logical reasoning paths
- Avoid hallucinated citations (grounds for immediate failure)

## Components

### 1. Synthetic Audit Cases (`synthetic_cases.py`)

Generates ground-truth test cases covering realistic Kenyan financial crime typologies:

- **Structuring**: Multiple small payments to evade KES 1M CTR threshold
- **Agent Fraud**: Abnormal agent activity patterns
- **Digital Lending**: M-Shwari/Fuliza misuse for layering
- **Agent Collusion**: Rapid cash-in/cash-out at single agents
- **Trade-Based Money Laundering**: Over-invoicing schemes
- **PEP Transactions**: Politically exposed person monitoring
- **Terrorist Financing**: Structured payments to high-risk regions
- **Shell Company Layering**: Complex corporate structures
- **Cash Intensive Business**: Smurfing through legitimate businesses
- **Real Estate**: Large cash property purchases
- **Cryptocurrency**: VASP-related money laundering

Each case includes:
- Scenario description
- Mock transaction logs (evidence)
- Target citation from Kenyan regulatory corpus
- Expected reasoning path
- Expected audit decision

### 2. Evaluation Harness (`evaluate.py`)

Implements key RAG evaluation metrics:

- **Precision@k**: Precision at rank k for retrieved documents
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant document
- **Citation Fidelity**: Accuracy of cited regulatory sections
- **Reasoning Alignment**: Alignment between expected and actual reasoning paths
- **Decision Accuracy**: Correctness of audit decisions

## Usage

### Generate Synthetic Cases

```python
from eval.synthetic_cases import generate_synthetic_cases, get_case_by_id

# Generate all cases
cases = generate_synthetic_cases()
print(f"Generated {len(cases)} synthetic audit cases")

# Get specific case
case_001 = get_case_by_id("001")
print(f"Case: {case_001.description}")
print(f"Target Citation: {case_001.target_citation}")
```

### Run Evaluation

```python
import asyncio
from eval.evaluate import RAGEvaluator
from src.retrieval.vector_db import VectorStore
from src.retrieval.ingestion import DocumentIngestor

# Initialize components
vector_store = VectorStore()
document_ingestor = DocumentIngestor()
evaluator = RAGEvaluator(vector_store, document_ingestor)

# Evaluate all cases
evaluation = await evaluator.evaluate_all_cases()

# Print comprehensive report
evaluator.print_evaluation_report(evaluation)
```

### Evaluate Single Case

```python
from eval.synthetic_cases import get_case_by_id

case = get_case_by_id("001")
query = f"{case.scenario_category}: {case.description}"

result = await evaluator.evaluate_single_case(
    case=case,
    query=query,
    actual_reasoning=["Identified pattern...", "Aggregated transactions..."],
    actual_decision="FILE_STR"
)

print(f"Precision@5: {result.precision_at_k:.4f}")
print(f"Citation Match: {result.citation_match}")
print(f"Decision Match: {result.decision_match}")
```

## Regulatory Framework Coverage

The synthetic cases prioritize Kenyan statutory law as absolute authority:

- **POCAMLA** (Proceeds of Crime and Anti-Money Laundering Act)
- **DPA 2019** (Data Protection Act)
- **CBK Guidelines** (Central Bank of Kenya)
- **FRC National Risk Assessment** (Financial Reporting Centre)

## Compliance Requirements

- **Tiered Compliance**: Kenyan statutory law takes precedence
- **Audit-First Logic**: All outputs must be verifiable against legal corpus
- **Citation Integrity**: Hallucinated citations are grounds for immediate failure
- **Realistic Scenarios**: Based on actual Kenyan financial crime typologies

## Evaluation Metrics Explained

### Precision@k
Measures the fraction of retrieved documents that are relevant at rank k.
- Higher values indicate better retrieval precision
- Critical for compliance where accuracy matters more than recall

### Mean Reciprocal Rank (MRR)
Measures how highly the first relevant document is ranked.
- Range: 0 to 1, where 1 indicates first result is relevant
- Important for user experience in audit workflows

### Citation Fidelity
Measures the accuracy of regulatory citations retrieved.
- Uses semantic similarity to match citations
- Critical for audit-first compliance requirements

### Reasoning Alignment
Measures how well the agent's reasoning matches expected audit logic.
- Evaluates step-by-step decision paths
- Ensures audit decisions are justifiable

### Decision Accuracy
Measures correctness of final audit decisions.
- Binary metric: correct/incorrect
- Most important for operational compliance

## Integration with LangGraph/LangChain

The evaluation module is designed to work with:
- **LangGraph** for multi-agent orchestration
- **LangChain** for document processing and retrieval
- **ChromaDB** for vector storage
- **Sentence-Transformers** for embeddings
- **Cross-Encoders** for reranking

## Testing

Run the synthetic cases generator to verify installation:

```bash
python eval/synthetic_cases.py
```

Expected output: 12 synthetic audit cases with full details.

## Future Enhancements

- Add more synthetic cases for edge cases
- Implement automated agent reasoning extraction
- Add benchmarking against baseline systems
- Create visualization dashboard for results
- Add integration with continuous evaluation pipelines
