"""
Evaluation module for SentinAI agentic RAG system.

This module provides tools for evaluating the system's ability to identify,
cite, and justify compliance-related audit decisions against ground-truth datasets.
"""

from eval.synthetic_cases import (
    SyntheticAuditCase,
    TransactionLog,
    generate_synthetic_cases,
    get_case_by_id,
    get_all_cases
)

from eval.evaluate import (
    EvaluationMetric,
    EvaluationResult,
    OverallEvaluation,
    CitationMatcher,
    ReasoningEvaluator,
    RAGEvaluator
)

__all__ = [
    "SyntheticAuditCase",
    "TransactionLog",
    "generate_synthetic_cases",
    "get_case_by_id",
    "get_all_cases",
    "EvaluationMetric",
    "EvaluationResult",
    "OverallEvaluation",
    "CitationMatcher",
    "ReasoningEvaluator",
    "RAGEvaluator"
]
