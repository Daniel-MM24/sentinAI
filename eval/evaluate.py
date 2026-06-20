"""
Evaluation Harness for Agentic RAG System

This module implements evaluation metrics to test the Agent's ability to retrieve
correct statutes and justify findings against the ground-truth dataset.

Metrics computed:
- Precision@k: Precision at rank k for retrieved documents
- MRR (Mean Reciprocal Rank): Average reciprocal rank of first relevant document
- Citation Fidelity: Accuracy of cited regulatory sections
- Reasoning Alignment: Alignment between expected and actual reasoning paths
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from eval.synthetic_cases import SyntheticAuditCase, get_all_cases
from src.retrieval.vector_db import VectorStore
from src.retrieval.ingestion import DocumentIngestor
from src.retrieval.schemas import SearchResult

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Evaluation metrics for RAG system performance."""
    PRECISION_AT_K = "precision_at_k"
    MRR = "mean_reciprocal_rank"
    CITATION_FIDELITY = "citation_fidelity"
    REASONING_ALIGNMENT = "reasoning_alignment"


@dataclass
class EvaluationResult:
    """Result of evaluating a single synthetic case."""
    case_id: str
    scenario_category: str
    precision_at_k: float
    reciprocal_rank: float
    citation_match: bool
    citation_similarity: float
    reasoning_alignment_score: float
    retrieved_documents: List[SearchResult]
    target_citation: str
    retrieved_citations: List[str]
    expected_decision: str
    actual_decision: str
    decision_match: bool


@dataclass
class OverallEvaluation:
    """Overall evaluation results across all test cases."""
    total_cases: int
    avg_precision_at_1: float
    avg_precision_at_3: float
    avg_precision_at_5: float
    mean_reciprocal_rank: float
    citation_fidelity: float
    reasoning_alignment: float
    decision_accuracy: float
    per_category_results: Dict[str, Dict[str, float]]
    individual_results: List[EvaluationResult]


class CitationMatcher:
    """Matches retrieved documents to target citations."""
    
    @staticmethod
    def normalize_citation(citation: str) -> str:
        """Normalize citation string for comparison."""
        citation = citation.lower().strip()
        # Remove common variations
        citation = citation.replace("section", "sec")
        citation = citation.replace("clause", "cl")
        citation = citation.replace("regulation", "reg")
        citation = citation.replace("guideline", "guide")
        return citation
    
    @staticmethod
    def extract_key_terms(citation: str) -> List[str]:
        """Extract key terms from citation."""
        citation = citation.lower()
        # Look for act names, section numbers, etc.
        terms = []
        
        # Common Kenyan regulatory terms
        if "pocamla" in citation:
            terms.append("pocamla")
        if "dpa" in citation or "data protection" in citation:
            terms.append("dpa")
        if "cbk" in citation:
            terms.append("cbk")
        if "frc" in citation:
            terms.append("frc")
        
        # Extract section numbers
        import re
        sections = re.findall(r'section\s*(\d+)', citation)
        terms.extend([f"sec{sec}" for sec in sections])
        
        return terms
    
    @staticmethod
    def calculate_similarity(target: str, retrieved: str) -> float:
        """Calculate similarity between target and retrieved citation."""
        target_norm = CitationMatcher.normalize_citation(target)
        retrieved_norm = CitationMatcher.normalize_citation(retrieved)
        
        target_terms = set(CitationMatcher.extract_key_terms(target))
        retrieved_terms = set(CitationMatcher.extract_key_terms(retrieved))
        
        if not target_terms:
            return 0.0
        
        # Jaccard similarity
        intersection = target_terms & retrieved_terms
        union = target_terms | retrieved_terms
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)


class ReasoningEvaluator:
    """Evaluates alignment between expected and actual reasoning."""
    
    @staticmethod
    def calculate_alignment(expected: List[str], actual: List[str]) -> float:
        """Calculate alignment score between expected and actual reasoning steps."""
        if not expected:
            return 1.0 if not actual else 0.0
        
        expected_lower = [step.lower() for step in expected]
        actual_lower = [step.lower() for step in actual]
        
        # Count how many expected reasoning steps are covered in actual reasoning
        covered_steps = 0
        for exp_step in expected_lower:
            for act_step in actual_lower:
                # Simple keyword matching for reasoning alignment
                if any(word in act_step for word in exp_step.split()):
                    covered_steps += 1
                    break
        
        return covered_steps / len(expected_lower)


class RAGEvaluator:
    """Main evaluator for the agentic RAG system."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        document_ingestor: DocumentIngestor,
        k_values: List[int] = [1, 3, 5]
    ):
        self.vector_store = vector_store
        self.document_ingestor = document_ingestor
        self.k_values = k_values
        self.citation_matcher = CitationMatcher()
        self.reasoning_evaluator = ReasoningEvaluator()
        
    async def evaluate_single_case(
        self,
        case: SyntheticAuditCase,
        query: str,
        actual_reasoning: Optional[List[str]] = None,
        actual_decision: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a single synthetic audit case."""
        
        # Perform hybrid search
        retrieved_docs = await self.vector_store.hybrid_search(
            query=query,
            k=max(self.k_values)
        )
        
        # Calculate Precision@k for each k
        precision_scores = {}
        for k in self.k_values:
            precision_scores[f"precision_at_{k}"] = self._calculate_precision_at_k(
                retrieved_docs[:k],
                case.target_citation
            )
        
        # Calculate Reciprocal Rank
        reciprocal_rank = self._calculate_reciprocal_rank(
            retrieved_docs,
            case.target_citation
        )
        
        # Calculate Citation Fidelity
        retrieved_citations = [doc.content for doc in retrieved_docs]
        citation_similarities = [
            self.citation_matcher.calculate_similarity(case.target_citation, citation)
            for citation in retrieved_citations
        ]
        citation_match = max(citation_similarities) > 0.7  # Threshold for match
        citation_similarity = max(citation_similarities)
        
        # Calculate Reasoning Alignment
        if actual_reasoning:
            reasoning_alignment = self.reasoning_evaluator.calculate_alignment(
                case.expected_reasoning,
                actual_reasoning
            )
        else:
            reasoning_alignment = 0.0  # No actual reasoning provided
        
        # Decision Match
        if actual_decision:
            decision_match = case.expected_decision == actual_decision
        else:
            decision_match = False  # No actual decision provided
        
        return EvaluationResult(
            case_id=case.case_id,
            scenario_category=case.scenario_category,
            precision_at_k=precision_scores.get("precision_at_5", 0.0),
            reciprocal_rank=reciprocal_rank,
            citation_match=citation_match,
            citation_similarity=citation_similarity,
            reasoning_alignment_score=reasoning_alignment,
            retrieved_documents=retrieved_docs,
            target_citation=case.target_citation,
            retrieved_citations=retrieved_citations,
            expected_decision=case.expected_decision,
            actual_decision=actual_decision or "N/A",
            decision_match=decision_match
        )
    
    def _calculate_precision_at_k(
        self,
        retrieved_docs: List[SearchResult],
        target_citation: str,
        threshold: float = 0.7
    ) -> float:
        """Calculate Precision@k."""
        if not retrieved_docs:
            return 0.0
        
        relevant_count = 0
        for doc in retrieved_docs:
            similarity = self.citation_matcher.calculate_similarity(
                target_citation,
                doc.content
            )
            if similarity >= threshold:
                relevant_count += 1
        
        return relevant_count / len(retrieved_docs)
    
    def _calculate_reciprocal_rank(
        self,
        retrieved_docs: List[SearchResult],
        target_citation: str,
        threshold: float = 0.7
    ) -> float:
        """Calculate Reciprocal Rank."""
        for rank, doc in enumerate(retrieved_docs, start=1):
            similarity = self.citation_matcher.calculate_similarity(
                target_citation,
                doc.content
            )
            if similarity >= threshold:
                return 1.0 / rank
        
        return 0.0
    
    async def evaluate_all_cases(
        self,
        cases: Optional[List[SyntheticAuditCase]] = None,
        query_builder: Optional[callable] = None
    ) -> OverallEvaluation:
        """Evaluate all synthetic audit cases."""
        
        if cases is None:
            cases = get_all_cases()
        
        individual_results = []
        per_category_results: Dict[str, List[EvaluationResult]] = {}
        
        for case in cases:
            # Build query from case description
            if query_builder:
                query = query_builder(case)
            else:
                query = f"{case.scenario_category}: {case.description}"
            
            # Evaluate single case
            result = await self.evaluate_single_case(case, query)
            individual_results.append(result)
            
            # Group by category
            if case.scenario_category not in per_category_results:
                per_category_results[case.scenario_category] = []
            per_category_results[case.scenario_category].append(result)
        
        # Calculate overall metrics
        total_cases = len(individual_results)
        
        # Average Precision@k
        avg_precision_at_1 = sum(r.precision_at_k for r in individual_results) / total_cases
        avg_precision_at_3 = sum(r.precision_at_k for r in individual_results) / total_cases
        avg_precision_at_5 = sum(r.precision_at_k for r in individual_results) / total_cases
        
        # Mean Reciprocal Rank
        mean_reciprocal_rank = sum(r.reciprocal_rank for r in individual_results) / total_cases
        
        # Citation Fidelity
        citation_fidelity = sum(r.citation_similarity for r in individual_results) / total_cases
        
        # Reasoning Alignment
        reasoning_alignment = sum(r.reasoning_alignment_score for r in individual_results) / total_cases
        
        # Decision Accuracy
        decision_accuracy = sum(1 for r in individual_results if r.decision_match) / total_cases
        
        # Per-category results
        category_metrics = {}
        for category, results in per_category_results.items():
            category_metrics[category] = {
                "avg_precision": sum(r.precision_at_k for r in results) / len(results),
                "avg_rr": sum(r.reciprocal_rank for r in results) / len(results),
                "citation_fidelity": sum(r.citation_similarity for r in results) / len(results),
                "decision_accuracy": sum(1 for r in results if r.decision_match) / len(results)
            }
        
        return OverallEvaluation(
            total_cases=total_cases,
            avg_precision_at_1=avg_precision_at_1,
            avg_precision_at_3=avg_precision_at_3,
            avg_precision_at_5=avg_precision_at_5,
            mean_reciprocal_rank=mean_reciprocal_rank,
            citation_fidelity=citation_fidelity,
            reasoning_alignment=reasoning_alignment,
            decision_accuracy=decision_accuracy,
            per_category_results=category_metrics,
            individual_results=individual_results
        )
    
    def print_evaluation_report(self, evaluation: OverallEvaluation):
        """Print a comprehensive evaluation report."""
        print("=" * 80)
        print("RAG SYSTEM EVALUATION REPORT")
        print("=" * 80)
        print(f"\nTotal Cases Evaluated: {evaluation.total_cases}")
        print("\n" + "-" * 80)
        print("OVERALL METRICS")
        print("-" * 80)
        print(f"Precision@1:  {evaluation.avg_precision_at_1:.4f}")
        print(f"Precision@3:  {evaluation.avg_precision_at_3:.4f}")
        print(f"Precision@5:  {evaluation.avg_precision_at_5:.4f}")
        print(f"MRR:          {evaluation.mean_reciprocal_rank:.4f}")
        print(f"Citation Fidelity:  {evaluation.citation_fidelity:.4f}")
        print(f"Reasoning Alignment: {evaluation.reasoning_alignment:.4f}")
        print(f"Decision Accuracy:   {evaluation.decision_accuracy:.4f}")
        
        print("\n" + "-" * 80)
        print("PER-CATEGORY RESULTS")
        print("-" * 80)
        for category, metrics in evaluation.per_category_results.items():
            print(f"\n{category}:")
            print(f"  Avg Precision:  {metrics['avg_precision']:.4f}")
            print(f"  Avg RR:         {metrics['avg_rr']:.4f}")
            print(f"  Citation Fidelity:  {metrics['citation_fidelity']:.4f}")
            print(f"  Decision Accuracy:   {metrics['decision_accuracy']:.4f}")
        
        print("\n" + "-" * 80)
        print("INDIVIDUAL CASE RESULTS")
        print("-" * 80)
        for result in evaluation.individual_results:
            print(f"\nCase {result.case_id} ({result.scenario_category}):")
            print(f"  Precision@5:  {result.precision_at_k:.4f}")
            print(f"  Reciprocal Rank: {result.reciprocal_rank:.4f}")
            print(f"  Citation Match: {result.citation_match}")
            print(f"  Citation Similarity: {result.citation_similarity:.4f}")
            print(f"  Reasoning Alignment: {result.reasoning_alignment_score:.4f}")
            print(f"  Decision Match: {result.decision_match}")
            print(f"  Expected: {result.expected_decision}")
            print(f"  Actual: {result.actual_decision}")
        
        print("\n" + "=" * 80)


async def main():
    """Main evaluation function."""
    import sys
    sys.path.append('/home/dan/project/sentinAI')
    
    # Initialize components
    vector_store = VectorStore()
    document_ingestor = DocumentIngestor()
    
    # Create evaluator
    evaluator = RAGEvaluator(vector_store, document_ingestor)
    
    # Load synthetic cases
    cases = get_all_cases()
    print(f"Loaded {len(cases)} synthetic audit cases for evaluation")
    
    # Note: In a real implementation, you would need to:
    # 1. Load and index the Kenyan regulatory corpus
    # 2. Run the evaluation against the indexed corpus
    # 3. Provide actual reasoning and decisions from the agent
    
    # For demonstration, we'll show the structure
    print("\nEvaluation harness ready.")
    print("To run full evaluation:")
    print("1. Index Kenyan regulatory documents using DocumentIngestor")
    print("2. Add documents to VectorStore using add_documents()")
    print("3. Run evaluator.evaluate_all_cases() with agent outputs")
    
    # Example of how it would work:
    # evaluation = await evaluator.evaluate_all_cases()
    # evaluator.print_evaluation_report(evaluation)


if __name__ == "__main__":
    asyncio.run(main())
