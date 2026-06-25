"""Auditor agent for compliance evaluation and risk assessment."""
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from src.agents.base import BaseAgent
from src.agents.prompts import get_prompt
from src.agents.state import AgentState, ComplianceEvaluation
from src.core.config import settings

logger = logging.getLogger(__name__)


class AuditorAgent(BaseAgent):
    """Agent responsible for evaluating compliance and assigning risk scores."""

    def __init__(self, llm=None, **kwargs):
        system_prompt = get_prompt("auditor")
        super().__init__(name="auditor", llm=llm, system_prompt=system_prompt, **kwargs)

    def run(self, state: AgentState) -> Dict[str, Any]:
        """Execute audit: evaluate retrieved context against compliance criteria."""
        retrieved_context = state.get("retrieved_context", [])
        context_str = "\n\n".join(retrieved_context) if retrieved_context else "No context retrieved."

        logger.info("Auditor agent evaluating context with %d items", len(retrieved_context))

        # Build messages for structured output
        messages = self._build_messages(
            state,
            additional_context=f"Retrieved Context:\n{context_str}",
        )

        # Get structured evaluation
        try:
            evaluation = self._invoke_llm_structured(messages, ComplianceEvaluation)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Structured evaluation failed: %s", exc)
            # Fallback to a default evaluation
            evaluation = ComplianceEvaluation(
                is_compliant=False,
                risk_score=0.8,
                justification=f"Evaluation error: {exc}",
                flagged_sections=["evaluation_error"],
            )

        # Determine escalation based on risk score
        escalation_threshold = getattr(settings, "RISK_ESCALATION_THRESHOLD", 0.7)
        requires_escalation = evaluation.risk_score >= escalation_threshold

        logger.info(
            "Audit complete: compliant=%s, risk=%.2f, escalate=%s",
            evaluation.is_compliant,
            evaluation.risk_score,
            requires_escalation,
        )

        return {
            "evaluation": evaluation,
            "requires_escalation": requires_escalation,
            "messages": [
                AIMessage(
                    content=f"Compliance evaluation: {'PASS' if evaluation.is_compliant else 'FAIL'} "
                    f"(risk score: {evaluation.risk_score:.2f})",
                    name=self.name,
                )
            ],
        }
