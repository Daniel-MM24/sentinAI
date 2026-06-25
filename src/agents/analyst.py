"""Analyst agent for final report synthesis and recommendations."""
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from src.agents.base import BaseAgent
from src.agents.prompts import get_prompt
from src.agents.state import AgentState, AnalystReport

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Agent responsible for synthesizing final audit reports."""

    def __init__(self, llm=None, **kwargs):
        system_prompt = get_prompt("analyst")
        super().__init__(name="analyst", llm=llm, system_prompt=system_prompt, **kwargs)

    def run(self, state: AgentState) -> Dict[str, Any]:
        """Execute analysis: synthesize evaluation and context into final report."""
        evaluation = state.get("evaluation")
        retrieved_context = state.get("retrieved_context", [])
        requires_escalation = state.get("requires_escalation", False)

        logger.info("Analyst agent synthesizing report (escalation=%s)", requires_escalation)

        if evaluation is None:
            logger.warning("No evaluation available, generating fallback report")
            report = AnalystReport(
                summary="Audit could not be completed - no evaluation was produced.",
                compliance_status="UNKNOWN",
                risk_assessment="Unable to assess risk due to missing evaluation.",
                recommended_actions=["Re-run audit; no evaluation was produced."],
                citations=[],
                escalated=True,
            )
        else:
            # Build context string for the LLM
            context_str = "\n\n".join(retrieved_context) if retrieved_context else "No context retrieved."
            eval_str = (
                f"Compliance: {'PASS' if evaluation.is_compliant else 'FAIL'}\n"
                f"Risk Score: {evaluation.risk_score:.2f}\n"
                f"Justification: {evaluation.justification}\n"
                f"Flagged Sections: {', '.join(evaluation.flagged_sections)}"
            )

            messages = self._build_messages(
                state,
                additional_context=f"Evaluation:\n{eval_str}\n\nContext:\n{context_str}",
            )

            try:
                report = self._invoke_llm_structured(messages, AnalystReport)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Report synthesis failed: %s", exc)
                report = AnalystReport(
                    summary=f"Report synthesis error: {exc}",
                    compliance_status="ERROR",
                    risk_assessment=f"Error during analysis: {exc}",
                    recommended_actions=["Review error logs and retry audit."],
                    citations=[],
                    escalated=requires_escalation,
                )

        # Ensure escalation flag is set correctly
        report.escalated = requires_escalation

        logger.info("Analyst report generated: %s", report.summary[:100])

        return {
            "final_report": report,
            "messages": [
                AIMessage(
                    content=f"Final report generated: {report.compliance_status} (escalated={report.escalated})",
                    name=self.name,
                )
            ],
        }
