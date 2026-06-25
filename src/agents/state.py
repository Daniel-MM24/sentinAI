"""Agent state definitions for LangGraph orchestration."""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ComplianceEvaluation(BaseModel):
    """Structured compliance evaluation output from the Auditor agent."""

    is_compliant: bool = Field(
        ..., description="Whether the query/data passes compliance checks."
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Numerical risk score (0.0 = safe, 1.0 = high risk)."
    )
    justification: str = Field(
        ..., description="Textual justification for the compliance decision."
    )
    flagged_sections: List[str] = Field(
        default_factory=list,
        description="List of specific sections or data points that were flagged.",
    )


class AnalystReport(BaseModel):
    """Final synthesized report from the Analyst agent."""

    summary: str = Field(..., description="Executive summary of the audit findings.")
    compliance_status: str = Field(..., description="Overall compliance status.")
    risk_assessment: str = Field(..., description="Risk assessment narrative.")
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="List of recommended actions based on findings.",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Source IDs referenced in the report.",
    )
    escalated: bool = Field(
        default=False,
        description="Whether the audit was escalated to human review.",
    )


class AgentState(TypedDict):
    """State for the multi-agent audit workflow."""

    messages: Annotated[List, add_messages]
    retrieved_context: List[str]
    evaluation: Optional[ComplianceEvaluation]
    requires_escalation: bool
    final_report: Optional[AnalystReport]
    calculation_metrics: Optional[Dict[str, Any]]
