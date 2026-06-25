"""LangGraph orchestration for the multi-agent audit workflow."""
import logging
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, Optional, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END

from src.agents.analyst import AnalystAgent
from src.agents.auditor import AuditorAgent
from src.agents.researcher import ResearcherAgent
from src.agents.state import AgentState
from src.agents.tools import set_vector_store
from src.retrieval.vector_db import VectorStore

logger = logging.getLogger(__name__)


class AuditGraph:
    """Orchestrates the multi-agent compliance audit workflow using LangGraph."""

    def __init__(
        self,
        researcher: ResearcherAgent,
        auditor: AuditorAgent,
        analyst: AnalystAgent,
        checkpointer: Optional[BaseCheckpointSaver[Any]] = None,
    ):
        self.researcher = researcher
        self.auditor = auditor
        self.analyst = analyst
        self.checkpointer = checkpointer

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph with nodes and edges."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("researcher", self.researcher)
        workflow.add_node("auditor", self.auditor)
        workflow.add_node("analyst", self.analyst)

        # Define edges
        workflow.add_edge(START, "researcher")
        workflow.add_edge("researcher", "auditor")
        workflow.add_conditional_edges(
            "auditor",
            self._route_after_audit,
            {
                "analyst": "analyst",
                "escalate": END,
            },
        )
        workflow.add_edge("analyst", END)

        # Compile with checkpointer if provided
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()

    def _route_after_audit(self, state: AgentState) -> str:
        """Route after audit based on escalation flag."""
        if state.get("requires_escalation", False):
            logger.info("Routing to escalation (human review)")
            return "escalate"
        return "analyst"

    def _initial_state(self, query: str) -> AgentState:
        """Create the initial state for a new audit."""
        return AgentState(
            messages=[HumanMessage(content=query)],
            retrieved_context=[],
            evaluation=None,
            requires_escalation=False,
            final_report=None,
            calculation_metrics={},
        )

    def _config(self, thread_id: str) -> Dict[str, Any]:
        """Create configuration for graph execution."""
        config = {"configurable": {"thread_id": thread_id}}
        return config

    def run(self, query: str, thread_id: Optional[str] = None) -> AgentState:
        """Execute the full workflow synchronously and return the final state."""
        thread_id = thread_id or str(uuid.uuid4())
        result = self.graph.invoke(
            self._initial_state(query), self._config(thread_id)
        )
        return cast(AgentState, result)

    async def arun(self, query: str, thread_id: Optional[str] = None) -> AgentState:
        """Execute the full workflow asynchronously and return the final state."""
        thread_id = thread_id or str(uuid.uuid4())
        result = await self.graph.ainvoke(
            self._initial_state(query), self._config(thread_id)
        )
        return cast(AgentState, result)

    def stream(
        self, query: str, thread_id: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Stream the workflow execution node by node."""
        thread_id = thread_id or str(uuid.uuid4())
        for update in self.graph.stream(
            self._initial_state(query), self._config(thread_id), stream_mode="updates"
        ):
            yield update

    async def astream(
        self, query: str, thread_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream the workflow execution asynchronously node by node."""
        thread_id = thread_id or str(uuid.uuid4())
        async for update in self.graph.astream(
            self._initial_state(query), self._config(thread_id), stream_mode="updates"
        ):
            yield update


def create_audit_graph(
    vector_store: VectorStore,
    llm: Optional[BaseChatModel] = None,
    checkpoint_connection_string: Optional[str] = None,
) -> AuditGraph:
    """Factory function to create an AuditGraph with configured agents."""
    # Set the global vector store for tools
    set_vector_store(vector_store)

    # Create agents
    researcher = ResearcherAgent(llm=llm)
    auditor = AuditorAgent(llm=llm)
    analyst = AnalystAgent(llm=llm)

    # Create PostgresSaver checkpointer if connection string provided
    checkpointer = None
    if checkpoint_connection_string:
        checkpointer = PostgresSaver.from_conn_string(checkpoint_connection_string)

    return AuditGraph(
        researcher=researcher,
        auditor=auditor,
        analyst=analyst,
        checkpointer=checkpointer,
    )
