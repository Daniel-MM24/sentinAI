"""Researcher agent for document retrieval and context gathering."""
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.base import BaseAgent
from src.agents.prompts import get_prompt
from src.agents.state import AgentState
from src.agents.tools import query_vector_db

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Agent responsible for retrieving relevant compliance documents."""

    def __init__(self, llm=None, **kwargs):
        system_prompt = get_prompt("researcher")
        super().__init__(name="researcher", llm=llm, system_prompt=system_prompt, **kwargs)
        self.tools = [query_vector_db]

    def run(self, state: AgentState) -> Dict[str, Any]:
        """Execute research: query vector DB and summarize findings."""
        # Get the latest user query from messages
        query = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "content") and isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            query = str(state.get("messages", [])[-1]) if state.get("messages") else ""

        logger.info("Researcher agent executing with query: %s", query)

        # Query the vector database
        context = ""
        try:
            import asyncio

            context = asyncio.run(query_vector_db.invoke({"query": query, "k": 3}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vector DB query failed: %s", exc)
            context = f"Error retrieving documents: {exc}"

        # Generate a summary note using the LLM
        messages = self._build_messages(
            state,
            additional_context=f"Query: {query}\n\nRetrieved Context:\n{context}",
        )

        note = self._invoke_llm(messages)

        # Update state
        return {
            "retrieved_context": [context] if context else [],
            "messages": [AIMessage(content=note, name=self.name)],
        }
