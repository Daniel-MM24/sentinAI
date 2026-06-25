"""Base agent abstraction for LangGraph nodes."""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from pydantic import SecretStr

from src.agents.state import AgentState

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseModel")


def content_to_text(content: Any) -> str:
    """Extract text content from various LLM response formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return str(content)


class BaseAgent(ABC):
    """Abstract base class for all agents in the SentinAI workflow."""

    def __init__(
        self,
        name: str,
        llm: Optional[BaseChatModel] = None,
        system_prompt: str = "",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        use_openrouter: bool = False,
        openrouter_headers: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature

        if llm is None:
            from langchain_openai import ChatOpenAI

            # OpenRouter-specific configuration
            if use_openrouter:
                openrouter_base_url = base_url or "https://openrouter.ai/api/v1"
                openrouter_api_key = api_key or SecretStr("")
                
                # Default OpenRouter headers
                default_headers = {
                    "X-Title": "SentinAI",
                    "HTTP-Referer": "https://github.com/Daniel-MM24/sentinAI",
                }
                
                # Merge with custom headers if provided
                if openrouter_headers:
                    default_headers.update(openrouter_headers)
                
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=openrouter_api_key,
                    base_url=openrouter_base_url,
                    default_headers=default_headers,
                )
            else:
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=SecretStr(api_key) if api_key else None,
                    base_url=base_url,
                )
        else:
            self.llm = llm

    @abstractmethod
    def run(self, state: AgentState) -> Dict[str, Any]:
        """Execute the agent's logic and return state updates."""
        ...

    def _invoke_llm(self, messages: List[BaseMessage]) -> str:
        """Invoke the LLM and return its response as plain text."""
        response = self.llm.invoke(list(messages))
        return content_to_text(response.content)

    def _invoke_llm_structured(
        self, messages: List[BaseMessage], model_cls: type[T]
    ) -> T:
        """Invoke the LLM and parse the response as a structured Pydantic model using native structured output."""
        structured_llm = self.llm.with_structured_output(model_cls)
        result = structured_llm.invoke(messages)
        return result

    def _build_messages(
        self, state: AgentState, additional_context: Optional[str] = None
    ) -> List[BaseMessage]:
        """Build the message list for LLM invocation."""
        messages: List[BaseMessage] = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Add conversation history
        for msg in state.get("messages", []):
            messages.append(msg)

        # Add additional context if provided
        if additional_context:
            messages.append(HumanMessage(content=additional_context))

        return messages

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Execute the agent with error handling."""
        try:
            return self.run(state)
        except Exception as exc:  # noqa: BLE001 - node-level safety net
            logger.exception("Agent '%s' failed: %s", self.name, exc)
            raise
