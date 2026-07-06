from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from customer_service_app.domain.schemas import ChatRequest


class CustomerServiceState(TypedDict):
    """LangGraph state for the customer service agent.

    Graph nodes read from and write to this state.  Non-serializable runtime
    objects (DB session, cache, search client, chat model, etc.) are passed
    through ``RunnableConfig.configurable`` rather than state, keeping state
    JSON-serializable for future checkpointing.
    """

    # ── input ──────────────────────────────────────────────────────────
    request: ChatRequest

    # ── pipeline intermediates ──────────────────────────────────────────
    conversation_id: str
    knowledge: list[dict[str, Any]]  # serialized KnowledgeChunk
    trace: list[dict[str, Any]]  # serialized ChatTraceStep
    cache_hit: bool

    # ── agent loop ─────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int

    # ── output ─────────────────────────────────────────────────────────
    answer: str
    error: NotRequired[str]
