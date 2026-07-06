from __future__ import annotations

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.agents.customer_service_agent import build_customer_service_graph
from customer_service_app.core.config import get_settings
from customer_service_app.infrastructure.cache.redis_semantic_cache import RedisSemanticCache
from customer_service_app.infrastructure.embeddings.langchain_factory import build_embedding_model
from customer_service_app.infrastructure.llm.langchain_factory import build_chat_model
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient
from customer_service_app.infrastructure.vector_store.langchain_store import TenantAwareQdrantStore
from customer_service_app.tools.confirmation_gated_node import ConfirmationGatedToolNode
from customer_service_app.tools.langchain_tools import (
    CONFIRMATION_REQUIRED_TOOLS,
    create_refund_ticket,
    query_order_status,
    search_public_web,
    transfer_to_human,
)


@dataclass(slots=True)
class AgentRuntime:
    """Holds compiled graph and non-serializable objects for config injection."""

    graph: CompiledStateGraph
    chat_model: object
    embedding_model: object
    qdrant_store: TenantAwareQdrantStore
    semantic_cache: RedisSemanticCache | None
    search_client: SerpApiSearchClient
    tools: list
    tool_node: ConfirmationGatedToolNode


def build_langgraph_agent(session: AsyncSession) -> AgentRuntime:
    """Assemble LangChain components and return the compiled graph + runtime."""
    settings = get_settings()

    chat_model = build_chat_model(settings)
    embedding_model = build_embedding_model(settings)
    qdrant_store = TenantAwareQdrantStore(settings, embedding_model)
    search_client = SerpApiSearchClient(settings)

    semantic_cache: RedisSemanticCache | None = None
    if settings.semantic_cache_enabled:
        semantic_cache = RedisSemanticCache(settings, embedding_model)

    tool_list = [
        query_order_status,
        search_public_web,
        create_refund_ticket,
        transfer_to_human,
    ]
    tool_node = ConfirmationGatedToolNode(
        tool_list,
        confirmation_required=CONFIRMATION_REQUIRED_TOOLS,
    )

    graph = build_customer_service_graph()
    return AgentRuntime(
        graph=graph,
        chat_model=chat_model,
        embedding_model=embedding_model,
        qdrant_store=qdrant_store,
        semantic_cache=semantic_cache,
        search_client=search_client,
        tools=tool_list,
        tool_node=tool_node,
    )


def build_config(
    session: AsyncSession,
    runtime: AgentRuntime,
    *,
    tenant_id: str,
    user_id: str,
    confirmed_tools: set[str] | None = None,
) -> RunnableConfig:
    """Build the RunnableConfig carrying all runtime dependencies."""
    return {
        "configurable": {
            "session": session,
            "settings": get_settings(),
            "chat_model": runtime.chat_model,
            "embedding_model": runtime.embedding_model,
            "qdrant_store": runtime.qdrant_store,
            "semantic_cache": runtime.semantic_cache,
            "search_client": runtime.search_client,
            "tools": runtime.tools,
            "tool_node": runtime.tool_node,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "confirmed_tools": confirmed_tools or set(),
        }
    }
