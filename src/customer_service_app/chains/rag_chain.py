from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from customer_service_app.core.config import Settings
from customer_service_app.domain.schemas import KnowledgeChunk
from customer_service_app.infrastructure.vector_store.langchain_store import TenantAwareQdrantStore
from customer_service_app.prompts.customer_service import format_knowledge_context


def build_rag_chain(
    settings: Settings,
    embedding_model: Embeddings,
    qdrant_store: TenantAwareQdrantStore,
):
    """Build an LCEL chain that retrieves knowledge for a question.

    Input dict: {"question": str, "tenant_id": str}

    Returns a formatted string suitable for insertion into the system prompt.
    """

    async def _retrieve(inputs: dict[str, Any]) -> list[KnowledgeChunk]:
        question: str = inputs["question"]
        tenant_id: str = inputs["tenant_id"]

        if not settings.rag_enabled:
            return []

        await qdrant_store.ensure_collection()

        retriever = qdrant_store.as_tenant_retriever(
            tenant_id,
            top_k=settings.rag_top_k,
            score_threshold=settings.rag_score_threshold,
        )
        docs = await retriever.ainvoke(question)
        return qdrant_store.docs_to_knowledge_chunks(docs)

    def _format(chunks: list[KnowledgeChunk]) -> str:
        return format_knowledge_context([c.model_dump() for c in chunks])

    chain = (
        RunnablePassthrough.assign(
            knowledge=RunnableLambda(_retrieve),
        )
        | RunnablePassthrough.assign(
            knowledge_context=lambda d: _format(d["knowledge"]),
        )
    )
    return chain
