from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ExternalServiceError
from customer_service_app.domain.schemas import KnowledgeChunk


class TenantAwareQdrantStore:
    """Wraps langchain_qdrant.QdrantVectorStore with multi-tenant filtering.

    Every search is scoped to a tenant_id via a Qdrant payload filter,
    matching the tenancy model in the existing QdrantKnowledgeVectorStore.
    """

    def __init__(self, settings: Settings, embedding: Embeddings) -> None:
        self._settings = settings
        self._embedding = embedding
        self._client: AsyncQdrantClient | None = None
        self._store: QdrantVectorStore | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            kwargs: dict = {"url": self._settings.qdrant_url, "timeout": 30}
            if self._settings.qdrant_api_key:
                kwargs["api_key"] = self._settings.qdrant_api_key
            self._client = AsyncQdrantClient(**kwargs)
        return self._client

    @property
    def store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self.client,
                collection_name=self._settings.qdrant_collection,
                embedding=self._embedding,
            )
        return self._store

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        try:
            collections = await self.client.get_collections()
            names = {c.name for c in collections.collections}
            if self._settings.qdrant_collection not in names:
                await self.client.create_collection(
                    collection_name=self._settings.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self._settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise ExternalServiceError(f"Qdrant collection init failed: {exc}") from exc

    def as_tenant_retriever(
        self,
        tenant_id: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> BaseRetriever:
        """Return a retriever scoped to the given tenant.

        Uses a Qdrant payload filter so that only chunks belonging to
        *tenant_id* are considered during similarity search.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        top_k = top_k or self._settings.rag_top_k
        score_threshold = score_threshold or self._settings.rag_score_threshold

        qdrant_filter = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )

        retriever = self.store.as_retriever(
            search_kwargs={
                "k": top_k,
                "score_threshold": score_threshold,
                "filter": qdrant_filter,
            }
        )
        return retriever

    @staticmethod
    def docs_to_knowledge_chunks(docs: list[Document]) -> list[KnowledgeChunk]:
        """Convert LangChain Documents to the domain KnowledgeChunk model."""
        chunks: list[KnowledgeChunk] = []
        for doc in docs:
            metadata = doc.metadata or {}
            chunks.append(
                KnowledgeChunk(
                    id=metadata.get("id", metadata.get("chunk_id", "")),
                    source=metadata.get("source", ""),
                    title=metadata.get("title", ""),
                    content=doc.page_content,
                    score=metadata.get("score", metadata.get("_score", 0.0)),
                    metadata=metadata,
                )
            )
        return chunks
