from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_service_app.core.config import get_settings  # noqa: E402
from customer_service_app.domain.schemas import KnowledgeChunk  # noqa: E402
from customer_service_app.infrastructure.embeddings.langchain_factory import (  # noqa: E402
    build_embedding_model,
)
from customer_service_app.infrastructure.vector_store.langchain_store import (  # noqa: E402
    TenantAwareQdrantStore,
)


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """把长文档切成适合 embedding 的小块。"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, 0)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


async def ingest(directory: Path, tenant_id: str) -> None:
    """导入 Markdown 知识库到 Qdrant。

    流程：读取 md → 切 chunk → 调 embedding → 写入 Qdrant。
    这就是 RAG 项目里"离线知识入库"的最小生产流程。
    """
    settings = get_settings()
    embedding_model = build_embedding_model(settings)
    qdrant_store = TenantAwareQdrantStore(settings, embedding_model)

    chunks: list[KnowledgeChunk] = []
    for path in directory.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for index, chunk_text in enumerate(split_text(text)):
            raw_id = f"{tenant_id}:{path}:{index}"
            chunks.append(
                KnowledgeChunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id)),
                    source=str(path),
                    title=path.stem,
                    content=chunk_text,
                    score=1.0,
                    metadata={"chunk_index": index},
                )
            )

    if not chunks:
        print("No markdown documents found.")
        return

    await qdrant_store.ensure_collection()
    vectors = await embedding_model.aembed_documents([chunk.content for chunk in chunks])

    # Build Qdrant points and upsert via the async client.
    from qdrant_client.models import PointStruct

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "tenant_id": tenant_id,
            "source": chunk.source,
            "title": chunk.title,
            "content": chunk.content,
            "score": chunk.score,
            "chunk_id": chunk.id,
            **chunk.metadata,
        }
        points.append(PointStruct(id=chunk.id, vector=vector, payload=payload))

    await qdrant_store.client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )
    print(f"Ingested {len(chunks)} chunks into Qdrant collection {settings.qdrant_collection}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tenant-id", default="default")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(ingest(args.directory, args.tenant_id))
