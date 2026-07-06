from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from customer_service_app.core.config import Settings


def build_embedding_model(settings: Settings) -> Embeddings:
    """Create a LangChain Embeddings instance from application settings.

    Returns OllamaEmbeddings or OpenAIEmbeddings based on EMBEDDING_PROVIDER.
    AliCloud DashScope / Bailian providers default to OpenAIEmbeddings since
    they expose an OpenAI-compatible embeddings endpoint.
    """
    provider = settings.embedding_provider

    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.require("EMBEDDING_MODEL", settings.embedding_model),
            base_url=settings.require("EMBEDDING_BASE_URL", settings.embedding_base_url),
        )

    # All other providers use the OpenAI-compatible embeddings API.
    # This covers: openai_compatible, openai, deepseek, dashscope, bailian, aliyun
    kwargs: dict = {
        "model": settings.require("EMBEDDING_MODEL", settings.embedding_model),
        "base_url": settings.require("EMBEDDING_BASE_URL", settings.embedding_base_url),
        "timeout": settings.embedding_timeout_seconds,
    }

    api_key = settings.embedding_api_key or settings.llm_api_key
    if api_key:
        kwargs["api_key"] = api_key

    if provider in ("dashscope", "bailian", "aliyun") or str(
        settings.embedding_model
    ).startswith("text-embedding-v4"):
        kwargs["dimensions"] = settings.embedding_dimension

    return OpenAIEmbeddings(**kwargs)
