from __future__ import annotations

from langchain_openai import ChatOpenAI

from customer_service_app.core.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance from application settings.

    All OpenAI-compatible providers (DeepSeek, Qwen, Claude via gateway, etc.)
    use the same ChatOpenAI class since they share the chat/completions API shape.
    """
    return ChatOpenAI(
        model=settings.require("LLM_MODEL", settings.llm_model),
        api_key=settings.require("LLM_API_KEY", settings.llm_api_key),
        base_url=settings.require("LLM_BASE_URL", settings.llm_base_url),
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
