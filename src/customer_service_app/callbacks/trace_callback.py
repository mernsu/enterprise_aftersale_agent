from __future__ import annotations

import time
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from customer_service_app.domain.schemas import ChatTraceStep


class CustomerServiceTraceCallback(AsyncCallbackHandler):
    """LangChain callback that records agent steps as ChatTraceStep entries.

    Attach this handler via ``config["callbacks"]`` and read ``trace`` after
    the graph run completes.
    """

    def __init__(self) -> None:
        self.trace: list[ChatTraceStep] = []
        self._tool_starts: dict[str, float] = {}

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        pass

    async def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> None:
        generations = response.generations
        tool_calls_count = 0
        if generations and generations[0]:
            msg = generations[0][0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_count = len(msg.tool_calls)

        self.trace.append(
            ChatTraceStep(
                stage="llm_decision",
                detail="模型完成回答或工具调用决策",
                metadata={"tool_calls_count": tool_calls_count},
            )
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._tool_starts[run_id] = time.monotonic()

    async def on_tool_end(
        self,
        output: Any,
        **kwargs: Any,
    ) -> None:
        self.trace.append(
            ChatTraceStep(
                stage="tools",
                detail="已执行模型请求的工具",
                metadata={"tool_result": str(output)[:200]},
            )
        )

    async def on_retriever_end(
        self,
        documents: list[Any],
        **kwargs: Any,
    ) -> None:
        self.trace.append(
            ChatTraceStep(
                stage="rag",
                detail="完成知识库检索",
                metadata={"chunk_count": len(documents)},
            )
        )
