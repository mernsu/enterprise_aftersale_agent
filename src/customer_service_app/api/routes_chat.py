from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage

from customer_service_app.api.dependencies import AgentContext, get_langgraph_agent
from customer_service_app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTraceStep,
    KnowledgeChunk,
    ToolCallView,
)
from customer_service_app.services.container import build_config

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ctx: AgentContext = Depends(get_langgraph_agent),
) -> ChatResponse:
    """处理一轮客服问答：预处理 → 缓存 → RAG → LLM → 工具 → 保存 → 返回。"""
    raw_confirmed = request.metadata.get("confirmed_tools", [])
    if isinstance(raw_confirmed, str):
        raw_confirmed = [raw_confirmed]
    confirmed_tools = {str(t) for t in raw_confirmed if str(t).strip()}

    config = build_config(
        ctx.session,
        ctx.runtime,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        confirmed_tools=confirmed_tools,
    )

    state = await ctx.runtime.graph.ainvoke(
        {"request": request, "iteration_count": 0, "messages": []},
        config,
    )
    await ctx.session.commit()
    return _state_to_response(state)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    ctx: AgentContext = Depends(get_langgraph_agent),
) -> StreamingResponse:
    """SSE 流式接口，通过 LangGraph astream_events 实现 token 级流式输出。"""
    raw_confirmed = request.metadata.get("confirmed_tools", [])
    if isinstance(raw_confirmed, str):
        raw_confirmed = [raw_confirmed]
    confirmed_tools = {str(t) for t in raw_confirmed if str(t).strip()}

    config = build_config(
        ctx.session,
        ctx.runtime,
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        confirmed_tools=confirmed_tools,
    )

    async def event_stream() -> AsyncIterator[str]:
        async for event in ctx.runtime.graph.astream_events(
            {"request": request, "iteration_count": 0, "messages": []},
            config,
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield _sse("token", {"content": chunk.content})
            elif kind == "on_chat_model_end":
                msg = event["data"]["output"]
                if isinstance(msg, AIMessage):
                    yield _sse("llm_done", {
                        "content": msg.content,
                        "tool_calls": bool(msg.tool_calls),
                    })
            elif kind == "on_tool_start":
                yield _sse("tool_start", {
                    "name": event["name"],
                    "input": event["data"].get("input"),
                })
            elif kind == "on_tool_end":
                yield _sse("tool_result", {
                    "name": event["name"],
                    "output": str(event["data"]["output"]),
                })

        await ctx.session.commit()
        yield _sse("done", {"status": "complete"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _state_to_response(state: dict[str, Any]) -> ChatResponse:
    """Convert final graph state into a ChatResponse."""
    messages = state.get("messages", [])
    knowledge = [KnowledgeChunk(**c) for c in state.get("knowledge", [])]
    trace = [ChatTraceStep(**s) for s in state.get("trace", [])]

    tool_calls: list[ToolCallView] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCallView(
                    id=tc["id"], name=tc["name"], arguments=tc.get("args", {}),
                ))

    return ChatResponse(
        conversation_id=state.get("conversation_id", ""),
        answer=state.get("answer", ""),
        cache_hit=state.get("cache_hit", False),
        knowledge=knowledge,
        tool_calls=tool_calls,
        tool_results=[],
        trace=trace,
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
