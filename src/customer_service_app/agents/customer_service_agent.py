from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from customer_service_app.agents.state import CustomerServiceState
from customer_service_app.domain.schemas import ChatTraceStep
from customer_service_app.prompts.customer_service import (
    CUSTOMER_SERVICE_SYSTEM_PROMPT,
    format_knowledge_context,
)
from customer_service_app.services.conversation_service import ConversationService
from customer_service_app.services.question_preprocessor import QuestionPreprocessor

MAX_AGENT_ITERATIONS = 3


# ── node implementations ────────────────────────────────────────────────────


async def preprocess_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Normalize the user question and detect intent before any LLM / DB work."""
    request = state["request"]
    trace = list(state.get("trace", []))

    profile = QuestionPreprocessor().analyze(request.question)
    normalized = profile.normalized_question
    if normalized != request.question:
        request = request.model_copy(update={"question": normalized})

    trace.append(
        ChatTraceStep(
            stage="preprocess",
            detail="完成用户问题标准化和轻量意图识别",
            metadata=profile.as_trace_metadata(),
        ).model_dump()
    )
    return {"request": request, "trace": trace}


async def conversation_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Load or create the conversation, scoped to tenant + user."""
    session = config["configurable"]["session"]
    request = state["request"]
    trace = list(state.get("trace", []))

    service = ConversationService(session)
    conversation = await service.ensure_conversation(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        first_question=request.question,
    )
    trace.append(ChatTraceStep(stage="conversation", detail="会话已加载或创建").model_dump())

    # Also stash conversation_id in config so tool handlers can reach it.
    config["configurable"]["conversation_id"] = conversation.id

    return {"conversation_id": conversation.id, "trace": trace}


async def cache_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Check the Redis semantic cache.  Sets cache_hit and answer on match."""
    request = state["request"]
    session = config["configurable"]["session"]
    semantic_cache = config["configurable"].get("semantic_cache")
    trace = list(state.get("trace", []))

    skip = bool(request.metadata.get("skip_cache"))
    if semantic_cache is None or skip:
        if skip:
            trace.append(ChatTraceStep(stage="cache", detail="本次请求已跳过语义缓存").model_dump())
        return {"cache_hit": False, "trace": trace}

    cached = await semantic_cache.lookup(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        question=request.question,
    )
    if cached is not None:
        trace.append(
            ChatTraceStep(
                stage="cache",
                detail="命中 Redis 语义缓存",
                metadata={"similarity": cached.similarity},
            ).model_dump()
        )
        conversation_id = state.get("conversation_id", "")
        service = ConversationService(session)
        await service.save_turn(
            conversation_id=conversation_id,
            question=request.question,
            answer=cached.answer,
            metadata={"assistant": {"cache_hit": True}},
        )
        await session.commit()
        return {
            "cache_hit": True,
            "answer": cached.answer,
            "trace": trace,
        }

    trace.append(ChatTraceStep(stage="cache", detail="语义缓存未命中").model_dump())
    return {"cache_hit": False, "trace": trace}


async def rag_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Retrieve relevant knowledge chunks from the vector store."""
    request = state["request"]
    settings = config["configurable"]["settings"]
    qdrant_store = config["configurable"]["qdrant_store"]
    trace = list(state.get("trace", []))

    if not settings.rag_enabled:
        return {"knowledge": [], "trace": trace}

    await qdrant_store.ensure_collection()
    retriever = qdrant_store.as_tenant_retriever(
        request.tenant_id,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    docs = await retriever.ainvoke(request.question)
    chunks = qdrant_store.docs_to_knowledge_chunks(docs)

    trace.append(
        ChatTraceStep(
            stage="rag",
            detail="完成知识库检索",
            metadata={"chunk_count": len(chunks)},
        ).model_dump()
    )
    return {
        "knowledge": [c.model_dump() for c in chunks],
        "trace": trace,
    }


async def build_messages_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Assemble the full LLM message list: system prompt + history + knowledge + question."""
    request = state["request"]
    session = config["configurable"]["session"]
    conversation_id = state.get("conversation_id", "")
    knowledge_dicts = state.get("knowledge", [])
    trace = list(state.get("trace", []))

    # Load recent history from DB.
    service = ConversationService(session)
    history = await service.recent_history(conversation_id, limit=12)
    original_count = len(history)

    # Apply deterministic window: keep last 8, compress earlier to one summary.
    keep_recent = 8
    if len(history) <= 12:
        window_messages = history
        compressed_count = 0
    else:
        earlier = history[:-keep_recent]
        recent = history[-keep_recent:]
        summary_lines = ["以下是较早历史对话的简要摘要，用于帮助理解当前上下文："]
        for msg in earlier:
            short = " ".join(msg.content.split())[:80]
            label = {"user": "用户", "assistant": "客服"}.get(msg.role, msg.role)
            if short:
                summary_lines.append(f"- {label}: {short}")
        from customer_service_app.domain.schemas import ChatMessage

        window_messages = [
            ChatMessage(
                role="system",
                content="\n".join(summary_lines),
                metadata={"memory_type": "history_summary", "source_message_count": len(earlier)},
            ),
            *recent,
        ]
        compressed_count = len(earlier)

    trace.append(
        ChatTraceStep(
            stage="memory",
            detail="已整理短期历史消息",
            metadata={
                "original_count": original_count,
                "compressed_count": compressed_count,
                "final_count": len(window_messages),
                "compressed": compressed_count > 0,
            },
        ).model_dump()
    )

    # Build LangChain messages.
    knowledge_context = format_knowledge_context(knowledge_dicts)
    system_prompt = CUSTOMER_SERVICE_SYSTEM_PROMPT.format(
        tenant_id=request.tenant_id,
        knowledge_context=knowledge_context,
    )
    messages: list = [SystemMessage(content=system_prompt)]
    for msg in window_messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))

    messages.append(HumanMessage(content=request.question))
    return {"messages": messages, "trace": trace}


async def agent_llm_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Call the LLM with tools bound.  Appends the AIMessage to state.messages."""
    chat_model = config["configurable"]["chat_model"]
    tools = config["configurable"]["tools"]
    trace = list(state.get("trace", []))

    model_with_tools = chat_model.bind_tools(tools)
    response: AIMessage = await model_with_tools.ainvoke(state["messages"])
    trace.append(
        ChatTraceStep(
            stage="llm_decision",
            detail="模型完成首轮回答或工具调用决策",
            metadata={"tool_calls_count": len(response.tool_calls)},
        ).model_dump()
    )
    return {"messages": [response], "trace": trace}


def should_continue(state: CustomerServiceState) -> Literal["tools", "end"]:
    """Route to tools if the last AIMessage has tool_calls, otherwise finish."""
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        iteration = state.get("iteration_count", 0)
        if iteration >= MAX_AGENT_ITERATIONS:
            return "end"
        return "tools"
    return "end"


async def agent_tools_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Execute tool calls via the confirmation-gated node."""
    tool_node = config["configurable"]["tool_node"]
    messages = list(state.get("messages", []))
    iteration = state.get("iteration_count", 0)

    result = await tool_node.ainvoke({"messages": messages}, config)
    new_messages = result.get("messages", [])
    return {"messages": new_messages, "iteration_count": iteration + 1}


async def save_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Persist the user / assistant turn to MySQL."""
    request = state["request"]
    session = config["configurable"]["session"]
    conversation_id = state.get("conversation_id", "")
    trace = list(state.get("trace", []))

    # Extract final answer from the last AIMessage (skip tool-call-only messages).
    answer = ""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break

    # Collect tool call views for metadata.
    tool_calls_meta = []
    tool_results_meta = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_meta.append({"id": tc["id"], "name": tc["name"], "arguments": tc.get("args", {})})

    service = ConversationService(session)
    await service.save_turn(
        conversation_id=conversation_id,
        question=request.question,
        answer=answer,
        metadata={
            "assistant": {
                "knowledge_count": len(state.get("knowledge", [])),
                "tool_calls": tool_calls_meta,
                "tool_results": tool_results_meta,
            }
        },
    )
    trace.append(ChatTraceStep(stage="done", detail="回答已生成并落库").model_dump())
    return {"answer": answer, "trace": trace}


async def cache_update_node(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """Write the Q&A pair into the semantic cache (if enabled and not skipped)."""
    request = state["request"]
    semantic_cache = config["configurable"].get("semantic_cache")
    answer = state.get("answer", "")
    trace = list(state.get("trace", []))

    skip = bool(request.metadata.get("skip_cache"))
    if semantic_cache is not None and answer and not skip:
        conversation_id = state.get("conversation_id", "")
        await semantic_cache.update(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            metadata={"conversation_id": conversation_id},
        )
        trace.append(ChatTraceStep(stage="cache", detail="已写入语义缓存").model_dump())
    return {"trace": trace}


# ── conditional routing ─────────────────────────────────────────────────────


def route_after_cache(state: CustomerServiceState) -> Literal["cached", "continue"]:
    if state.get("cache_hit"):
        return "cached"
    return "continue"


# ── graph builder ───────────────────────────────────────────────────────────


def build_customer_service_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph for the customer service agent.

    Returns a compiled graph.  Runtime objects (chat model, DB session, etc.)
    must be supplied via ``RunnableConfig.configurable`` when invoking.
    """

    builder = StateGraph(CustomerServiceState)

    # ── add nodes ──────────────────────────────────────────────────────
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("conversation", conversation_node)
    builder.add_node("check_cache", cache_node)
    builder.add_node("return_cached", _noop)
    builder.add_node("retrieve_knowledge", rag_node)
    builder.add_node("build_messages", build_messages_node)
    builder.add_node("agent", agent_llm_node)
    builder.add_node("tools", agent_tools_node)
    builder.add_node("save", save_node)
    builder.add_node("update_cache", cache_update_node)

    # ── wire edges ──────────────────────────────────────────────────────
    builder.set_entry_point("preprocess")
    builder.add_edge("preprocess", "conversation")
    builder.add_edge("conversation", "check_cache")

    builder.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {"cached": "return_cached", "continue": "retrieve_knowledge"},
    )
    builder.add_edge("return_cached", END)

    builder.add_edge("retrieve_knowledge", "build_messages")
    builder.add_edge("build_messages", "agent")

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": "save"},
    )
    builder.add_edge("tools", "agent")

    builder.add_edge("save", "update_cache")
    builder.add_edge("update_cache", END)

    return builder.compile()


async def _noop(state: CustomerServiceState, config: RunnableConfig) -> dict[str, Any]:
    """No-op node reached on cache hit — the response is already set."""
    return {}
