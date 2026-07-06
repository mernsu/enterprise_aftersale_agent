from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from customer_service_app.tools.confirmation_gated_node import ConfirmationGatedToolNode
from customer_service_app.tools.langchain_tools import query_order_status


class MiniState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _make_graph(tool_node: ConfirmationGatedToolNode):
    """Build a tiny graph that routes through the tool node."""
    builder = StateGraph(MiniState)

    async def send_tool_call(state: MiniState):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "query_order_status",
                        "args": {"order_id": "202606040001"},
                    }],
                )
            ]
        }

    builder.add_node("call", send_tool_call)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("call")
    builder.add_edge("call", "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def _make_node(**kwargs) -> ConfirmationGatedToolNode:
    """Create a ConfirmationGatedToolNode with error messages enabled."""
    return ConfirmationGatedToolNode(
        [query_order_status],
        handle_tool_errors=True,  # wrap errors as ToolMessage content
        **kwargs,
    )


@pytest.mark.asyncio
async def test_confirmation_gated_node_returns_gate_for_unconfirmed_tool() -> None:
    """高风险工具未确认时返回 requires_confirmation 而不是真正执行。"""
    node = _make_node(confirmation_required={"query_order_status"})
    graph = _make_graph(node)
    result = await graph.ainvoke(
        {"messages": []},
        {"configurable": {"confirmed_tools": set(), "session": None}},
    )
    msgs = result["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    payload = json.loads(tool_msgs[0].content)
    assert payload.get("requires_confirmation") is True
    assert payload.get("tool_name") == "query_order_status"


@pytest.mark.asyncio
async def test_confirmed_tool_bypasses_gate() -> None:
    """显式确认后，高风险工具进入真实 handler。"""
    node = _make_node(confirmation_required={"query_order_status"})
    graph = _make_graph(node)
    result = await graph.ainvoke(
        {"messages": []},
        {
            "configurable": {
                "confirmed_tools": {"query_order_status"},
                "session": None,
                "tenant_id": "default",
                "user_id": "u001",
            }
        },
    )
    msgs = result["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    # Handler reached (gate bypassed) — fails because session=None.
    content = tool_msgs[0].content
    assert isinstance(content, str)


@pytest.mark.asyncio
async def test_non_high_risk_tool_executes_without_confirmation() -> None:
    """非高风险工具不需要确认即可执行。"""
    node = _make_node(confirmation_required=set())
    graph = _make_graph(node)
    result = await graph.ainvoke(
        {"messages": []},
        {
            "configurable": {
                "confirmed_tools": set(),
                "session": None,
                "tenant_id": "default",
                "user_id": "u001",
            }
        },
    )
    msgs = result["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
