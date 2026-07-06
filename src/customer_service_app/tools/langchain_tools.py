from __future__ import annotations

from typing import Any

from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt.tool_node import ToolRuntime

from customer_service_app.infrastructure.db.repositories import OrderRepository, TicketRepository
from customer_service_app.infrastructure.search.serpapi_client import SerpApiSearchClient


@tool
async def query_order_status(
    order_id: str,
    runtime: ToolRuntime = InjectedToolArg(),
) -> dict[str, Any]:
    """查询当前用户名下指定订单的状态、物流公司和运单号。"""
    config = runtime.config
    session = config["configurable"]["session"]
    tenant_id = config["configurable"]["tenant_id"]
    user_id = config["configurable"]["user_id"]

    order = await OrderRepository(session).get_by_order_id(
        tenant_id=tenant_id,
        user_id=user_id,
        order_id=order_id,
    )
    if order is None:
        return {"found": False, "order_id": order_id, "message": "未找到该用户名下的订单"}
    return {
        "found": True,
        "order_id": order.order_id,
        "status": order.status,
        "logistics_company": order.logistics_company,
        "tracking_number": order.tracking_number,
        "metadata": order.metadata_json,
    }


@tool
async def create_refund_ticket(
    order_id: str,
    reason: str,
    priority: str = "normal",
    runtime: ToolRuntime = InjectedToolArg(),
) -> dict[str, Any]:
    """当用户要申请退款或退货退款时，创建退款工单。该工具属于高风险写操作，需要确认后执行。"""
    config = runtime.config
    session = config["configurable"]["session"]
    tenant_id = config["configurable"]["tenant_id"]
    user_id = config["configurable"]["user_id"]
    conversation_id = config["configurable"].get("conversation_id", "")

    ticket = await TicketRepository(session).create(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        category="refund",
        title=f"退款申请：{order_id}",
        detail=reason,
        priority=priority,
        metadata={"order_id": order_id},
    )
    return {"ticket_id": ticket.id, "status": ticket.status, "category": ticket.category}


@tool
async def transfer_to_human(
    reason: str,
    priority: str = "high",
    runtime: ToolRuntime = InjectedToolArg(),
) -> dict[str, Any]:
    """当用户明确要求人工客服、投诉升级、情绪强烈或模型无法解决时创建人工客服工单。该工具需要确认后执行。"""
    config = runtime.config
    session = config["configurable"]["session"]
    tenant_id = config["configurable"]["tenant_id"]
    user_id = config["configurable"]["user_id"]
    conversation_id = config["configurable"].get("conversation_id", "")

    ticket = await TicketRepository(session).create(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        category="human_handoff",
        title="转人工处理",
        detail=reason,
        priority=priority,
    )
    return {"ticket_id": ticket.id, "status": ticket.status, "message": "已创建人工客服工单"}


@tool
async def search_public_web(
    query: str,
    runtime: ToolRuntime = InjectedToolArg(),
) -> dict[str, Any]:
    """搜索公开互联网信息，适合实时新闻、外部公告、需要最新信息的问题。"""
    config = runtime.config
    search_client: SerpApiSearchClient = config["configurable"]["search_client"]
    results = await search_client.search(query)
    return {"query": query, "results": results}


# Tool names that require explicit user confirmation before execution.
CONFIRMATION_REQUIRED_TOOLS: set[str] = {"create_refund_ticket", "transfer_to_human"}
