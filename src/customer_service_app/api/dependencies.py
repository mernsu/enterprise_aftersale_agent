from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from customer_service_app.infrastructure.db.session import get_db_session
from customer_service_app.services.container import AgentRuntime, build_langgraph_agent


@dataclass(slots=True)
class AgentContext:
    """Bundle returned by the FastAPI dependency for LangGraph-based routes."""

    runtime: AgentRuntime
    session: AsyncSession


def get_langgraph_agent(
    session: AsyncSession = Depends(get_db_session),
) -> AgentContext:
    """FastAPI dependency: assemble the LangGraph agent for the current request."""
    return AgentContext(runtime=build_langgraph_agent(session), session=session)
