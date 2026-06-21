from __future__ import annotations

from typing import Any

import pytest

from customer_service_app.services.tool_registry import ToolExecutionContext, ToolRegistry, ToolSpec


async def sample_handler(arguments: dict[str, Any], _: ToolExecutionContext) -> dict[str, Any]:
    """测试用工具函数，返回传入文本。"""
    return {"echo": arguments["text"]}


@pytest.mark.asyncio
async def test_tool_registry_executes_registered_tool() -> None:
    """验证工具注册后可以按名称执行。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=sample_handler,
        )
    )

    result = await registry.execute(
        name="echo",
        arguments_json='{"text": "hello"}',
        context=None,
    )

    assert result == {"echo": "hello"}
