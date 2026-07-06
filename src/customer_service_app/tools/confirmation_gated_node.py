from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolNode


class ConfirmationGatedToolNode(ToolNode):
    """ToolNode that gates high-risk tools behind explicit user confirmation.

    When a tool name is in *confirmation_required* and not present in
    ``config["configurable"]["confirmed_tools"]``, execution is skipped and
    a synthetic ToolMessage with ``requires_confirmation: true`` is returned
    instead.  This preserves the contract from the original ToolSpec system.
    """

    confirmation_required: set[str]

    def __init__(
        self,
        tools: list,
        *,
        confirmation_required: set[str] | None = None,
        name: str = "tools",
        **kwargs: Any,
    ) -> None:
        super().__init__(tools, name=name, **kwargs)
        self.confirmation_required = confirmation_required or set()

    async def _execute_tool_async(
        self,
        request: ToolCallRequest,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage | Any:
        tool_name = request.tool_call["name"]

        if tool_name in self.confirmation_required:
            confirmed: set[str] = set(
                config.get("configurable", {}).get("confirmed_tools", set())
            )
            if tool_name not in confirmed:
                try:
                    args = json.loads(request.tool_call.get("args", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}

                return ToolMessage(
                    content=json.dumps(
                        {
                            "requires_confirmation": True,
                            "tool_name": tool_name,
                            "arguments": args,
                            "message": "该操作需要用户或人工客服确认后再执行。",
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=request.tool_call["id"],
                    name=tool_name,
                )

        return await super()._execute_tool_async(request, input_type, config)
