from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from customer_service_app.memory.chat_memory import DatabaseBackedChatMessageHistory


class FakeMessage:
    """Minimal fake to satisfy the interface expected by DatabaseBackedChatMessageHistory."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class FakeRepo:
    """Fake ConversationRepository that returns canned messages."""

    def __init__(self, messages: list[FakeMessage]) -> None:
        self._messages = messages
        self.appended: list[dict] = []

    async def recent_messages(self, *, conversation_id: str, limit: int) -> list[FakeMessage]:
        return self._messages[-limit:]

    async def append_message(self, *, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.appended.append({"role": role, "content": content})


async def test_database_backed_history_loads_and_appends() -> None:
    """加载历史返回 LangChain BaseMessage，追加时写回 repo。"""
    repo = FakeRepo([
        FakeMessage("user", "你好"),
        FakeMessage("assistant", "你好，有什么可以帮你？"),
    ])
    history = DatabaseBackedChatMessageHistory("c001", repo, max_messages=10)
    messages = await history.aget_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "你好"

    await history.aadd_messages([HumanMessage(content="新问题")])
    await history.aadd_messages([AIMessage(content="新回答")])
    assert len(repo.appended) == 2


async def test_database_backed_history_filters_system_messages() -> None:
    """只有 user/assistant 消息被加载，system 消息应被过滤。"""
    repo = FakeRepo([
        FakeMessage("system", "摘要"),
        FakeMessage("user", "问题"),
    ])
    history = DatabaseBackedChatMessageHistory("c001", repo, max_messages=10)
    messages = await history.aget_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
