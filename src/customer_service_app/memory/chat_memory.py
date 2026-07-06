from __future__ import annotations

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from customer_service_app.infrastructure.db.repositories import ConversationRepository


class DatabaseBackedChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by the project's MySQL message store.

    Reads recent messages from the *messages* table via ConversationRepository
    and appends new messages after each turn.  Only user and assistant messages
    are loaded (system / tool messages are excluded).
    """

    def __init__(
        self,
        conversation_id: str,
        repo: ConversationRepository,
        *,
        max_messages: int = 12,
    ) -> None:
        self._conversation_id = conversation_id
        self._repo = repo
        self._max_messages = max_messages
        self._messages: list[BaseMessage] | None = None  # lazy-loaded

    async def _ensure_loaded(self) -> None:
        if self._messages is not None:
            return
        db_messages = await self._repo.recent_messages(
            conversation_id=self._conversation_id,
            limit=self._max_messages,
        )
        self._messages = []
        for msg in db_messages:
            if msg.role == "user":
                self._messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                self._messages.append(AIMessage(content=msg.content))

    @property
    async def messages(self) -> list[BaseMessage]:
        """Return messages in chronological order."""
        await self._ensure_loaded()
        return list(self._messages) if self._messages else []

    async def aget_messages(self) -> list[BaseMessage]:
        await self._ensure_loaded()
        return list(self._messages) if self._messages else []

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist new messages to the database."""
        await self._ensure_loaded()
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            await self._repo.append_message(
                conversation_id=self._conversation_id,
                role=role,
                content=msg.content,
            )
            self._messages.append(msg)

    async def add_message(self, message: BaseMessage) -> None:
        """Synonym for single-message append."""
        await self.aadd_messages([message])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        raise NotImplementedError("Use async version: aadd_messages")

    async def clear(self) -> None:
        self._messages = []
