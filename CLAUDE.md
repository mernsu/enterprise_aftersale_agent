# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Environment & dependencies
uv sync                                           # install all dependencies
uv sync --dev                                     # include dev dependencies
uv run <command>                                  # run within venv

# Development server
uv run python scripts/run_dev.py                  # start with reload
uv run python scripts/run_dev.py --port 9090      # custom port

# Infrastructure (Docker)
docker compose up -d mysql redis qdrant           # start dependencies
docker compose down                               # stop all

# Database setup (after first clone)
uv run python scripts/init_db.py                  # create tables
uv run python scripts/seed_demo_data.py           # insert demo orders
uv run python scripts/check_env.py                # verify config

# Knowledge base
uv run python scripts/ingest_docs.py sample_knowledge --tenant-id default

# Lint & test
uv run ruff check src tests scripts               # lint
uv run pytest -q                                  # run all tests
uv run pytest tests/test_tool_registry.py -q      # single test file
```

## Architecture

### Request Flow (LangGraph StateGraph)

A chat request goes through 10 nodes in `src/customer_service_app/agents/customer_service_agent.py`:

```text
preprocess → conversation → check_cache ──(hit)──→ return_cached → END
                              │ (miss)
                              ▼
                        retrieve_knowledge → build_messages → agent ←──┐
                                                               │       │
                                                               ▼       │
                                                          [tool_calls?]─┘
                                                               │ (no)
                                                               ▼
                                                              save → update_cache → END
```

1. **Preprocess** — `QuestionPreprocessor` normalizes whitespace, detects intent keywords, rejects blank input.
2. **Conversation** — `ConversationService` loads or creates a session, scoped to `tenant_id` + `user_id`.
3. **Cache** — `RedisSemanticCache` checks for semantically similar prior questions (embedding + cosine). Skip via `metadata.skip_cache=true`.
4. **RAG** — `TenantAwareQdrantStore.as_tenant_retriever()` searches Qdrant with a tenant-scoped filter.
5. **Build Messages** — System prompt + compressed history (last 8 verbatim, earlier summarized) + knowledge context + user question.
6. **Agent LLM** — `ChatOpenAI.bind_tools(tools)` with the 4 registered tools.
7. **Tools** — `ConfirmationGatedToolNode` executes tools; high-risk tools (`create_refund_ticket`, `transfer_to_human`) require explicit confirmation in `metadata.confirmed_tools`.
8. **Save** — User/assistant messages written to MySQL via `ConversationService.save_turn()`.
9. **Cache Update** — Redis cache updated with the new Q&A pair.

### Dependency Injection

`src/customer_service_app/services/container.py` is the DI container. `build_langgraph_agent(session)` assembles ChatOpenAI, embedding model, Qdrant store, semantic cache, search client, `@tool`-decorated functions, and `ConfirmationGatedToolNode` into an `AgentRuntime`. Non-serializable runtime objects flow through `RunnableConfig.configurable`, not graph state.

### Tool System

Tools live in `src/customer_service_app/tools/langchain_tools.py` as `@tool`-decorated async functions. Runtime dependencies (`AsyncSession`, tenant/user, search client) are injected via `ToolRuntime = InjectedToolArg()`. Tools requiring confirmation are listed in `CONFIRMATION_REQUIRED_TOOLS`. The `ConfirmationGatedToolNode(ToolNode)` in `confirmation_gated_node.py` intercepts high-risk tool calls and returns a `requires_confirmation: true` gate.

### Key LangChain Components

| Component | LangChain Class | Location |
| --- | --- | --- |
| LLM | `ChatOpenAI` | `infrastructure/llm/langchain_factory.py` |
| Embeddings | `OllamaEmbeddings` / `OpenAIEmbeddings` | `infrastructure/embeddings/langchain_factory.py` |
| Vector Store | `QdrantVectorStore` (wrapped) | `infrastructure/vector_store/langchain_store.py` |
| Agent Graph | `StateGraph` | `agents/customer_service_agent.py` |
| Tools | `@tool` + `StructuredTool` | `tools/langchain_tools.py` |
| Memory | `DatabaseBackedChatMessageHistory` | `memory/chat_memory.py` |
| Streaming | `graph.astream_events()` | `api/routes_chat.py` |
| Tracing | `AsyncCallbackHandler` | `callbacks/trace_callback.py` |

### Multi-tenancy

Every DB query and Qdrant search is scoped by `tenant_id` + `user_id`. These values flow through `RunnableConfig.configurable` and are injected into tool handlers via `ToolRuntime`.

### Settings

`core/config.py` uses `pydantic-settings` with `.env` file loading. `get_settings()` is cached via `@lru_cache`. The `settings.require(name, value)` method raises `ConfigurationError` for missing required config — use it at the point of consumption, not at startup.

### Preserved Modules (unchanged from original)

DB models/session/repositories, `ConversationService`, `QuestionPreprocessor`, `Settings`, exception hierarchy, middleware, logging, health/ops/conversations routes, prompts, domain schemas, `RedisSemanticCache`, `SerpApiSearchClient`.
