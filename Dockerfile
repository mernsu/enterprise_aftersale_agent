FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

COPY pyproject.toml .
RUN uv sync --frozen --no-dev

COPY src ./src
COPY scripts ./scripts
COPY sample_knowledge ./sample_knowledge
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "customer_service_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
