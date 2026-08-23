FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    VTA_CACHE_DIR=/cache

WORKDIR /app

# Dependencies first for layer caching
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY data/corpus ./data/corpus
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 app && mkdir -p /cache && chown app /cache
USER app

# Mount a volume at /cache to persist embeddings across restarts; otherwise the
# corpus is re-embedded on every cold start.
VOLUME ["/cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["/app/.venv/bin/uvicorn", "--factory", "virtual_ta.api:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
