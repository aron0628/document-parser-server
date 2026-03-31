FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
COPY app/ ./app/

# Create volume directories
RUN mkdir -p /app/data/jobs /app/result /app/uploads

# Healthcheck using python httpx (no curl in slim image)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD uv run python -c "import httpx; httpx.get('http://localhost:${PORT:-9997}/health').raise_for_status()"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
