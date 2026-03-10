FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
COPY app/ ./app/

# Create volume directories
RUN mkdir -p /app/data/jobs /app/result /app/uploads

# Healthcheck using python httpx (no curl in slim image)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=5s \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT:-9997}/health').raise_for_status()"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
