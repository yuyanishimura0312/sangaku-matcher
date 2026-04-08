FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch for smaller image)
COPY pyproject.toml .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch --index-strategy unsafe-first-match && \
    pip install --no-cache-dir . && \
    rm -rf /root/.cache

# Copy application code and data
COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

# Expose port
ENV PORT=10000
EXPOSE 10000

# Run the app — Render sets PORT env var
CMD ["sh", "-c", "python -m uvicorn sangaku_matcher.web.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
