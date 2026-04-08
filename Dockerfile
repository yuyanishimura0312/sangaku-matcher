FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than default CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install app dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir . && rm -rf /root/.cache

# --- Runtime stage ---
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and pre-built database
COPY src/ src/
COPY data/matcher.db data/matcher.db

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "python -m uvicorn sangaku_matcher.web.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
