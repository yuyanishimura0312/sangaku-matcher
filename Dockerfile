FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than default CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy source and install app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir . && \
    apt-get purge -y gcc g++ && apt-get autoremove -y && \
    rm -rf /root/.cache

# Copy static assets and templates into installed package location
# (pip install doesn't include non-Python data files by default)
RUN SITE_PKG=$(python -c "import sangaku_matcher.web; import os; print(os.path.dirname(sangaku_matcher.web.__file__))") && \
    cp -r src/sangaku_matcher/web/static "$SITE_PKG/static" && \
    cp -r src/sangaku_matcher/web/templates "$SITE_PKG/templates"

# Copy pre-built database
COPY data/matcher.db data/matcher.db

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "python -m uvicorn sangaku_matcher.web.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
