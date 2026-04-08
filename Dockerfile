FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install lightweight dependencies (NO PyTorch)
RUN pip install --no-cache-dir \
    click pydantic pydantic-settings python-dotenv requests \
    numpy scipy pandas tenacity pdfplumber \
    fastapi "uvicorn[standard]" jinja2 python-multipart \
    onnxruntime optimum transformers tokenizers

# Copy source and install app (editable-like)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . && \
    rm -rf /root/.cache

# Copy static assets and templates into installed package
RUN SITE_PKG=$(python -c "import sangaku_matcher.web; import os; print(os.path.dirname(sangaku_matcher.web.__file__))") && \
    cp -r src/sangaku_matcher/web/static "$SITE_PKG/static" && \
    cp -r src/sangaku_matcher/web/templates "$SITE_PKG/templates"

# Pre-download and cache the ONNX model during build
RUN python -c "\
from optimum.onnxruntime import ORTModelForFeatureExtraction; \
from transformers import AutoTokenizer; \
m = ORTModelForFeatureExtraction.from_pretrained('intfloat/multilingual-e5-small', export=True); \
t = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small'); \
m.save_pretrained('/app/model-cache'); \
t.save_pretrained('/app/model-cache'); \
print('Model cached')"

# Clean up build deps to save space
RUN apt-get purge -y gcc g++ && apt-get autoremove -y

# Copy pre-built database
COPY data/matcher.db data/matcher.db

# Use ONNX backend in production
ENV USE_ONNX=1
ENV EMBEDDING_MODEL=/app/model-cache
ENV PORT=10000
EXPOSE 10000

CMD ["python", "-m", "uvicorn", "sangaku_matcher.web.app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
