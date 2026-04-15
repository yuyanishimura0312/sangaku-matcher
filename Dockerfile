FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached unless requirements change)
RUN pip install --no-cache-dir \
    click pydantic pydantic-settings python-dotenv requests \
    numpy scipy pandas tenacity pdfplumber \
    fastapi "uvicorn[standard]" jinja2 python-multipart slowapi \
    onnxruntime "optimum[onnxruntime]" transformers tokenizers \
    sentence-transformers

# Export model to ONNX and quantize to int8 (~113MB vs 448MB)
RUN python -c "\
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer; \
from optimum.onnxruntime.configuration import AutoQuantizationConfig; \
from transformers import AutoTokenizer; \
m = ORTModelForFeatureExtraction.from_pretrained('intfloat/multilingual-e5-small', export=True); \
m.save_pretrained('/tmp/onnx-fp32'); \
q = ORTQuantizer.from_pretrained('/tmp/onnx-fp32'); \
qc = AutoQuantizationConfig.avx2(is_static=False); \
q.quantize(save_dir='/app/model-cache', quantization_config=qc); \
t = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small'); \
t.save_pretrained('/app/model-cache'); \
print('Quantized model cached')" && \
    rm -rf /tmp/onnx-fp32

# Clean up build deps
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && \
    rm -rf /root/.cache /tmp/*

# Create non-root user before copying app files
RUN useradd -m -u 1001 appuser

# Download large DB from GitHub Release (avoids LFS budget issues)
# chmod 666 ensures WAL mode can create -wal/-shm files even under appuser
RUN mkdir -p data && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -fSL --retry 3 --retry-delay 5 -o data/matcher.db \
    https://github.com/yuyanishimura0312/sangaku-matcher/releases/download/v0.1.0-data/matcher.db && \
    echo "DB downloaded: $(stat -c%s data/matcher.db) bytes" && \
    test $(stat -c%s data/matcher.db) -gt 1000000 && \
    chmod 666 data/matcher.db && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy data files
COPY data/theme_taxonomy.json data/theme_taxonomy.json
COPY data/tech_taxonomy.json data/tech_taxonomy.json
COPY data/ambition_taxonomy.json data/ambition_taxonomy.json
COPY data/exit_weights.json data/exit_weights.json

# Copy source and install (this layer rebuilds on code changes)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# Overwrite site-packages with latest source to ensure all files are current
RUN SITE_PKG=$(python -c "import sangaku_matcher; print(sangaku_matcher.__path__[0])") && \
    cp -r src/sangaku_matcher/* "$SITE_PKG/"

ENV USE_ONNX=1
ENV EMBEDDING_MODEL=/app/model-cache
ENV PORT=10000
EXPOSE 10000

# Transfer ownership of /app to appuser, then drop root privileges
RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "uvicorn", "sangaku_matcher.web.app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
