FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached unless requirements change)
RUN pip install --no-cache-dir \
    click pydantic pydantic-settings python-dotenv requests \
    numpy scipy pandas tenacity pdfplumber \
    fastapi "uvicorn[standard]" jinja2 python-multipart \
    onnxruntime "optimum[onnxruntime]" transformers tokenizers \
    sentence-transformers

# Export model to ONNX and quantize to int8 (~113MB vs 448MB)
# This expensive layer depends ONLY on pip packages above, not on source code.
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

# Clean up build deps and caches
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && \
    rm -rf /root/.cache /tmp/*

# Copy pre-built database, taxonomy files, and weight config
COPY data/matcher.db data/matcher.db
COPY data/theme_taxonomy.json data/theme_taxonomy.json
COPY data/tech_taxonomy.json data/tech_taxonomy.json
COPY data/ambition_taxonomy.json data/ambition_taxonomy.json
COPY data/exit_weights.json data/exit_weights.json

# ── Source code layer (only this invalidates on code changes) ──
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . && \
    SITE_PKG=$(python -c "import sangaku_matcher; import os; print(os.path.dirname(sangaku_matcher.__file__))") && \
    cp -r src/sangaku_matcher/* "$SITE_PKG/" && \
    rm -rf /root/.cache

# Use ONNX backend with quantized local model
ENV USE_ONNX=1
ENV EMBEDDING_MODEL=/app/model-cache
ENV PORT=10000
EXPOSE 10000

CMD ["python", "-m", "uvicorn", "sangaku_matcher.web.app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
