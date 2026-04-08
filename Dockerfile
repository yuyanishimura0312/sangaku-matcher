FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies (NO PyTorch in final image)
RUN pip install --no-cache-dir \
    click pydantic pydantic-settings python-dotenv requests \
    numpy scipy pandas tenacity pdfplumber \
    fastapi "uvicorn[standard]" jinja2 python-multipart \
    onnxruntime "optimum[onnxruntime]" transformers tokenizers \
    sentence-transformers

# Copy source and install app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . && \
    rm -rf /root/.cache

# Copy static assets and templates into installed package
RUN SITE_PKG=$(python -c "import sangaku_matcher.web; import os; print(os.path.dirname(sangaku_matcher.web.__file__))") && \
    cp -r src/sangaku_matcher/web/static "$SITE_PKG/static" && \
    cp -r src/sangaku_matcher/web/templates "$SITE_PKG/templates"

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

# Clean up build deps and caches
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && \
    rm -rf /root/.cache /tmp/*

# Copy pre-built database
COPY data/matcher.db data/matcher.db

# Use ONNX backend with quantized local model
ENV USE_ONNX=1
ENV EMBEDDING_MODEL=/app/model-cache
ENV PORT=10000
EXPOSE 10000

CMD ["python", "-m", "uvicorn", "sangaku_matcher.web.app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
