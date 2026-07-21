# StockIntel backend — production image for Render (or any Docker host).
#
# Design notes:
#
# * CPU-only PyTorch. On Linux, `pip install torch` pulls the CUDA build by
#   default (~2.5GB) — useless without a GPU and large enough to blow past
#   image-size and memory limits. torch is installed first from the CPU wheel
#   index so the later `-r requirements.txt` sees it already satisfied and does
#   not re-pull the CUDA variant.
#
# * FinBERT is NOT baked in. It lazy-loads on the first news request, and only
#   if MARKETAUX_API_KEY is set. A FRED-only deployment never loads it, so
#   baking 440MB of weights into every image would be waste. The trade-off:
#   if news keys are added later, the first news request downloads the weights.
#
# * Single worker. FinBERT and torch each hold their own copy of large tensors,
#   so N workers means N times the memory. One worker keeps the memory
#   footprint predictable; requests are serialised, which is acceptable for a
#   research/portfolio deployment.

FROM python:3.13-slim AS base

# libgomp1 is LightGBM's only runtime system dependency (OpenMP).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STOCKINTEL_ENV=production \
    # Hugging Face cache inside the writable app dir, so a lazy FinBERT load
    # has somewhere to write on hosts with a read-only root filesystem.
    HF_HOME=/app/.hf_cache

COPY requirements.txt .

# CPU-only torch first, then everything else (torch is then already satisfied).
RUN pip install --upgrade pip \
 && pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Only the application package is needed at runtime — not tests, scripts,
# the local venv, caches or artifacts (all excluded via .dockerignore).
COPY app ./app

EXPOSE 8000

# Render (and most hosts) inject $PORT. Default to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
