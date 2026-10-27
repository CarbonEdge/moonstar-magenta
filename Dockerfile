# GPU benchmark image: JAX CUDA wheels are Linux-only, so run Magenta RT here.
# CUDA libs ship inside the jax[cuda12] wheels; only the host NVIDIA driver is needed.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir magenta-rt "jax[cuda12]"
ENV PYTHONUTF8=1 PYTHONIOENCODING=utf-8
WORKDIR /app
COPY moonstar_magenta ./moonstar_magenta
ENV PYTHONPATH=/app
