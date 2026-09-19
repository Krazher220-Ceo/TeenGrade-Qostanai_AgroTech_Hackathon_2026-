# AgroVision AI (Case 1) — CPU-only image for the FastAPI server and the
# Streamlit dashboard. Both services share this image; docker-compose.yml
# selects which process to run per-service via `command:`.
#
# IMPORTANT: model weights (case1/models/*.pt) and image assets are stored
# via Git LFS. Run `git lfs pull` on the HOST before `docker build`/`docker
# compose build` — otherwise only small LFS pointer text files get copied
# into the image and the server will fail to load the model at startup.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# System dependencies: libgl1/libglib for opencv-python-headless & Pillow,
# git for tooling that shells out to `git` (e.g. dataset scripts).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY case1/requirements.txt /app/case1/requirements.txt

# CPU-only PyTorch/torchvision wheels (much smaller than the default CUDA
# build) plus the rest of the project's dependencies.
RUN pip install --upgrade pip && \
    pip install "torch>=2.2.0" "torchvision>=0.17.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r case1/requirements.txt

COPY . /app

EXPOSE 8000 8501

# Default: run the API server. docker-compose.yml overrides `command:` for
# the dashboard service.
CMD ["python", "-m", "uvicorn", "case1.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
