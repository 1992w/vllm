# ---------- Build stage ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=8
ENV PIP_NO_BUILD_ISOLATION=1

#Replace this with the compute capability for your GPU's
ENV TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 8.9 9.0 10.0+PTX'

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build curl ca-certificates \
    python3 python3-venv python3-pip python3-dev python-is-python3 && \
    python3 -m pip install --upgrade pip numpy==1.26.4

RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124

# Clone vLLM and build wheel
#RUN git clone --branch v0.9.2 https://github.com/vllm-project/vllm.git /vllm

WORKDIR /vllm

COPY . .

# This is slow as balls
RUN python3 -m pip wheel . -w /tmp/wheels

# ---------- Runtime stage ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python-is-python3 python3-dev gcc g++ build-essential && \
    python3 -m pip install --upgrade pip numpy==1.26.4

# **Install C compiler and build tools**
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ build-essential

COPY --from=build /tmp/wheels /tmp/wheels
RUN python3 -m pip install --no-cache-dir /tmp/wheels/vllm-*.whl

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
