FROM NVIDIA/cuda-samples/CUDA13.0-Torch2.7.0-Python3.11:v2

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace/ACC_Train

COPY requirements.txt /tmp/acc_requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja cmake && \
    python -m pip install -r /tmp/acc_requirements.txt && \
    MAX_JOBS=8 python -m pip install --no-build-isolation "flash-attn>=2.7.0"

COPY . /workspace/ACC_Train

CMD ["/bin/bash"]
