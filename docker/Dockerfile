ARG BASE_IMAGE=2.11.0-cuda12.8-py3.12.3-devel
FROM ${BASE_IMAGE}

ARG INSTALL_FLASH_ATTN=1
ARG FLASH_ATTN_VERSION=">=2.8.3"

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace/ACC_Train

COPY requirements.txt /tmp/acc_requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja cmake && \
    python -m pip install -r /tmp/acc_requirements.txt && \
    if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
      MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation "flash-attn${FLASH_ATTN_VERSION}"; \
    fi

COPY . /workspace/ACC_Train

CMD ["/bin/bash"]
