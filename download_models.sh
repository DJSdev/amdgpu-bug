#!/bin/bash

# Install HuggingFace CLI in container
docker exec -it llama-server /bin/bash -c \
    'apt update && \
    apt install -y python3-venv && \
    python3 -m venv .venv && \
    source ./.venv/bin/activate && \
    pip3 install huggingface_hub && \
    \
    echo Downloading Llama3.2 GGUF && \
    hf download QuantFactory/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct.Q4_K_M.gguf && \
    \
    echo Downloading Gemma4 GUFF && \
    hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q8_0.gguf'

# Restart to take effect
docker restart llama-server
