# Pre-reqs

- Python
- uv
- docker
- make

# Getting Started

- `make start` - Start the llama.cpp server
- `make download-models` - Install the huggingface cli and download models

# Running benchmark

- `make run-bench`

# Extras

- `make stop` - Remove the container (doesn't delete model files)
- `rm -rf ./models` - Delete the local model files
