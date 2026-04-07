.PHONY: start stop ollama-pull zip-models download-models

COMFY_DIR = ./config/comfy-ui

WORKFLOW_FILE = $(COMFY_DIR)/workflows/mopSDXL_api.json
WORKFLOW_NODE_ID = $(shell jq -r 'to_entries[] | select(.value.class_type == "CLIPTextEncode") | .key' $(WORKFLOW_FILE) | head -n 1)

WORKFLOW_EDIT_FILE = $(COMFY_DIR)/workflow/edit_workflow.json

MODELS_DIR := ./
#URL := http://blob-server.home:8880/comfyui_models.tar.gz

start:
	@
	export COMFYUI_WORKFLOW_JSON=$$(cat $(WORKFLOW_FILE)); \
	export COMFYUI_WORKFLOW_NODE_ID=$(WORKFLOW_NODE_ID); \
	export COMFYUI_WORKFLOW_EDIT_JSON=[]; \
	export COMFYUI_WORKFLOW_EDIT_NODE_ID=0; \
	docker compose up -d

stop:
	@
	export COMFYUI_WORKFLOW_JSON="{}"; \
	export COMFYUI_WORKFLOW_NODE_ID="0"; \
	export COMFYUI_WORKFLOW_EDIT_JSON=[]; \
	export COMFYUI_WORKFLOW_EDIT_NODE_ID=0; \
	docker compose down

install-hf:
	docker exec -it llama-server /bin/bash -c \
	'apt update && \
	apt install -y python3-venv && \
	python3 -m venv .venv && \
	source ./.venv/bin/activate && \
	pip3 install huggingface_hub'

download-model:
	docker exec -it llama-server /bin/bash -c \
	'source ./.venv/bin/activate && \
	echo Downloading $(MODEL) from $(REPO) && \
	hf download $(REPO) $(MODEL)'

rm-ollama-models:
	@for i in $$(ls ./config/ollama/modelfiles/); do docker exec -it ollama ollama rm $$i; done

#zip-models:
#	tar -I pigz -cvf comfyui_models.tar.gz ./config/comfy-ui/models

#download-models:
#	@echo "Downloading and updating ComfyUI models"
#	curl -L $(URL) | tar -xz --overwrite
