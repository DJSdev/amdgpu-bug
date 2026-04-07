.PHONY: start stop download-models

start:
	docker compose up -d

stop:
	docker compose down

download-models:
	docker exec -it llama-server /bin/bash -c \
	'apt update && \
	apt install -y python3-venv && \
	\
	python3 -m venv .venv && \
	source ./.venv/bin/activate && \
	pip3 install huggingface_hub && \
	\
	echo Downloading Llama3.2 GGUF && \
	hf download QuantFactory/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct.Q4_K_M.gguf && \
	\
	echo Downloading Gemma4 GUFF && \
	hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q8_0.gguf'

.venv:
	uv sync

run-bench: .venv
	@
	# Restart to ensure 1st run is a cold start
	docker restart llama-server
	sleep 3
	./bench.py --host http://localhost:8080 --model llama-3.2-3B:Q4_K_M

	docker restart llama-server
	sleep 3
	./bench.py --host http://localhost:8080 --model gemma-4-26B-A4B
