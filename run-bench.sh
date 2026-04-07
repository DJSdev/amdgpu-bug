source .venv/bin/activate

# Restart to ensure 1st run is a cold start
docker restart llama-server
sleep 3
./bench.py --host http://localhost:8080 --model llama-3.2-3B:Q4_K_M

docker restart llama-server
sleep 3
./bench.py --host http://localhost:8080 --model gemma-4-32B
