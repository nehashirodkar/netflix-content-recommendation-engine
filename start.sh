#!/bin/bash
set -e

echo "Starting FastAPI on port ${API_PORT:-8000}..."
uvicorn src.api.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" &
API_PID=$!

echo "Waiting for FastAPI to be ready..."
for i in {1..60}; do
  if curl -sf "http://localhost:${API_PORT:-8000}/health" >/dev/null 2>&1; then
    echo "FastAPI is up."
    break
  fi
  sleep 2
done

echo "Starting Streamlit on port ${PORT:-7860}..."
exec streamlit run dashboard/app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
