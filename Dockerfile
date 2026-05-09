# Dockerfile for Hugging Face Spaces (Docker SDK).
# Runs FastAPI (internal :8000) and Streamlit (exposed :7860) together.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    API_PORT=8000 \
    API_BASE_URL=http://localhost:8000 \
    PULL_MODELS_ON_STARTUP=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY dashboard ./dashboard
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# HF Spaces sets writable HOME to /home/user; honor it for the model cache.
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod -R 777 /home/user
RUN mkdir -p /app/models /app/data && chmod -R 777 /app/models /app/data

EXPOSE 7860

CMD ["./start.sh"]
