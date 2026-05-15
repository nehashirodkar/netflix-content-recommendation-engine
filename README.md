---
title: Netflix-Style Recommendation Engine
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Netflix-Style Recommendation Engine

A side-by-side comparison of four recommendation approaches trained on **MovieLens 1M**:

| Model | Type | Library |
|---|---|---|
| **SVD** | Matrix factorization with biases | PyTorch (custom) |
| **Content-Based** | TF-IDF over title + genres | scikit-learn |
| **Neural CF** | User/item embeddings + MLP, BCE with negative sampling | PyTorch |
| **Hybrid** | Min-max normalized weighted ensemble of SVD + NCF | — |

Serves recommendations through a **FastAPI** service with cold-start fallback to content-based filtering, and a **Streamlit** dashboard for live model switching and side-by-side comparison.

Trained on Google Colab GPU, tracked with **Weights & Biases**, model artifacts hosted on **Hugging Face Hub**, demo deployed on **Hugging Face Spaces**.

---

## Architecture

```
GitHub repo (source code)
        │
        ▼
   Colab (training, GPU)  ──►  Weights & Biases (experiment tracking)
        │
        ▼
   Hugging Face Hub (model artifacts: svd.pt, ncf.pt, content.joblib, indices, metadata)
        │
        ▼
   Hugging Face Space (Docker)
        ├── FastAPI on :8000  (loads models from HF Hub at startup)
        └── Streamlit on :7860 (calls FastAPI; user-facing)
```

---

## Repo layout

```
src/
  config.py              # paths, hyperparameters, env-var overrides
  data/
    loader.py            # download + parse MovieLens 1M
    preprocess.py        # build indices, time-based train/val/test split
  models/
    svd.py               # PyTorch matrix factorization
    content.py           # TF-IDF content-based
    ncf.py               # Neural CF with negative sampling
    hybrid.py            # weighted ensemble
  evaluation/
    metrics.py           # Precision@K, Recall@K, NDCG@K, RMSE
  training/
    train_all.py         # full pipeline + W&B + optional HF Hub push
  api/
    main.py              # FastAPI app (/recommend, /compare, /health, /movies)
    recommender.py       # inference layer (loads models, serves predictions)
    schemas.py           # Pydantic request/response models
  utils/
    hf_hub.py            # push/pull models from Hugging Face Hub
dashboard/
  app.py                 # Streamlit UI
notebooks/
  train_colab.ipynb      # end-to-end Colab training notebook
Dockerfile               # HF Spaces deployment (FastAPI + Streamlit)
start.sh                 # container entrypoint
requirements.txt
```

---

## Setup — accounts (free, ~10 min)

1. **Hugging Face** — https://huggingface.co/join
   - Create a write-scope token: https://huggingface.co/settings/tokens → New token → role: `write`. Copy and keep it.
   - Create a model repo: https://huggingface.co/new → name it `netflix-recsys-models` (model type, public). Note the full ID, e.g. `NehaS98/netflix-recsys-models`.
   - Create a Space: https://huggingface.co/new-space → name it `netflix-recsys-demo` → SDK: **Docker** → public. Note the URL, e.g. `https://huggingface.co/spaces/NehaS98/netflix-recsys-demo`.
2. **Weights & Biases** — https://wandb.ai/signup
   - Get your API key: https://wandb.ai/authorize. Copy and keep it.

You should now have:
- HF token (write)
- HF model repo ID
- HF Space URL
- W&B API key

---

## Train (on Google Colab)

1. Open https://colab.research.google.com → File → Upload notebook → select `notebooks/train_colab.ipynb` from this repo.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run cells top to bottom. You'll be prompted for:
   - HF token
   - HF model repo (default: `NehaS98/netflix-recsys-models`)
   - W&B API key (leave blank to skip W&B)
4. The notebook will:
   - Clone the repo
   - Install deps (~3 min)
   - Download MovieLens 1M (~10s)
   - Train SVD (~3 min on T4), NCF (~10 min), Content (~30s)
   - Evaluate on test set (Precision@10, NDCG@10, Recall@10, RMSE)
   - Push trained models + indices + metadata to your HF model repo

Total Colab runtime: **~20–30 min**. You can close the tab once it's done — artifacts are on HF Hub.

---

## Test locally (optional)

If you want to sanity-check before deploying — but note this requires PyTorch installed locally (~2GB):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:HF_TOKEN = "<your-token>"
$env:HF_MODEL_REPO = "NehaS98/netflix-recsys-models"

# Pull the trained models you just pushed from Colab
python -c "from src.utils.hf_hub import pull_models; pull_models()"

# Run FastAPI
uvicorn src.api.main:app --reload
# In another terminal:
$env:API_BASE_URL = "http://localhost:8000"
streamlit run dashboard/app.py
```

Open http://localhost:8501 — you should see the dashboard. Pick user_id=1, hit Recommend.

---

## Deploy to Hugging Face Spaces

Once models are on HF Hub:

```powershell
# Add HF Space as a git remote (one-time)
git remote add space https://huggingface.co/spaces/NehaS98/netflix-recsys-demo

# Set the HF Space's repo secret so it can pull models from your private/public model repo:
# Go to your Space → Settings → Variables and secrets:
#   HF_TOKEN        = <your write-scope token>   (Secret)
#   HF_MODEL_REPO   = NehaS98/netflix-recsys-models   (Variable)

# Push code to the Space — it auto-builds the Docker image and deploys
git push space main
```

The Space will:
1. Build the Docker image (~5 min first time)
2. Pull models from your HF Hub model repo at container startup
3. Start FastAPI on internal :8000, Streamlit on :7860 (the public-facing port)

Once it's live: `https://huggingface.co/spaces/NehaS98/netflix-recsys-demo`

---

## API endpoints

- `GET /health` — service status + model metadata
- `GET /movies?q=matrix&limit=20` — search movies by title
- `POST /recommend` — single-model recommendations
  ```json
  {"user_id": 1, "model": "hybrid", "k": 10}
  ```
- `POST /compare` — runs all 4 models and returns each result
  ```json
  {"user_id": 1, "k": 10}
  ```
- Cold-start: omit `user_id`, optionally pass `history_movie_ids` — auto-falls back to content-based.

---

## Stack

Python · PyTorch · scikit-learn · pandas · FastAPI · Streamlit · Plotly · Weights & Biases · Hugging Face Hub · Hugging Face Spaces · Docker
