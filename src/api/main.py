"""FastAPI service exposing recommendations from SVD / NCF / Content / Hybrid models."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.api.recommender import get_recommender
from src.api.schemas import (
    CompareResponse,
    HealthResponse,
    MovieSearchResponse,
    RecommendRequest,
    RecommendResponse,
)

app = FastAPI(
    title="Netflix-Style Recommendation API",
    description="SVD, NCF, Content-based, and Hybrid recommenders trained on MovieLens 1M.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # Try to pull models from HF Hub on first boot (e.g. on HF Spaces).
    if os.environ.get("PULL_MODELS_ON_STARTUP", "0") == "1":
        try:
            from src.utils.hf_hub import pull_models
            pull_models()
        except Exception as e:
            print(f"pull_models failed: {e}")
    # Warm the singleton
    try:
        get_recommender()
        print("Recommender loaded.")
    except Exception as e:
        print(f"Recommender load failed: {e}")


@app.get("/", include_in_schema=False)
def root():
    return {"name": "Netflix-Style Recommendation API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        rec = get_recommender()
        return HealthResponse(
            status="ok",
            n_users=rec.n_users,
            n_items=rec.n_items,
            metadata=rec.metadata,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {e}")


@app.get("/movies", response_model=MovieSearchResponse)
def list_movies(q: str | None = Query(default=None, description="Search by title substring"), limit: int = 20):
    rec = get_recommender()
    return MovieSearchResponse(movies=rec.list_movies(query=q, limit=limit))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    rec = get_recommender()
    try:
        result = rec.recommend(
            model=req.model,
            user_id=req.user_id,
            history_movie_ids=req.history_movie_ids,
            k=req.k,
        )
        return RecommendResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
def compare(req: RecommendRequest):
    rec = get_recommender()
    results = rec.compare(user_id=req.user_id, history_movie_ids=req.history_movie_ids, k=req.k)
    return CompareResponse(user_id=req.user_id, k=req.k, results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=config.API_HOST, port=config.API_PORT, reload=False)
