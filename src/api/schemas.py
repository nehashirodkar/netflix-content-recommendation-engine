from typing import Literal, Optional

from pydantic import BaseModel, Field


ModelChoice = Literal["svd", "ncf", "content", "hybrid", "auto"]


class RecommendRequest(BaseModel):
    user_id: Optional[int] = Field(default=None, description="MovieLens user_id; omit for cold-start")
    history_movie_ids: Optional[list[int]] = Field(default=None, description="Optional MovieLens movie_ids to seed content-based")
    model: ModelChoice = "hybrid"
    k: int = Field(default=10, ge=1, le=100)


class Recommendation(BaseModel):
    rank: int
    item_idx: int
    movie_id: int
    title: str
    score: Optional[float] = None


class RecommendResponse(BaseModel):
    model_used: str
    cold_start: bool
    recommendations: list[Recommendation]


class CompareResponse(BaseModel):
    user_id: Optional[int]
    k: int
    results: dict


class HealthResponse(BaseModel):
    status: str
    n_users: int
    n_items: int
    metadata: dict


class MovieSearchResponse(BaseModel):
    movies: list[dict]
