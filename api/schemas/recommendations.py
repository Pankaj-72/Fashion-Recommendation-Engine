from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    article_id: str
    score: float
    metadata: dict = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list[RecommendationItem]
    source: str = "ann"
