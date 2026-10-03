from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas.recommendations import RecommendationResponse

router = APIRouter()


@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: str, request: Request, k: int = Query(default=10, ge=1, le=100)):
    try:
        result = request.app.state.recommender.recommend(user_id, k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecommendationResponse(customer_id=user_id, recommendations=result)
