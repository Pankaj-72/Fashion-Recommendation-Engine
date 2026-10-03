from fastapi import FastAPI, Request

from api.routes.recommendations import router
from api.services.recommender import RecommenderService


def create_app(recommender=None):
    app = FastAPI(title="Context-Aware Fashion Recommendation Engine", version="0.1.0",
                  description="Student project API for H&M purchase-based recommendations")
    app.state.recommender = recommender or RecommenderService()
    app.include_router(router)

    @app.get("/health", tags=["health"])
    def health(request: Request):
        return request.app.state.recommender.health()

    return app


app = create_app()
