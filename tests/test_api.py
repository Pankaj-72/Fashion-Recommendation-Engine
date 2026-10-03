from fastapi.testclient import TestClient

from api.main import create_app


class FakeRecommender:
    def health(self):
        return {"status": "ok", "model_ready": True, "item_count": 2}

    def recommend(self, customer_id, k):
        if customer_id == "unknown":
            raise KeyError(customer_id)
        return [{"article_id": "0001", "score": 0.8, "metadata": {"colour_group_name": "Blue"}}][:k]


def test_health_and_recommendation_endpoint():
    client = TestClient(create_app(FakeRecommender()))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/recommendations/u1?k=1").json()["recommendations"][0]["article_id"] == "0001"
    assert client.get("/recommendations/u1?k=0").status_code == 422
    assert client.get("/recommendations/unknown").status_code == 404


def test_missing_artifacts_report_not_ready(tmp_path):
    from api.services.recommender import RecommenderService

    client = TestClient(create_app(RecommenderService(artifact_dir=tmp_path)))
    assert client.get("/health").json()["model_ready"] is False
    assert client.get("/recommendations/u1").status_code == 503
