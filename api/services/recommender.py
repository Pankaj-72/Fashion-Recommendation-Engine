"""Business service for recommendations; HTTP routing stays deliberately thin."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.cache.redis_store import RedisFeatureStore
from src.retrieval.ann import CandidateIndex


class RecommenderService:
    def __init__(self, artifact_dir=None, redis_store=None):
        self.artifact_dir = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))
        self.redis = redis_store or RedisFeatureStore()
        self.users = self._read_table("user_features.csv", "customer_id")
        self.items = self._read_table("item_features.csv", "article_id")
        try:
            self.index = CandidateIndex.load(self.artifact_dir)
        except (OSError, ValueError):
            self.index = None

    def _read_table(self, name, index):
        path = self.artifact_dir / name
        if path.exists():
            try:
                frame = pd.read_csv(path, dtype={index: "string"})
                if "recent_article_ids" in frame:
                    frame["recent_article_ids"] = frame["recent_article_ids"].apply(lambda value: json.loads(value) if isinstance(value, str) and value else [])
                return frame.set_index(index, drop=False)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def health(self):
        return {"status": "ok", "model_ready": self.index is not None, "item_count": len(self.index.item_ids) if self.index else 0}

    def recommend(self, customer_id: str, k: int = 10):
        if self.index is None:
            raise RuntimeError("Recommendation artifacts are not ready; train a model and build the ANN index")
        user = self.users.loc[customer_id] if customer_id in self.users.index else None
        if user is None:
            # Cold start: recommend training-popular articles if scores are available.
            if self.items.empty or "purchase_count" not in self.items:
                raise KeyError(f"Unknown customer: {customer_id}")
            candidates = self.items.sort_values("purchase_count", ascending=False).head(k)
            return [{"article_id": str(row.article_id), "score": 0.0, "metadata": _metadata(row)} for row in candidates.itertuples()]
        cached = self.redis.get_json("recommendations", customer_id)
        if cached:
            return cached[:k]
        # Exported per-user embeddings are optional precomputed serving artifacts.
        vector = self.redis.get_json("user_embedding", customer_id)
        embedding_file = self.artifact_dir / "user_embeddings.json"
        if vector is None and embedding_file.exists():
            import json
            vector = json.loads(embedding_file.read_text(encoding="utf-8")).get(customer_id)
        if vector is None:
            raise RuntimeError("No user embedding found; generate user embeddings after training")
        results = self.index.search(np.asarray(vector, dtype="float32"), k)
        history = set(user.get("recent_article_ids", []) or [])
        results = [item for item in results if item["article_id"] not in history][:k]
        for item in results:
            cached_metadata = self.redis.get_json("item", item["article_id"])
            if cached_metadata:
                item["metadata"] = cached_metadata
            elif item["article_id"] in self.items.index:
                item["metadata"] = _metadata(self.items.loc[item["article_id"]])
            else:
                item["metadata"] = {}
        self.redis.set_json("recommendations", customer_id, results, ttl_seconds=900)
        return results


def _metadata(row):
    values = row._asdict() if hasattr(row, "_asdict") else row.to_dict()
    metadata = {}
    for key, value in values.items():
        if key in {"article_id", "purchase_count", "log_popularity"} or value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        metadata[key] = value.item() if hasattr(value, "item") else value
    return metadata
