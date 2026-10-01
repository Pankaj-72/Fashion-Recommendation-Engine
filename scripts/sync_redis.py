"""Refresh Redis user-vector and item metadata keys from exported artifacts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.cache.redis_store import RedisFeatureStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    root = Path(args.artifacts)
    store = RedisFeatureStore()
    if store.client is None:
        raise RuntimeError("Redis client is unavailable; install redis and start the Redis service")
    try:
        store.client.ping()
    except Exception as exc:
        raise RuntimeError("Redis is not reachable; start it before refreshing feature keys") from exc
    users = json.loads((root / "user_embeddings.json").read_text(encoding="utf-8"))
    items = pd.read_csv(root / "item_features.csv")
    stored = 0
    for user_id, vector in users.items():
        stored += int(store.set_json("user_embedding", user_id, vector, ttl_seconds=7 * 24 * 3600))
    for row in items.to_dict("records"):
        article_id = str(row.pop("article_id"))
        # Keep cache payload compact and JSON-safe.
        metadata = {key: (value.item() if hasattr(value, "item") else value) for key, value in row.items() if key in {"product_type_name", "product_group_name", "colour_group_name", "section_name", "purchase_count"}}
        stored += int(store.set_json("item", article_id, metadata, ttl_seconds=7 * 24 * 3600))
    expected = len(users) + len(items)
    if stored != expected:
        raise RuntimeError(f"Redis feature refresh incomplete: stored {stored} of {expected} records")
    print(f"Attempted feature refresh; successful writes: {stored}")


if __name__ == "__main__":
    main()