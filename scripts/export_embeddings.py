"""Restore trained model, export its item vectors and user vectors for serving."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.models.two_tower import build_model
from src.retrieval.embeddings import export_item_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["data"]["processed_dir"])
    artifacts = Path(config["project"]["artifact_dir"])
    users, items = pd.read_csv(root / "user_features.csv", dtype={"customer_id": "string"}), pd.read_csv(root / "item_features.csv", dtype={"article_id": "string"})
    user_side = [c for c in ("club_member_status", "fashion_news_frequency", "last_purchase_weekday", "last_purchase_month") if c in users]
    item_side = [c for c in ("product_type_name", "product_group_name", "colour_group_name", "index_group_name") if c in items]
    def vocab(frame, columns):
        return {c: frame[c].fillna("UNKNOWN").astype(str).unique().tolist() for c in columns}
    model = build_model(users.customer_id.astype(str).unique(), items.article_id.astype(str).unique(), vocab(users, user_side), vocab(items, item_side), config["features"]["embedding_dim"], config["training"]["learning_rate"])
    # Build variables before restoring weights.
    numeric_cols = [c for c in ("age", "purchase_count", "purchase_frequency", "recency_days") if c in users]
    user_build = {"entity_id": np.asarray(users.customer_id.astype(str).head(1)), **{c: np.asarray(users[c].fillna("UNKNOWN").astype(str).head(1)) for c in user_side}}
    if numeric_cols:
        user_build["numeric"] = users[numeric_cols].head(1).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype="float32")
    model.user_tower(user_build)
    model.item_tower({"entity_id": np.asarray(items.article_id.astype(str).head(1)), **{c: np.asarray(items[c].fillna("UNKNOWN").astype(str).head(1)) for c in item_side}})
    model.load_weights(artifacts / "two_tower.weights.h5")
    export_item_embeddings(model, items, artifacts)
    user_vectors = {}
    for row in users.to_dict("records"):
        features = {"entity_id": np.asarray([str(row["customer_id"])])}
        features.update({c: np.asarray([str(row.get(c) or "UNKNOWN")]) for c in user_side})
        numeric_cols = [c for c in ("age", "purchase_count", "purchase_frequency", "recency_days") if c in users]
        if numeric_cols:
            features["numeric"] = np.asarray([[float(row.get(c) or 0) for c in numeric_cols]], dtype="float32")
        user_vectors[str(row["customer_id"])] = model.user_tower(features, training=False).numpy()[0].tolist()
    (artifacts / "user_embeddings.json").write_text(json.dumps(user_vectors), encoding="utf-8")
    users.to_csv(artifacts / "user_features.csv", index=False)
    items.to_csv(artifacts / "item_features.csv", index=False)
    print(f"Exported vectors for {len(user_vectors)} customers and {len(items)} articles")


if __name__ == "__main__":
    main()
