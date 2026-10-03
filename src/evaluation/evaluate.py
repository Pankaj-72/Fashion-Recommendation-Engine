"""Evaluate ranked validation items against a simple train-popularity baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.metrics import evaluate_rankings
from src.retrieval.ann import CandidateIndex


def evaluate_popularity(config_path="config/config.yaml"):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base = Path(config["data"]["processed_dir"])
    train = pd.read_csv(base / "train.csv", dtype={"customer_id": "string", "article_id": "string"}, parse_dates=["t_dat"])
    valid = pd.read_csv(base / "validation.csv", dtype={"customer_id": "string", "article_id": "string"}, parse_dates=["t_dat"])
    # Always provide a baseline, and evaluate learned vectors when artifacts exist.
    popular = train.groupby("article_id").size().sort_values(ascending=False).index.astype(str).tolist()
    seen = train.groupby("customer_id")["article_id"].apply(lambda x: set(x.astype(str))).to_dict()
    relevant = valid.groupby("customer_id")["article_id"].apply(lambda x: set(x.astype(str))).to_dict()
    out = Path(config["project"]["artifact_dir"])
    relevant = {str(user): set(map(str, items)) for user, items in relevant.items()}
    learned = out / "user_embeddings.json"
    if (out / "item_ids.json").exists() and learned.exists():
        index = CandidateIndex.load(out)
        user_vectors = json.loads(learned.read_text(encoding="utf-8"))
        rankings = {user: [row["article_id"] for row in index.search(user_vectors[user], max(config["evaluation"]["ks"]) + len(seen.get(user, set())))
                           if row["article_id"] not in seen.get(user, set())]
                    for user in relevant if user in user_vectors}
        result = evaluate_rankings(rankings, relevant, config["evaluation"]["ks"])
        result["evaluation_source"] = "two_tower_ann"
    else:
        rankings = {str(user): [item for item in popular if item not in seen.get(user, set())] for user in relevant}
        result = evaluate_rankings(rankings, relevant, config["evaluation"]["ks"])
        result["evaluation_source"] = "popularity_baseline"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_json(out / "evaluation_report.json", orient="records", indent=2)
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    evaluate_popularity(args.config)


if __name__ == "__main__":
    main()