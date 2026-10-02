"""Config-driven local TFRS training entry point and reusable training function."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.models.two_tower import build_model
from src.training.dataset import sample_training_pairs

LOGGER = logging.getLogger(__name__)


def _feature_vocab(frame: pd.DataFrame, columns: list[str]):
    return {column: frame[column].fillna("UNKNOWN").astype(str).unique().tolist() for column in columns if column in frame}


def train_from_config(config_path="config/config.yaml"):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_dir = Path(config["data"]["processed_dir"])
    train = pd.read_csv(data_dir / "train.csv", dtype={"customer_id": "string", "article_id": "string"}, parse_dates=["t_dat"])
    users = pd.read_csv(data_dir / "user_features.csv", dtype={"customer_id": "string"})
    items = pd.read_csv(data_dir / "item_features.csv", dtype={"article_id": "string"})
    pairs = sample_training_pairs(train, config["training"]["negative_ratio"], config["project"]["random_seed"])
    if pairs.empty:
        raise ValueError("No training interactions found; run preprocessing and feature engineering first")
    user_side = ["club_member_status", "fashion_news_frequency", "last_purchase_weekday", "last_purchase_month"]
    item_side = ["product_type_name", "product_group_name", "colour_group_name", "index_group_name"]
    user_vocab = _feature_vocab(users, user_side)
    item_vocab = _feature_vocab(items, item_side)
    model = build_model(users.customer_id.astype(str).unique(), items.article_id.astype(str).unique(),
                        user_vocab, item_vocab, config["features"]["embedding_dim"], config["training"]["learning_rate"])
    user_lookup = users.set_index("customer_id")
    item_lookup = items.set_index("article_id")
    user_input = {"entity_id": pairs.customer_id.astype(str).to_numpy()}
    item_input = {"entity_id": pairs.article_id.astype(str).to_numpy()}
    for name in user_vocab:
        user_input[name] = pairs.customer_id.map(user_lookup[name]).fillna("UNKNOWN").astype(str).to_numpy()
    for name in item_vocab:
        item_input[name] = pairs.article_id.map(item_lookup[name]).fillna("UNKNOWN").astype(str).to_numpy()
    numeric_user = [column for column in ("age", "purchase_count", "purchase_frequency", "recency_days") if column in users]
    if numeric_user:
        numeric_table = users.set_index("customer_id")[numeric_user].apply(pd.to_numeric, errors="coerce").fillna(0)
        user_input["numeric"] = numeric_table.reindex(pairs.customer_id).to_numpy(dtype="float32")
    # User tower numeric input is a vector of age/count/frequency/recency values.
    dataset = __import__("tensorflow").data.Dataset.from_tensor_slices({"user": user_input, "item": item_input, "label": pairs.label.to_numpy(dtype="float32")})
    dataset = dataset.shuffle(min(len(pairs), 100_000), seed=config["project"]["random_seed"]).batch(config["training"]["batch_size"]).prefetch(__import__("tensorflow").data.AUTOTUNE)
    model.fit(dataset, epochs=config["training"]["epochs"], verbose=1)
    out = Path(config["project"]["artifact_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_weights(out / "two_tower.weights.h5")
    (out / "model_metadata.yaml").write_text(yaml.safe_dump({"user_ids": users.customer_id.astype(str).tolist(), "item_ids": items.article_id.astype(str).tolist(), "embedding_dim": config["features"]["embedding_dim"]}), encoding="utf-8")
    LOGGER.info("Saved model weights to %s", out)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_from_config(args.config)


if __name__ == "__main__":
    main()