"""Build consistent train-only customer and article feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.item_features import build_item_features
from src.features.user_features import build_user_features


def build_feature_tables(train: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame, output_dir: str | Path,
                        history_days: int = 90, max_history_items: int = 20):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    users = build_user_features(train, customers, history_days, max_history_items)
    items = build_item_features(train, articles)
    import json
    users.assign(recent_article_ids=users["recent_article_ids"].map(json.dumps)).to_csv(output / "user_features.csv", index=False)
    items.to_csv(output / "item_features.csv", index=False)
    return users, items