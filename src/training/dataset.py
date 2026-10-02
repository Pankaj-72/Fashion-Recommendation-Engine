"""Build implicit-feedback pairs with uniform, train-only negative sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_training_pairs(train: pd.DataFrame, negative_ratio: int = 4, seed: int = 42) -> pd.DataFrame:
    if negative_ratio < 0:
        raise ValueError("negative_ratio cannot be negative")
    if train.empty:
        return pd.DataFrame(columns=["customer_id", "article_id", "label"])
    positives = train[["customer_id", "article_id"]].drop_duplicates().copy()
    positives["label"] = 1.0
    items = train["article_id"].drop_duplicates().to_numpy()
    positive_by_user = train.groupby("customer_id")["article_id"].apply(set).to_dict()
    rng = np.random.default_rng(seed)
    negatives = []
    for user in positives["customer_id"].unique():
        eligible = np.asarray([item for item in items if item not in positive_by_user[user]], dtype=object)
        count = min(len(eligible), len(positive_by_user[user]) * negative_ratio)
        if count:
            for item in rng.choice(eligible, size=count, replace=False):
                negatives.append((user, item, 0.0))
    negative_frame = pd.DataFrame(negatives, columns=["customer_id", "article_id", "label"])
    return pd.concat([positives, negative_frame], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)