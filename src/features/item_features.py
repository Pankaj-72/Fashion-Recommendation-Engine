"""Article-side categorical attributes and training-period popularity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_item_features(train: pd.DataFrame, articles: pd.DataFrame):
    items = articles.drop_duplicates("article_id").copy()
    counts = train.groupby("article_id").size().rename("purchase_count").reset_index() if not train.empty else pd.DataFrame(columns=["article_id", "purchase_count"])
    items = items.merge(counts, on="article_id", how="left")
    items["purchase_count"] = items["purchase_count"].fillna(0).astype("int32")
    items["log_popularity"] = np.log1p(items["purchase_count"]).astype("float32")
    # Keep useful, available H&M metadata; normalize absent categorical values.
    category_columns = [column for column in (
        "product_type_name", "product_group_name", "department_name", "section_name",
        "colour_group_name", "garment_group_name", "index_group_name", "index_name",
    ) if column in items]
    for column in category_columns:
        items[column] = items[column].fillna("UNKNOWN").astype(str)
    return items