"""Training-history-only customer aggregates and recent purchase history."""

from __future__ import annotations

import pandas as pd


def build_user_features(train: pd.DataFrame, customers: pd.DataFrame, history_days: int = 90, max_history_items: int = 20):
    if history_days < 1 or max_history_items < 1:
        raise ValueError("history_days and max_history_items must be positive")
    if train.empty:
        base = customers[["customer_id"]].drop_duplicates().copy()
        base["purchase_count"] = 0
        base["purchase_frequency"] = 0.0
        base["recency_days"] = float(history_days)
        base["recent_article_ids"] = [[] for _ in range(len(base))]
    else:
        latest = train["t_dat"].max()
        tx = train.copy()
        tx["age_days"] = (latest - tx["t_dat"]).dt.days.clip(lower=0)
        grouped = tx.groupby("customer_id").agg(
            purchase_count=("article_id", "size"),
            recency_days=("age_days", "min"),
            active_days=("t_dat", "nunique"),
        ).reset_index()
        last_dates = tx.sort_values("t_dat").groupby("customer_id")["t_dat"].last().rename("last_purchase_date").reset_index()
        grouped = grouped.merge(last_dates, on="customer_id", how="left")
        grouped["last_purchase_weekday"] = grouped["last_purchase_date"].dt.dayofweek.astype(str)
        grouped["last_purchase_month"] = grouped["last_purchase_date"].dt.month.astype(str)
        grouped["purchase_frequency"] = grouped["purchase_count"] / grouped["active_days"].clip(lower=1)
        recent = tx[tx["t_dat"] >= latest - pd.Timedelta(days=history_days - 1)]
        histories = (recent.sort_values("t_dat").groupby("customer_id")["article_id"]
                     .apply(lambda items: list(dict.fromkeys(items.astype(str)))[-max_history_items:]).rename("recent_article_ids").reset_index())
        base = grouped.merge(histories, on="customer_id", how="left")
        base["recent_article_ids"] = base["recent_article_ids"].apply(lambda value: value if isinstance(value, list) else [])
    # Retain all known users, including cold-start users with zero history.
    result = customers.drop_duplicates("customer_id").merge(base, on="customer_id", how="left")
    result["purchase_count"] = result["purchase_count"].fillna(0).astype("int32")
    result["purchase_frequency"] = result["purchase_frequency"].fillna(0.0).astype("float32")
    result["recency_days"] = result["recency_days"].fillna(float(history_days)).astype("float32")
    result["recent_article_ids"] = result["recent_article_ids"].apply(lambda value: value if isinstance(value, list) else [])
    for column in ("last_purchase_weekday", "last_purchase_month"):
        if column not in result:
            result[column] = "UNKNOWN"
        result[column] = result[column].fillna("UNKNOWN").astype(str)
    if "age" in result:
        result["age"] = pd.to_numeric(result["age"], errors="coerce").fillna(-1).astype("float32")
    for column in ("club_member_status", "fashion_news_frequency"):
        if column in result:
            result[column] = result[column].fillna("UNKNOWN").astype(str)
    return result