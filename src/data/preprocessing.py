"""Validate source tables, normalize interactions, and split by event time."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import ARTICLE_COLUMNS, CUSTOMER_COLUMNS, TRANSACTION_COLUMNS, require_columns


def clean_tables(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame):
    """Return cleaned copies; training split is strictly earlier than validation."""
    require_columns(transactions, TRANSACTION_COLUMNS, "transactions")
    require_columns(customers, CUSTOMER_COLUMNS, "customers")
    require_columns(articles, ARTICLE_COLUMNS, "articles")
    tx = transactions.copy()
    tx["customer_id"] = tx["customer_id"].astype("string").str.strip()
    tx["article_id"] = tx["article_id"].astype("string").str.strip().str.zfill(10)
    tx["t_dat"] = pd.to_datetime(tx["t_dat"], errors="coerce", utc=False)
    tx = tx.dropna(subset=["customer_id", "article_id", "t_dat"])
    tx = tx[(tx["customer_id"] != "") & (tx["article_id"] != "")]
    tx = tx.drop_duplicates(["customer_id", "article_id", "t_dat"]).sort_values("t_dat")

    customer = customers.copy()
    customer["customer_id"] = customer["customer_id"].astype("string").str.strip()
    customer = customer.dropna(subset=["customer_id"]).drop_duplicates("customer_id", keep="last")
    article = articles.copy()
    article["article_id"] = article["article_id"].astype("string").str.strip().str.zfill(10)
    article = article.dropna(subset=["article_id"]).drop_duplicates("article_id", keep="last")
    return tx.reset_index(drop=True), customer.reset_index(drop=True), article.reset_index(drop=True)


def chronological_split(transactions: pd.DataFrame, validation_days: int = 7):
    """Hold out the most recent N calendar days; keep users/items train-only as-is."""
    if validation_days < 1:
        raise ValueError("validation_days must be at least 1")
    if transactions.empty:
        return transactions.copy(), transactions.copy()
    cutoff = transactions["t_dat"].max().normalize() - pd.Timedelta(days=validation_days - 1)
    train = transactions[transactions["t_dat"] < cutoff].copy()
    valid = transactions[transactions["t_dat"] >= cutoff].copy()
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def save_processed(train, valid, customers, articles, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", valid), ("customers", customers), ("articles", articles)):
        frame.to_csv(output / f"{name}.csv", index=False)
        