"""Load the three Kaggle CSVs using their documented column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import ARTICLE_COLUMNS, CUSTOMER_COLUMNS, TRANSACTION_COLUMNS, require_columns


def load_tables(transactions_path: str | Path, customers_path: str | Path, articles_path: str | Path):
    transactions = pd.read_csv(transactions_path, dtype={"customer_id": "string", "article_id": "string"})
    customers = pd.read_csv(customers_path, dtype={"customer_id": "string"})
    articles = pd.read_csv(articles_path, dtype={"article_id": "string"})
    require_columns(transactions, TRANSACTION_COLUMNS, "transactions")
    require_columns(customers, CUSTOMER_COLUMNS, "customers")
    require_columns(articles, ARTICLE_COLUMNS, "articles")
    return transactions, customers, articles