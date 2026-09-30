"""Create deterministic synthetic H&M-shaped data for development plumbing only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_tables():
    customers = pd.DataFrame({
        "customer_id": [f"customer_{i:03d}" for i in range(1, 9)],
        "age": [22, 31, 44, 27, 36, 52, 19, 41],
        "club_member_status": ["ACTIVE" if i % 2 else "PRE-CREATE" for i in range(8)],
        "fashion_news_frequency": ["Regularly" if i % 3 else "NONE" for i in range(8)],
    })
    articles = pd.DataFrame({
        "article_id": [f"{i:010d}" for i in range(1001, 1013)],
        "product_type_name": ["Trousers", "Dress", "Sweater"] * 4,
        "product_group_name": ["Garment Lower body", "Garment Full body", "Garment Upper body"] * 4,
        "colour_group_name": ["Black", "Blue", "Beige", "Red"] * 3,
        "index_group_name": ["Ladieswear"] * 12,
    })
    rows = []
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    for day, date in enumerate(dates):
        for user in range(1, 9):
            # Deterministic preference signal and repeated purchases.
            item = 1001 + ((user * 3 + day // 4) % 12)
            rows.append((f"customer_{user:03d}", f"{item:010d}", date))
            if day % 6 == 0:
                rows.append((f"customer_{user:03d}", f"{1001 + (user % 12):010d}", date))
    transactions = pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat"])
    return transactions, customers, articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/sample")
    args = parser.parse_args()
    from src.data.preprocessing import chronological_split, clean_tables, save_processed

    tx, customers, articles = clean_tables(*sample_tables())
    train, valid = chronological_split(tx, validation_days=7)
    save_processed(train, valid, customers, articles, Path(args.output))
    print(f"Wrote development-only fixture: {len(train)} train, {len(valid)} validation events")


if __name__ == "__main__":
    main()