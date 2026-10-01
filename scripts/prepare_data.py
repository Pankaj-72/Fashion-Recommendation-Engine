"""CLI: clean Kaggle source CSVs and create a chronological split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.loader import load_tables
from src.data.preprocessing import chronological_split, clean_tables, save_processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = config["data"]
    tx, customers, articles = clean_tables(*load_tables(data["transactions"], data["customers"], data["articles"]))
    if data.get("max_rows"):
        tx = tx.head(int(data["max_rows"]))
    train, valid = chronological_split(tx, int(data["validation_days"]))
    save_processed(train, valid, customers, articles, data["processed_dir"])
    print(f"Saved {len(train):,} training and {len(valid):,} validation transactions")


if __name__ == "__main__":
    main()