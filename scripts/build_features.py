"""Build features from the processed transaction/customer/article tables."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.features.build_features import build_feature_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["data"]["processed_dir"])
    build_feature_tables(pd.read_csv(root / "train.csv", dtype={"customer_id": "string", "article_id": "string"}, parse_dates=["t_dat"]),
                         pd.read_csv(root / "customers.csv", dtype={"customer_id": "string"}),
                         pd.read_csv(root / "articles.csv", dtype={"article_id": "string"}), root,
                         config["features"]["history_days"], config["features"]["max_history_items"])
    print(f"Wrote user and item features under {root}")


if __name__ == "__main__":
    main()
