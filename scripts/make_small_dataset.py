"""CLI: make a tiny copy of the H&M raw data (~N transaction rows) for fast experiments.

Whole customers (with their full purchase history) are sampled instead of cutting rows, so the
chronological train/validation split and the per-user features still make sense.
The 3.5 GB CSV is streamed in chunks, so this needs very little RAM.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

NEEDED = {"t_dat", "customer_id", "article_id"}


def read_transactions(path, customer_ids, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream the huge CSV in chunks and keep only rows of the wanted customers."""
    keep, parts, total, kept = set(customer_ids), [], 0, 0
    reader = pd.read_csv(path, usecols=lambda c: c in NEEDED, dtype={"customer_id": str, "article_id": str}, chunksize=chunksize)
    for chunk in reader:
        total += len(chunk)
        chunk = chunk[chunk["customer_id"].isin(keep)]
        kept += len(chunk)
        parts.append(chunk)
        print(f"  read {total:,} rows, kept {kept:,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/raw_small")
    parser.add_argument("--rows", type=int, default=20_000, help="max transaction rows to keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(raw / "customers.csv", dtype={"customer_id": str})
    # H&M averages ~23 purchases/customer, so oversample candidates ~2x and trim to the row budget below.
    n_candidates = min(len(customers), max(1000, args.rows // 8))
    candidates = customers.sample(n=n_candidates, random_state=args.seed)
    print(f"Scanning transactions for {n_candidates:,} candidate customers ...", flush=True)
    tx = read_transactions(raw / "transactions_train.csv", candidates["customer_id"])

    rng = np.random.default_rng(args.seed)
    counts = tx.groupby("customer_id").size()
    order = counts.index.to_numpy()[rng.permutation(len(counts))]
    cumulative = counts.loc[order].cumsum().to_numpy()
    chosen = set(order[cumulative <= args.rows])
    tx = tx[tx["customer_id"].isin(chosen)].sort_values("t_dat")
    if len(tx) < args.rows * 0.8:
        print(f"WARNING: only {len(tx):,} rows found; rerun with a different --seed or a larger --rows")

    tx.to_csv(out / "transactions_train.csv", index=False)
    candidates[candidates["customer_id"].isin(chosen)].to_csv(out / "customers.csv", index=False)
    shutil.copy(raw / "articles.csv", out / "articles.csv")
    print(f"Wrote {len(tx):,} transactions, {len(chosen):,} customers, dates {tx['t_dat'].min()} .. {tx['t_dat'].max()} -> {out}")
    print("Now set in config/config.yaml:  data.transactions/customers/articles -> " + str(out) + "/...  and  max_users: null")


if __name__ == "__main__":
    main()