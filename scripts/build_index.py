"""Build an ANN index from an existing exported item embedding matrix."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.retrieval.ann import CandidateIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    path = Path(args.artifacts)
    index = CandidateIndex(np.load(path / "item_embeddings.npy"), json.loads((path / "item_ids.json").read_text(encoding="utf-8")))
    index.save(path)
    print(f"Built index for {len(index.item_ids)} articles")


if __name__ == "__main__":
    main()
