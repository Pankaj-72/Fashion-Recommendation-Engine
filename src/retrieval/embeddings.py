"""Export ordered item vectors and IDs from a trained two-tower model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_item_embeddings(model, items, output_dir: str | Path, batch_size: int = 1024):
    ids = items["article_id"].astype(str).tolist()
    vectors = []
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        inputs = {"entity_id": np.asarray(batch_ids)}
        for name in model.item_tower.side:
            inputs[name] = items.iloc[start:start + batch_size][name].fillna("UNKNOWN").astype(str).to_numpy()
        vectors.append(model.item_tower(inputs, training=False).numpy())
    matrix = np.concatenate(vectors).astype("float32") if vectors else np.empty((0, 0), dtype="float32")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "item_embeddings.npy", matrix)
    (output / "item_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return matrix, ids