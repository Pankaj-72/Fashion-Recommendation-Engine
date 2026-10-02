"""FAISS inner-product candidate index with exact NumPy fallback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CandidateIndex:
    def __init__(self, vectors: np.ndarray, item_ids: list[str]):
        if len(vectors) != len(item_ids):
            raise ValueError("Each item vector must have exactly one corresponding item ID")
        self.vectors = np.asarray(vectors, dtype="float32")
        self.item_ids = [str(item) for item in item_ids]
        self._index = None
        if len(self.vectors):
            try:
                import faiss
                faiss.normalize_L2(self.vectors)
                self._index = faiss.IndexFlatIP(self.vectors.shape[1])
                self._index.add(self.vectors)
            except ImportError:
                norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                self.vectors = self.vectors / np.maximum(norms, 1e-12)

    def search(self, user_vector, k: int = 10):
        if k < 1:
            raise ValueError("k must be positive")
        if not len(self.item_ids):
            return []
        query = np.asarray(user_vector, dtype="float32").reshape(1, -1)
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        query /= np.maximum(norm, 1e-12)
        count = min(k, len(self.item_ids))
        if self._index is not None:
            scores, indices = self._index.search(query, count)
            pairs = zip(indices[0], scores[0])
        else:
            scores = self.vectors @ query[0]
            indices = np.argsort(-scores)[:count]
            pairs = ((index, scores[index]) for index in indices)
        return [{"article_id": self.item_ids[int(index)], "score": float(score)} for index, score in pairs if index >= 0]

    def save(self, directory: str | Path):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "item_embeddings.npy", self.vectors)
        (output / "item_ids.json").write_text(json.dumps(self.item_ids), encoding="utf-8")
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(output / "items.faiss"))

    @classmethod
    def load(cls, directory: str | Path):
        path = Path(directory)
        result = cls(np.load(path / "item_embeddings.npy"), json.loads((path / "item_ids.json").read_text(encoding="utf-8")))
        try:
            import faiss
            index_path = path / "items.faiss"
            if index_path.exists():
                result._index = faiss.read_index(str(index_path))
        except ImportError:
            pass
        return result
