"""
Step 6: Vector Database (FAISS)
---------------------------------
Stores every chunk's embedding for fast similarity search, and keeps the
parallel metadata (which chunk, which document, which page) needed to
answer "where did this come from?" after retrieval.
"""
import json
import os
from dataclasses import asdict
from typing import List, Tuple

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray):
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        query_vec = query_vec.reshape(1, -1)
        scores, idxs = self.index.search(query_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.json"), "w") as f:
            json.dump([asdict(c) for c in self.chunks], f)

    @classmethod
    def load(cls, folder: str, dim: int):
        store = cls(dim)
        store.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.json")) as f:
            store.chunks = [Chunk(**d) for d in json.load(f)]
        return store
