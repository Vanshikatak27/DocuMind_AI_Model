"""
Step 5: Embeddings
--------------------
Wraps a Sentence-Transformers model that converts text into a numerical
vector capturing its meaning, so that "Jury 2 requirements" and "what
happens during the second jury" land close together in vector space even
though they share almost no words.

A tiny hash-based fallback is included purely so the rest of the pipeline
can be exercised in offline environments with no internet access (e.g. a
sandboxed CI check). It is NOT a semantic embedding and should not be used
for a real demo — `get_embedder()` always prefers the real model and only
falls back with a loud warning if the model truly cannot be downloaded.
"""
import hashlib
import numpy as np
from typing import List


class SentenceTransformerEmbedder:
    """Real embedding backend used for the actual DocuMind AI demo."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class SimpleHashEmbedder:
    """Deterministic, offline, NON-semantic fallback (dev/testing only)."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.model_name = "offline-hash-fallback (NOT semantic — dev/testing only)"

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", allow_fallback: bool = True):
    """Try to load the real embedding model; fall back to the offline dev
    embedder only if `allow_fallback=True` and the download genuinely fails
    (e.g. no internet). Always prints which one ended up in use."""
    try:
        embedder = SentenceTransformerEmbedder(model_name)
        print(f"[embeddings] using real model: {model_name}")
        return embedder
    except Exception as e:
        if not allow_fallback:
            raise
        print(f"[embeddings] WARNING: could not load '{model_name}' ({e}).")
        print("[embeddings] Falling back to the NON-semantic offline embedder — "
              "for a real demo, run this on a machine/Colab with internet access.")
        return SimpleHashEmbedder()
