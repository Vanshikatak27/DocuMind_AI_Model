
"""
End-to-end index build: Steps 1-6 chained together
-----------------------------------------------------
folder of PDFs -> pages -> cleaned text -> chunks -> embeddings -> VectorStore
"""

import numpy as np
from .ingest import load_and_clean
from .chunking import chunk_pages
from .vectorstore import VectorStore


def build_index(
    doc_folder: str,
    embedder,
    chunk_size: int = 800,
    overlap: int = 150,
    verbose: bool = True
) -> VectorStore:

    pages = load_and_clean(doc_folder)

    if verbose:
        print(
            f"[indexer] extracted {len(pages)} non-empty pages from "
            f"{len({p.doc_name for p in pages})} document(s)"
        )

    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap
    )

    if verbose:
        print(
            f"[indexer] split into {len(chunks)} chunks "
            f"(chunk_size={chunk_size}, overlap={overlap})"
        )

    vectors = np.asarray(
        embedder.encode([c.text for c in chunks]),
        dtype="float32"
    )

    vectors /= np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True),
        1e-12
    )

    store = VectorStore(dim=vectors.shape[1])
    store.add(chunks, vectors)

    if verbose:
        print(
            f"[indexer] indexed {len(chunks)} chunk embeddings "
            f"(dim={vectors.shape[1]}) into the vector store"
        )

    return store
