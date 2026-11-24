"""
Steps 7-9: Retrieval, Prompt Construction, Generation, Citation & Grounding
-----------------------------------------------------------------------------
This module ties the whole pipeline together: embed the query, search the
vector store, build a grounded prompt from the retrieved chunks, call the
LLM, and attach a citation — or refuse to answer if nothing relevant enough
was retrieved.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .chunking import Chunk

NOT_FOUND_MESSAGE = "I could not find sufficient information about this in the provided documents."

SYSTEM_INSTRUCTION = (
    "You are an academic assistant. Answer the user's question using ONLY the "
    "provided document context. Do not use any outside knowledge. "
    "If the context does not contain enough information to answer, respond "
    f'exactly with: "{NOT_FOUND_MESSAGE}"'
)


@dataclass
class RagResult:
    question: str
    answer: str
    grounded: bool
    citations: List[str]
    retrieved: List[Chunk]
    scores: List[float]


# ---------------------------------------------------------------------------
# Step 7: Retrieval helper (query embedding + FAISS search) lives in rag.py
# so it can apply the grounding-threshold decision right where the score is.
# ---------------------------------------------------------------------------



def retrieve(store, embedder, query: str, top_k: int = 4):
    q_vec = np.asarray(
        embedder.encode([query])[0],
        dtype="float32"
    )
    q_vec /= max(float(np.linalg.norm(q_vec)), 1e-12)
    return store.search(q_vec, top_k=top_k)


# ---------------------------------------------------------------------------
# Step 8: Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(question: str, retrieved: List[Chunk]) -> str:
    context_blocks = []
    for i, c in enumerate(retrieved, start=1):
        context_blocks.append(f"[{i}] (Source: {c.doc_name}, Page {c.page_num})\n{c.text}")
    context = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite which numbered source you used):"
    )


# ---------------------------------------------------------------------------
# Step 9-11: full pipeline orchestration, incl. grounding + citations
# ---------------------------------------------------------------------------

class DocuMindRAG:
    def __init__(self, store, embedder, llm, top_k: int = 4, min_similarity: float = 0.30):
        """
        store:          a VectorStore already populated with document chunks
        embedder:       the same embedder used to build the store
        llm:            an object exposing .generate(prompt: str) -> str
        top_k:          how many chunks to retrieve per question
        min_similarity: retrieval-score floor below which we refuse to
                         answer rather than let the LLM guess from weak
                         context (extra grounding safeguard on top of the
                         prompt instruction itself)
        """
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.min_similarity = min_similarity

    def answer(self, question: str) -> RagResult:
        results = retrieve(self.store, self.embedder, question, top_k=self.top_k)
        if not results or results[0][1] < self.min_similarity:
            return RagResult(
                question=question, answer=NOT_FOUND_MESSAGE, grounded=False,
                citations=[], retrieved=[c for c, _ in results], scores=[s for _, s in results],
            )

        chunks = [c for c, _ in results]
        scores = [s for _, s in results]
        prompt = build_prompt(question, chunks)
        raw_answer = self.llm.generate(prompt).strip()

        grounded = NOT_FOUND_MESSAGE.lower() not in raw_answer.lower()
        citations = []
        if grounded:
            # cite the single most relevant chunk; multi-source answers can
            # cite more by scanning which [n] markers the model referenced
            top_chunk = chunks[0]
            citations = [f"{top_chunk.doc_name} — Page {top_chunk.page_num}"]

        return RagResult(
            question=question, answer=raw_answer, grounded=grounded,
            citations=citations, retrieved=chunks, scores=scores,
        )
