"""
Step 4: Chunking
-----------------
Splits each page's text into smaller, overlapping chunks so that (a) each
retrieved piece of context is focused enough for the LLM to use well, and
(b) we stay well within the LLM's context window even with several
documents in play. Every chunk keeps a reference to exactly where it came
from — document name, page number and a chunk id — which is what makes
citations possible later.
"""
from dataclasses import dataclass, field
from typing import List
import itertools

from .ingest import Page

_counter = itertools.count()


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_num: int
    text: str


def chunk_pages(pages: List[Page], chunk_size: int = 800, overlap: int = 150) -> List[Chunk]:
    """Character-based sliding-window chunking with overlap.

    chunk_size / overlap are in characters, not tokens — simple, fast, and a
    reasonable default for short academic-style documents. For longer or
    denser documents, a sentence-aware splitter would reduce mid-sentence
    cuts, at the cost of a bit more complexity.
    """
    chunks: List[Chunk] = []
    for page in pages:
        text = page.text
        if len(text) <= chunk_size:
            chunks.append(_make_chunk(page, text))
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end]
            chunks.append(_make_chunk(page, piece))
            if end == len(text):
                break
            start = end - overlap  # step forward, re-covering `overlap` chars for context continuity
    return chunks


def _make_chunk(page: Page, text: str) -> Chunk:
    cid = f"c{next(_counter):05d}"
    return Chunk(chunk_id=cid, doc_name=page.doc_name, page_num=page.page_num, text=text.strip())
