"""
Step 1-2: Document Ingestion & Text Extraction
-----------------------------------------------
Loads every PDF in a folder and extracts text page-by-page, keeping the
document name and page number attached to every page — this metadata is
what lets us cite an exact source later.
"""
import os
import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class Page:
    doc_name: str
    page_num: int          # 1-indexed, matches what a human would cite
    text: str


def load_pdfs(folder: str) -> List[Page]:
    """Load every .pdf file in `folder` and return one Page per PDF page."""
    pages: List[Page] = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Document folder not found: {folder}")

    pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder}")

    for fname in pdf_files:
        path = os.path.join(folder, fname)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            pages.append(Page(doc_name=fname, page_num=i + 1, text=raw))
    return pages


# Step 3: Text Cleaning ------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")  # re-join words split across a line by a hyphen


def clean_text(text: str) -> str:
    """Light-touch cleanup: fix hyphenated line-breaks, collapse whitespace,
    drop stray page-number-only lines. Deliberately conservative — we want
    to keep the document's actual wording intact for grounded answers."""
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    # drop lines that are just a page number or empty after stripping
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.strip().isdigit()]
    return "\n".join(lines).strip()


def load_and_clean(folder: str) -> List[Page]:
    pages = load_pdfs(folder)
    for p in pages:
        p.text = clean_text(p.text)
    return [p for p in pages if p.text]  # drop blank pages (e.g. scanned images with no text layer)
