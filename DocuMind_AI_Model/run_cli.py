"""
Quick terminal test of the DocuMind AI pipeline, using the modular
`documind` package (the same code used by `documind/app.py`'s Gradio UI).

Usage:
    python3 run_cli.py                          # ask a couple of demo questions
    python3 run_cli.py "your question here"      # ask one question
"""
import sys

from documind.embeddings import get_embedder
from documind.indexer import build_index
from documind.llm import get_llm
from documind.rag import DocuMindRAG

DOCS_DIR = "documents"


def main():
    embedder = get_embedder("sentence-transformers/all-MiniLM-L6-v2", allow_fallback=False)
    store = build_index(DOCS_DIR, embedder)
    llm = get_llm("hf", model_name="google/flan-t5-base")
    rag = DocuMindRAG(store, embedder, llm, top_k=4, min_similarity=0.30)

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = [
            "What happens in Jury 2 for a Model Making project?",
            "Can we change our Model Making topic after Jury 1?",
            "What is the minimum attendance percentage required to appear for the ESE?",
            "What is the exact attendance percentage for an event not mentioned in the documents?",
        ]

    for q in questions:
        result = rag.answer(q)
        print("\nQ:", q)
        print("A:", result.answer)
        if result.citations:
            print("Source:", "; ".join(result.citations))


if __name__ == "__main__":
    main()
