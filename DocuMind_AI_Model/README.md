# DocuMind AI
Multi-Document Retrieval-Augmented Generation with Source Citations — **Topic 183**

A working implementation of the pipeline: **ingest PDFs → chunk → embed → FAISS vector
database → retrieve → LLM generation grounded in retrieved context → answer with source
citation → evaluate**.

## The two ways to use this project

### 1. `DocuMind_AI_Colab.ipynb` — the main deliverable
A single self-contained notebook, meant to be opened in **Google Colab** and run top to
bottom. It builds the whole pipeline inline, step by step (matching the implementation flow
in the problem statement), generates five sample academic PDFs to demo against out of the
box, runs the 27-question evaluation set, and finishes with a live Gradio demo (`share=True`
gives you a public link for a jury demo). This is the notebook to present.

Just upload it to [colab.research.google.com](https://colab.research.google.com), or open it
from Google Drive, and run every cell in order. The first run downloads two small open models
from Hugging Face (an embedding model and a generation model) — this needs the internet
access a normal Colab runtime already has, no API key required.

### 2. `documind/` — the same pipeline as a proper Python package
A cleaner, modular version of the exact same logic, useful if you'd rather run things from a
terminal, reuse pieces in another script, or show the jury a "real project structure" instead
of one big notebook.

```
documind/
  ingest.py       Step 1-2: load PDFs, extract text per page (PyPDF)
  chunking.py     Step 3-4: clean + split into overlapping chunks with metadata
  embeddings.py   Step 5: Sentence-Transformers embedding wrapper
  vectorstore.py  Step 6: FAISS index + metadata store (save/load)
  indexer.py      chains ingest -> chunk -> embed -> vector store
  llm.py          Step 9: pluggable LLM backends (local HF model / OpenAI API / dev stub)
  rag.py          Step 7-11: retrieval, prompt construction, grounding, citations
  evaluate.py     Step 12: runs eval_questions.csv through the pipeline, scores retrieval
                  accuracy + refusal rate automatically, leaves answer/citation correctness
                  for manual grading
  app.py          Step 13: the Gradio interface
```

## Quick start (local machine or Colab terminal)

```bash
pip install -r requirements.txt
python3 make_sample_docs.py      # regenerates the 5 sample PDFs into documents/ (already included)
python3 run_cli.py               # asks a few demo questions from the terminal
python3 -m documind.app          # launches the Gradio UI at http://127.0.0.1:7860
```

To evaluate:
```python
from documind.embeddings import get_embedder
from documind.indexer import build_index
from documind.llm import get_llm
from documind.rag import DocuMindRAG
from documind.evaluate import run_evaluation

embedder = get_embedder(allow_fallback=False)
store = build_index("documents", embedder)
rag = DocuMindRAG(store, embedder, get_llm("hf"))
run_evaluation(rag, "eval_questions.csv")   # writes eval_results.csv
```

## Using your own documents instead of the sample set
Replace the files in `documents/` with your own PDFs (or upload them in the notebook's
optional upload cell) — nothing else needs to change. The pipeline works on whatever PDFs are
in that folder.

## Swapping the LLM
`documind/llm.py` exposes three interchangeable backends behind the same `.generate(prompt)`
interface:
- `get_llm("hf")` — default, `google/flan-t5-base`, open, no API key, runs on CPU.
- `get_llm("openai", model_name="gpt-4o-mini")` — needs `OPENAI_API_KEY` set and `pip install openai`.
- `get_llm("dev")` — offline stand-in used only for testing the surrounding pipeline with no
  model download at all; **do not use it for the actual demo or evaluation.**

## Grounding & hallucination control
Two independent safeguards, matching what the problem statement asks for:
1. **Prompt-level**: the system instruction tells the LLM to answer only from retrieved
   context and to say the answer wasn't found otherwise (see `rag.SYSTEM_INSTRUCTION`).
2. **Retrieval-score floor**: if even the best-matching chunk's similarity score is below
   `min_similarity` (default `0.30`), the pipeline refuses to answer *before* even calling the
   LLM, saving a wasted generation call. Tune this threshold against your own documents.

## Evaluation set
`eval_questions.csv` has 27 questions: 19 single-document, 4 cross-document (need evidence
from two documents), and 4 deliberately unanswerable (to test refusal/grounding). Retrieval
accuracy and refusal-correctness are computed automatically; `answer_correct` and
`citation_correct` are left blank because judging free-text correctness reliably needs a human
reading the transcript against the source PDF — fill those in before reporting final numbers
in your report or jury presentation. **Do not present numbers computed with the `"dev"` LLM
backend or with `allow_fallback=True`'s offline embedder — both exist purely for testing the
pipeline's plumbing without a model download, and are not semantically meaningful.**

## Notes on where this was tested
The full pipeline (document generation → extraction → cleaning → chunking → embedding →
FAISS indexing → retrieval → prompt construction → generation → citation → grounding refusal
→ CSV evaluation → Gradio UI start/stop) was executed end-to-end successfully. The sandbox
this was built in has no outbound access to huggingface.co, so the two real model downloads
(`sentence-transformers/all-MiniLM-L6-v2` and `google/flan-t5-base`) were substituted with
deterministic stand-ins for that one test pass to confirm there are no code-level bugs; every
non-model-download step was verified against the real generated PDFs. Running this in Colab
(or any machine with normal internet access) downloads and uses the real models — no code
changes needed.
