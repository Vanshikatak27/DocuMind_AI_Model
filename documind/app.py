
import os
import shutil
import tempfile
import gradio as gr

from .embeddings import get_embedder
from .indexer import build_index
from .llm import get_llm
from .rag import DocuMindRAG

DEFAULT_DOCS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "documents")
)


CSS = """
:root {
    color-scheme: dark;
}

body, .gradio-container {
    background: #090b16 !important;
    color: #edf0ff !important;
    font-family: Inter, "Segoe UI", sans-serif !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    padding: 28px !important;
}

#app-header {
    position: relative;
    overflow: hidden;
    padding: 42px 44px;
    margin-bottom: 28px;
    border: 1px solid #38305f;
    border-radius: 26px;
    background:
        radial-gradient(ellipse at 90% 0%, #52369b66, transparent 42%),
        linear-gradient(125deg, #171a35, #101426 65%, #211637);
    box-shadow: 0 18px 55px #00000030;
}

#app-header h1 {
    color: #ffffff;
    font-size: clamp(32px, 4vw, 46px);
    font-weight: 750;
    letter-spacing: -1.8px;
    margin: 14px 0;
}

#app-header p {
    color: #b9c1df;
    font-size: 16px;
    line-height: 1.8;
}

.eyebrow {
    color: #c4a9ff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.gradio-container .block,
.gradio-container .form {
    background: transparent;
}

.gradio-container .panel,
.gradio-container .gr-group {
    background: #111629 !important;
    border: 1px solid #292f4a !important;
    border-radius: 21px !important;
    box-shadow: 0 12px 30px #00000018;
}

.gradio-container .panel {
    padding: 22px !important;
}

.gradio-container h3,
.gradio-container label {
    color: #f0f1ff !important;
}

.gradio-container .prose,
.gradio-container .prose p {
    color: #b4bdd9;
    line-height: 1.75;
}

.gradio-container input,
.gradio-container textarea {
    background: #0b1020 !important;
    color: #f1f3ff !important;
    border: 1px solid #303956 !important;
    border-radius: 13px !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #9a7bff !important;
    box-shadow: 0 0 0 3px #8c6bff20 !important;
}

#process-btn, #ask-btn {
    min-height: 50px;
    border: 1px solid #9b82ff40 !important;
    border-radius: 13px !important;
    color: white !important;
    font-weight: 750 !important;
    background: linear-gradient(110deg, #7955ed, #5b6df0) !important;
    box-shadow: 0 8px 22px #7955ed25;
    transition: transform .2s, box-shadow .2s;
}

#process-btn:hover, #ask-btn:hover {
    box-shadow: 0 10px 28px #7955ed45;
    transform: translateY(-1px);
}

#answer-box {
    min-height: 180px;
    padding: 22px !important;
    border: 1px solid #39315f !important;
    border-radius: 17px !important;
    background: linear-gradient(145deg, #171b32, #121629) !important;
    color: #f4f4ff !important;
    line-height: 1.9;
}

#citation-box {
    padding: 18px !important;
    border: 1px solid #293650 !important;
    border-radius: 15px !important;
    background: #0e1425 !important;
}

#citation-box a {
    color: #b9a5ff !important;
}

#status-box textarea {
    min-height: 100px;
    background: #0b1020 !important;
    border-color: #303956 !important;
    line-height: 1.7;
}

.gradio-container button {
    transition: all .2s ease;
}

footer {
    display: none !important;
}
"""


def build_app(
    llm_backend: str = "hf",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    state = {"rag": None, "embedder": None}

    def load_documents(files):
        try:
            if state["embedder"] is None:
                state["embedder"] = get_embedder(
                    embed_model, allow_fallback=False
                )

            if files:
                folder = tempfile.mkdtemp(prefix="documind_upload_")
                for f in files:
                    if isinstance(f, dict):
                        source = f.get("path") or f.get("name")
                    else:
                        source = getattr(f, "name", None) or str(f)

                    if not source or not os.path.isfile(source):
                        continue

                    shutil.copy2(
                        source,
                        os.path.join(folder, os.path.basename(source))
                    )
            else:
                folder = DEFAULT_DOCS_DIR

            store = build_index(folder, state["embedder"])
            if not store.chunks:
                return "No readable PDF content found. Please upload valid PDFs."

            llm = get_llm(llm_backend)
            state["rag"] = DocuMindRAG(
                store, state["embedder"], llm,
                top_k=4, min_similarity=0.30
            )

            names = sorted({c.doc_name for c in store.chunks})
            return (
                f"Ready! Indexed {len(store.chunks)} chunks "
                f"from {len(names)} document(s): "
                + ", ".join(names)
            )
        except Exception as e:
            return f"Could not process documents: {e}"

    def ask(question):
        if state["rag"] is None:
            return (
                "Please process your PDFs first using the button in the "
                "Document Library panel.",
                "No sources available yet."
            )

        if not question or not question.strip():
            return "Enter a question to get started.", ""

        try:
            result = state["rag"].answer(question.strip())
            citations = (
                "\n".join(f"- {c}" for c in result.citations)
                if result.citations
                else "_No source citation available for this answer._"
            )
            return result.answer, citations
        except Exception as e:
            return f"Something went wrong while answering: {e}", ""

    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate"
    )

    with gr.Blocks(
        title="DocuMind AI | Research Workspace",
        theme=theme,
        css=CSS
    ) as demo:

        gr.HTML("""
        <div id="app-header">
            <div class="eyebrow">✦ Your intelligent research workspace</div>
            <h1>DocuMind AI</h1>
            <p>Ask questions across your PDFs. Get focused answers
            with source citations.</p>
        </div>
        """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Document Library")
                    gr.Markdown(
                        '<p class="muted">Upload one or more PDFs, '
                        'or use the sample documents in your project.</p>'
                    )
                    files = gr.File(
                        label="Your PDF files",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                    process_btn = gr.Button(
                        "✦  Process Documents",
                        elem_id="process-btn"
                    )
                    status = gr.Textbox(
                        label="Library status",
                        value="Waiting for documents",
                        interactive=False,
                        lines=4,
                        elem_id="status-box"
                    )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### How it works")
                    gr.Markdown("""
                    <p class="muted">01 — Upload your documents</p>
                    <p class="muted">02 — Build the searchable index</p>
                    <p class="muted">03 — Ask a question</p>
                    <p class="muted">04 — Review the answer and sources</p>
                    """)

            with gr.Column(scale=2, min_width=320):
                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Ask your documents")
                    gr.Markdown(
                        '<p class="muted">Your questions are answered '
                        'using the indexed document context.</p>'
                    )

                    question = gr.Textbox(
                        label="Your question",
                        placeholder=(
                            "Ask something about your PDFs..."
                        ),
                        lines=3,
                        elem_id="question-box"
                    )
                    with gr.Row():
                        ask_btn = gr.Button(
                            "Ask DocuMind  →",
                            elem_id="ask-btn"
                        )
                        clear_btn = gr.ClearButton(
                            [question], value="Clear question"
                        )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Answer")
                    answer = gr.Markdown(
                        "Your answer will appear here after you ask a question.",
                        elem_id="answer-box"
                    )

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### Source citations")
                    citation = gr.Markdown(
                        "Sources will appear here when available.",
                        elem_id="citation-box"
                    )

        process_btn.click(
            load_documents, inputs=[files], outputs=[status]
        )
        ask_btn.click(
            ask, inputs=[question], outputs=[answer, citation]
        )
        question.submit(
            ask, inputs=[question], outputs=[answer, citation]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch()
