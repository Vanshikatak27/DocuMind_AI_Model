"""
LLM backends
-------------
Two interchangeable ways to satisfy the "LLM / API" box in the architecture
diagram. Both expose the same `.generate(prompt) -> str` method so the rest
of the pipeline (rag.py) never needs to know which one is in use.

HFLocalLLM   - a small, open, ungated instruction-tuned model pulled from
               Hugging Face and run locally (free, works in Colab's CPU or
               GPU runtime, no API key needed).
OpenAILLM    - a thin wrapper around an OpenAI-compatible chat completion
               API, for groups who prefer/require an API-based LLM. Only
               used if an API key is supplied.
EchoDevLLM   - a trivial offline stand-in (returns the top context sentence)
               used solely so the surrounding pipeline can be smoke-tested
               with no model download and no network — never use this for
               the actual demo or evaluation.
"""
import os


class HFLocalLLM:
    """Default backend: a small open instruction-tuned model, e.g.
    'google/flan-t5-base'. No API key required; the first call downloads
    the model (needs internet — this is fine in Colab)."""

    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 200):
        from transformers import pipeline  # imported lazily
        self.model_name = model_name
        self.pipe = pipeline("text2text-generation", model=model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]["generated_text"]


class OpenAILLM:
    """Optional API-based backend. Set OPENAI_API_KEY (and optionally
    OPENAI_MODEL, default 'gpt-4o-mini') to use this instead of the local
    Hugging Face model."""

    def __init__(self, model_name: str = None, api_key: str = None):
        from openai import OpenAI  # imported lazily; `pip install openai` if used
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content


class EchoDevLLM:
    """Offline dev/testing stand-in only — NOT a real LLM. Returns the most
    relevant context sentence verbatim so the pipeline plumbing (retrieval,
    prompting, citation, grounding) can be verified without any model
    download or network access."""

    def generate(self, prompt: str) -> str:
        if 'Context:\n' not in prompt:
            return ""
        context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:")[0]
        first_block = context.split("\n\n")[0]
        # strip the "[1] (Source: ...)" header line, keep the sentence(s) after it
        lines = first_block.split("\n", 1)
        return lines[1].split(".")[0].strip() + "." if len(lines) > 1 else first_block


def get_llm(backend: str = "hf", **kwargs):
    backend = backend.lower()
    if backend == "hf":
        return HFLocalLLM(**kwargs)
    if backend == "openai":
        return OpenAILLM(**kwargs)
    if backend == "dev":
        return EchoDevLLM()
    raise ValueError(f"Unknown LLM backend: {backend!r} (use 'hf', 'openai', or 'dev')")
