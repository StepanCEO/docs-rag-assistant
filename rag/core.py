"""RAG core: retrieve relevant chunks, answer strictly from them, cite sources."""

from dataclasses import dataclass, field
from typing import Callable

from .store import Hit, VectorStore

# complete(system, user) -> str
CompleteFn = Callable[[str, str], str]

NO_ANSWER = "NO_ANSWER"

SYSTEM_PROMPT = f"""You are a documentation assistant. Answer the question using ONLY
the provided context fragments.

Rules:
- If the context does not contain the answer, reply with exactly: {NO_ANSWER}
- Never use outside knowledge; never guess.
- Answer in the language of the question.
- Be concise and specific."""


@dataclass(frozen=True)
class Source:
    """A citation pointing back into the ingested documents."""

    file: str
    chunk: int


@dataclass(frozen=True)
class Answer:
    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)


def ask(question: str, store: VectorStore, complete: CompleteFn, top_k: int = 5) -> Answer:
    """Answer a question from the vector store, refusing when context is missing."""
    hits = store.query(question, top_k=top_k)
    if not hits:
        return Answer(text="Индекс пуст или ничего релевантного не найдено.", found=False)

    context = _format_context(hits)
    user = f"Context fragments:\n{context}\n\nQuestion: {question}"
    raw = complete(SYSTEM_PROMPT, user).strip()

    if NO_ANSWER in raw:
        return Answer(text="В документах нет ответа на этот вопрос.", found=False)

    sources = _dedup_sources(hits)
    return Answer(text=raw, found=True, sources=sources)


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {h.doc_name}, chunk {h.chunk_index})\n{h.text}"
        for i, h in enumerate(hits)
    )


def _dedup_sources(hits: list[Hit]) -> list[Source]:
    seen = set()
    sources = []
    for h in hits:
        key = (h.doc_name, h.chunk_index)
        if key not in seen:
            seen.add(key)
            sources.append(Source(file=h.doc_name, chunk=h.chunk_index))
    return sources
