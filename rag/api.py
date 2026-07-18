"""FastAPI app: POST /ask over the indexed documents.

Run with: uvicorn rag.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_settings
from .core import Source, ask
from .openai_backend import OpenAIBackend
from .store import VectorStore

app = FastAPI(
    title="docs-rag-assistant",
    description="Ask questions over your own documents. Answers include citations.",
    version="0.1.0",
)

_settings = None
_backend = None
_store = None


def _components() -> tuple:
    """Lazy init so importing the module (e.g. in tests) needs no API key."""
    global _settings, _backend, _store
    if _store is None:
        _settings = load_settings()
        _backend = OpenAIBackend(_settings)
        _store = VectorStore(_settings.chroma_dir, _backend.embed)
    return _settings, _backend, _store


class AskRequest(BaseModel):
    question: str = Field(min_length=1, examples=["What is chunk overlap for?"])


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    try:
        settings, backend, store = _components()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    answer = ask(request.question, store, backend.complete, top_k=settings.top_k)
    return AskResponse(answer=answer.text, found=answer.found, sources=answer.sources)
