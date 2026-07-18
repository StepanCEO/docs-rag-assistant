"""End-to-end RAG tests with a fake embedder and fake LLM — no API key needed.

The fake embedder maps texts to deterministic vectors based on word overlap
with a small vocabulary, so "similar" texts really do land closer in the
vector space and retrieval behaves realistically.
"""

import pytest

from rag.chunker import Chunk
from rag.core import NO_ANSWER, ask
from rag.store import VectorStore

VOCAB = ["python", "chunking", "embeddings", "docker", "cats", "pricing"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lower = text.lower()
        vec = [float(lower.count(word)) for word in VOCAB]
        # avoid the zero vector, which breaks cosine distance
        vectors.append(vec if any(vec) else [0.01] * len(VOCAB))
    return vectors


@pytest.fixture()
def store(tmp_path):
    s = VectorStore(str(tmp_path / "chroma"), fake_embed)
    s.add(
        [
            Chunk("guide.md", 0, "python chunking splits text into pieces"),
            Chunk("guide.md", 1, "embeddings turn text into vectors"),
            Chunk("ops.md", 0, "docker packages the app for deployment"),
        ]
    )
    return s


def test_query_returns_most_relevant_chunk(store):
    hits = store.query("how does chunking work in python", top_k=1)
    assert hits[0].doc_name == "guide.md"
    assert hits[0].chunk_index == 0


def test_ask_returns_answer_with_sources(store):
    def fake_llm(system: str, user: str) -> str:
        assert "chunking splits text" in user  # retrieved context reached the LLM
        return "Chunking splits text into pieces."

    answer = ask("What is chunking?", store, fake_llm, top_k=2)
    assert answer.found
    assert answer.text == "Chunking splits text into pieces."
    assert any(s.file == "guide.md" for s in answer.sources)


def test_ask_refuses_when_llm_says_no_answer(store):
    answer = ask("What are your prices?", store, lambda s, u: NO_ANSWER, top_k=2)
    assert not answer.found
    assert answer.sources == []


def test_ask_on_empty_store(tmp_path):
    empty = VectorStore(str(tmp_path / "empty"), fake_embed)
    answer = ask("anything", empty, lambda s, u: "should not be called")
    assert not answer.found
