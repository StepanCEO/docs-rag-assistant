"""Chunker tests: pure logic, no dependencies."""

import pytest

from rag.chunker import chunk_documents, chunk_text
from rag.loader import Document


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", chunk_size=500) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_respect_size_and_overlap():
    text = words(1000)
    chunks = chunk_text(text, chunk_size=400, overlap=100)

    assert all(len(c.split()) <= 400 for c in chunks)
    # consecutive chunks share exactly `overlap` words
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-100:] == second[:100]


def test_no_words_are_lost():
    text = words(950)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(text.split())


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=100, overlap=100)


def test_chunk_documents_tags_source_and_index():
    docs = [Document(name="a.md", text=words(700)), Document(name="b.md", text="short")]
    chunks = chunk_documents(docs, chunk_size=400, overlap=50)

    a_chunks = [c for c in chunks if c.doc_name == "a.md"]
    b_chunks = [c for c in chunks if c.doc_name == "b.md"]
    assert len(a_chunks) == 2
    assert [c.index for c in a_chunks] == [0, 1]
    assert len(b_chunks) == 1
    assert b_chunks[0].id == "b.md#0"
