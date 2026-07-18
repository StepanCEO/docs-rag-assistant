"""Text chunking: split documents into overlapping word-based chunks.

Pure functions, no dependencies — the easiest part of RAG to get subtly
wrong, so it is fully unit-tested.
"""

from dataclasses import dataclass

from .loader import Document


@dataclass(frozen=True)
class Chunk:
    """A slice of a source document, ready for embedding."""

    doc_name: str
    index: int  # position of the chunk within its document
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_name}#{self.index}"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Split text into chunks of ~``chunk_size`` words with ``overlap`` words shared.

    Word-based rather than character-based so we never split mid-word, and
    overlap preserves context that would otherwise be lost at boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = 500, overlap: int = 80
) -> list[Chunk]:
    """Chunk every document, tagging each chunk with its source and position."""
    chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            chunks.append(Chunk(doc_name=doc.name, index=i, text=text))
    return chunks
