"""Vector store: ChromaDB wrapper with pluggable embeddings.

The embedder is injected as a plain callable so tests can use a fake one
and the rest of the code never touches the OpenAI SDK directly.
"""

from dataclasses import dataclass
from typing import Callable

import chromadb

from .chunker import Chunk

# embed(texts) -> list of vectors, one per text
EmbedFn = Callable[[list[str]], list[list[float]]]

COLLECTION = "documents"


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with its source metadata."""

    doc_name: str
    chunk_index: int
    text: str
    distance: float


class VectorStore:
    """Persistent ChromaDB collection of embedded document chunks."""

    def __init__(self, persist_dir: str, embed: EmbedFn):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed = embed

    def reset(self) -> None:
        """Drop the collection so a re-ingest starts clean."""
        try:
            self._client.delete_collection(COLLECTION)
        except Exception:
            pass  # collection may not exist yet

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Embed and store chunks; returns the number stored."""
        collection = self._client.get_or_create_collection(COLLECTION)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self._embed([c.text for c in batch]),
                metadatas=[{"doc_name": c.doc_name, "chunk_index": c.index} for c in batch],
            )
        return len(chunks)

    def query(self, question: str, top_k: int = 5) -> list[Hit]:
        """Return the ``top_k`` chunks most similar to the question."""
        collection = self._client.get_or_create_collection(COLLECTION)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=self._embed([question]),
            n_results=min(top_k, collection.count()),
        )
        return [
            Hit(
                doc_name=meta["doc_name"],
                chunk_index=meta["chunk_index"],
                text=text,
                distance=dist,
            )
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
