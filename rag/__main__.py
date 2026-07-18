"""CLI: ``python -m rag ingest <path>`` and ``python -m rag ask "<question>"``."""

import argparse
import logging
import sys
from pathlib import Path

from openai import OpenAIError

from .chunker import chunk_documents
from .config import load_settings
from .core import ask
from .loader import load_documents
from .openai_backend import OpenAIBackend
from .store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag", description="Ask questions over your own documents, with citations."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a folder or file of documents")
    p_ingest.add_argument("path", type=Path, help="Folder or file (.pdf/.md/.txt)")

    p_ask = sub.add_parser("ask", help="Ask a question over the indexed documents")
    p_ask.add_argument("question", help="Your question")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        settings = load_settings()
        backend = OpenAIBackend(settings)
        store = VectorStore(settings.chroma_dir, backend.embed)

        if args.command == "ingest":
            documents = load_documents(args.path)
            if not documents:
                print("No supported documents found (.pdf/.md/.txt).", file=sys.stderr)
                return 1
            chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
            store.reset()
            stored = store.add(chunks)
            print(f"Indexed {len(documents)} document(s) into {stored} chunk(s).")
            return 0

        answer = ask(args.question, store, backend.complete, top_k=settings.top_k)
        print(f"\n{answer.text}\n")
        if answer.sources:
            print("Sources:")
            for s in answer.sources:
                print(f"  - {s.file} (chunk {s.chunk})")
        return 0

    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except OpenAIError as exc:
        print(f"OpenAI API error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
