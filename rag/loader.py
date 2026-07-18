"""Document loading: PDF, Markdown and plain text files."""

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


@dataclass(frozen=True)
class Document:
    """One loaded source file."""

    name: str  # file name relative to the ingest root
    text: str


def load_documents(root: Path) -> list[Document]:
    """Load every supported file under ``root`` (recursively)."""
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    paths = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = []
    base = root.parent if root.is_file() else root
    for path in paths:
        text = _read_file(path)
        if text.strip():
            documents.append(Document(name=str(path.relative_to(base)), text=text))
        else:
            logger.warning("Skipping %s: no extractable text", path)
    return documents


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")
