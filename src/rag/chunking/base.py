from typing import Protocol

from rag.models import Chunk, Document


class TextChunker(Protocol):
    """Split normalized documents into stable text chunks."""

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Return chunks in deterministic document and chunk order."""
