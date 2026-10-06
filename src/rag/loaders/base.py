from pathlib import Path
from typing import Protocol

from rag.models import Document


class DocumentLoader(Protocol):
    """Load and normalize source files into RAG documents."""

    def load(self, path: str | Path) -> list[Document]:
        """Load one logical input into one or more normalized documents."""
