import hashlib
import logging
from pathlib import Path

from rag.chunking.base import TextChunker
from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import EmbeddingError, VectorStoreError
from rag.loaders.base import DocumentLoader
from rag.models import Chunk, IndexingSummary
from rag.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Coordinate deterministic document load, chunk, embed, and index steps."""

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def index(
        self,
        path: str | Path,
        *,
        document_id: str | None = None,
        user_id: str | None = None,
        replace_existing: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> IndexingSummary:
        metadata = metadata or {}
        # Read the file before splitting its text into smaller pieces.
        documents = self._loader.load(path)

        chunks = self._chunker.split(documents)
        if document_id is not None:
            chunks = self._chunks_for_logical_document(
                chunks,
                document_id=document_id,
                user_id=user_id,
                metadata=metadata,
            )

        if not chunks:
            msg = "Indexing produced no chunks."
            raise VectorStoreError(msg)

        # Create one search vector for each chunk.
        embeddings = self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks],
        )
        if len(embeddings) != len(chunks):
            msg = (
                "Embedding provider returned a different number of embeddings "
                "than chunks."
            )
            raise EmbeddingError(msg)

        # Finish embedding first, then replace the old saved chunks.
        if replace_existing and document_id is not None:
            self._vector_store.delete(document_id=document_id, user_id=user_id)
        self._vector_store.upsert(chunks, embeddings)

        summary = IndexingSummary(
            document_count=len(documents),
            chunk_count=len(chunks),
            document_ids=(
                (document_id,)
                if document_id is not None
                else tuple(document.id for document in documents)
            ),
            chunk_ids=tuple(chunk.id for chunk in chunks),
        )
        logger.info(
            "Document indexing completed",
            extra={
                "document_count": summary.document_count,
                "chunk_count": summary.chunk_count,
            },
        )
        return summary

    @staticmethod
    def _chunks_for_logical_document(
        chunks: list[Chunk],
        *,
        document_id: str,
        user_id: str | None,
        metadata: dict[str, object],
    ) -> list[Chunk]:
        # Give all pages or sections of the file the same document ID.
        logical_chunks = []
        for logical_index, chunk in enumerate(chunks):
            chunk_metadata = {
                **chunk.metadata,
                **metadata,
                "document_id": document_id,
                "user_id": user_id,
                "loader_document_id": chunk.document_id,
                "source_chunk_index": chunk.chunk_index,
            }
            logical_chunks.append(
                Chunk(
                    id=_logical_chunk_id(document_id, logical_index, chunk.text),
                    document_id=document_id,
                    text=chunk.text,
                    chunk_index=logical_index,
                    metadata=chunk_metadata,
                    token_count=chunk.token_count,
                )
            )
        return logical_chunks


def _logical_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{chunk_index}:{text}".encode()).hexdigest()
    return f"chunk_{digest[:24]}"
