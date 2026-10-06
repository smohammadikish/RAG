from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import RetrieverError
from rag.models import RetrievedChunk
from rag.vectorstores.base import MetadataFilter, VectorStore


class Retriever:
    """Embed a query and retrieve relevant chunks from a vector store."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            msg = "Retrieval query must not be empty."
            raise RetrieverError(msg)
        if top_k <= 0:
            msg = "top_k must be greater than zero."
            raise RetrieverError(msg)

        # Turn the question into a vector, then find similar document vectors.
        query_embedding = self._embedding_provider.embed_query(query)
        results = self._vector_store.search(
            query_embedding,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
        return results

