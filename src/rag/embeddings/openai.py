import logging
from typing import Any

from pydantic import BaseModel, Field

from rag.config import Settings, get_settings
from rag.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_EMBEDDING_DIMENSION = 1536


class OpenAIEmbeddingConfig(BaseModel):
    """Configuration for the OpenAI embedding adapter."""

    model: str = Field(min_length=1)
    dimension: int = Field(default=DEFAULT_OPENAI_EMBEDDING_DIMENSION, ge=1)
    batch_size: int = Field(default=100, ge=1)
    api_key: str | None = None


class OpenAIEmbeddingProvider:
    """OpenAI-backed embedding provider behind the RAG interface."""

    def __init__(
        self,
        *,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int = 100,
        api_key: str | None = None,
        client: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        defaults = settings or get_settings()
        self._config = OpenAIEmbeddingConfig(
            model=model if model is not None else defaults.OPENAI_EMBEDDING_MODEL,
            dimension=dimension if dimension is not None else defaults.RAG_EMBEDDING_DIMENSION,
            batch_size=batch_size,
            api_key=api_key if api_key is not None else defaults.OPENAI_API_KEY,
        )
        self._client = client if client is not None else self._build_client()

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        # Send smaller batches instead of one large request.
        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start : start + self._config.batch_size]
            embeddings.extend(self._embed(batch))

        if len(embeddings) != len(texts):
            msg = (
                "OpenAI embedding response count did not match input count: "
                f"expected {len(texts)}, got {len(embeddings)}."
            )
            raise EmbeddingError(msg)

        logger.info(
            "Embedding batch generated",
            extra={"text_count": len(texts), "model": self._config.model},
        )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            msg = "Query text must not be empty."
            raise EmbeddingError(msg)

        embeddings = self._embed([text])
        return embeddings[0]

    def _build_client(self) -> Any:
        if not self._config.api_key:
            msg = "OPENAI_API_KEY must be set before creating OpenAI embeddings."
            raise EmbeddingError(msg)

        try:
            from openai import OpenAI
        except ImportError as exc:
            msg = "The openai package is required for OpenAI embeddings."
            raise EmbeddingError(msg) from exc

        return OpenAI(api_key=self._config.api_key)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._config.model,
                input=texts,
            )
        except Exception as exc:
            msg = f"OpenAI embedding request failed for model: {self._config.model}"
            raise EmbeddingError(msg) from exc

        data = list(getattr(response, "data", []))
        if len(data) != len(texts):
            msg = (
                "OpenAI embedding response count did not match batch input count: "
                f"expected {len(texts)}, got {len(data)}."
            )
            raise EmbeddingError(msg)

        embeddings: list[list[float]] = []
        # Keep vectors in the same order as the input texts.
        for item in sorted(data, key=lambda value: getattr(value, "index", 0)):
            embedding = list(getattr(item, "embedding", []))
            if len(embedding) != self._config.dimension:
                msg = (
                    "OpenAI embedding dimension mismatch: "
                    f"expected {self._config.dimension}, got {len(embedding)}."
                )
                raise EmbeddingError(msg)
            embeddings.append([float(value) for value in embedding])

        return embeddings
