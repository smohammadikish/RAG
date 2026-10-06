"""Settings for indexing and searching local documents."""

from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", min_length=1)
    OPENAI_RESPONSE_MODEL: str = Field(default="gpt-4.1-mini", min_length=1)
    RAG_EMBEDDING_DIMENSION: int = Field(default=1536, ge=1)
    RAG_CHUNK_SIZE: int = Field(default=800, ge=1)
    RAG_CHUNK_OVERLAP: int = Field(default=100, ge=0)
    RAG_RETRIEVAL_TOP_K: int = Field(default=5, ge=1)
    RAG_CONTEXT_MAX_CHARS: int = Field(default=12000, ge=1)
    QDRANT_PATH: str = Field(default=".rag_data", min_length=1)
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = Field(default="rag_chunks", min_length=1)

    # Read .env too; ignore settings that belong to optional tools.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore",
    )

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.RAG_CHUNK_OVERLAP >= self.RAG_CHUNK_SIZE:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self


# Load settings once and reuse them during this run.
@lru_cache
def get_settings() -> Settings:
    return Settings()
