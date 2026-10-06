from pathlib import Path

from rag.config import Settings
from rag import ContextBuilder, DocumentIndexer, RecursiveTextChunker, Retriever
from rag.loaders import SharedDocumentLoader
from rag.vectorstores.qdrant import QdrantVectorStore


class KeywordEmbeddings:
    dimension = 2

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.lower().count("alpha")), float(text.lower().count("beta"))]


def test_local_pipeline_replaces_persists_and_filters(tmp_path: Path):
    settings = Settings(_env_file=None, QDRANT_PATH=str(tmp_path / "index"))
    embedder = KeywordEmbeddings()
    file = tmp_path / "notes.txt"
    file.write_text("alpha old passage", encoding="utf-8")
    store = QdrantVectorStore(dimension=2, settings=settings)
    try:
        indexer = DocumentIndexer(loader=SharedDocumentLoader(),
                                  chunker=RecursiveTextChunker(),
                                  embedding_provider=embedder, vector_store=store)
        indexer.index(file, document_id="notes", replace_existing=True)
        file.write_text("alpha updated passage", encoding="utf-8")
        indexer.index(file, document_id="notes", replace_existing=True)
        other = tmp_path / "other.txt"
        other.write_text("alpha unrelated passage", encoding="utf-8")
        indexer.index(other, document_id="other")
    finally:
        store.close()

    reopened = QdrantVectorStore(dimension=2, settings=settings)
    try:
        results = Retriever(embedding_provider=embedder, vector_store=reopened).retrieve(
            "alpha", filters={"document_id": "notes"})
        assert len(results) == 1
        context = ContextBuilder().build(results)
        assert "updated passage" in context.text
        assert "old passage" not in context.text
        assert "unrelated passage" not in context.text
        assert "File: notes.txt" in context.text
    finally:
        reopened.close()
