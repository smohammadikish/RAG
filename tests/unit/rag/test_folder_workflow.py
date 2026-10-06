from pathlib import Path

from rag import DocumentIndexer, RecursiveTextChunker
from rag.cli import main, sync_folder
from rag.config import Settings
from rag.loaders import SharedDocumentLoader
from rag.vectorstores.qdrant import QdrantVectorStore


class CountingEmbeddings:
    dimension = 2

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count("alpha")), 1.0]


def test_folder_sync_updates_reuses_removes_and_retries(tmp_path, capsys):
    folder = tmp_path / "user_input"
    folder.mkdir()
    file = folder / "notes.txt"
    file.write_text("alpha old passage")
    (folder / "unsupported.xls").write_text("unsupported")
    settings = Settings(_env_file=None, QDRANT_PATH=str(tmp_path / "index"),
                        RAG_EMBEDDING_DIMENSION=2)
    embeddings = CountingEmbeddings()
    store = QdrantVectorStore(dimension=2, settings=settings)
    indexer = DocumentIndexer(loader=SharedDocumentLoader(),
                              chunker=RecursiveTextChunker(),
                              embedding_provider=embeddings, vector_store=store)
    try:
        assert sync_folder(folder, indexer, store, settings) == 1
        assert embeddings.calls == 1
        assert sync_folder(folder, indexer, store, settings) == 1
        assert embeddings.calls == 1
        file.write_text("alpha new passage")
        assert sync_folder(folder, indexer, store, settings) == 1
        assert embeddings.calls == 2
        results = store.search([1.0, 1.0], top_k=5)
        assert len(results) == 1
        assert "new passage" in results[0].chunk.text
        # A failed refresh must remove stale passages and retry next time.
        file.write_text("")
        assert sync_folder(folder, indexer, store, settings) == 0
        assert store.search([1.0, 1.0], top_k=5) == []
        file.write_text("alpha recovered")
        assert sync_folder(folder, indexer, store, settings) == 1
        # Chunking changes invalidate the saved hashes too.
        settings.RAG_CHUNK_SIZE = 900
        assert sync_folder(folder, indexer, store, settings) == 1
        assert embeddings.calls == 4
        file.unlink()
        assert sync_folder(folder, indexer, store, settings) == 0
        assert store.search([1.0, 1.0], top_k=5) == []
        assert "Skipped unsupported" in capsys.readouterr().out
    finally:
        store.close()


def test_interactive_questions_search_all_folder_documents(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_input"
    folder.mkdir()
    (folder / "notes.txt").write_text("alpha useful passage")
    settings = Settings(_env_file=None, QDRANT_PATH=str(tmp_path / "index"),
                        RAG_EMBEDDING_DIMENSION=2)
    monkeypatch.setattr("rag.cli.get_settings", lambda: settings)
    monkeypatch.setattr("rag.cli.OpenAIEmbeddingProvider", lambda **kwargs: CountingEmbeddings())
    def answer(question, context, settings):
        assert "alpha useful passage" in context.text
        return "A concise answer [Source 1]."

    monkeypatch.setattr("rag.cli.generate_answer", answer)
    monkeypatch.setattr("sys.argv", ["rag-search"])
    questions = iter(["", "alpha", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    assert main() == 0
    output = capsys.readouterr().out
    assert "Ready: 1 documents" in output
    assert "[Source 1] notes.txt — lines:1-1" in output
    assert "A concise answer" in output
    assert "alpha useful passage" not in output
    assert output.index("Answer:") < output.index("Sources (provided")


def test_empty_folder_needs_no_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, OPENAI_API_KEY=None,
                        QDRANT_PATH=str(tmp_path / "index"))
    monkeypatch.setattr("rag.cli.get_settings", lambda: settings)
    monkeypatch.setattr("sys.argv", ["rag-search"])
    assert main() == 0
    assert (tmp_path / "user_input").is_dir()
    assert "No documents ready" in capsys.readouterr().out
