"""Check local documents, then answer questions with supporting sources."""

import argparse
import hashlib
import json
from pathlib import Path

from rag.config import get_settings
from rag import ContextBuilder, DocumentIndexer, RecursiveTextChunker, Retriever
from rag.embeddings.openai import OpenAIEmbeddingProvider
from rag.extraction import SUPPORTED_EXTRACTION_EXTENSIONS
from rag.generation import generate_answer
from rag.loaders import SharedDocumentLoader
from rag.vectorstores.qdrant import QdrantVectorStore


def sync_folder(folder, indexer, store, settings, *, force=False):
    """Remember successful file hashes so unchanged documents need no embeddings."""
    folder = folder.resolve()
    # Keep a separate file list for each folder and collection.
    identity = f"{folder}:{settings.QDRANT_URL}:{settings.QDRANT_COLLECTION_NAME}"
    manifest_path = Path(settings.QDRANT_PATH) / (
        "files_" + hashlib.sha256(identity.encode()).hexdigest()[:16] + ".json"
    )
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        previous = {}
    # New embedding or chunk settings mean the files must be indexed again.
    config = [settings.OPENAI_EMBEDDING_MODEL, settings.RAG_EMBEDDING_DIMENSION,
              settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP]
    old_files = previous.get("files", {})
    unchanged_config = previous.get("config") == config and not force
    files = {}
    paths = {str(p.resolve()): p for p in sorted(folder.rglob("*"))
             if p.is_file() and not p.name.startswith((".", "~$"))}
    supported = {key: p for key, p in paths.items()
                 if p.suffix.lower() in SUPPORTED_EXTRACTION_EXTENSIONS}
    for key, path in paths.items():
        if key not in supported:
            print(f"Skipped unsupported file: {path.relative_to(folder)}")
    # Remove saved passages for files that are no longer in the folder.
    for key in old_files.keys() - supported.keys():
        store.delete(document_id=key)
        print(f"Removed from index: {Path(key).name}")
    for key, path in supported.items():
        try:
            # Compare file contents, not just the filename or modification date.
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if unchanged_config and old_files.get(key) == digest:
                print(f"Unchanged: {path.relative_to(folder)}")
            else:
                summary = indexer.index(
                    path, document_id=key, replace_existing=True,
                    metadata={"input_folder": str(folder)},
                )
                print(f"Indexed: {path.relative_to(folder)} ({summary.chunk_count} chunks)")
            files[key] = digest
        except Exception as exc:
            # Do not serve an outdated version of a file that failed to refresh.
            store.delete(document_id=key)
            print(f"Failed: {path.relative_to(folder)} — {exc}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"config": config, "files": files}), encoding="utf-8")
    # Replace the file only after the full new list has been written.
    temporary.replace(manifest_path)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, nargs="?", help="optional single-file search")
    parser.add_argument("question", nargs="?", help="question for single-file search")
    args = parser.parse_args()
    if (args.file is None) != (args.question is None):
        parser.error("Provide both a file and a question, or neither for interactive mode.")
    if args.file is not None:
        path = args.file.expanduser().resolve()
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
        if not args.question.strip():
            parser.error("Question must not be empty.")
    else:
        folder = Path("user_input").resolve()
        folder.mkdir(exist_ok=True)
        print(f"Checking documents in {folder}")

    settings = get_settings()
    store = QdrantVectorStore(
        dimension=settings.RAG_EMBEDDING_DIMENSION, settings=settings,
        ensure_collection=False, wait=True,
    )
    try:
        # Rebuild embeddings if the collection was deleted or has never existed.
        fresh_collection = not store.collection_exists()
        store.ensure_collection()
        has_files = args.file is not None or any(
            p.is_file() and not p.name.startswith((".", "~$"))
            and p.suffix.lower() in SUPPORTED_EXTRACTION_EXTENSIONS
            for p in folder.rglob("*")
        )
        # An empty folder does not need an OpenAI client or API key.
        embeddings = OpenAIEmbeddingProvider(settings=settings) if has_files else None
        indexer = DocumentIndexer(
            loader=SharedDocumentLoader(),
            chunker=RecursiveTextChunker(
                chunk_size=settings.RAG_CHUNK_SIZE, chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            ),
            embedding_provider=embeddings, vector_store=store,
        ) if has_files else None
        if args.file is not None:
            input_folder = Path("user_input").resolve()
            metadata = {"input_folder": str(input_folder)} if path.is_relative_to(input_folder) else {}
            summary = indexer.index(
                path, document_id=str(path), replace_existing=True, metadata=metadata,
            )
            print(f"Indexed {summary.chunk_count} chunks from {path.name}.")
            filters = {"document_id": str(path)}
        else:
            count = sync_folder(folder, indexer, store, settings, force=fresh_collection)
            if not count:
                print("No documents ready. Add supported files to user_input/ and run again.")
                return 0
            print(f"\nReady: {count} documents. Ask a question, or type exit.")
            # Search only documents belonging to this input folder.
            filters = {"input_folder": str(folder)}
        retriever = Retriever(embedding_provider=embeddings, vector_store=store)
        builder = ContextBuilder(max_chars=settings.RAG_CONTEXT_MAX_CHARS)
        # Keep taking questions until the user exits.
        while True:
            if args.question is not None:
                question = args.question
            else:
                try:
                    question = input("\nAsk a question: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if question.lower() in {"exit", "quit"}:
                    break
                if not question:
                    continue
            try:
                results = retriever.retrieve(
                    question, top_k=settings.RAG_RETRIEVAL_TOP_K, filters=filters,
                )
                context = builder.build(results)
                answer = generate_answer(question, context, settings)
                # Show the answer first, then list the evidence supplied to the model.
                print(f"\nAnswer:\n{answer}")
                if context.sources:
                    print("\nSources (provided to the answer model):")
                    for source in context.sources:
                        name = source.get("filename") or source.get("source") or "Document"
                        location = source.get("source_location")
                        if not location and source.get("page_number") is not None:
                            location = f"page:{source['page_number']}"
                        suffix = f" — {location}" if location else ""
                        print(f"[Source {source['source_number']}] {name}{suffix}")
            except Exception as exc:
                print(f"Search failed: {exc}")
                if args.question is not None:
                    return 1
            if args.question is not None:
                break
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
