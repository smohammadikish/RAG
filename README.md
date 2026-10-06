# RAG

A small Python project that reads a file, splits its text into chunks, creates
OpenAI embeddings, and retrieves relevant passages for a question.

Files and the search index stay on your machine. Document text and questions
are sent to OpenAI to create embeddings, so you need internet access and an
OpenAI API key. The script prints source passages; it does not generate an answer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
rag-search ./manual.pdf "How do I reset my password?"
```

Supported files: PDF (with text), TXT, Markdown, CSV, JSON, DOCX, and XLSX.
Scanned PDFs need OCR before indexing.

The index is saved in `.rag_data/`. No Docker, database server, or tracing setup
is needed. Re-running the example replaces that file's indexed chunks and searches
only that file. Run one process at a time against the local index.

## Configuration

Only `OPENAI_API_KEY` is required. Optional settings in `.env`:

| Setting | Default |
| --- | --- |
| `QDRANT_PATH` | `.rag_data` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `RAG_EMBEDDING_DIMENSION` | `1536` |
| `RAG_CHUNK_SIZE` | `800` words |
| `RAG_CHUNK_OVERLAP` | `100` words |
| `RAG_RETRIEVAL_TOP_K` | `5` |
| `RAG_CONTEXT_MAX_CHARS` | `12000` |

If you change the embedding model or dimension, use a new index directory.
For an existing Qdrant server, set `QDRANT_URL` instead; `QDRANT_API_KEY` is optional.

## Code

- `src/rag/cli.py`: workflow behind `rag-search` and `python -m rag`.
- `src/rag/config.py`: settings.
- `src/rag/extraction.py`: file readers and source locations.
- `src/rag/`: chunking, embeddings, indexing, search, and context formatting.

Python imports use `rag`, for example:

```python
from rag import DocumentIndexer, Retriever, ContextBuilder
from rag.config import Settings
```

Installation generates `src/rag.egg-info/`, which contains package metadata
and is ignored by Git. You do not need to edit or copy it; reinstalling recreates
it. The optional `examples/` folder was removed because `rag-search` provides
the complete workflow.

The default collection is now `rag_chunks`. To search an index built before
this rename, set `QDRANT_COLLECTION_NAME` to your previous collection name.

## Tests

```bash
pip install -e ".[test]"
pytest tests/unit/rag
```

Unit tests use fake embedding clients. The local storage test uses fixed vectors
and a temporary directory, with no API key or server required.

## Code analysis with Graphify

[Graphify](https://github.com/Graphify-Labs/graphify) is optional development tooling.
Install it separately from the document-search dependencies:

```bash
pip install -e ".[analysis]"
graphify extract . --code-only
graphify cluster-only . --no-label
graphify query "How does document indexing work?"
graphify explain "DocumentIndexer"
```

Open `graphify-out/graph.html` to explore the code graph. The generated files are
ignored by Git. Re-run extraction and clustering after changing the code.
Code-only analysis uses local AST parsing and skips semantic analysis of docs.
It needs no API key. Graphify does not participate in document search.
