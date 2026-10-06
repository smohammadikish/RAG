# RAG

Search your local documents from the terminal. Put files in `user_input/`, run
the app, and ask questions. The app uses retrieved passages to write a concise answer, then lists supporting
filenames and source locations separately.

Python 3.12 or newer, internet access, and an OpenAI API key are required.
Qdrant stores the search index locally, so no Docker or database server is needed.
Document text and questions are sent to OpenAI for embeddings. Questions and
selected passages are also sent to OpenAI to generate answers.

Two commands cover the main workflows:

| Task | Command |
| --- | --- |
| Search your documents | `python -m rag` |
| Debug retrieval and generated answers | `RUN_RAG_EVAL=1 pytest -s -v tests/rag_eval/test_rag.py` |

The evaluation command needs the optional evaluation dependencies described below.

## Setup

Run these commands from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you do not already have a `.env` file, create one:

```bash
cp .env.example .env
```

Set your key in `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

## Add documents and run

Copy your documents into `user_input/`. Subfolders are supported.

```text
user_input/
├── manual.pdf
├── report.docx
└── sales.xlsx
```

Start the app from the project folder:

```bash
source .venv/bin/activate
python -m rag
```

Before accepting questions, the app checks the folder:

| File status | Action |
| --- | --- |
| New or changed | Extract text, create embeddings, and update the index |
| Unchanged | Reuse the existing embeddings |
| Deleted | Remove its passages from the index |
| Unsupported | Skip and report the file |
| Failed to process | Report the failure and exclude outdated passages |

Once the check finishes, ask questions:

```text
Ready: 3 documents. Ask a question, or type exit.

Ask a question: What were the sales results?
```

Questions search all successfully indexed files in `user_input/`. The answer
model receives the selected passages together so it can combine facts across
sources. For experience questions, it is instructed to combine employment
periods without counting overlapping dates twice, and to state any uncertainty.
It cannot account for employment periods missing from the retrieved context.

The answer appears first, followed by source references instead of raw chunks:

```text
Answer:
She has approximately six years of experience. [Source 1] [Source 2]

Sources (provided to the answer model):
[Source 1] resume.pdf — page:1
[Source 2] resume.pdf — page:2
```

This is an illustrative example. The sources list shows the passages supplied
to the model; inline citations indicate which sources the answer refers to.
If the context is insufficient, the model is instructed to say so.
Type `exit` or `quit`, or press Ctrl+C at the prompt, to stop.
After adding, editing, or removing files, restart the app to refresh the index.
If no documents are ready, the app exits with instructions to add files.

`rag-search` with no arguments starts the same interactive workflow.
To index and search just one file instead:

```bash
rag-search ./user_input/manual.pdf "How do I reset my password?"
```

## Supported files

| Format | Extension | Extracted content |
| --- | --- | --- |
| PDF | `.pdf` | Text by page |
| Word | `.docx` | Paragraphs and tables |
| Excel | `.xlsx` | Sheets and cell values |
| Plain text | `.txt` | Text |
| Markdown | `.md` | Text |
| CSV | `.csv` | Rows and columns |
| JSON | `.json` | Values with JSON paths |

Scanned PDFs need OCR before use. Older `.doc` and `.xls` files must be converted
to `.docx` and `.xlsx`. Hidden files and Excel temporary files starting with `~$`
are skipped.

## Configuration

Settings are read from `.env` or environment variables. Only `OPENAI_API_KEY`
needs to be supplied for the default local workflow.

| Setting | Default |
| --- | --- |
| `QDRANT_PATH` | `.rag_data` |
| `QDRANT_COLLECTION_NAME` | `rag_chunks` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `OPENAI_RESPONSE_MODEL` | `gpt-4.1-mini` |
| `RAG_EMBEDDING_DIMENSION` | `1536` |
| `RAG_CHUNK_SIZE` | `800` words |
| `RAG_CHUNK_OVERLAP` | `100` words |
| `RAG_RETRIEVAL_TOP_K` | `5` passages |
| `RAG_CONTEXT_MAX_CHARS` | `12000` characters |

Chunk overlap must be smaller than chunk size. Changing chunk settings triggers
reindexing on the next run. When changing the embedding model or dimension,
use a new `QDRANT_PATH` to avoid mixing incompatible vectors.

To connect to an existing Qdrant server, set `QDRANT_URL` and, if required,
`QDRANT_API_KEY`. The default setup uses local storage.

## Project files

```text
user_input/       Your documents
src/rag/         Python package
  cli.py         Folder checks and question prompt
  config.py      Settings
  extraction.py  Document readers
  generation.py  Answer generation from retrieved context
  chunking/      Text splitting
  embeddings/    OpenAI embeddings
  vectorstores/  Qdrant storage
  retrieval/     Passage search
  context/       Source formatting
  indexing.py    Read, split, embed, and save documents
tests/
  unit/rag/      Tests without paid API calls
  rag_eval/      Optional DeepEval checks with real model calls
  integration/   Optional Qdrant server test
.rag_data/       Generated index and saved file hashes
```

Documents, `.env`, and generated data are ignored by Git. Installation also
creates `src/rag.egg-info/`; this is generated package metadata and requires no
manual editing. Run one app process at a time against a local index.

## Tests

```bash
pip install -e ".[test]"
pytest tests/unit/rag
```

These tests use fake embeddings and temporary local indexes, without an API key
or Qdrant server. The optional server integration test uses `RAG_TEST_QDRANT_URL`.

## RAG evaluation (optional)

Use this test to check two different parts of RAG:

- **Retrieval:** did the search find passages related to the question?
- **Generation:** did the model use those passages to write a useful answer?

The evaluation test reuses the app's folder check, retriever, context builder,
and answer generator. Generation scores therefore evaluate the same answer
model and prompt used by the app.

Install the evaluation extra, add documents to `user_input/`, and set
`OPENAI_API_KEY` in `.env`. Run from the project folder with the app stopped:

```bash
pip install -e ".[eval]"
RUN_RAG_EVAL=1 pytest -s -v tests/rag_eval/test_rag.py
```

Evaluation is skipped by default during ordinary test runs. The explicit command
makes paid OpenAI calls for embeddings, answers, and judging. No DeepEval account
or dashboard is needed. Use `-k "dates"` to run only the example question about dates.

Add or replace questions in the `QUESTIONS` list in `tests/rag_eval/test_rag.py`.
Choose questions relevant to your own documents; no expected answers are required.
Set these optional values in `.env` to change the models or passing score:

```dotenv
OPENAI_RESPONSE_MODEL=gpt-4.1-mini
EVAL_JUDGE_MODEL=gpt-4.1-mini
EVAL_THRESHOLD=0.7
```

For each question, the console shows:

1. The question and retrieved passages, including search similarity scores.
2. Which passages fit into the final context sent to the answer model.
3. The generated answer.
4. The three evaluation scores, pass/fail results, and reasons.

Search similarity scores and evaluation scores measure different things.
A metric passes when its score is at least `EVAL_THRESHOLD` (default `0.7`).
The test prints all metric results before reporting failures. An API or judge
error is shown as `ERROR`, rather than as a low quality score.

Example layout (scores below are illustrative):

```text
Question:
What dates or deadlines are mentioned?

Retrieved chunks:
1. score=0.8123; in prompt
The report is due on 15 November.

RAG answer:
The report is due on 15 November. [Source 1]

Evaluation (threshold=0.7):
Retriever — Contextual Relevancy: 0.82 [PASS]
Reason: ...
Generator — Faithfulness: 0.95 [PASS]
Reason: ...
Generator — Answer Relevancy: 0.90 [PASS]
Reason: ...
```

| Stage | Metric | What a low score suggests |
| --- | --- | --- |
| Retrieval | Contextual Relevancy | Retrieved passages contain information unrelated to the question |
| Generation | Faithfulness | Answer claims are unsupported by the context actually sent to the generator |
| Generation | Answer Relevancy | The answer does not directly address the question |

Contextual Relevancy judges all retrieved chunks. Faithfulness judges only chunks
included by the context builder, so discarded passages cannot justify an answer.
Scores are reported individually, with no combined RAG score. These are model
judgments and can vary across runs. A relevant passage can still miss important
information. A model may correctly say that it cannot answer and still receive
a low answer relevancy score. Read the passages and reasons before deciding
which part needs fixing.

No expected answers, Contextual Precision, Contextual Recall, database, or
dashboard are used. To remove evaluation later, delete `tests/rag_eval/` and the
`eval` dependency extra; the document-search code does not depend on them.

DeepEval metric references: [Contextual Relevancy](https://deepeval.com/docs/metrics-contextual-relevancy),
[Faithfulness](https://deepeval.com/docs/metrics-faithfulness), and
[Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy).

## Optional code analysis

Graphify maps the Python code for development; it is separate from document search.

```bash
pip install -e ".[analysis]"
graphify extract . --code-only
graphify cluster-only . --no-label
graphify query "How does document indexing work?"
```

Open `graphify-out/graph.html` to explore the map. Re-run these analysis commands
after changing the code. Code-only analysis runs locally without an API key.

