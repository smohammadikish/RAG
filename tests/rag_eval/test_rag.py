"""Opt-in, paid RAG debugging: RUN_RAG_EVAL=1 pytest -s tests/rag_eval."""

import os
from pathlib import Path

import pytest
from pydantic import Field

from rag import ContextBuilder, DocumentIndexer, RecursiveTextChunker, Retriever
from rag.cli import sync_folder
from rag.config import Settings
from rag.embeddings.openai import OpenAIEmbeddingProvider
from rag.loaders import SharedDocumentLoader
from rag.generation import generate_answer
from rag.vectorstores.qdrant import QdrantVectorStore

# Turn off optional DeepEval usage telemetry for these tests.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

# Replace these with questions relevant to your documents. No expected answers needed.
QUESTIONS = [
    "What are the main findings in the documents?",
    "What actions or recommendations are described?",
    "What dates or deadlines are mentioned?",
]

# Normal tests skip these paid calls unless the user enables them.
pytestmark = [
    pytest.mark.rag_eval,
    pytest.mark.skipif(os.getenv("RUN_RAG_EVAL") != "1", reason="Set RUN_RAG_EVAL=1 to run paid evaluation."),
]


class EvalSettings(Settings):
    # Evaluation-only settings; production configuration stays unchanged.
    EVAL_JUDGE_MODEL: str = "gpt-4.1-mini"
    EVAL_THRESHOLD: float = Field(default=0.7, ge=0, le=1)


# Check the documents once, then reuse the retriever for all questions.
@pytest.fixture(scope="module")
def pipeline():
    settings = EvalSettings()
    if not settings.OPENAI_API_KEY:
        pytest.fail("Set OPENAI_API_KEY in .env before running evaluation.")
    folder = Path("user_input").resolve()
    if not folder.is_dir():
        pytest.fail("Add documents to user_input/ and run from the project folder.")
    embeddings = OpenAIEmbeddingProvider(settings=settings)
    store = QdrantVectorStore(
        dimension=embeddings.dimension, settings=settings, ensure_collection=False, wait=True,
    )
    try:
        fresh = not store.collection_exists()
        store.ensure_collection()
        indexer = DocumentIndexer(
            loader=SharedDocumentLoader(),
            chunker=RecursiveTextChunker(
                chunk_size=settings.RAG_CHUNK_SIZE, chunk_overlap=settings.RAG_CHUNK_OVERLAP,
            ),
            embedding_provider=embeddings, vector_store=store,
        )
        if not sync_folder(folder, indexer, store, settings, force=fresh):
            pytest.fail("No documents ready for evaluation; check the indexing messages above.")
        yield settings, folder, Retriever(embedding_provider=embeddings, vector_store=store)
    finally:
        store.close()


@pytest.mark.parametrize("question", QUESTIONS)
def test_rag_answer(question, pipeline):
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
    from deepeval.models import OpenAIModel
    from deepeval.test_case import LLMTestCase

    settings, folder, retriever = pipeline
    print(f"\nQuestion:\n{question}\n\nRetrieved chunks:", flush=True)
    results = retriever.retrieve(
        question, top_k=settings.RAG_RETRIEVAL_TOP_K, filters={"input_folder": str(folder)},
    )
    context = ContextBuilder(max_chars=settings.RAG_CONTEXT_MAX_CHARS).build(results)
    # The context builder may leave out chunks that exceed its text limit.
    included = {source["chunk_id"] for source in context.sources}
    for number, result in enumerate(results, 1):
        status = "in prompt" if result.chunk.id in included else "excluded by context builder"
        print(f"\n{number}. score={result.score:.4f}; {status}\n{result.chunk.text}", flush=True)
    if not context.text:
        pytest.fail("Retrieval/context produced no usable passages; generation was not run.")
    print(f"\nFinal context sent to generator:\n{context.text}", flush=True)
    answer = generate_answer(question, context, settings)
    print(f"\nRAG answer:\n{answer}", flush=True)
    # Retrieval quality uses all chunks returned by the search.
    retrieval_case = LLMTestCase(
        input=question, actual_output=answer,
        retrieval_context=[result.chunk.text for result in results],
    )
    # Judge faithfulness against only the passages actually visible to the generator.
    generation_case = LLMTestCase(
        input=question, actual_output=answer,
        retrieval_context=[result.chunk.text for result in results if result.chunk.id in included],
    )
    # The judge scores the answer; it is separate from the answer model.
    judge = OpenAIModel(model=settings.EVAL_JUDGE_MODEL, api_key=settings.OPENAI_API_KEY)
    options = dict(threshold=settings.EVAL_THRESHOLD, model=judge, include_reason=True,
                   async_mode=False, eval_mode="llm")
    metrics = [
        ("Retriever", "Contextual Relevancy", ContextualRelevancyMetric(**options), retrieval_case),
        ("Generator", "Faithfulness", FaithfulnessMetric(**options), generation_case),
        ("Generator", "Answer Relevancy", AnswerRelevancyMetric(**options), generation_case),
    ]
    print(f"\nEvaluation (threshold={settings.EVAL_THRESHOLD}):", flush=True)
    # Show every metric result before failing the test.
    failures = []
    for stage, label, metric, case in metrics:
        try:
            metric.measure(case)
            passed = metric.is_successful()
            print(f"{stage} — {label}: {metric.score:.2f} [{'PASS' if passed else 'FAIL'}]")
            print(f"Reason: {metric.reason or 'No reason provided.'}", flush=True)
            if not passed:
                failures.append(f"{label}: {metric.score:.2f} — {metric.reason}")
        except Exception as exc:
            print(f"{stage} — {label}: ERROR — {exc}", flush=True)
            failures.append(f"{label}: evaluation error — {exc}")
    assert not failures, "\n".join(failures)
