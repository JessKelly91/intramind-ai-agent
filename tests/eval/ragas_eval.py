"""Ragas eval harness for IntraMind AI Agent.

Loads tests/eval/data/golden_qa.jsonl, drives IntraMindAgent.search() against
a fixture collection, then scores results with Ragas using a local Ollama judge.

Outputs:
  tests/eval/results/latest.json - structured per-question and aggregate scores

Usage:
  # Make sure the platform is up (Weaviate + vector-service + api-gateway)
  # and Ollama has the judge model pulled (default llama3.1:8b).
  python -m tests.eval.ragas_eval --collection eval_corpus --seed
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
GOLDEN_PATH = DATA_DIR / "golden_qa.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
LATEST_RESULTS = RESULTS_DIR / "latest.json"

DEFAULT_COLLECTION = "eval_corpus"
DEFAULT_NUM_RESULTS = 5


def _load_golden() -> list[dict[str, Any]]:
    """Load the golden Q&A JSONL file."""
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(f"Golden Q&A file not found: {GOLDEN_PATH}")

    entries: list[dict[str, Any]] = []
    with GOLDEN_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


async def _seed_corpus(collection: str) -> None:
    """Ingest the fixture corpus into the eval collection.

    Each file in tests/eval/data/corpus/ is ingested as a single document via
    the IntraMindAgent ingestion workflow. Idempotency is achieved by attempting
    to delete the collection first and recreating it.
    """
    # Defer imports until called - keeps `--help` cheap and avoids importing
    # the agent in environments without dependencies installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from agent.main import IntraMindAgent  # type: ignore[import]
    from tools.api_client import APIGatewayClient  # type: ignore[import]

    async with APIGatewayClient() as client:
        try:
            await client.delete_collection(collection)
            logger.info("Deleted existing collection: %s", collection)
        except Exception as exc:  # noqa: BLE001 - delete may legitimately 404
            logger.debug("Collection delete returned: %s (likely did not exist)", exc)

        await client.create_collection(
            name=collection, description="IntraMind RAG eval fixture corpus"
        )
        logger.info("Created eval collection: %s", collection)

    agent = IntraMindAgent(thread_id=False)  # disable conversation memory for evals

    for path in sorted(CORPUS_DIR.glob("*.txt")):
        result = await agent.ingest_document(
            file_path=str(path),
            collection_name=collection,
            document_metadata={"source": path.name, "consent_basis": "fixture"},
        )
        if not result.get("success"):
            raise RuntimeError(f"Failed to ingest {path.name}: {result.get('error')}")
        logger.info(
            "Seeded %s (%d chunks)", path.name, result.get("chunks_stored", 0)
        )


async def _run_agent(
    entries: list[dict[str, Any]], collection: str
) -> list[dict[str, Any]]:
    """Run the agent on every golden question and capture context + answer."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from agent.main import IntraMindAgent  # type: ignore[import]

    agent = IntraMindAgent(thread_id=False)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        question = entry["question"]
        result = await agent.search(
            query=question,
            collection_name=collection,
            num_results=DEFAULT_NUM_RESULTS,
            min_score=0.0,
        )
        contexts = [
            r.get("content", "") for r in result.get("results", []) if r.get("content")
        ]
        rows.append(
            {
                "id": entry.get("id", hashlib.sha1(question.encode()).hexdigest()[:8]),
                "question": question,
                "ground_truth": entry.get("ground_truth", ""),
                "expected_source": entry.get("expected_source"),
                "answer": result.get("response", ""),
                "contexts": contexts,
                "success": bool(result.get("success")),
            }
        )
    return rows


def _score_with_ragas(
    rows: list[dict[str, Any]], judge_model: str
) -> dict[str, Any]:
    """Score rows with Ragas using a local Ollama judge.

    Returns a dict with per-question and aggregate scores.
    """
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ragas dependencies are not installed. "
            "Install with: pip install ragas langchain-ollama datasets"
        ) from exc

    judge_llm = LangchainLLMWrapper(ChatOllama(model=judge_model, temperature=0.0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=judge_model)
    )

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"] or [""],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    aggregate = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

    # Per-row scores. Ragas returns a Dataset/DataFrame; convert to records.
    try:
        per_row = result.to_pandas().to_dict(orient="records")
    except Exception:
        per_row = []

    return {"aggregate": aggregate, "per_row": per_row}


def _write_results(
    rows: list[dict[str, Any]], scores: dict[str, Any], judge_model: str
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "judge_model": judge_model,
        "num_questions": len(rows),
        "aggregate": scores["aggregate"],
        "per_row": scores["per_row"],
        "raw_rows": rows,
    }
    LATEST_RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LATEST_RESULTS


async def _async_main(args: argparse.Namespace) -> int:
    if args.seed:
        await _seed_corpus(args.collection)

    entries = _load_golden()
    if not entries:
        print("No golden Q&A entries found.", file=sys.stderr)
        return 1

    rows = await _run_agent(entries, args.collection)
    scores = _score_with_ragas(rows, args.judge_model)
    out_path = _write_results(rows, scores, args.judge_model)

    print(f"\nRagas evaluation complete. Results written to: {out_path}")
    print(f"\nAggregate scores (judge: {args.judge_model}):")
    for metric, value in scores["aggregate"].items():
        print(f"  {metric:<20} {value:.3f}")
    return 0


def main() -> int:
    """Console entry point."""
    parser = argparse.ArgumentParser(description="Run Ragas evals on IntraMind agent")
    parser.add_argument(
        "--collection",
        default=os.environ.get("RAGAS_COLLECTION", DEFAULT_COLLECTION),
        help="Collection name to seed and search against",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Re-ingest the fixture corpus before running evals",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("RAGAS_JUDGE_MODEL", "llama3.1:8b"),
        help="Ollama model used as the LLM judge",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Python logging level (default INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())
