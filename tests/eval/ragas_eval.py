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

import httpx

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


_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _install_ragas_compat_stubs() -> None:
    """Stub langchain_community symbols that ragas hard-imports but no longer ship.

    ragas imports ``ChatVertexAI`` / ``VertexAI`` from ``langchain_community`` at
    module load, but ``langchain-community`` 0.4.x removed those paths. They are
    never used with an Ollama judge, so inject harmless placeholders to keep the
    import chain working. No-op when the real modules are importable.
    """
    import types

    try:  # langchain_community.chat_models.vertexai.ChatVertexAI
        __import__(
            "langchain_community.chat_models.vertexai", fromlist=["ChatVertexAI"]
        )
    except Exception:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")
        mod.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = mod

    try:  # langchain_community.llms.VertexAI
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        try:
            import langchain_community.llms as _llms  # type: ignore

            if not hasattr(_llms, "VertexAI"):
                _llms.VertexAI = type("VertexAI", (), {})  # type: ignore[attr-defined]
        except Exception:
            pass


def _score_with_ragas(
    rows: list[dict[str, Any]], judge_model: str
) -> dict[str, Any]:
    """Score rows with Ragas using a local Ollama judge.

    Runs serialized (``RunConfig(max_workers=1)``): Ragas' default concurrent
    async executor deadlocks on Windows with the current langchain stack.
    Aggregates via pandas column means so it stays robust to ragas
    result-accessor changes across versions.

    NOTE: must be called OUTSIDE a running event loop - ``ragas.evaluate()``
    starts its own loop and deadlocks if nested inside ``asyncio.run()``.
    """
    _install_ragas_compat_stubs()

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
        from ragas.run_config import RunConfig
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
        # Serialize: the concurrent executor deadlocks on this stack/OS.
        run_config=RunConfig(max_workers=1, timeout=240),
        # A weak local judge occasionally emits unparseable output; record it as
        # NaN-and-skip instead of aborting the whole run.
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    aggregate: dict[str, float] = {}
    for metric in _METRIC_NAMES:
        if metric in df.columns:
            aggregate[metric] = float(df[metric].astype(float).mean(skipna=True))
        else:
            aggregate[metric] = float("nan")

    per_row = df.to_dict(orient="records")
    return {"aggregate": aggregate, "per_row": per_row}


def _prompt_versions() -> dict[str, Any] | None:
    """Fingerprint the agent's prompts so each baseline records what produced it.

    Returns {prompt_id: {id, version, hash}} or None if the registry can't be
    imported (e.g. running the eval against an older agent checkout).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    try:
        from prompts.registry import fingerprints  # type: ignore[import]

        return fingerprints()
    except Exception:  # noqa: BLE001 - prompt stamping is best-effort metadata
        return None


def _active_prompt_versions(label: str) -> dict[str, Any] | None:
    """Resolve active prompt versions from the runtime registry when configured."""

    fallback_versions = _prompt_versions()
    registry_url = os.environ.get("PROMPT_REGISTRY_URL", "").rstrip("/")
    api_key = os.environ.get("PROMPT_REGISTRY_API_KEY", "")
    if not registry_url or not api_key or not fallback_versions:
        return fallback_versions

    active: dict[str, Any] = {}
    for prompt_id in fallback_versions:
        try:
            response = httpx.get(
                f"{registry_url}/api/v1/prompts/{prompt_id}",
                params={"label": label},
                headers={"X-API-Key": api_key},
                timeout=5.0,
            )
            response.raise_for_status()
            body = response.json()
            active[prompt_id] = {
                "id": body["id"],
                "version": body["version"],
                "hash": body["hash"],
                "label": body.get("label", label),
                "source": "registry",
            }
        except Exception as exc:  # noqa: BLE001 - eval metadata is best-effort
            logger.warning("Could not resolve runtime prompt %s: %s", prompt_id, exc)
            active[prompt_id] = fallback_versions[prompt_id]
    return active


def _metric_threshold(metric: str) -> float:
    env_key = f"RAGAS_THRESHOLD_{metric.upper()}"
    if env_key in os.environ:
        return float(os.environ[env_key])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    try:
        from config import settings  # type: ignore[import]

        return float(getattr(settings, f"ragas_threshold_{metric}"))
    except Exception:
        return 0.7


def _eval_passed(aggregate: dict[str, Any]) -> bool:
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    return all(float(aggregate.get(metric, 0.0)) >= _metric_threshold(metric) for metric in metrics)


def _post_prompt_registry_evals(payload: dict[str, Any]) -> None:
    """Best-effort POST of aggregate Ragas metrics to active prompt versions."""

    registry_url = os.environ.get("PROMPT_REGISTRY_URL", "").rstrip("/")
    api_key = os.environ.get("PROMPT_REGISTRY_API_KEY", "")
    prompt_versions = payload.get("prompt_versions") or {}
    if not registry_url or not api_key or not prompt_versions:
        return

    passed = _eval_passed(payload.get("aggregate", {}))
    for prompt_id, prompt_version in prompt_versions.items():
        version = prompt_version.get("version")
        if not version:
            continue
        try:
            response = httpx.post(
                f"{registry_url}/api/v1/prompts/{prompt_id}/versions/{version}/evals",
                headers={"X-API-Key": api_key},
                json={
                    "judge_model": payload["judge_model"],
                    "metrics": payload["aggregate"],
                    "passed": passed,
                    "results_ref": str(LATEST_RESULTS),
                },
                timeout=5.0,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - posting evals must not fail CI
            logger.warning(
                "Could not post Ragas eval metrics for %s v%s: %s",
                prompt_id,
                version,
                exc,
            )


def _write_results(
    rows: list[dict[str, Any]], scores: dict[str, Any], judge_model: str
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label = os.environ.get("PROMPT_REGISTRY_LABEL", "production")
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "judge_model": judge_model,
        "num_questions": len(rows),
        # Records which prompt versions produced this baseline so score deltas
        # can be attributed to specific prompt changes.
        "prompt_versions": _active_prompt_versions(label),
        "aggregate": scores["aggregate"],
        "per_row": scores["per_row"],
        "raw_rows": rows,
    }
    LATEST_RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _post_prompt_registry_evals(payload)
    return LATEST_RESULTS


async def _gather_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Async phase: (optionally) seed the corpus, then drive the agent.

    Kept separate from scoring so that ``ragas.evaluate()`` (which spins up its
    own event loop) runs only after this loop has fully closed - calling it from
    within ``asyncio.run()`` deadlocks.
    """
    if args.seed:
        await _seed_corpus(args.collection)

    entries = _load_golden()
    if not entries:
        raise RuntimeError("No golden Q&A entries found.")

    return await _run_agent(entries, args.collection)


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

    # Ragas' internal async executor is more stable on the Selector loop;
    # Windows' default Proactor loop can hang it.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Phase 1 (async): seed + run the agent over the golden questions.
    rows = asyncio.run(_gather_rows(args))

    # Phase 2 (sync, OUTSIDE the event loop): score with Ragas and persist.
    scores = _score_with_ragas(rows, args.judge_model)
    out_path = _write_results(rows, scores, args.judge_model)

    print(f"\nRagas evaluation complete. Results written to: {out_path}")
    print(f"\nAggregate scores (judge: {args.judge_model}):")
    for metric, value in scores["aggregate"].items():
        print(f"  {metric:<20} {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
