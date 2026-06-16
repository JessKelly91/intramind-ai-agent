"""Unit tests for deterministic retrieval metrics in the Ragas eval harness."""

import math

from tests.eval.ragas_eval import (
    _eval_passed,
    _json_safe,
    _ragas_quality,
    _retrieved_contexts,
    _row_retrieval_metrics,
    _score_retrieval,
)


def test_retrieved_contexts_extracts_source_from_metadata() -> None:
    contexts = _retrieved_contexts(
        [
            {
                "id": "chunk-1",
                "content": "Relevant chunk",
                "metadata": {"source": "docs/source-a.txt"},
                "score": 0.91,
            },
            {
                "id": "chunk-2",
                "content": "",
                "metadata": {"source": "docs/empty.txt"},
                "score": 0.5,
            },
            {
                "id": "chunk-3",
                "content": "Fallback filename chunk",
                "metadata": {"filename": "source-b.txt"},
                "score": 0.82,
            },
        ]
    )

    assert contexts == [
        {
            "rank": 1,
            "id": "chunk-1",
            "source": "source-a.txt",
            "score": 0.91,
            "content": "Relevant chunk",
        },
        {
            "rank": 3,
            "id": "chunk-3",
            "source": "source-b.txt",
            "score": 0.82,
            "content": "Fallback filename chunk",
        },
    ]


def test_row_retrieval_metrics_find_expected_source_rank() -> None:
    metrics = _row_retrieval_metrics(
        {
            "expected_source": "source-b.txt",
            "retrieved_contexts": [
                {"rank": 1, "source": "source-a.txt"},
                {"rank": 2, "source": "source-b.txt"},
            ],
        }
    )

    assert metrics["expected_source_rank"] == 2
    assert metrics["hit_at_1"] is False
    assert metrics["hit_at_3"] is True
    assert metrics["mrr"] == 0.5


def test_score_retrieval_aggregates_hits_and_misses() -> None:
    scores = _score_retrieval(
        [
            {
                "id": "q1",
                "question": "Question 1",
                "expected_source": "source-a.txt",
                "retrieved_contexts": [
                    {"rank": 1, "source": "source-a.txt"},
                    {"rank": 2, "source": "source-b.txt"},
                ],
            },
            {
                "id": "q2",
                "question": "Question 2",
                "expected_source": "source-b.txt",
                "retrieved_contexts": [
                    {"rank": 1, "source": "source-a.txt"},
                    {"rank": 2, "source": "source-b.txt"},
                ],
            },
            {
                "id": "q3",
                "question": "Question 3",
                "expected_source": "source-c.txt",
                "retrieved_contexts": [
                    {"rank": 1, "source": "source-a.txt"},
                    {"rank": 2, "source": "source-b.txt"},
                ],
            },
        ]
    )

    assert scores["hit_at_1"] == 1 / 3
    assert scores["hit_at_3"] == 2 / 3
    assert scores["mrr"] == 0.5
    assert scores["misses"] == [
        {
            "id": "q3",
            "question": "Question 3",
            "expected_source": "source-c.txt",
            "retrieved_sources": ["source-a.txt", "source-b.txt"],
        }
    ]


def test_json_safe_converts_nan_and_infinity_to_none() -> None:
    payload = {
        "aggregate": {
            "faithfulness": math.nan,
            "answer_relevancy": math.inf,
            "context_precision": -math.inf,
            "context_recall": 1.0,
        },
        "rows": [{"faithfulness": math.nan}],
    }

    assert _json_safe(payload) == {
        "aggregate": {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": 1.0,
        },
        "rows": [{"faithfulness": None}],
    }


def test_ragas_quality_counts_parse_failures(monkeypatch) -> None:
    monkeypatch.setenv("RAGAS_MAX_PARSE_FAILURE_RATE", "0.5")

    quality = _ragas_quality(
        [
            {
                "faithfulness": math.nan,
                "answer_relevancy": 0.9,
                "context_precision": math.nan,
                "context_recall": 1.0,
            },
            {
                "faithfulness": 0.8,
                "answer_relevancy": math.nan,
                "context_precision": 1.0,
                "context_recall": 1.0,
            },
        ]
    )

    assert quality["nan_counts"] == {
        "faithfulness": 1,
        "answer_relevancy": 1,
        "context_precision": 1,
        "context_recall": 0,
    }
    assert quality["parse_failure_count"] == 3
    assert quality["total_metric_cells"] == 8
    assert quality["parse_failure_rate"] == 3 / 8
    assert quality["passed"] is True


def test_eval_passed_fails_on_parse_failure_rate(monkeypatch) -> None:
    monkeypatch.setenv("RAGAS_THRESHOLD_FAITHFULNESS", "0.7")
    monkeypatch.setenv("RAGAS_THRESHOLD_ANSWER_RELEVANCY", "0.7")
    monkeypatch.setenv("RAGAS_THRESHOLD_CONTEXT_PRECISION", "0.7")
    monkeypatch.setenv("RAGAS_THRESHOLD_CONTEXT_RECALL", "0.7")

    aggregate = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.9,
        "context_precision": 0.9,
        "context_recall": 0.9,
    }

    assert _eval_passed(aggregate, {"passed": True}) is True
    assert _eval_passed(aggregate, {"passed": False}) is False


def test_eval_passed_fails_on_unparseable_metric(monkeypatch) -> None:
    monkeypatch.setenv("RAGAS_THRESHOLD_FAITHFULNESS", "0.7")
    monkeypatch.setenv("RAGAS_THRESHOLD_ANSWER_RELEVANCY", "0.7")
    monkeypatch.setenv("RAGAS_THRESHOLD_CONTEXT_PRECISION", "0.7")
    monkeypatch.setenv("RAGAS_THRESHOLD_CONTEXT_RECALL", "0.7")

    aggregate = {
        "faithfulness": None,
        "answer_relevancy": 0.9,
        "context_precision": 0.9,
        "context_recall": 0.9,
    }

    assert _eval_passed(aggregate, {"passed": True}) is False
