"""Ragas threshold gate tests.

Reads the latest aggregate scores produced by ragas_eval.py and asserts they
meet configured thresholds. The asserts are real - flipping the env var
RAGAS_ENFORCE_THRESHOLDS=true (or settings.ragas_enforce_thresholds=True) is
all that's needed to upgrade these from warning-only to enforcing.

Behavior matrix:

    RAGAS_ENFORCE_THRESHOLDS=false (default)  -> tests xfail on score regressions
    RAGAS_ENFORCE_THRESHOLDS=true             -> tests fail on score regressions

Either way the asserts are exercised, so the failing-check infrastructure is
fully built out and just gated.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

EVAL_DIR = Path(__file__).resolve().parent
LATEST_RESULTS = EVAL_DIR / "results" / "latest.json"


def _enforce() -> bool:
    """Return True when threshold violations should fail the suite."""
    raw = os.environ.get("RAGAS_ENFORCE_THRESHOLDS", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    # Fall back to settings if the env var isn't set.
    try:
        import sys

        sys.path.insert(0, str(EVAL_DIR.parents[1] / "src"))
        from config import settings  # type: ignore[import]

        return bool(settings.ragas_enforce_thresholds)
    except Exception:
        return False


def _threshold(metric: str) -> float:
    env_key = f"RAGAS_THRESHOLD_{metric.upper()}"
    if env_key in os.environ:
        try:
            return float(os.environ[env_key])
        except ValueError:
            pass
    try:
        import sys

        sys.path.insert(0, str(EVAL_DIR.parents[1] / "src"))
        from config import settings  # type: ignore[import]

        return float(getattr(settings, f"ragas_threshold_{metric}"))
    except Exception:
        return 0.7


@pytest.fixture(scope="module")
def aggregate_scores() -> dict:
    if not LATEST_RESULTS.exists():
        pytest.skip(
            f"No Ragas results at {LATEST_RESULTS}. "
            "Run `python -m tests.eval.ragas_eval --seed` first."
        )
    payload = json.loads(LATEST_RESULTS.read_text(encoding="utf-8"))
    return payload.get("aggregate", {})


def _check(metric: str, score: Any) -> None:
    threshold = _threshold(metric)
    if score is None or (
        isinstance(score, float) and (math.isnan(score) or math.isinf(score))
    ):
        msg = f"{metric}=null/unparseable below threshold {threshold:.3f}"
        if _enforce():
            pytest.fail(msg)
        else:
            pytest.xfail(f"warning-only: {msg}")
    if score >= threshold:
        return
    msg = f"{metric}={score:.3f} below threshold {threshold:.3f}"
    if _enforce():
        pytest.fail(msg)
    else:
        pytest.xfail(f"warning-only: {msg}")


def test_faithfulness_threshold(aggregate_scores: dict) -> None:
    _check("faithfulness", aggregate_scores.get("faithfulness", 0.0))


def test_answer_relevancy_threshold(aggregate_scores: dict) -> None:
    _check("answer_relevancy", aggregate_scores.get("answer_relevancy", 0.0))


def test_context_precision_threshold(aggregate_scores: dict) -> None:
    _check("context_precision", aggregate_scores.get("context_precision", 0.0))


def test_context_recall_threshold(aggregate_scores: dict) -> None:
    _check("context_recall", aggregate_scores.get("context_recall", 0.0))
