"""Versioned prompt registry for the IntraMind AI agent.

Every LLM prompt the agent uses lives here with an explicit ``version`` and a
derived content ``hash``. The rules:

- Editing a prompt's ``template`` MUST be accompanied by bumping its ``version``
  and regenerating the lock file (``python -m prompts.registry --update-locks``).
- A guard test (``tests/test_prompt_registry.py``) fails if the registry drifts
  from ``locks.json``, so no prompt can change silently and every change shows
  up as an auditable diff (template + version + hash) in git history.

The active fingerprint (id -> version + hash) is stamped onto OTEL spans (so
Phoenix shows which prompt version produced each answer) and into Ragas eval
result files (so every baseline records exactly which prompts produced it).
This is what lets us attribute eval-score deltas to specific prompt changes.

To change a prompt:
    1. Edit the ``template`` below.
    2. Bump that prompt's ``version``.
    3. Run ``python -m prompts.registry --update-locks``.
    4. Add a ``CHANGELOG.md`` entry recording WHY it changed and the resulting
       eval delta (rerun the Ragas harness) - that rationale + numbers are the
       whole point of the changelog, since git/locks already track the rest.
    5. Commit template + version + locks.json + CHANGELOG together.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
LOCKS_PATH = PROMPTS_DIR / "locks.json"


@dataclass(frozen=True)
class Prompt:
    """A single, versioned prompt template.

    Attributes:
        id: Stable identifier (used as the key in fingerprints, span attributes,
            and the lock file). Never rename without treating it as a new prompt.
        version: Monotonically increasing integer. Bump on every template edit.
        template: The exact prompt text sent to the model.
    """

    id: str
    version: int
    template: str

    @property
    def hash(self) -> str:
        """Short, stable SHA-256 of the template text."""
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:12]

    def fingerprint(self) -> dict[str, object]:
        """Return the {id, version, hash} record used for stamping and locks."""
        return {"id": self.id, "version": self.version, "hash": self.hash}


# ---------------------------------------------------------------------------
# Prompts
#
# v1 templates are captured VERBATIM from the original inline literals in
# search_workflow.py (including their incidental indentation) so introducing
# this registry is a pure, behavior-preserving refactor. Any cleanup is a
# tracked version bump, not a silent edit.
# ---------------------------------------------------------------------------

QUERY_CLASSIFIER = Prompt(
    id="query_classifier",
    version=1,
    template="""You are a query classifier for a document search system.
    Classify the user's query as either 'simple' or 'complex'.

    Simple queries:
    - Direct fact lookups
    - Single-concept searches
    - Questions that can be answered with one search

    Complex queries:
    - Multi-part questions
    - Queries requiring aggregation of multiple documents
    - Comparative or analytical questions

    Respond with ONLY 'simple' or 'complex', nothing else.""",
)

QUERY_EXPANSION = Prompt(
    id="query_expansion",
    version=1,
    template="""You are a query expansion expert for document search.
    Given a complex query, generate 2-3 related search queries that will help find relevant documents.
    Each query should focus on a different aspect of the original question.

    Format your response as a numbered list:
    1. [first query]
    2. [second query]
    3. [third query]""",
)

RESULT_SYNTHESIS = Prompt(
    id="result_synthesis",
    version=1,
    template="""You are a helpful assistant that answers questions based on document search results.
    Use the provided documents to answer the user's question.
    Be concise and accurate. If the documents don't contain enough information, say so.
    Mention which document numbers support your answer (e.g., "According to Documents 1 and 2...").""",
)


REGISTRY: dict[str, Prompt] = {
    p.id: p
    for p in (
        QUERY_CLASSIFIER,
        QUERY_EXPANSION,
        RESULT_SYNTHESIS,
    )
}


def fingerprints() -> dict[str, dict[str, object]]:
    """Return {prompt_id: {id, version, hash}} for every registered prompt."""
    return {pid: p.fingerprint() for pid, p in REGISTRY.items()}


def annotate_span(prompt: Prompt) -> None:
    """Best-effort: tag the active OTEL span with the prompt fingerprint.

    No-op when tracing is disabled or no span is recording. Never raises -
    observability must never break the request path.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("prompt.id", prompt.id)
            span.set_attribute("prompt.version", prompt.version)
            span.set_attribute("prompt.hash", prompt.hash)
    except Exception:  # noqa: BLE001 - tracing must never break execution
        pass


def write_locks() -> None:
    """Write the current fingerprints to ``locks.json`` (sorted, trailing nl)."""
    LOCKS_PATH.write_text(
        json.dumps(fingerprints(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_locks() -> dict[str, dict[str, object]]:
    """Load the committed lock file."""
    return json.loads(LOCKS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt registry utilities")
    parser.add_argument(
        "--update-locks",
        action="store_true",
        help="Regenerate locks.json from the current registry",
    )
    args = parser.parse_args()

    if args.update_locks:
        write_locks()
        print(f"Wrote {LOCKS_PATH}")
    else:
        print(json.dumps(fingerprints(), indent=2, sort_keys=True))
