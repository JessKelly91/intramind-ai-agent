"""PII detection and redaction for IntraMind ingestion.

Step 3 of the Free RAI Stack. Decision locked in: redact-on-ingest only,
using Presidio's `replace` operator with stable per-document tokenized
pseudonyms (e.g. <PERSON_1>, <EMAIL_2>).

Why this specific approach:
  * Raw PII never enters Weaviate (data minimization).
  * Tokenized replacement preserves sentence structure so retrieval quality
    on non-PII concepts stays high.
  * Stable token IDs within a single document preserve co-reference, which
    keeps the synthesizer coherent.
  * Audit metadata (`pii_findings`) records type and offsets but NOT raw
    values - so we can prove redaction happened without retaining PII.

Trade-off: irreversible. Future role-based unmasking would require a
separate originals store and is explicitly out of scope.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
]
DEFAULT_SCORE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class PIIFinding:
    """A single detected PII instance, stripped of the raw value."""

    entity_type: str
    start: int
    end: int
    score: float
    token: str  # the replacement token, e.g. "<PERSON_1>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": round(self.score, 3),
            "token": self.token,
        }


class PIIRedactor:
    """Detect PII via Presidio and replace with stable tokenized pseudonyms.

    Token assignment is stable within a single `redact()` call: identical
    detected text within one document gets the same token (e.g. "John Doe"
    appearing 5 times all become "<PERSON_1>"), but tokens are NOT stable
    across calls/documents - so there is no cross-document linkage of PII.
    """

    def __init__(
        self,
        entities: Iterable[str] | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.entities = list(entities) if entities is not None else list(DEFAULT_ENTITIES)
        self.score_threshold = score_threshold
        self._analyzer = None  # lazily constructed
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """True if Presidio + spaCy model are importable and usable."""
        if self._available is None:
            self._lazy_init()
        return bool(self._available)

    def _lazy_init(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            logger.warning(
                "Presidio is not installed - PII redaction disabled: %s", exc
            )
            self._available = False
            return

        try:
            self._analyzer = AnalyzerEngine()
            self._available = True
        except Exception as exc:  # noqa: BLE001 - typically missing spaCy model
            logger.warning(
                "Presidio AnalyzerEngine init failed - PII redaction disabled. "
                "Did you run `python -m spacy download en_core_web_lg`? %s",
                exc,
            )
            self._available = False

    def redact(
        self, text: str, doc_id: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Detect and replace PII with type-tagged tokens.

        Args:
            text: Raw extracted document text.
            doc_id: Optional identifier used only for logging - tokens are
                NOT salted with this value, so there is no cross-document
                linkage even if the same caller passes the same doc_id.

        Returns:
            (redacted_text, findings) where:
              * redacted_text has PII replaced with `<TYPE_N>` tokens.
              * findings is a list of dicts (no raw values stored).
        """
        if not text:
            return text, []

        if not self.available:
            return text, []

        analyzer_results = self._analyzer.analyze(  # type: ignore[union-attr]
            text=text,
            entities=self.entities,
            language="en",
            score_threshold=self.score_threshold,
        )
        # Sort by start ascending; on ties prefer longer matches first.
        sorted_results = sorted(
            analyzer_results, key=lambda r: (r.start, -(r.end - r.start))
        )

        # Greedy non-overlap filtering so we don't double-tag overlapping spans
        # (Presidio occasionally returns nested PERSON inside LOCATION etc.).
        filtered: list[Any] = []
        last_end = -1
        for r in sorted_results:
            if r.start >= last_end:
                filtered.append(r)
                last_end = r.end

        # Stable token assignment: same surface text within this document
        # gets the same token. Counter is per-entity-type.
        per_type_counters: dict[str, int] = defaultdict(int)
        text_to_token: dict[tuple[str, str], str] = {}
        findings: list[PIIFinding] = []

        for r in filtered:
            surface = text[r.start : r.end]
            key = (r.entity_type, surface)
            token = text_to_token.get(key)
            if token is None:
                per_type_counters[r.entity_type] += 1
                token = f"<{r.entity_type}_{per_type_counters[r.entity_type]}>"
                text_to_token[key] = token
            findings.append(
                PIIFinding(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=float(r.score),
                    token=token,
                )
            )

        # Build redacted text by walking the original and substituting spans.
        if not findings:
            logger.debug("No PII found in document (doc_id=%s)", doc_id)
            return text, []

        out_parts: list[str] = []
        cursor = 0
        for f in findings:
            if f.start > cursor:
                out_parts.append(text[cursor : f.start])
            out_parts.append(f.token)
            cursor = f.end
        if cursor < len(text):
            out_parts.append(text[cursor:])

        redacted = "".join(out_parts)
        logger.info(
            "Redacted %d PII span(s) in document (doc_id=%s)",
            len(findings),
            doc_id,
        )
        return redacted, [f.to_dict() for f in findings]


# Convenience module-level singleton, initialized lazily.
_default_redactor: PIIRedactor | None = None


def get_default_redactor(
    entities: Iterable[str] | None = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> PIIRedactor:
    """Return a process-wide PIIRedactor configured from settings."""
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = PIIRedactor(
            entities=entities, score_threshold=score_threshold
        )
    return _default_redactor
