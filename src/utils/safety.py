"""Output safety classification via Llama Guard on Ollama.

Step 4 of the Free RAI Stack. Decision locked in: hard-block on unsafe
verdicts. The original (flagged) response text is never returned to the
user; we replace it with a templated fallback and discard citations.

Llama Guard returns text in this canonical shape::

    safe

    OR

    unsafe
    S1, S5

We parse both forms and record categories without ever surfacing the
original response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """Outcome of a Llama Guard classification call.

    Attributes:
        is_safe: True when the classifier returns "safe".
        categories: Llama Guard violation categories (e.g. "S1", "S5") if any.
        raw_verdict: Raw model output (kept for debugging, never returned to
            the user along with the flagged response).
        unavailable: True when the classifier could not produce a reliable
            verdict because dependencies or infrastructure were unavailable.
    """

    is_safe: bool
    categories: list[str] = field(default_factory=list)
    raw_verdict: str = ""
    unavailable: bool = False
    error_reason: str | None = None

    def to_metadata(self, checked_at: Optional[str] = None) -> dict[str, Any]:
        """Convert to a JSON-safe dict for state.safety_flag.

        Notably does NOT include the prompt or response text.
        """
        return {
            "flagged": not self.is_safe,
            "categories": list(self.categories),
            "checked_at": checked_at,
            "classifier_unavailable": self.unavailable,
            "error_reason": self.error_reason,
        }


def _parse_llama_guard_output(text: str) -> SafetyResult:
    """Parse Llama Guard's plain-text verdict into a SafetyResult."""
    raw = (text or "").strip()
    if not raw:
        # Default to safe in permissive mode rather than blocking everything,
        # but mark it as unavailable so required mode can fail closed.
        logger.warning("Llama Guard returned empty output - defaulting to safe")
        return SafetyResult(
            is_safe=True,
            categories=["EMPTY_VERDICT"],
            raw_verdict=raw,
            unavailable=True,
            error_reason="empty_verdict",
        )

    first_line = raw.splitlines()[0].strip().lower()
    if first_line == "safe":
        return SafetyResult(is_safe=True, raw_verdict=raw)

    if first_line == "unsafe":
        # Subsequent lines (if any) list comma-separated category codes.
        categories: list[str] = []
        for line in raw.splitlines()[1:]:
            for token in line.split(","):
                token = token.strip()
                if token:
                    categories.append(token)
        return SafetyResult(is_safe=False, categories=categories, raw_verdict=raw)

    # Unknown / malformed verdict. Fail closed (treat as unsafe) so the
    # hard-block policy still applies.
    logger.warning("Llama Guard returned unexpected verdict: %r", raw)
    return SafetyResult(is_safe=False, categories=["MALFORMED_VERDICT"], raw_verdict=raw)


async def classify_output(
    prompt: str,
    response: str,
    model: str = "llama-guard3",
    base_url: Optional[str] = None,
) -> SafetyResult:
    """Classify a synthesized response with Llama Guard via Ollama.

    Args:
        prompt: Original user prompt that produced the response.
        response: Synthesized assistant response to evaluate.
        model: Ollama model tag (default ``llama-guard3``).
        base_url: Override Ollama base URL. If None, uses the
            ``langchain_ollama`` default which respects ``OLLAMA_HOST`` /
            ``OLLAMA_BASE_URL``.

    Returns:
        SafetyResult. On unrecoverable errors (Ollama unreachable, package
        missing) returns ``is_safe=True`` in permissive mode metadata with
        categories=["CLASSIFIER_UNAVAILABLE"] and ``unavailable=True``. Required
        mode in the workflow uses this metadata to fail closed.
    """
    try:
        from langchain_core.messages import HumanMessage
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        logger.warning(
            "langchain-ollama not installed - safety classifier disabled: %s", exc
        )
        return SafetyResult(
            is_safe=True,
            categories=["CLASSIFIER_UNAVAILABLE"],
            raw_verdict="<langchain-ollama not installed>",
            unavailable=True,
            error_reason="dependency_missing",
        )

    kwargs: dict[str, Any] = {"model": model, "temperature": 0.0}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        guard = ChatOllama(**kwargs)
        # Llama Guard models accept the standard chat message format and
        # internally render the moderation prompt template.
        guard_messages = [
            HumanMessage(content=prompt),
            # langchain-ollama doesn't expose AIMessage with role="assistant"
            # for guard models; we pass the assistant text as a second human
            # turn prefixed with a clear marker. Llama Guard 3 understands
            # this pattern.
            HumanMessage(content=f"[ASSISTANT_RESPONSE]\n{response}"),
        ]
        result = await guard.ainvoke(guard_messages)
        verdict_text = getattr(result, "content", "") or ""
        return _parse_llama_guard_output(verdict_text)
    except Exception as exc:  # noqa: BLE001 - workflow decides fail-open/closed
        logger.warning("Safety classifier call failed: %s", exc)
        return SafetyResult(
            is_safe=True,
            categories=["CLASSIFIER_UNAVAILABLE"],
            raw_verdict=f"<error: {exc.__class__.__name__}>",
            unavailable=True,
            error_reason=exc.__class__.__name__,
        )
