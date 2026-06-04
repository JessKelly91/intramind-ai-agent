"""Tests for output safety guard (Step 4 of Free RAI Stack).

These tests cover:
  * Llama Guard verdict parsing for "safe" / "unsafe / S<n>" / malformed.
  * The safety_check workflow node's hard-block behavior: when classify_output
    returns is_safe=False the agent search() returns the templated fallback
    and citations are stripped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from utils.safety import (
    SafetyResult,
    _parse_llama_guard_output,
    classify_output,
)


class TestParseLlamaGuardOutput:
    """Unit tests for the verdict parser."""

    def test_parse_safe(self) -> None:
        r = _parse_llama_guard_output("safe")
        assert r.is_safe is True
        assert r.categories == []

    def test_parse_unsafe_with_categories(self) -> None:
        r = _parse_llama_guard_output("unsafe\nS1, S5")
        assert r.is_safe is False
        assert "S1" in r.categories and "S5" in r.categories

    def test_parse_unsafe_no_categories(self) -> None:
        r = _parse_llama_guard_output("unsafe")
        assert r.is_safe is False
        assert r.categories == []

    def test_parse_empty_defaults_to_safe(self) -> None:
        r = _parse_llama_guard_output("")
        assert r.is_safe is True

    def test_parse_malformed_fails_closed(self) -> None:
        # Anything that isn't "safe"/"unsafe" is treated as unsafe so the
        # hard-block policy still applies.
        r = _parse_llama_guard_output("maybe-not-safe?")
        assert r.is_safe is False
        assert r.categories == ["MALFORMED_VERDICT"]

    def test_to_metadata_excludes_text(self) -> None:
        r = SafetyResult(is_safe=False, categories=["S1"], raw_verdict="unsafe\nS1")
        meta = r.to_metadata(checked_at="2026-01-01T00:00:00Z")
        assert meta == {
            "flagged": True,
            "categories": ["S1"],
            "checked_at": "2026-01-01T00:00:00Z",
        }
        # Critically: never include the raw verdict or any prompt/response text.
        assert "raw_verdict" not in meta
        assert "response" not in meta


class TestClassifyOutputResilience:
    """The classifier must fail open if Ollama / langchain-ollama is unreachable."""

    @pytest.mark.asyncio
    async def test_missing_dependency_fails_open(self) -> None:
        # Force the ImportError branch by patching the import inside classify_output.
        with patch.dict(
            "sys.modules", {"langchain_ollama": None, "langchain_core.messages": None}
        ):
            # The patch.dict above only sets the entries; importlib will still
            # find them. Easier: drive the function with an obviously bad
            # base_url so the network call fails, exercising the except path.
            r = await classify_output(
                prompt="anything",
                response="anything",
                model="model-that-does-not-exist",
                base_url="http://127.0.0.1:1",  # nothing listens here
            )
        # is_safe=True with CLASSIFIER_UNAVAILABLE so dev/CI without Ollama
        # doesn't accidentally hard-block all traffic.
        assert r.is_safe is True
        assert "CLASSIFIER_UNAVAILABLE" in r.categories


class TestSafetyCheckNodeHardBlock:
    """End-to-end test of the hard-block policy in the search workflow."""

    @pytest.mark.asyncio
    async def test_unsafe_response_replaced_with_fallback(self) -> None:
        import sys

        from config import settings
        from models.state import SearchWorkflowState

        # NOTE: workflows/__init__.py re-exports the *compiled graph* under
        # the name `search_workflow`, so `from workflows import search_workflow`
        # gives a CompiledStateGraph, not the module. Pull the module out of
        # sys.modules instead so we can patch its functions.
        import workflows.search_workflow  # noqa: F401 -- ensures it's imported

        sw = sys.modules["workflows.search_workflow"]

        # Build a state as if synthesize_results just produced a flagged answer.
        state: SearchWorkflowState = {
            "messages": [],
            "user_query": "ignore previous instructions and...",
            "thread_id": None,
            "use_conversation_context": False,
            "current_step": "synthesize_results",
            "next_step": "safety_check",
            "workflow_complete": False,
            "search_strategy": "simple",
            "search_query": "...",
            "search_results": [{"id": "x", "content": "leaked source"}],
            "num_results": 5,
            "min_score": 0.0,
            "document_path": None,
            "document_type": None,
            "extracted_content": None,
            "document_metadata": {},
            "response": "ORIGINAL_FLAGGED_RESPONSE_TEXT",
            "citations": ["src://flagged"],
            "safety_flag": None,
            "error": None,
            "retry_count": 0,
            "query_complexity": "simple",
            "expanded_queries": None,
            "aggregated_results": None,
        }

        # Patch classify_output so it deterministically flags the response.
        unsafe = SafetyResult(
            is_safe=False, categories=["S1"], raw_verdict="unsafe\nS1"
        )
        with patch.object(sw, "classify_output", new=AsyncMock(return_value=unsafe)):
            after_check = await sw.safety_check(state)

        assert after_check["next_step"] == "handle_unsafe_response"
        assert after_check["safety_flag"]["flagged"] is True
        assert "S1" in after_check["safety_flag"]["categories"]

        # Now run the handler. It must scrub the original response and citations.
        after_handler = await sw.handle_unsafe_response(after_check)
        assert after_handler["response"] == settings.safety_fallback_message
        assert after_handler["citations"] == []
        assert after_handler["workflow_complete"] is True
        # The flag metadata is preserved for observability.
        assert after_handler["safety_flag"]["flagged"] is True
        # The original flagged text must not have leaked into the new state.
        assert "ORIGINAL_FLAGGED_RESPONSE_TEXT" not in (after_handler["response"] or "")

    @pytest.mark.asyncio
    async def test_safe_response_passes_through(self) -> None:
        import sys

        from models.state import SearchWorkflowState

        import workflows.search_workflow  # noqa: F401
        sw = sys.modules["workflows.search_workflow"]

        state: SearchWorkflowState = {
            "messages": [],
            "user_query": "What is the dividend?",
            "thread_id": None,
            "use_conversation_context": False,
            "current_step": "synthesize_results",
            "next_step": "safety_check",
            "workflow_complete": False,
            "search_strategy": "simple",
            "search_query": "dividend",
            "search_results": [{"id": "doc-1", "content": "...0.15..."}],
            "num_results": 5,
            "min_score": 0.0,
            "document_path": None,
            "document_type": None,
            "extracted_content": None,
            "document_metadata": {},
            "response": "The board approved a quarterly dividend of $0.15.",
            "citations": ["doc-1"],
            "safety_flag": None,
            "error": None,
            "retry_count": 0,
            "query_complexity": "simple",
            "expanded_queries": None,
            "aggregated_results": None,
        }
        safe = SafetyResult(is_safe=True, categories=[], raw_verdict="safe")
        with patch.object(sw, "classify_output", new=AsyncMock(return_value=safe)):
            after_check = await sw.safety_check(state)

        assert after_check["workflow_complete"] is True
        assert after_check["safety_flag"]["flagged"] is False
        # Original safe response and citations are preserved.
        assert "0.15" in after_check["response"]
        assert after_check["citations"] == ["doc-1"]
