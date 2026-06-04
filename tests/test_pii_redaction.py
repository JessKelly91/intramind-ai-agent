"""Tests for PII redaction (Step 3 of Free RAI Stack).

These tests use the synthetic strings only - no real PII.
"""

from __future__ import annotations

import pytest

from utils.pii import PIIRedactor


def _redactor() -> PIIRedactor:
    """Build a fresh redactor and skip the test if Presidio isn't installed."""
    r = PIIRedactor()
    if not r.available:
        pytest.skip(
            "Presidio not available - install presidio-analyzer + spaCy model "
            "(`python -m spacy download en_core_web_lg`) to enable these tests"
        )
    return r


class TestPIIRedactor:
    """Unit tests for the PIIRedactor utility."""

    def test_no_pii_passthrough(self) -> None:
        r = _redactor()
        text = "The quarterly report covered revenue and operating expenses."
        redacted, findings = r.redact(text, doc_id="doc-1")
        assert redacted == text
        assert findings == []

    def test_email_is_tokenized(self) -> None:
        r = _redactor()
        redacted, findings = r.redact(
            "Please email jane@example.com for details.", doc_id="doc-1"
        )
        assert "jane@example.com" not in redacted
        assert "<EMAIL_ADDRESS_1>" in redacted
        types = {f["entity_type"] for f in findings}
        assert "EMAIL_ADDRESS" in types
        # No raw value should be retained in findings
        for f in findings:
            assert "value" not in f
            assert f["token"] == "<EMAIL_ADDRESS_1>"

    def test_repeated_value_gets_same_token(self) -> None:
        r = _redactor()
        text = (
            "Contact jane@example.com for billing. "
            "We also cc'd jane@example.com on the renewal."
        )
        redacted, findings = r.redact(text, doc_id="doc-1")
        # The same email should yield the same token both times
        assert redacted.count("<EMAIL_ADDRESS_1>") == 2
        assert "jane@example.com" not in redacted
        # And we should see two findings (one per occurrence)
        email_findings = [f for f in findings if f["entity_type"] == "EMAIL_ADDRESS"]
        assert len(email_findings) == 2
        assert {f["token"] for f in email_findings} == {"<EMAIL_ADDRESS_1>"}

    def test_distinct_values_get_distinct_tokens(self) -> None:
        r = _redactor()
        text = "Reach out to alice@example.com or bob@example.com."
        redacted, findings = r.redact(text, doc_id="doc-1")
        assert "alice@example.com" not in redacted
        assert "bob@example.com" not in redacted
        tokens = {f["token"] for f in findings if f["entity_type"] == "EMAIL_ADDRESS"}
        assert tokens == {"<EMAIL_ADDRESS_1>", "<EMAIL_ADDRESS_2>"}

    def test_findings_have_no_raw_values(self) -> None:
        r = _redactor()
        text = "Contact Jane Smith at jane@example.com or 555-1212."
        _, findings = r.redact(text, doc_id="doc-1")
        for f in findings:
            assert set(f.keys()) <= {"entity_type", "start", "end", "score", "token"}

    def test_empty_text_returns_empty(self) -> None:
        r = _redactor()
        redacted, findings = r.redact("", doc_id="doc-1")
        assert redacted == ""
        assert findings == []


class TestPIIRedactorWhenUnavailable:
    """When Presidio isn't installed, redact() should pass content through."""

    def test_unavailable_redactor_is_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        r = PIIRedactor()
        # Force the "not available" branch even if Presidio is installed.
        r._available = False
        text = "Contact jane@example.com"
        redacted, findings = r.redact(text, doc_id="doc-1")
        assert redacted == text
        assert findings == []
