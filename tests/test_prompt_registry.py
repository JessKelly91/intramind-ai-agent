"""Guard tests for the versioned prompt registry.

The key test is ``test_registry_matches_locks``: it fails if any prompt template
was edited without bumping its version and regenerating ``locks.json``. This is
what makes prompt changes auditable - you cannot silently change a prompt.
"""

import hashlib
import sys

import pytest

from prompts.registry import (
    REGISTRY,
    Prompt,
    fingerprints,
    load_locks,
    write_locks,
)

# NOTE: the `workflows` package re-exports a `search_workflow` graph object that
# shadows the submodule attribute, so `import workflows.search_workflow as x`
# would bind the graph, not the module. Pull the module from sys.modules.
import workflows.search_workflow  # noqa: F401  (ensures the submodule is loaded)

search_workflow_module = sys.modules["workflows.search_workflow"]


def test_registry_matches_locks():
    """Every registered prompt must match the committed lock file.

    If this fails because you intentionally changed a prompt:
      1. Bump that prompt's `version` in prompts/registry.py
      2. Run `python -m prompts.registry --update-locks` (from src/)
      3. Add a CHANGELOG.md entry
      4. Commit the template + version + locks.json + changelog together
    """
    assert fingerprints() == load_locks(), (
        "Prompt registry drifted from locks.json. A prompt template changed "
        "without a tracked version bump. See this test's docstring for the fix."
    )


def test_hash_is_sha256_of_template():
    for prompt in REGISTRY.values():
        expected = hashlib.sha256(prompt.template.encode("utf-8")).hexdigest()[:12]
        assert prompt.hash == expected


def test_prompt_ids_unique_and_consistent():
    # The dict key must equal the prompt's own id.
    for key, prompt in REGISTRY.items():
        assert key == prompt.id


def test_versions_are_positive_ints():
    for prompt in REGISTRY.values():
        assert isinstance(prompt.version, int)
        assert prompt.version >= 1


def test_write_locks_is_idempotent(tmp_path, monkeypatch):
    """write_locks() round-trips: writing then reloading yields the registry."""
    import prompts.registry as reg

    target = tmp_path / "locks.json"
    monkeypatch.setattr(reg, "LOCKS_PATH", target)
    reg.write_locks()
    assert reg.load_locks() == fingerprints()


def test_search_workflow_uses_registry_prompts():
    """The workflow must source its prompts from the registry, not new literals."""
    assert search_workflow_module.QUERY_CLASSIFIER is REGISTRY["query_classifier"]
    assert search_workflow_module.QUERY_EXPANSION is REGISTRY["query_expansion"]
    assert search_workflow_module.RESULT_SYNTHESIS is REGISTRY["result_synthesis"]


def test_annotate_span_is_safe_without_tracing():
    """annotate_span must never raise when tracing is off / no active span."""
    from prompts.registry import annotate_span

    # Should be a no-op, not an error.
    annotate_span(REGISTRY["result_synthesis"])


def test_prompt_is_frozen():
    p = Prompt(id="x", version=1, template="hello")
    with pytest.raises(Exception):
        p.version = 2  # type: ignore[misc]
