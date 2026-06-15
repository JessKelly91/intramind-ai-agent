"""Runtime prompt client with TTL caching and baked-in fallback."""

from __future__ import annotations

import logging
import time
from dataclasses import replace

import httpx

from config import settings
from prompts.registry import REGISTRY, Prompt

logger = logging.getLogger(__name__)

_CacheKey = tuple[str, str]
_cache: dict[_CacheKey, tuple[float, Prompt]] = {}


def _fallback(prompt_id: str, label: str) -> Prompt:
    prompt = REGISTRY[prompt_id]
    return replace(prompt, label=label, source="fallback")


def clear_prompt_cache() -> None:
    """Clear the in-memory cache; used by tests and long-running reload hooks."""

    _cache.clear()


def get_prompt(prompt_id: str, label: str | None = None) -> Prompt:
    """Return a prompt from the registry service or the baked-in fallback.

    Any registry failure, timeout, non-200 response, or missing configuration
    falls back to ``prompts.registry.REGISTRY`` so the agent request path keeps
    today's behavior when the service is absent.
    """

    active_label = label or settings.prompt_registry_label
    registry_url = (settings.prompt_registry_url or "").rstrip("/")
    if not registry_url:
        return _fallback(prompt_id, active_label)

    cache_key = (prompt_id, active_label)
    now = time.monotonic()
    cached = _cache.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    try:
        response = httpx.get(
            f"{registry_url}/api/v1/prompts/{prompt_id}",
            params={"label": active_label},
            headers={"X-API-Key": settings.prompt_registry_api_key or ""},
            timeout=2.0,
        )
        response.raise_for_status()
        body = response.json()
        prompt = Prompt(
            id=body["id"],
            version=int(body["version"]),
            template=body["template"],
            label=body.get("label", active_label),
            source="registry",
        )
        ttl = max(0, settings.prompt_registry_cache_ttl)
        if ttl:
            _cache[cache_key] = (now + ttl, prompt)
        return prompt
    except Exception as exc:  # noqa: BLE001 - fallback must catch all registry errors
        logger.warning(
            "Prompt registry lookup failed for %s@%s; using baked-in fallback: %s",
            prompt_id,
            active_label,
            exc,
        )
        return _fallback(prompt_id, active_label)
