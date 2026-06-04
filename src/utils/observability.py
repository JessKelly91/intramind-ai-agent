"""OpenTelemetry tracing for IntraMind AI Agent.

Self-hosted Phoenix integration via OpenInference. Idempotent so calling
init_tracing multiple times (e.g. once from CLI and again from agent module
import) is safe.
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

_TRACING_INITIALIZED = False
_TRACING_LOCK = Lock()


def _is_truthy(value: Optional[str]) -> bool:
    """Parse common truthy strings from environment variables."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def init_tracing(
    service_name: str = "intramind-ai-agent",
    endpoint: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Register an OTEL TracerProvider pointing at Phoenix and instrument LangChain.

    Idempotent: subsequent calls are no-ops. Safe to call from multiple entry
    points (CLI startup, agent module import, FastAPI startup event).

    Args:
        service_name: Logical service name for spans (e.g. "intramind-ai-agent").
        endpoint: Phoenix collector endpoint. Defaults to PHOENIX_ENDPOINT env
            var or http://localhost:6006.
        enabled: Force enable/disable. Defaults to ENABLE_TRACING env var.

    Returns:
        True if tracing was initialized in this call, False if it was already
        initialized or is disabled.
    """
    global _TRACING_INITIALIZED

    with _TRACING_LOCK:
        if _TRACING_INITIALIZED:
            return False

        if enabled is None:
            enabled = _is_truthy(os.environ.get("ENABLE_TRACING"))
        if not enabled:
            logger.debug("Tracing disabled (ENABLE_TRACING is not truthy)")
            return False

        endpoint = endpoint or os.environ.get(
            "PHOENIX_ENDPOINT", "http://localhost:6006"
        )

        try:
            from phoenix.otel import register
        except ImportError as exc:
            logger.warning(
                "Tracing requested but arize-phoenix-otel is not installed: %s",
                exc,
            )
            return False

        try:
            register(
                project_name=service_name,
                endpoint=f"{endpoint.rstrip('/')}/v1/traces",
                set_global_tracer_provider=True,
                auto_instrument=False,
            )
        except Exception as exc:
            logger.warning("Failed to register Phoenix tracer provider: %s", exc)
            return False

        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument()
        except ImportError as exc:
            logger.warning(
                "openinference-instrumentation-langchain not installed: %s", exc
            )
        except Exception as exc:
            logger.warning("Failed to instrument LangChain: %s", exc)

        _TRACING_INITIALIZED = True
        logger.info(
            "Tracing initialized (service=%s, endpoint=%s)", service_name, endpoint
        )
        return True


def is_tracing_initialized() -> bool:
    """Return True if init_tracing has successfully initialized tracing."""
    return _TRACING_INITIALIZED
