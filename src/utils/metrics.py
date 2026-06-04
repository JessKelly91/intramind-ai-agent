"""Metrics collection for observability.

This module provides persistent metrics tracking for the IntraMind agent.
Metrics are stored in a JSON file to persist across CLI invocations.
Can be upgraded to Prometheus or other observability platforms later.
"""

import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable

# Metrics file location
METRICS_FILE = Path.home() / ".intramind" / "metrics.json"

# Default metrics structure
DEFAULT_METRICS = {
    "queries_total": 0,
    "queries_simple": 0,
    "queries_complex": 0,
    "ingestions_total": 0,
    "ingestions_success": 0,
    "ingestions_failed": 0,
    "errors_total": 0,
    "total_latency_ms": 0,
    "llm_calls_router": 0,
    "llm_calls_primary": 0,
    # Step 4: Llama Guard output safety counters
    "safety_flags_total": 0,
    "safety_flags_by_category": {},
    "start_time": datetime.now().isoformat(),
}


def record_safety_flag(categories: list[str]) -> None:
    """Increment safety flag counters when Llama Guard blocks a response.

    Args:
        categories: Llama Guard category codes (e.g. ["S1", "S5"]). Pass an
            empty list if the response was flagged but no categories were
            parsed from the verdict.
    """
    metrics = _load_metrics()
    metrics["safety_flags_total"] = metrics.get("safety_flags_total", 0) + 1
    by_cat = metrics.get("safety_flags_by_category") or {}
    for cat in categories or ["UNCATEGORIZED"]:
        by_cat[cat] = int(by_cat.get(cat, 0)) + 1
    metrics["safety_flags_by_category"] = by_cat
    _save_metrics(metrics)


def _load_metrics() -> dict[str, Any]:
    """Load metrics from file."""
    try:
        if METRICS_FILE.exists():
            return json.loads(METRICS_FILE.read_text())
        return DEFAULT_METRICS.copy()
    except Exception:
        return DEFAULT_METRICS.copy()


def _save_metrics(metrics: dict[str, Any]) -> None:
    """Save metrics to file."""
    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    except Exception:
        pass  # Silently fail if we can't write metrics


# Load metrics on module import
METRICS = _load_metrics()


def track_query(func: Callable) -> Callable:
    """Decorator to track query metrics.
    
    Automatically tracks:
    - Total queries
    - Query complexity (simple vs complex)
    - Latency
    - Errors
    
    Args:
        func: The async function to track
        
    Returns:
        Wrapped function with metrics tracking
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> dict[str, Any]:
        metrics = _load_metrics()
        start = time.time()
        metrics["queries_total"] += 1

        try:
            result = await func(*args, **kwargs)

            # Track complexity if available
            if result.get("complexity") == "simple":
                metrics["queries_simple"] += 1
                metrics["llm_calls_router"] += 1  # Router always runs
                # Simple queries typically use router + 1 search call (no LLM)
            elif result.get("complexity") == "complex":
                metrics["queries_complex"] += 1
                metrics["llm_calls_router"] += 1  # Router always runs
                metrics["llm_calls_primary"] += (
                    1 + (len(result.get("expanded_queries", [])) or 1)
                )  # Query expansion + synthesis

            # Track latency
            latency_ms = (time.time() - start) * 1000
            metrics["total_latency_ms"] += latency_ms

            # Track errors
            if not result.get("success", True):
                metrics["errors_total"] += 1

            _save_metrics(metrics)
            return result

        except Exception as e:
            metrics["errors_total"] += 1
            _save_metrics(metrics)
            raise

    return wrapper


def track_ingestion(func: Callable) -> Callable:
    """Decorator to track document ingestion metrics.
    
    Tracks:
    - Total ingestions
    - Success/failure rates
    - Latency
    
    Args:
        func: The async function to track
        
    Returns:
        Wrapped function with metrics tracking
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> dict[str, Any]:
        metrics = _load_metrics()
        start = time.time()
        metrics["ingestions_total"] += 1

        try:
            result = await func(*args, **kwargs)

            # Track success/failure
            if result.get("success", False):
                metrics["ingestions_success"] += 1
            else:
                metrics["ingestions_failed"] += 1
                metrics["errors_total"] += 1

            # Track latency
            latency_ms = (time.time() - start) * 1000
            metrics["total_latency_ms"] += latency_ms

            _save_metrics(metrics)
            return result

        except Exception as e:
            metrics["ingestions_failed"] += 1
            metrics["errors_total"] += 1
            _save_metrics(metrics)
            raise

    return wrapper


def get_metrics() -> dict[str, Any]:
    """Get current metrics with computed values.
    
    Returns:
        Dictionary containing all metrics plus computed statistics
    """
    metrics = _load_metrics()
    total_queries = metrics["queries_total"]
    total_ingestions = metrics["ingestions_total"]
    total_operations = total_queries + total_ingestions
    
    # Calculate average latency
    avg_latency_ms = (
        metrics["total_latency_ms"] / total_operations if total_operations > 0 else 0
    )

    # Calculate percentages
    simple_pct = (
        (metrics["queries_simple"] / total_queries * 100) if total_queries > 0 else 0
    )
    complex_pct = (
        (metrics["queries_complex"] / total_queries * 100) if total_queries > 0 else 0
    )
    error_rate = (
        (metrics["errors_total"] / total_operations * 100)
        if total_operations > 0
        else 0
    )
    ingestion_success_rate = (
        (metrics["ingestions_success"] / total_ingestions * 100)
        if total_ingestions > 0
        else 0
    )

    # Estimate costs (approximate based on typical usage)
    # Router: Claude Haiku ~$0.00025 per call
    # Primary: Claude Haiku ~$0.00025 per call  
    # These are rough estimates for ~500 tokens per call
    router_cost = metrics["llm_calls_router"] * 0.00025
    primary_cost = metrics["llm_calls_primary"] * 0.00025
    total_cost = router_cost + primary_cost

    return {
        "queries": {
            "total": total_queries,
            "simple": metrics["queries_simple"],
            "simple_pct": simple_pct,
            "complex": metrics["queries_complex"],
            "complex_pct": complex_pct,
        },
        "ingestions": {
            "total": total_ingestions,
            "success": metrics["ingestions_success"],
            "failed": metrics["ingestions_failed"],
            "success_rate": ingestion_success_rate,
        },
        "performance": {
            "avg_latency_ms": avg_latency_ms,
            "avg_latency_s": avg_latency_ms / 1000,
            "total_operations": total_operations,
        },
        "errors": {
            "total": metrics["errors_total"],
            "rate": error_rate,
        },
        "costs": {
            "router_calls": metrics["llm_calls_router"],
            "router_cost_usd": router_cost,
            "primary_calls": metrics["llm_calls_primary"],
            "primary_cost_usd": primary_cost,
            "total_cost_usd": total_cost,
        },
        "system": {
            "start_time": metrics["start_time"],
            "uptime": _get_uptime(metrics["start_time"]),
        },
    }


def _get_uptime(start_time_iso: str) -> str:
    """Calculate uptime since metrics started.
    
    Args:
        start_time_iso: ISO format timestamp of when metrics started
    
    Returns:
        Human-readable uptime string
    """
    start = datetime.fromisoformat(start_time_iso)
    uptime_seconds = (datetime.now() - start).total_seconds()
    
    if uptime_seconds < 60:
        return f"{uptime_seconds:.1f}s"
    elif uptime_seconds < 3600:
        return f"{uptime_seconds / 60:.1f}m"
    else:
        return f"{uptime_seconds / 3600:.1f}h"


def reset_metrics() -> None:
    """Reset all metrics to initial values."""
    new_metrics = DEFAULT_METRICS.copy()
    new_metrics["start_time"] = datetime.now().isoformat()
    _save_metrics(new_metrics)

