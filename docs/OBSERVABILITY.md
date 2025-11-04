# Observability & Metrics

This document describes the observability and metrics system implemented in the IntraMind AI Agent.

## Overview

The metrics system provides real-time tracking of agent performance, costs, and usage patterns. It uses in-memory storage for simplicity and can be upgraded to Prometheus or other observability platforms later.

## Architecture

### Components

1. **`src/utils/metrics.py`** - Core metrics collection module
   - In-memory metrics storage
   - Decorator-based tracking (`@track_query`, `@track_ingestion`)
   - Computed statistics and cost estimation
   
2. **`src/agent/main.py`** - Integration points
   - `@track_query` on `search()` method
   - `@track_ingestion` on `ingest_document()` method
   
3. **`src/cli/main.py`** - CLI interface
   - `intramind metrics` command to display statistics
   - `intramind metrics --reset` to reset counters

## Metrics Tracked

### Query Metrics
- **Total Queries**: Total number of search operations
- **Simple Queries**: Queries handled by direct search (no query expansion)
- **Complex Queries**: Queries requiring multi-query expansion and synthesis

### Ingestion Metrics
- **Total Ingestions**: Total document ingestion attempts
- **Successful Ingestions**: Successfully processed documents
- **Failed Ingestions**: Failed document processing

### Performance Metrics
- **Average Latency**: Mean response time across all operations
- **Total Operations**: Combined queries + ingestions
- **Error Rate**: Percentage of operations that failed

### Cost Metrics (Estimated)
- **Router LLM Calls**: Number of calls to query classification LLM
- **Primary LLM Calls**: Number of calls to main LLM (expansion + synthesis)
- **Estimated Costs**: Approximate costs based on Claude Haiku pricing (~$0.00025/call)

### System Metrics
- **Uptime**: Time since metrics started tracking
- **Start Time**: When metrics collection began

## Usage

### View Current Metrics

```bash
# Display all metrics
intramind metrics

# Example output:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Query Metrics              ┃                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Total Queries              │ 42                 │
│ Simple Queries             │ 28 (66.7%)         │
│ Complex Queries            │ 14 (33.3%)         │
└────────────────────────────┴────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Performance Metrics        ┃                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Average Latency            │ 3.24s              │
│ Total Operations           │ 42                 │
│ Error Count                │ 2 (4.8%)           │
└────────────────────────────┴────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Cost Metrics (Estimated)   ┃                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ Router LLM Calls           │ 42 calls           │
│ Router Cost                │ $0.0105            │
│ Primary LLM Calls          │ 28 calls           │
│ Primary Cost               │ $0.0070            │
│ Total Estimated Cost       │ $0.0175            │
└────────────────────────────┴────────────────────┘
```

### Reset Metrics

```bash
# Reset all metrics counters to zero
intramind metrics --reset
```

### Programmatic Access

You can also access metrics programmatically in Python:

```python
from utils.metrics import get_metrics

# Get computed metrics
metrics_data = get_metrics()

# Access specific metrics
total_queries = metrics_data["queries"]["total"]
avg_latency = metrics_data["performance"]["avg_latency_s"]
total_cost = metrics_data["costs"]["total_cost_usd"]

print(f"Processed {total_queries} queries with {avg_latency:.2f}s average latency")
print(f"Estimated cost: ${total_cost:.4f}")
```

## Implementation Details

### Automatic Tracking

Metrics are automatically tracked when using the `IntraMindAgent` class:

```python
from agent import IntraMindAgent

agent = IntraMindAgent()

# This search is automatically tracked
result = await agent.search("What is machine learning?")
# Metrics updated:
# - queries_total += 1
# - queries_simple += 1 (or queries_complex)
# - llm_calls_router += 1
# - total_latency_ms += <execution_time>
```

### Custom Tracking

You can use the decorators on your own functions:

```python
from utils.metrics import track_query

@track_query
async def my_custom_search(query: str) -> dict:
    # Your search logic
    return {
        "success": True,
        "complexity": "simple",  # or "complex"
        "response": "...",
    }
```

### Cost Estimation

Cost estimates are based on:
- **Router LLM**: Claude Haiku at ~$0.00025 per call (500 tokens avg)
- **Primary LLM**: Claude Haiku at ~$0.00025 per call (500 tokens avg)

These are approximate values. Actual costs depend on:
- Prompt length
- Response length
- Model pricing at time of use

Update the cost estimates in `metrics.py` if needed:

```python
# In src/utils/metrics.py, function get_metrics()
router_cost = METRICS["llm_calls_router"] * 0.00025  # Adjust this value
primary_cost = METRICS["llm_calls_primary"] * 0.00025  # Adjust this value
```

## Future Enhancements

### Prometheus Integration

For production deployments, consider upgrading to Prometheus:

```python
# Example Prometheus integration
from prometheus_client import Counter, Histogram, start_http_server

queries_total = Counter('intramind_queries_total', 'Total queries')
query_latency = Histogram('intramind_query_latency_seconds', 'Query latency')

@track_query
async def search(query: str):
    queries_total.inc()
    with query_latency.time():
        # Search logic
        pass
```

### Persistent Storage

To persist metrics across restarts:

```python
import json
from pathlib import Path

def save_metrics():
    """Save metrics to disk."""
    Path("metrics.json").write_text(json.dumps(METRICS))

def load_metrics():
    """Load metrics from disk."""
    if Path("metrics.json").exists():
        METRICS.update(json.loads(Path("metrics.json").read_text()))
```

### Advanced Analytics

Add more detailed tracking:
- Query patterns and common terms
- Most expensive queries
- Peak usage times
- User-specific metrics
- Result quality scores

## Troubleshooting

### Metrics Not Updating

If metrics aren't updating, verify:

1. **Decorators are applied**: Check `@track_query` and `@track_ingestion` decorators
2. **Using correct methods**: Use `IntraMindAgent.search()` not direct workflow calls
3. **Async execution**: Ensure you're using `await` or `asyncio.run()`

### Inaccurate Cost Estimates

Cost estimates are approximate. To improve accuracy:

1. **Log actual LLM calls**: Add instrumentation to LLM providers
2. **Track token usage**: Count prompt and completion tokens
3. **Update pricing**: Adjust cost per call based on current pricing

### Memory Usage

Metrics are stored in-memory and reset on restart. For long-running processes:

1. **Monitor memory**: Large operation counts won't significantly impact memory
2. **Periodic resets**: Call `reset_metrics()` periodically if needed
3. **Export metrics**: Save to disk or push to external systems

## Examples

### Portfolio Demo

Use metrics to demonstrate system efficiency:

```bash
# Run some searches
intramind search -q "machine learning basics"
intramind search -q "compare supervised vs unsupervised learning"
intramind search -q "neural networks"

# Show metrics for demo
intramind metrics

# Highlight:
# - Low latency (2-4s simple, 5-8s complex)
# - Low cost (~$0.001 per query)
# - High success rate (>95%)
```

### Cost Analysis

Track costs during development:

```python
# At start of session
from utils.metrics import reset_metrics
reset_metrics()

# Run your tests...
# ... many operations ...

# At end of session
from utils.metrics import get_metrics
metrics = get_metrics()

print(f"Session cost: ${metrics['costs']['total_cost_usd']:.4f}")
print(f"Cost per query: ${metrics['costs']['total_cost_usd'] / metrics['queries']['total']:.6f}")
```

## Related Documentation

- [Architecture Overview](../ARCHITECTURE.md)
- [CLI Usage Guide](../QUICKSTART.md)
- [API Reference](../README.md)

