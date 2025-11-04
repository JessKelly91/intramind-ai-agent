# Integration Tests Guide

This document explains how to run the end-to-end integration tests for IntraMind AI Agent.

## Overview

Integration tests verify the complete system workflow by testing with **real services running**:
- ✅ Weaviate Vector Database
- ✅ Vector Database Service (gRPC)
- ✅ API Gateway (.NET)
- ✅ Ollama (local LLM)

Unlike unit tests (which use mocks), integration tests make real API calls and verify the entire stack works together.

## Test Files

### `test_e2e_ingestion_search.py` ⭐ (New!)
**End-to-end tests for the complete workflow: ingest → search → verify**

Tests included:
1. `test_ingest_then_simple_search` - Core E2E test
2. `test_ingest_then_complex_search` - Complex query with expansion
3. `test_ingest_multiple_docs_then_search` - Multi-document scenarios
4. `test_ingest_then_search_with_min_score` - Score filtering
5. `test_search_empty_collection_no_crash` - Edge case handling
6. `test_ingest_markdown_then_search` - Markdown support
7. `test_ingest_large_document` - Large document handling

### `test_min_score_filtering.py`
**Tests for similarity score filtering feature**

Tests included:
1. `test_min_score_filtering` - Score threshold verification
2. `test_min_score_returns_valid_scores` - Score validity
3. `test_high_min_score_filters_aggressively` - High threshold behavior

---

## Prerequisites

### 1. Start All Services

Before running integration tests, ensure all services are running:

```powershell
# Terminal 1: Start Weaviate
cd vector-db-service
docker-compose up -d

# Terminal 2: Start Vector Service
cd vector-db-service
.\venv\Scripts\Activate.ps1
python -m src.service.server

# Terminal 3: Start API Gateway
cd api-gateway
dotnet run --project src/IntraMind.ApiGateway

# Terminal 4: Verify Ollama is running
# Ollama should already be running on port 11434
```

### 2. Verify Health

```powershell
cd ai-agent
.\.venv\Scripts\Activate.ps1
python -m src.cli.main health --url http://127.0.0.1:64536
```

You should see: `API Gateway is healthy`

---

## Running Integration Tests

### Run All Integration Tests

```powershell
cd ai-agent
.\.venv\Scripts\Activate.ps1

# Run all integration tests
pytest -m integration -v

# With detailed output
pytest -m integration -v -s
```

### Run Specific Integration Test File

```powershell
# Run only E2E ingestion+search tests
pytest tests/test_e2e_ingestion_search.py -v -m integration

# Run only min_score tests
pytest tests/test_min_score_filtering.py -v -m integration
```

### Run Specific Test

```powershell
# Run a single test
pytest tests/test_e2e_ingestion_search.py::test_ingest_then_simple_search -v -m integration
```

---

## Running Unit Tests Only

To run unit tests (which use mocks and don't require services):

```powershell
# Run all unit tests (excludes integration tests)
pytest -m "not integration" -v

# Or just run pytest without markers
pytest tests/ -v --ignore=tests/test_min_score_filtering.py --ignore=tests/test_e2e_ingestion_search.py
```

---

## Running All Tests (Unit + Integration)

```powershell
# Run everything
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

## Test Organization

```
tests/
├── Unit Tests (use mocks, fast, no services needed):
│   ├── test_search_workflow.py         (18 tests)
│   ├── test_agent.py                   (13 tests)
│   ├── test_agent_tools.py             (18 tests)
│   ├── test_api_client.py              (3 tests)
│   └── test_ingestion_workflow.py      (32 tests)
│
└── Integration Tests (require services, slower):
    ├── test_e2e_ingestion_search.py    (8 tests) ⭐ NEW
    └── test_min_score_filtering.py     (3 tests)
```

**Total: 95 tests (87 unit + 8 integration)**

---

## Expected Output

### Successful Integration Test Run

```
tests/test_e2e_ingestion_search.py::test_ingest_then_simple_search PASSED
✓ Ingested document: 4 chunks stored
✓ Search found 3 relevant results
✓ Response: According to the document, Q4 2024 revenue was...

tests/test_e2e_ingestion_search.py::test_ingest_then_complex_search PASSED
✓ Ingested 5 chunks
✓ Classified as: complex
✓ Expanded into 3 queries
✓ Found 4 results

... (more tests)

========================= 8 passed in 45.23s ==========================
```

---

## Troubleshooting

### "Connection refused" errors
**Problem**: Services are not running
**Solution**: Start all services (see Prerequisites above)

### "Collection not found" errors
**Problem**: Collection doesn't exist in Weaviate
**Solution**: Tests create unique collections automatically, but if persisting, restart Weaviate:
```powershell
cd vector-db-service
docker-compose down
docker-compose up -d
```

### Tests are very slow
**Expected**: Integration tests are slower than unit tests (30-60 seconds total)
**Why**: Real API calls, LLM inference, vector search operations

### LLM timeouts
**Problem**: Ollama is slow or not responding
**Solution**: 
1. Verify Ollama is running: `ollama list`
2. Use a smaller model: `ollama pull llama3.2:3b`
3. Increase timeout in API client if needed

---

## Test Database Isolation

Integration tests use **unique test collection names** (e.g., `test_e2e_collection_a3b9f2d1`) to avoid conflicts:
- Each test run creates its own collection
- No interference between tests
- No pollution of production data

### Cleaning Up Test Collections

If you want to clean up test collections after runs:

```python
# In Python/IPython:
from tools.api_client import APIGatewayClient
import asyncio

async def cleanup():
    async with APIGatewayClient() as client:
        collections = await client.list_collections()
        for col in collections:
            if col["name"].startswith("test_e2e_collection_"):
                await client.delete_collection(col["name"])
                print(f"Deleted: {col['name']}")

asyncio.run(cleanup())
```

---

## CI/CD Integration

For GitHub Actions or other CI pipelines:

```yaml
# .github/workflows/test.yml
- name: Run Unit Tests
  run: pytest -m "not integration" -v --cov=src

- name: Start Services
  run: docker-compose up -d

- name: Run Integration Tests
  run: pytest -m integration -v
```

---

## Next Steps

After running these tests successfully:
1. ✅ Verify all 8 integration tests pass
2. ✅ Update NEXT_STEPS.md to mark integration tests complete
3. 🎯 Move on to Architecture Diagrams (Week 2 final task)
4. 🎯 Begin Week 3: Portfolio Preparation

---

## Questions?

If integration tests fail:
1. Check that all services are healthy
2. Review service logs for errors
3. Ensure Ollama has the llama3.2:3b model
4. Verify network connectivity (localhost, correct ports)
5. Try running tests individually to isolate issues

