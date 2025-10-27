# IntraMind AI Agent - Next Steps

This guide outlines the recommended path to complete your AI agent and make it portfolio-ready.

## 📋 Table of Contents

1. [Immediate Steps](#immediate-steps-week-1)
2. [Core Enhancements](#core-enhancements-week-2)
3. [Portfolio Preparation](#portfolio-preparation-week-3)
4. [Optional Advanced Features](#optional-advanced-features)

---

## Immediate Steps (Week 1)

### 1. ✅ Get Everything Running

**Goal**: End-to-end system verification

#### a. Start All Services

```bash
# Terminal 1: Start Weaviate (if not running)
cd vector-db-service
docker-compose up -d

# Terminal 2: Start Vector Service
cd vector-db-service
python src/service/server.py

# Terminal 3: Start API Gateway
cd api-gateway
dotnet run

# Terminal 4: Start Ollama
ollama serve
```

#### b. Verify Health

```bash
cd ai-agent
python -m src.cli.main health
```

**Expected Output**:
```
✓ API Gateway is healthy
```

#### c. Test with Sample Data

Create `scripts/add_sample_data.py`:

```python
"""Add sample documents for testing."""

import asyncio
from tools.api_client import APIGatewayClient


async def main():
    async with APIGatewayClient() as client:
        # Create collection
        print("Creating collection...")
        try:
            await client.create_collection(
                name="intramind_documents",
                description="IntraMind sample documents"
            )
            print("✓ Collection created")
        except Exception as e:
            print(f"Collection may already exist: {e}")

        # Sample documents
        documents = [
            {
                "content": "Q4 2024 revenue projections show 25% year-over-year growth. "
                          "Key drivers include new product launches and market expansion in Asia.",
                "metadata": {
                    "title": "Q4 Revenue Projections",
                    "type": "financial",
                    "date": "2024-10-15",
                    "department": "Finance"
                }
            },
            {
                "content": "The new AI-powered analytics platform is scheduled for release in "
                          "December 2024. Features include real-time data visualization, "
                          "predictive modeling, and automated reporting.",
                "metadata": {
                    "title": "Product Launch Plan",
                    "type": "product",
                    "date": "2024-09-20",
                    "department": "Product"
                }
            },
            {
                "content": "Annual security audit completed successfully. All systems passed "
                          "compliance checks. Recommendations include implementing MFA for "
                          "all admin accounts and updating SSL certificates quarterly.",
                "metadata": {
                    "title": "Security Audit Report",
                    "type": "security",
                    "date": "2024-10-01",
                    "department": "IT Security"
                }
            },
            {
                "content": "Customer satisfaction scores increased to 92% in Q3, up from 87% "
                          "in Q2. Primary factors: improved response times and expanded "
                          "self-service options.",
                "metadata": {
                    "title": "Q3 Customer Satisfaction",
                    "type": "customer",
                    "date": "2024-10-10",
                    "department": "Customer Success"
                }
            },
            {
                "content": "Remote work policy update: Hybrid model will be permanent. "
                          "Employees required in office Tuesday-Thursday. Monday and Friday "
                          "flexible. Home office stipend increased to $500 annually.",
                "metadata": {
                    "title": "Remote Work Policy Update",
                    "type": "hr",
                    "date": "2024-10-05",
                    "department": "Human Resources"
                }
            },
        ]

        # Insert documents
        print(f"\nInserting {len(documents)} documents...")
        for i, doc in enumerate(documents, 1):
            try:
                result = await client.insert_document(
                    collection_name="intramind_documents",
                    content=doc["content"],
                    metadata=doc["metadata"]
                )
                print(f"  {i}. ✓ {doc['metadata']['title']} (ID: {result.id})")
            except Exception as e:
                print(f"  {i}. ✗ Failed: {e}")

        print("\n✅ Sample data loaded!")
        print("\nTry these queries:")
        print("  - 'What are our revenue projections?'")
        print("  - 'Tell me about the product launch'")
        print("  - 'What's the remote work policy?'")


if __name__ == "__main__":
    asyncio.run(main())
```

**Run it**:
```bash
python scripts/add_sample_data.py
```

#### d. Test Search Workflow

```bash
# Interactive mode
python -m src.cli.main search

# Test queries:
Search: What are our Q4 revenue projections?
Search: Compare customer satisfaction with security audit results
Search: What products are launching soon?
```

**Success Criteria**:
- ✅ All services start without errors
- ✅ Sample documents inserted successfully
- ✅ Search returns relevant results
- ✅ Simple queries work (1-2 seconds)
- ✅ Complex queries expand to multiple searches

---

## Core Enhancements (Week 2)

### 2. 🔨 Add Document Ingestion Workflow

**Goal**: Automate document processing and storage

Create `src/workflows/ingestion_workflow.py`:

```python
"""Document Ingestion Workflow using LangGraph."""

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from models.state import IngestionWorkflowState
from tools import get_api_client

logger = logging.getLogger(__name__)


async def validate_document(state: IngestionWorkflowState) -> IngestionWorkflowState:
    """Validate document before processing."""
    logger.info("Node: validate_document")

    # Check file exists, size limits, etc.
    # For now, simple validation
    if not state["file_path"]:
        return {**state, "error": "No file path provided", "next_step": "handle_error"}

    return {**state, "current_step": "validate_document", "next_step": "extract_content"}


async def extract_content(state: IngestionWorkflowState) -> IngestionWorkflowState:
    """Extract text content from document."""
    logger.info("Node: extract_content")

    # TODO: Add actual file reading logic
    # For now, assume content is already in state
    extracted = state.get("extracted_content", "")

    if not extracted:
        return {**state, "error": "No content to extract", "next_step": "handle_error"}

    return {
        **state,
        "current_step": "extract_content",
        "extracted_content": extracted,
        "next_step": "chunk_content",
    }


async def chunk_content(state: IngestionWorkflowState) -> IngestionWorkflowState:
    """Chunk content for optimal retrieval."""
    logger.info("Node: chunk_content")

    content = state["extracted_content"]
    chunk_size = state.get("chunk_size", 1000)
    chunk_overlap = state.get("chunk_overlap", 200)

    # Simple chunking by character count
    chunks = []
    for i in range(0, len(content), chunk_size - chunk_overlap):
        chunk_text = content[i : i + chunk_size]
        chunks.append({"content": chunk_text, "index": len(chunks)})

    logger.info(f"Created {len(chunks)} chunks")

    return {
        **state,
        "current_step": "chunk_content",
        "chunks": chunks,
        "next_step": "store_chunks",
    }


async def store_chunks(state: IngestionWorkflowState) -> IngestionWorkflowState:
    """Store chunks in vector database."""
    logger.info("Node: store_chunks")

    chunks = state["chunks"]
    collection = state["collection_name"]
    client = get_api_client()

    inserted_ids = []

    try:
        # Batch insert
        documents = [
            {
                "content": chunk["content"],
                "metadata": {
                    **state.get("document_metadata", {}),
                    "chunk_index": chunk["index"],
                    "total_chunks": len(chunks),
                },
            }
            for chunk in chunks
        ]

        result = await client.insert_documents_batch(
            collection_name=collection, documents=documents
        )

        inserted_ids = result.get("ids", [])
        logger.info(f"Stored {len(inserted_ids)} chunks")

        return {
            **state,
            "current_step": "store_chunks",
            "inserted_ids": inserted_ids,
            "workflow_complete": True,
        }

    except Exception as e:
        logger.error(f"Storage failed: {e}")
        return {**state, "error": str(e), "next_step": "handle_error"}


def create_ingestion_workflow() -> StateGraph:
    """Create document ingestion workflow."""
    workflow = StateGraph(IngestionWorkflowState)

    # Add nodes
    workflow.add_node("validate_document", validate_document)
    workflow.add_node("extract_content", extract_content)
    workflow.add_node("chunk_content", chunk_content)
    workflow.add_node("store_chunks", store_chunks)

    # Set entry
    workflow.set_entry_point("validate_document")

    # Add edges
    workflow.add_edge("validate_document", "extract_content")
    workflow.add_edge("extract_content", "chunk_content")
    workflow.add_edge("chunk_content", "store_chunks")
    workflow.add_edge("store_chunks", END)

    return workflow.compile()


ingestion_workflow = create_ingestion_workflow()
```

**Test it**:
```python
# Add to agent/main.py
async def ingest_document(
    self, content: str, collection_name: str = "intramind_documents"
) -> dict:
    """Ingest a document into the system."""
    # Implementation
```

### 3. 🎨 Create Architecture Diagrams

**Goal**: Visual documentation for your portfolio

Create `docs/ARCHITECTURE.md`:

```markdown
# IntraMind AI Agent Architecture

## System Overview

[Include Mermaid diagrams or ASCII art of your architecture]

## LangGraph State Machine

[Visual representation of your workflows]

## Component Interactions

[Sequence diagrams showing request flow]
```

**Tools to use**:
- [Mermaid](https://mermaid.js.org/) - Markdown-native diagrams
- [Excalidraw](https://excalidraw.com/) - Hand-drawn style
- [Draw.io](https://draw.io/) - Professional diagrams
- LangGraph's built-in `.get_graph().draw_mermaid()` method

**Export your workflow**:
```python
from workflows.search_workflow import search_workflow

# Get Mermaid diagram
print(search_workflow.get_graph().draw_mermaid())
```

### 4. 🧪 Add Integration Tests

Create `tests/test_search_workflow.py`:

```python
"""Integration tests for search workflow."""

import pytest
from unittest.mock import AsyncMock, patch

from agent import IntraMindAgent


@pytest.mark.asyncio
async def test_simple_query_classification():
    """Test that simple queries are classified correctly."""
    agent = IntraMindAgent()

    with patch("workflows.search_workflow.get_api_client") as mock_client:
        mock_client.return_value.search = AsyncMock(
            return_value={"results": [], "query": "test", "total_results": 0}
        )

        result = await agent.search("What is the weather?")

        assert result["complexity"] == "simple"


@pytest.mark.asyncio
async def test_complex_query_expansion():
    """Test that complex queries are expanded."""
    agent = IntraMindAgent()

    result = await agent.search(
        "Compare Q3 and Q4 revenue and explain the differences"
    )

    assert result["complexity"] == "complex"
    assert result.get("expanded_queries") is not None
    assert len(result["expanded_queries"]) >= 2
```

Run tests:
```bash
pytest tests/ -v
```

---

## Portfolio Preparation (Week 3)

### 5. 📹 Create Demo Materials

#### a. Record Demo Video (3-5 minutes)

**Script**:
1. **Introduction** (30s)
   - "IntraMind AI Agent - intelligent document search"
   - Show architecture diagram

2. **Code Walkthrough** (90s)
   - Show LangGraph state machine code
   - Highlight key design decisions
   - Explain hybrid LLM strategy

3. **Live Demo** (90s)
   - Start CLI
   - Run simple query → show it routes to simple_search
   - Run complex query → show multi-query expansion
   - Show streaming results

4. **Technical Highlights** (60s)
   - Cost analysis (~$1/month)
   - State machine benefits (testable, debuggable)
   - Production-ready patterns

**Tools**:
- OBS Studio (free screen recording)
- Loom (easy sharing)
- Add to YouTube/portfolio site

#### b. Create Portfolio Writeup

Create `docs/PORTFOLIO_WRITEUP.md`:

```markdown
# IntraMind AI Agent - Technical Deep Dive

## Problem Statement
Enterprise document search is challenging because...

## Solution Architecture
I chose LangGraph because...

## Technical Highlights
1. State machine workflow design
2. Hybrid LLM strategy for cost optimization
3. Production patterns: error handling, observability, testing

## Key Learnings
- LangGraph provides better control than pure LLM agents
- Hybrid approach reduces costs by 80%
- State machines make AI workflows testable

## Metrics
- Cost: $0.001 per query
- Latency: 2-4s (simple), 5-8s (complex)
- Accuracy: [Add your results]

## Code Samples
[Include key snippets with explanations]
```

#### c. Update Main Portfolio

**GitHub README Badge**:
```markdown
[![IntraMind](https://img.shields.io/badge/IntraMind-AI%20Agent-blue)](https://github.com/JessKelly91/intramind-ai-agent)
```

**Portfolio Site Section**:
```
## IntraMind AI Agent

AI-powered document search using LangGraph state machines

**Tech Stack**: Python, LangGraph, LangChain, Claude Haiku, Ollama
**Highlights**:
- Production-ready state machine architecture
- 80% cost reduction with hybrid LLM approach
- Comprehensive testing and observability

[Live Demo] [Code] [Architecture Docs]
```

### 6. 📊 Add Observability

#### a. Implement Metrics

Create `src/utils/metrics.py`:

```python
"""Metrics collection for observability."""

import time
from functools import wraps
from typing import Callable

# Simple in-memory metrics (upgrade to Prometheus later)
METRICS = {
    "queries_total": 0,
    "queries_simple": 0,
    "queries_complex": 0,
    "avg_latency_ms": 0,
    "errors_total": 0,
}


def track_query(func: Callable) -> Callable:
    """Decorator to track query metrics."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        METRICS["queries_total"] += 1

        try:
            result = await func(*args, **kwargs)

            # Track complexity
            if result.get("complexity") == "simple":
                METRICS["queries_simple"] += 1
            elif result.get("complexity") == "complex":
                METRICS["queries_complex"] += 1

            # Track latency
            latency = (time.time() - start) * 1000
            METRICS["avg_latency_ms"] = (
                METRICS["avg_latency_ms"] * 0.9 + latency * 0.1
            )

            return result

        except Exception as e:
            METRICS["errors_total"] += 1
            raise

    return wrapper
```

#### b. Add Metrics Endpoint to CLI

```python
@cli.command()
def metrics():
    """Display agent metrics."""
    from utils.metrics import METRICS

    table = Table(title="Agent Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in METRICS.items():
        table.add_row(key, str(value))

    console.print(table)
```

### 7. 📝 Finalize Documentation

**Complete these files**:
- [x] README.md
- [x] QUICKSTART.md
- [ ] docs/ARCHITECTURE.md (diagrams)
- [ ] docs/PORTFOLIO_WRITEUP.md (technical narrative)
- [ ] CHANGELOG.md (version history)
- [ ] CONTRIBUTING.md (if accepting contributions)

**Add code examples to README**:
- Streaming search
- Custom workflows
- Error handling
- Testing patterns

---

## Optional Advanced Features

### 8. 🖼️ Add Multimodal Processing

**Goal**: Process PDFs, images, presentations

Create `src/workflows/multimodal_workflow.py`:

```python
"""Multimodal document processing workflow."""

# Nodes:
# - detect_file_type
# - process_pdf
# - process_image (OCR)
# - process_pptx
# - extract_metadata
# - store_document
```

**Libraries to integrate**:
- PyPDF2 / pdfplumber (PDF text extraction)
- pytesseract (OCR for images)
- python-pptx (PowerPoint processing)
- Pillow (image handling)

### 9. 🧠 Implement Agentic RAG

**Goal**: Self-correcting retrieval with query refinement

**Features**:
- Query understanding and expansion
- Result evaluation
- Automatic re-querying if results are poor
- Source citation verification

### 10. 🌐 Add Web UI

**Goal**: Beautiful frontend for demos

**Option A: Streamlit** (Fastest)
```python
# app.py
import streamlit as st
from agent import IntraMindAgent

st.title("🧠 IntraMind AI Agent")

query = st.text_input("Search Query")
if st.button("Search"):
    agent = IntraMindAgent()
    result = agent.search(query)
    st.write(result["response"])
```

**Option B: FastAPI + React** (More impressive)
- FastAPI backend with SSE streaming
- React frontend with real-time updates
- Deploy to Vercel/Netlify

---

## Success Checklist

### Minimum Viable Portfolio Piece ✅
- [ ] All services running end-to-end
- [ ] Sample data loaded and searchable
- [ ] Search workflow handles simple and complex queries
- [ ] CLI works in interactive mode
- [ ] README with architecture diagram
- [ ] Code pushed to GitHub

### Portfolio-Ready ⭐
- [ ] Integration tests passing
- [ ] Demo video recorded
- [ ] Portfolio writeup completed
- [ ] Architecture diagrams created
- [ ] Metrics/observability implemented
- [ ] At least one additional workflow (ingestion or multimodal)

### Production-Grade 🚀
- [ ] Comprehensive test coverage (>80%)
- [ ] Error handling and recovery tested
- [ ] Performance benchmarks documented
- [ ] Deployment guide (Docker Compose)
- [ ] CI/CD pipeline setup
- [ ] Monitoring and alerting

---

## Timeline Estimate

| Phase | Time | Focus |
|-------|------|-------|
| Week 1 | 5-10 hrs | Get running, test workflows, sample data |
| Week 2 | 10-15 hrs | Add workflow, tests, architecture docs |
| Week 3 | 5-10 hrs | Demo materials, portfolio prep, polish |
| **Total** | **20-35 hrs** | **Portfolio-ready AI agent** |

---

## Questions to Consider

Before moving forward:

1. **LLM Strategy**: Stick with Ollama (free) or use Claude Haiku (better quality)?
2. **Additional Workflow**: Ingestion, multimodal, or multi-agent?
3. **Portfolio Focus**: Technical depth or breadth of features?
4. **Demo Format**: Video walkthrough, live demo, or both?
5. **Deployment**: Local demo only or deploy somewhere (Railway, Render)?

---

## Resources for Next Steps

- **LangGraph Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Multimodal RAG**: https://python.langchain.com/docs/use_cases/question_answering/
- **Testing Strategies**: https://python.langchain.com/docs/guides/testing
- **Deployment**: https://python.langchain.com/docs/guides/deployment

---

**You're ready to build! Start with Week 1 and iterate from there. 🚀**

Need help with any specific step? Let me know!
