# IntraMind AI Agent

> AI-powered intelligent document search agent using LangGraph state machine architecture

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://langchain-ai.github.io/langgraph/)
[![Coverage](https://img.shields.io/badge/Coverage-67%25-brightgreen)](htmlcov/index.html)
[![Tests](https://img.shields.io/badge/Tests-112%20passing-success)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

The IntraMind AI Agent is an intelligent document search system built with **LangGraph**, a state machine framework for building sophisticated AI workflows. It provides semantic search capabilities with automatic query classification, multi-query expansion, and intelligent result synthesis.

### Key Features

- **🤖 LangGraph State Machine**: Predictable, debuggable AI workflows
- **🔍 Smart Query Routing**: Automatic classification of simple vs complex queries
- **💬 Conversation Memory**: Context-aware interactions with smart cost optimization
- **📊 Multi-Query Expansion**: Complex queries are expanded into multiple searches
- **✨ Result Synthesis**: AI-powered aggregation and summarization of search results
- **💰 Cost-Effective Hybrid LLM**: Local models for routing, API models for synthesis
- **🎨 Beautiful CLI**: Rich terminal UI with streaming support
- **🔌 API Integration**: Seamless connection to IntraMind API Gateway

## 🏗️ Architecture

### LangGraph State Machine

The agent uses a state machine approach for reliable, observable AI workflows:

```
                    ┌──────────────────┐
                    │  User Query      │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Classify Query   │ ◄─── Router LLM (Ollama)
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
           ┌────────────────┐  ┌──────────────────┐
           │ Simple Search  │  │  Complex Search  │
           └────────┬───────┘  └────────┬─────────┘
                    │                   │
                    │    ┌──────────────┘
                    │    │  (Multi-Query + Dedup)
                    │    │
                    ▼    ▼
           ┌──────────────────────┐
           │  Synthesize Results  │ ◄─── Primary LLM (Claude/GPT)
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │   Response + Cites   │
           └──────────────────────┘
```

### Hybrid LLM Strategy

**Cost-Effective Approach:**
- **Router LLM** (Ollama - Free): Query classification, expansion
- **Primary LLM** (Claude Haiku / GPT-3.5): Response synthesis
- **Result**: ~80% of operations run locally, only synthesis uses API calls

## 📁 Project Structure

```
intramind-ai-agent/
├── src/
│   ├── agent/              # Main agent interface
│   │   └── main.py         # IntraMindAgent class
│   ├── workflows/          # LangGraph workflows
│   │   └── search_workflow.py  # Document search state machine
│   ├── models/             # Data models
│   │   ├── state.py        # LangGraph state definitions
│   │   └── api.py          # API request/response models
│   ├── tools/              # Agent tools
│   │   ├── api_client.py   # API Gateway client
│   │   └── agent_tools.py  # LangChain tools
│   ├── cli/                # CLI interface
│   │   └── main.py         # Click + Rich CLI
│   ├── utils/              # Utilities
│   │   ├── llm.py          # LLM initialization
│   │   └── logging.py      # Structured logging
│   └── config.py           # Configuration management
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── pyproject.toml          # Project metadata
└── .env.example            # Environment template
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM)
- IntraMind API Gateway running
- Anthropic or OpenAI API key (optional, for better synthesis)

### Installation

1. **Clone the repository** (if not already in the IntraMind monorepo):
```bash
cd ai-agent
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt

# One-time spaCy model download (required for Presidio PII redaction - Step 3 of Free RAI Stack)
python -m spacy download en_core_web_lg

# One-time Ollama model pulls (Free RAI Stack: Step 2 judge + Step 4 safety classifier)
ollama pull llama3.1:8b      # Ragas judge
ollama pull llama-guard3     # Output safety classifier
```

3. **Set up environment**:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
# API Gateway
API_GATEWAY_URL=http://localhost:5000

# Primary LLM (for synthesis)
PRIMARY_LLM_PROVIDER=anthropic  # or openai
ANTHROPIC_API_KEY=your-key-here

# Router LLM (for classification - free!)
ROUTER_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

4. **Install Ollama model** (for local routing):
```bash
ollama pull llama3.2:3b
```

### Usage

#### CLI - Interactive Search

```bash
python -m src.cli.main search
```

This launches an interactive search session:
```
╔═══════════════════════════════════════════╗
║                                           ║
║          🧠 IntraMind AI Agent            ║
║   Intelligent Document Search Platform    ║
║                                           ║
╚═══════════════════════════════════════════╝

Interactive Search Mode
Type your query (or 'exit' to quit)

Search: What are the Q4 revenue projections?
```

#### CLI - Single Query

```bash
python -m src.cli.main search --query "What are the Q4 revenue projections?" --limit 10
```

#### Check System Health

```bash
python -m src.cli.main health
```

#### View Configuration

```bash
python -m src.cli.main info
```

### Programmatic Usage

```python
import asyncio
from agent import IntraMindAgent

async def main():
    agent = IntraMindAgent()

    # Perform search
    result = await agent.search(
        query="What are the Q4 revenue projections?",
        collection_name="intramind_documents",
        num_results=10
    )

    if result["success"]:
        print(f"Response: {result['response']}")
        print(f"Citations: {result['citations']}")
        print(f"Query Complexity: {result['complexity']}")

asyncio.run(main())
```

### Streaming Results

```python
async def stream_example():
    agent = IntraMindAgent()

    async for update in agent.stream_search("Q4 revenue projections"):
        print(f"Step: {update['step']}")
        if update['response']:
            print(f"Response: {update['response']}")

asyncio.run(stream_example())
```

### Advanced Streaming Example

```python
import asyncio
from agent import IntraMindAgent
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

async def streaming_search_with_ui():
    """Stream search with live UI updates."""
    agent = IntraMindAgent()
    console = Console()
    
    with Live(console=console, refresh_per_second=4) as live:
        current_step = "Initializing..."
        results_count = 0
        
        async for update in agent.stream_search("What are our Q4 projections?"):
            current_step = update.get('step', 'Processing')
            results_count = update.get('results_count', 0)
            
            # Update live display
            live.update(
                Panel(
                    f"[bold blue]Step:[/bold blue] {current_step}\n"
                    f"[bold green]Results Found:[/bold green] {results_count}",
                    title="🧠 IntraMind AI Agent",
                    border_style="cyan"
                )
            )
            
            # Process final response
            if update.get('complete') and update.get('response'):
                console.print("\n[bold green]Final Response:[/bold green]")
                console.print(update['response'])

asyncio.run(streaming_search_with_ui())
```

## 🔧 Configuration

### LLM Providers

#### Anthropic (Recommended for Quality)
```env
PRIMARY_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

#### OpenAI
```env
PRIMARY_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
```

#### Ollama (Free, Local)
```env
PRIMARY_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:8b
```

### Agent Settings

```env
AGENT_MAX_ITERATIONS=10      # Max workflow iterations
AGENT_VERBOSE=true           # Enable verbose logging
ENABLE_STREAMING=true        # Stream results
DEFAULT_COLLECTION=intramind_documents
SEARCH_LIMIT=10              # Default result limit
```

### Conversation Memory Settings

```env
# Enable context-aware conversations (recommended)
ENABLE_CONVERSATION_MEMORY=true
MAX_CONVERSATION_HISTORY=5          # Max turns in context (cost optimization)
SMART_CONTEXT_SELECTION=true        # Only use history for complex queries (~20% cost increase)
CHECKPOINT_STORAGE_PATH=./data/checkpoints.db
```

**Learn more**: See [CONVERSATION_MEMORY.md](docs/CONVERSATION_MEMORY.md) for detailed usage and best practices.

## 🛡️ Error Handling

### Graceful Error Recovery

The agent includes comprehensive error handling at every workflow step:

```python
import asyncio
from agent import IntraMindAgent

async def search_with_error_handling():
    """Demonstrate error handling patterns."""
    agent = IntraMindAgent()
    
    try:
        result = await agent.search(
            query="What are our revenue projections?",
            collection_name="intramind_documents",
            num_results=10,
            min_score=0.7
        )
        
        # Check for success
        if result["success"]:
            print(f"✓ Response: {result['response']}")
            print(f"✓ Found {len(result.get('citations', []))} sources")
        else:
            # Workflow completed but with errors
            print(f"✗ Search failed: {result.get('error', 'Unknown error')}")
            
    except ConnectionError as e:
        print(f"✗ API Gateway connection failed: {e}")
        print("  → Check if API Gateway is running")
        
    except TimeoutError as e:
        print(f"✗ Request timeout: {e}")
        print("  → Try reducing num_results or increasing timeout")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        print("  → Check logs for details")

asyncio.run(search_with_error_handling())
```

### Custom Error Handling in Workflows

```python
# In your workflow node
async def my_workflow_node(state: AgentState) -> AgentState:
    """Node with proper error handling."""
    try:
        # Your logic here
        result = await some_operation()
        
        return {
            **state,
            "current_step": "my_node",
            "next_step": "next_node",
            "data": result
        }
        
    except ValueError as e:
        # Validation errors
        return {
            **state,
            "current_step": "my_node",
            "error": f"Validation error: {e}",
            "next_step": "handle_error"
        }
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Node failed: {e}", exc_info=True)
        return {
            **state,
            "current_step": "my_node",
            "error": f"Unexpected error: {e}",
            "next_step": "handle_error"
        }
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def search_with_retry():
    """Search with automatic retry on failure."""
    agent = IntraMindAgent()
    return await agent.search(query="What are our projections?")

# Usage
try:
    result = await search_with_retry()
    print(f"Success after retries: {result['response']}")
except Exception as e:
    print(f"Failed after 3 attempts: {e}")
```

## 🎨 LangGraph Workflows

### Search Workflow

The document search workflow is implemented as a LangGraph state machine:

**State Definition** (`models/state.py`):
```python
class SearchWorkflowState(AgentState):
    """State for document search workflow."""
    query_complexity: str | None        # "simple" | "complex"
    expanded_queries: list[str] | None  # For complex searches
    search_results: list[dict] | None   # Retrieved documents
    response: str | None                # Synthesized answer
    citations: list[str] | None         # Document IDs
```

**Workflow Nodes** (`workflows/search_workflow.py`):
1. **classify_query**: Determines if query is simple or complex
2. **simple_search**: Single semantic search
3. **complex_search**: Multi-query expansion + aggregation
4. **synthesize_results**: AI-powered response generation
5. **handle_error**: Error recovery

**Routing Logic**:
- After classification → route to simple_search OR complex_search
- After search → route to synthesize_results OR handle_error
- After synthesis → END

### Creating Custom Workflows

Here's a complete example of a custom workflow for document summarization:

```python
# src/workflows/summarization_workflow.py
from langgraph.graph import StateGraph, END
from models.state import AgentState
from tools.api_client import APIGatewayClient
from utils.llm import get_primary_llm
from langchain_core.messages import SystemMessage, HumanMessage

async def fetch_document(state: AgentState) -> AgentState:
    """Fetch document by ID."""
    doc_id = state["user_query"]  # Document ID passed as query
    
    try:
        async with APIGatewayClient() as client:
            doc = await client.get_document(
                collection_name="intramind_documents",
                document_id=doc_id
            )
        
        return {
            **state,
            "current_step": "fetch_document",
            "document_content": doc.content,
            "next_step": "summarize"
        }
    except Exception as e:
        return {
            **state,
            "current_step": "fetch_document",
            "error": str(e),
            "next_step": "handle_error"
        }

async def summarize_document(state: AgentState) -> AgentState:
    """Generate document summary."""
    content = state.get("document_content", "")
    llm = get_primary_llm()
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content="Summarize the following document in 3-5 sentences."),
            HumanMessage(content=content)
        ])
        
        return {
            **state,
            "current_step": "summarize_document",
            "response": response.content,
            "workflow_complete": True
        }
    except Exception as e:
        return {
            **state,
            "current_step": "summarize_document",
            "error": str(e),
            "next_step": "handle_error"
        }

async def handle_error(state: AgentState) -> AgentState:
    """Handle workflow errors."""
    error = state.get("error", "Unknown error")
    return {
        **state,
        "current_step": "handle_error",
        "response": f"Error: {error}",
        "workflow_complete": True
    }

def create_summarization_workflow() -> StateGraph:
    """Create the summarization workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("fetch_document", fetch_document)
    workflow.add_node("summarize_document", summarize_document)
    workflow.add_node("handle_error", handle_error)
    
    # Set entry point
    workflow.set_entry_point("fetch_document")
    
    # Add conditional edges
    def route_after_fetch(state):
        return state.get("next_step", "summarize_document")
    
    workflow.add_conditional_edges(
        "fetch_document",
        route_after_fetch,
        {
            "summarize": "summarize_document",
            "handle_error": "handle_error"
        }
    )
    
    # Terminal edges
    workflow.add_edge("summarize_document", END)
    workflow.add_edge("handle_error", END)
    
    return workflow.compile()

# Usage
async def main():
    workflow = create_summarization_workflow()
    
    result = await workflow.ainvoke({
        "user_query": "doc-id-123",
        "workflow_complete": False
    })
    
    print(result["response"])
```

## 🧪 Testing

### Test Suite Overview

The project includes **112 comprehensive tests** covering all major components:

| Test File | Tests | Coverage | Focus |
|-----------|-------|----------|-------|
| `test_search_workflow.py` | 18 | Workflow nodes, routing, integration | LangGraph search workflow |
| `test_ingestion_workflow.py` | 32 | Validation, parsing, chunking, storage | Document ingestion pipeline |
| `test_agent.py` | 13 | Agent interface, streaming, errors | IntraMindAgent API |
| `test_agent_tools.py` | 18 | All 5 LangChain tools | Tool implementations |
| `test_conversation_memory.py` | 18 | Checkpointing, context handling | Conversation memory feature |
| `test_e2e_ingestion_search.py` | 7 | End-to-end integration | Full system validation |
| `test_api_client.py` | 3 | HTTP client basics | API Gateway client |
| `test_min_score_filtering.py` | 3 | Score threshold filtering | Search quality feature |
| **Total** | **112** | **67% coverage** | **All passing ✅** |

### Testing Patterns

#### 1. Testing Workflow Nodes

```python
# tests/test_search_workflow.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from workflows.search_workflow import classify_query

@pytest.mark.asyncio
async def test_classify_query_simple():
    """Test classification of simple query."""
    state = {
        "user_query": "What is the revenue?",
        "num_results": 10,
        "workflow_complete": False
    }
    
    # Mock the LLM response
    with patch("workflows.search_workflow.get_router_llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = "simple"
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await classify_query(state)
        
        # Assertions
        assert result["query_complexity"] == "simple"
        assert result["current_step"] == "classify_query"
        assert "next_step" in result

@pytest.mark.asyncio
async def test_classify_query_complex():
    """Test classification of complex query."""
    state = {
        "user_query": "Compare Q3 and Q4 performance across all departments",
        "num_results": 10,
        "workflow_complete": False
    }
    
    with patch("workflows.search_workflow.get_router_llm") as mock_llm:
        mock_response = Mock()
        mock_response.content = "complex"
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await classify_query(state)
        
        assert result["query_complexity"] == "complex"
```

#### 2. Testing Agent Interface

```python
# tests/test_agent.py
import pytest
from unittest.mock import AsyncMock, patch
from agent import IntraMindAgent

@pytest.mark.asyncio
async def test_agent_search_success():
    """Test successful agent search."""
    agent = IntraMindAgent()
    
    # Mock the workflow execution
    mock_result = {
        "success": True,
        "response": "Q4 revenue is projected at $5M",
        "query_complexity": "simple",
        "search_results": [{"id": "doc1", "content": "Revenue data"}],
        "workflow_complete": True
    }
    
    with patch.object(agent, '_search_workflow') as mock_workflow:
        mock_workflow.ainvoke = AsyncMock(return_value=mock_result)
        
        result = await agent.search("What are revenue projections?")
        
        assert result["success"] is True
        assert "response" in result
        assert result["complexity"] == "simple"
```

#### 3. Testing Tools

```python
# tests/test_agent_tools.py
import pytest
from unittest.mock import AsyncMock, patch
from tools.agent_tools import search_documents

@pytest.mark.asyncio
async def test_search_documents_tool():
    """Test search_documents LangChain tool."""
    mock_response = AsyncMock()
    mock_response.results = [
        Mock(document_id="1", content="Test", score=0.9, metadata={})
    ]
    
    with patch("tools.agent_tools.APIGatewayClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.search = AsyncMock(
            return_value=mock_response
        )
        
        result = await search_documents._arun(
            query="test query",
            collection_name="docs",
            limit=5
        )
        
        assert "Found 1 results" in result
        assert "doc_id: 1" in result
```

#### 4. Integration Testing

```python
# tests/test_e2e_ingestion_search.py
import pytest
from agent import IntraMindAgent

@pytest.mark.integration
@pytest.mark.asyncio
async def test_ingest_then_search():
    """Test full workflow: ingest document then search."""
    agent = IntraMindAgent()
    collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # Step 1: Ingest document
    ingest_result = await agent.ingest_document(
        file_path="test_data/sample.txt",
        collection_name=collection
    )
    assert ingest_result["success"] is True
    assert ingest_result["chunks_stored"] > 0
    
    # Step 2: Search for content
    search_result = await agent.search(
        query="test content",
        collection_name=collection
    )
    assert search_result["success"] is True
    assert len(search_result["citations"]) > 0
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (fast, no services required)
pytest -m "not integration" -v

# Integration tests (requires running services)
pytest -m integration -v

# With coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Specific test file
pytest tests/test_search_workflow.py -v

# Specific test
pytest tests/test_agent.py::test_agent_search_success -v
```

### Coverage Report

**Overall: 67% coverage** (404/606 statements)

**Core Components (100% Coverage):**
- ✅ `agent/main.py` - Agent interface
- ✅ `models/api.py` - Data models
- ✅ `models/state.py` - State definitions
- ✅ `agent_tools.py` - LangChain tools
- ✅ `search_workflow.py` - 98% (LangGraph workflow)

View HTML coverage report:
```bash
# Opens detailed line-by-line coverage
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
```

## 📊 Performance

### Cost Analysis (Typical Query)

| Operation | Provider | Cost | Notes |
|-----------|----------|------|-------|
| Query Classification | Ollama (Local) | $0.00 | Free |
| Query Expansion | Ollama (Local) | $0.00 | Free |
| Synthesis | Claude Haiku | ~$0.001 | ~1K tokens |
| **Total per Query** | | **~$0.001** | |

**Monthly estimate** (1000 queries): **~$1.00**

### Latency

- Simple Query: 2-4 seconds
- Complex Query: 5-8 seconds
- Streaming: First token in ~1 second

## 🛠️ Development

### Code Quality

Format code:
```bash
black src/
```

Lint:
```bash
ruff check src/
```

Type checking:
```bash
mypy src/
```

### Project Setup

Install development dependencies:
```bash
pip install -e ".[dev]"
```

## 🛡️ Responsible AI

This project implements a **fully free, open-source Responsible AI stack** (the "Free RAI Stack"). All LLM-based judges and classifiers run on local Ollama, so the running cost is $0.

| Capability | Tool | Where it lives |
|---|---|---|
| Tracing | Phoenix (Apache 2.0) | `docker-compose.yml` service + `src/utils/observability.py` |
| RAG evals (CI) | Ragas with Ollama judge | `tests/eval/` + `.github/workflows/ci.yml` `rag-evals` job |
| PII redaction | Presidio | `src/utils/pii.py` + `redact_pii` node in `ingestion_workflow.py` |
| Output safety | Llama Guard via Ollama | `src/utils/safety.py` + `safety_check` node in `search_workflow.py` |
| Drift monitoring | Evidently AI | `scripts/drift_report.py` + scheduled GitHub Action |
| Governance docs | Model + dataset cards | `docs/cards/` |

### RAG Eval Outputs

`tests/eval/ragas_eval.py` writes `tests/eval/results/latest.json` with two complementary metric groups:

- **Ragas answer-quality metrics**: `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall`.
- **Deterministic retrieval metrics**: `hit_at_1`, `hit_at_3`, `mrr`, and a per-question miss list based on `expected_source` in `tests/eval/data/golden_qa.jsonl`.

The output also records eval parameters (`collection`, `num_results`, `min_score`, vectorizer mode, prompt label, and prompt versions) so baseline changes can be traced to either retrieval settings or prompt changes.

Ragas metrics are written as strict JSON: unparseable local-judge outputs are converted to `null` and summarized under `ragas_quality.nan_counts` plus `ragas_quality.parse_failure_rate`. `RAGAS_MAX_PARSE_FAILURE_RATE` defaults to `0.2`; runs above that tolerance are treated as failed for Prompt Registry posting/promotion decisions even while threshold tests remain warning-only.

### PII Policy: Redact-on-Ingest with Tokenized Pseudonyms

PII detected during document ingestion is replaced with **stable, type-tagged tokens** (e.g. `<PERSON_1>`, `<EMAIL_ADDRESS_2>`) before content is chunked and stored. Raw PII never enters Weaviate.

Why this specific policy:
- **Data minimization.** Raw PII never persists, so the blast radius of any backup, exfiltration, or unauthorized read is minimized.
- **Embedding fidelity.** Tokenized pseudonyms preserve sentence structure, so retrieval quality on non-PII concepts is unaffected.
- **Stable within a document.** The same surface text gets the same token across the document so the synthesizer can still resolve co-references coherently.
- **Auditability.** `pii_findings` metadata on every chunk records the entity type, span offsets, and the assigned token — but never the raw value.

Trade-off (accepted): this approach is **irreversible**. Once redacted on ingest, the original PII cannot be recovered. Future role-based unmasking would require a separate originals store with independent permissions and is explicitly out of scope here.

To disable redaction (e.g. for a fully synthetic test corpus), set `ENABLE_PII_REDACTION=false`.

## 🔮 Future Enhancements

- [ ] **Multimodal Processing Workflow**: PDF, images, presentations
- [ ] **Document Ingestion Workflow**: Automated chunking and indexing
- [ ] **Multi-Agent Collaboration**: Specialized agents for different tasks
- [ ] **Agentic RAG**: Self-correcting retrieval with evaluation
- [ ] **Memory & Context**: Conversation history and user preferences
- [ ] **Web UI**: Streamlit or FastAPI frontend

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools Guide](https://python.langchain.com/docs/modules/tools/)
- [IntraMind API Gateway](https://github.com/JessKelly91/intramind-api-gateway)
- [Ollama Models](https://ollama.ai/library)

## 🤝 Contributing

This is a portfolio project demonstrating modern AI agent architecture patterns. Feel free to explore, learn, and provide feedback!

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Part of the [IntraMind Platform](https://github.com/JessKelly91/IntraMind)** - AI-Powered Enterprise Document Search
