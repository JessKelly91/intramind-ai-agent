# IntraMind AI Agent

> AI-powered intelligent document search agent using LangGraph state machine architecture

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

The IntraMind AI Agent is an intelligent document search system built with **LangGraph**, a state machine framework for building sophisticated AI workflows. It provides semantic search capabilities with automatic query classification, multi-query expansion, and intelligent result synthesis.

### Key Features

- **🤖 LangGraph State Machine**: Predictable, debuggable AI workflows
- **🔍 Smart Query Routing**: Automatic classification of simple vs complex queries
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

### Adding New Workflows

Create a new workflow in `src/workflows/`:

```python
from langgraph.graph import StateGraph, END
from models.state import AgentState

def my_custom_node(state: AgentState) -> AgentState:
    # Your logic here
    return {**state, "current_step": "my_node"}

def create_my_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("my_node", my_custom_node)

    # Set entry and edges
    workflow.set_entry_point("my_node")
    workflow.add_edge("my_node", END)

    return workflow.compile()

my_workflow = create_my_workflow()
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
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
