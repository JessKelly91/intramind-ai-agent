# IntraMind AI Agent - Quick Start Guide

Get the AI agent running in **5 minutes**!

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed
- IntraMind API Gateway running (from the main IntraMind repo)

## Step 1: Install Dependencies

```bash
cd ai-agent
pip install -r requirements.txt
```

## Step 2: Setup Ollama (Free Local LLM)

```bash
# Install Ollama from https://ollama.ai/

# Pull the Llama 3.2 model (3B - fast and lightweight)
ollama pull llama3.2:3b

# Verify it's running
ollama list
```

## Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with minimal configuration
```

**Minimal `.env` for testing (no API keys needed):**
```env
# API Gateway
API_GATEWAY_URL=http://localhost:5000

# Use Ollama for everything (free!)
PRIMARY_LLM_PROVIDER=ollama
ROUTER_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Optional: Use Claude Haiku for better synthesis
# PRIMARY_LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your-key-here
```

## Step 4: Verify API Gateway

```bash
# Check if API Gateway is running
python -m src.cli.main health
```

Expected output:
```
✓ API Gateway is healthy
```

If this fails, make sure you've started the API Gateway service from the main IntraMind repo.

## Step 5: Run Your First Search!

### Interactive Mode

```bash
python -m src.cli.main search
```

This opens an interactive search session:
```
Search: What documents do we have about Q4?
```

### Single Query Mode

```bash
python -m src.cli.main search --query "Find documents about revenue projections"
```

## Next Steps

### Add Some Test Documents

You'll need some documents in your vector database first. You can:

1. Use the API Gateway directly to insert documents
2. Or use the agent's insert tool programmatically

Example Python script to add test documents:

```python
import asyncio
from tools.api_client import APIGatewayClient

async def add_test_docs():
    async with APIGatewayClient() as client:
        # Create collection
        await client.create_collection(
            name="intramind_documents",
            description="Test documents"
        )

        # Insert documents
        await client.insert_document(
            collection_name="intramind_documents",
            content="Q4 revenue projections show 25% growth year-over-year.",
            metadata={"title": "Q4 Projections", "type": "financial"}
        )

        await client.insert_document(
            collection_name="intramind_documents",
            content="The new product launch is scheduled for December 2024.",
            metadata={"title": "Product Launch", "type": "planning"}
        )

        print("✓ Test documents added!")

asyncio.run(add_test_docs())
```

### Upgrade to Better LLM

For production-quality responses, use Claude Haiku or GPT-3.5:

1. Get an API key:
   - Anthropic: https://console.anthropic.com/
   - OpenAI: https://platform.openai.com/

2. Update `.env`:
```env
PRIMARY_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

3. Keep Ollama for routing (saves money!):
```env
ROUTER_LLM_PROVIDER=ollama
```

## Troubleshooting

### "Connection refused" error

**Problem**: API Gateway not running

**Solution**:
```bash
cd ../api-gateway  # or wherever your API Gateway is
dotnet run
```

### "Ollama not found" error

**Problem**: Ollama not installed or not running

**Solution**:
1. Install from https://ollama.ai/
2. Run `ollama serve` in a terminal
3. Pull model: `ollama pull llama3.2:3b`

### "No API key" error

**Problem**: Trying to use Anthropic/OpenAI without key

**Solution**: Either:
1. Add your API key to `.env`
2. Or switch to Ollama: `PRIMARY_LLM_PROVIDER=ollama`

### Import errors

**Problem**: Package not found

**Solution**:
```bash
# Make sure you're in the ai-agent directory
cd ai-agent

# Reinstall dependencies
pip install -r requirements.txt
```

## Understanding the Architecture

The agent uses a **LangGraph state machine** for predictable workflows:

1. **Query Classification** (Router LLM - Ollama)
   - Simple or complex query?

2. **Search Strategy** (Based on classification)
   - Simple: Single semantic search
   - Complex: Multi-query expansion + deduplication

3. **Result Synthesis** (Primary LLM - Claude/GPT/Ollama)
   - Aggregate results
   - Generate coherent answer
   - Provide citations

This hybrid approach keeps costs low (~$0.001 per query with Claude Haiku) while maintaining quality.

## What's Next?

Check out the full [README.md](README.md) for:
- Programmatic usage examples
- Adding new workflows
- Multimodal document processing
- Testing and development

---

**Happy searching! 🚀**
