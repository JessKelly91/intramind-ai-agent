"""Tests for the IntraMindAgent main interface."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from agent.main import IntraMindAgent


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_agent_initialization():
    """Test that agent initializes correctly."""
    agent = IntraMindAgent()
    
    assert agent is not None
    assert agent.search_workflow is not None
    assert hasattr(agent, "search")
    assert hasattr(agent, "stream_search")


# ============================================================================
# SEARCH METHOD TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_search_simple_query_success():
    """Test successful search with simple query."""
    agent = IntraMindAgent()
    
    # Mock workflow response
    mock_workflow_result = {
        "workflow_complete": True,
        "query_complexity": "simple",
        "response": "The revenue is $1M according to Document 1.",
        "citations": ["doc1", "doc2"],
        "search_results": [
            {
                "id": "doc1",
                "content": "Revenue is $1M",
                "metadata": {"title": "Report"},
                "score": 0.95,
            }
        ],
        "expanded_queries": None,
    }
    
    with patch.object(agent.search_workflow, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_workflow_result
        
        result = await agent.search("What is the revenue?")
        
        # Verify result structure
        assert result["success"] is True
        assert result["query"] == "What is the revenue?"
        assert result["response"] == "The revenue is $1M according to Document 1."
        assert result["citations"] == ["doc1", "doc2"]
        assert result["complexity"] == "simple"
        assert len(result["results"]) == 1
        assert result["expanded_queries"] is None
        
        # Verify workflow was called with correct state
        call_args = mock_ainvoke.call_args[0][0]
        assert call_args["user_query"] == "What is the revenue?"
        assert call_args["num_results"] == 10  # default
        assert call_args["document_metadata"]["collection_name"] == "intramind_documents"


@pytest.mark.asyncio
async def test_search_complex_query_success():
    """Test successful search with complex query."""
    agent = IntraMindAgent()
    
    # Mock workflow response for complex query
    mock_workflow_result = {
        "workflow_complete": True,
        "query_complexity": "complex",
        "response": "Q3 revenue was $800K and Q4 was $1M, showing 25% growth.",
        "citations": ["doc1", "doc2", "doc3"],
        "search_results": [
            {"id": "doc1", "content": "Q3 revenue", "metadata": {}, "score": 0.92},
            {"id": "doc2", "content": "Q4 revenue", "metadata": {}, "score": 0.90},
            {"id": "doc3", "content": "Growth analysis", "metadata": {}, "score": 0.88},
        ],
        "expanded_queries": ["Q3 revenue figures", "Q4 revenue projections", "Revenue comparison"],
    }
    
    with patch.object(agent.search_workflow, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_workflow_result
        
        result = await agent.search("Compare Q3 and Q4 revenue")
        
        # Verify result structure
        assert result["success"] is True
        assert result["query"] == "Compare Q3 and Q4 revenue"
        assert result["response"] is not None
        assert result["complexity"] == "complex"
        assert len(result["results"]) == 3
        assert result["expanded_queries"] is not None
        assert len(result["expanded_queries"]) == 3


@pytest.mark.asyncio
async def test_search_with_custom_collection():
    """Test search with custom collection name."""
    agent = IntraMindAgent()
    
    mock_workflow_result = {
        "workflow_complete": True,
        "query_complexity": "simple",
        "response": "Test response",
        "citations": [],
        "search_results": [],
        "expanded_queries": None,
    }
    
    with patch.object(agent.search_workflow, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_workflow_result
        
        result = await agent.search(
            "test query",
            collection_name="custom_collection",
            num_results=5
        )
        
        # Verify custom parameters were used
        call_args = mock_ainvoke.call_args[0][0]
        assert call_args["document_metadata"]["collection_name"] == "custom_collection"
        assert call_args["num_results"] == 5


@pytest.mark.asyncio
async def test_search_handles_workflow_error():
    """Test that search handles workflow errors gracefully."""
    agent = IntraMindAgent()
    
    with patch.object(agent.search_workflow, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        # Simulate workflow exception
        mock_ainvoke.side_effect = Exception("Workflow execution failed")
        
        result = await agent.search("test query")
        
        # Verify error response
        assert result["success"] is False
        assert result["query"] == "test query"
        assert "error" in result
        assert "Workflow execution failed" in result["error"]


@pytest.mark.asyncio
async def test_search_no_results():
    """Test search when no results are found."""
    agent = IntraMindAgent()
    
    mock_workflow_result = {
        "workflow_complete": True,
        "query_complexity": "simple",
        "response": "I couldn't find any relevant documents for your query.",
        "citations": [],
        "search_results": [],
        "expanded_queries": None,
    }
    
    with patch.object(agent.search_workflow, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_workflow_result
        
        result = await agent.search("nonexistent topic")
        
        assert result["success"] is True
        assert len(result["results"]) == 0
        assert len(result["citations"]) == 0
        assert "couldn't find" in result["response"].lower()


@pytest.mark.asyncio
async def test_search_state_initialization():
    """Test that search initializes state correctly."""
    agent = IntraMindAgent()
    
    mock_workflow_result = {
        "workflow_complete": True,
        "query_complexity": "simple",
        "response": "Test",
        "citations": [],
        "search_results": [],
    }
    
    with patch.object(agent.search_workflow, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_workflow_result
        
        await agent.search("test query", collection_name="test_col", num_results=15)
        
        # Verify initial state structure
        call_args = mock_ainvoke.call_args[0][0]
        
        # Check all required state fields are present
        assert "messages" in call_args
        assert "user_query" in call_args
        assert "current_step" in call_args
        assert "workflow_complete" in call_args
        assert "num_results" in call_args
        assert "document_metadata" in call_args
        
        # Check specific values
        assert call_args["user_query"] == "test query"
        assert call_args["num_results"] == 15
        assert call_args["document_metadata"]["collection_name"] == "test_col"
        assert call_args["workflow_complete"] is False
        assert call_args["current_step"] == "start"


# ============================================================================
# STREAM SEARCH METHOD TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_stream_search_yields_updates():
    """Test that stream_search yields state updates."""
    agent = IntraMindAgent()
    
    # Mock streaming workflow responses
    mock_stream_events = [
        {
            "classify_query": {
                "current_step": "classify_query",
                "query_complexity": "simple",
                "workflow_complete": False,
            }
        },
        {
            "simple_search": {
                "current_step": "simple_search",
                "query_complexity": "simple",
                "search_results": [{"id": "doc1", "content": "test"}],
                "workflow_complete": False,
            }
        },
        {
            "synthesize_results": {
                "current_step": "synthesize_results",
                "query_complexity": "simple",
                "search_results": [{"id": "doc1", "content": "test"}],
                "response": "Test response",
                "workflow_complete": True,
            }
        },
    ]
    
    async def mock_astream(state):
        for event in mock_stream_events:
            yield event
    
    with patch.object(agent.search_workflow, "astream", side_effect=mock_astream):
        events = []
        async for event in agent.stream_search("test query"):
            events.append(event)
        
        # Verify we got events for each node
        assert len(events) == 3
        
        # Check first event (classification)
        assert events[0]["node"] == "classify_query"
        assert events[0]["step"] == "classify_query"
        assert events[0]["complexity"] == "simple"
        
        # Check second event (search)
        assert events[1]["node"] == "simple_search"
        assert events[1]["results_count"] == 1
        
        # Check third event (synthesis)
        assert events[2]["node"] == "synthesize_results"
        assert events[2]["response"] == "Test response"
        assert events[2]["complete"] is True


@pytest.mark.asyncio
async def test_stream_search_complex_query():
    """Test stream_search with complex query."""
    agent = IntraMindAgent()
    
    mock_stream_events = [
        {
            "classify_query": {
                "current_step": "classify_query",
                "query_complexity": "complex",
                "workflow_complete": False,
            }
        },
        {
            "complex_search": {
                "current_step": "complex_search",
                "query_complexity": "complex",
                "expanded_queries": ["query1", "query2"],
                "search_results": [{"id": "doc1"}, {"id": "doc2"}],
                "workflow_complete": False,
            }
        },
        {
            "synthesize_results": {
                "current_step": "synthesize_results",
                "query_complexity": "complex",
                "search_results": [{"id": "doc1"}, {"id": "doc2"}],
                "response": "Complex response",
                "workflow_complete": True,
            }
        },
    ]
    
    async def mock_astream(state):
        for event in mock_stream_events:
            yield event
    
    with patch.object(agent.search_workflow, "astream", side_effect=mock_astream):
        events = []
        async for event in agent.stream_search("Compare Q3 and Q4"):
            events.append(event)
        
        assert len(events) == 3
        assert events[0]["complexity"] == "complex"
        assert events[1]["results_count"] == 2
        assert events[2]["complete"] is True


@pytest.mark.asyncio
async def test_stream_search_handles_errors():
    """Test that stream_search handles errors gracefully."""
    agent = IntraMindAgent()
    
    async def mock_astream_error(state):
        raise Exception("Streaming failed")
        yield  # Make it a generator
    
    with patch.object(agent.search_workflow, "astream", side_effect=mock_astream_error):
        events = []
        async for event in agent.stream_search("test query"):
            events.append(event)
        
        # Should yield error event
        assert len(events) == 1
        assert events[0]["node"] == "error"
        assert "Streaming failed" in events[0]["error"]
        assert events[0]["complete"] is True


@pytest.mark.asyncio
async def test_stream_search_with_custom_parameters():
    """Test stream_search with custom collection and limit."""
    agent = IntraMindAgent()
    
    mock_stream_events = [
        {
            "classify_query": {
                "current_step": "classify_query",
                "query_complexity": "simple",
                "workflow_complete": False,
            }
        },
    ]
    
    async def mock_astream(state):
        # Verify state was initialized with custom parameters
        assert state["document_metadata"]["collection_name"] == "custom_col"
        assert state["num_results"] == 20
        for event in mock_stream_events:
            yield event
    
    with patch.object(agent.search_workflow, "astream", side_effect=mock_astream):
        events = []
        async for event in agent.stream_search(
            "test query",
            collection_name="custom_col",
            num_results=20
        ):
            events.append(event)
        
        assert len(events) >= 1


@pytest.mark.asyncio
async def test_stream_search_no_results():
    """Test stream_search when no results are found."""
    agent = IntraMindAgent()
    
    mock_stream_events = [
        {
            "classify_query": {
                "current_step": "classify_query",
                "query_complexity": "simple",
                "workflow_complete": False,
            }
        },
        {
            "simple_search": {
                "current_step": "simple_search",
                "search_results": [],  # No results
                "workflow_complete": False,
            }
        },
        {
            "synthesize_results": {
                "current_step": "synthesize_results",
                "search_results": [],
                "response": "I couldn't find any relevant documents.",
                "workflow_complete": True,
            }
        },
    ]
    
    async def mock_astream(state):
        for event in mock_stream_events:
            yield event
    
    with patch.object(agent.search_workflow, "astream", side_effect=mock_astream):
        events = []
        async for event in agent.stream_search("nonexistent topic"):
            events.append(event)
        
        # Verify we got updates even with no results
        assert len(events) == 3
        assert events[1]["results_count"] == 0
        assert events[2]["complete"] is True


@pytest.mark.asyncio
async def test_stream_search_yields_correct_format():
    """Test that stream_search yields correctly formatted events."""
    agent = IntraMindAgent()
    
    mock_stream_events = [
        {
            "test_node": {
                "current_step": "test_step",
                "query_complexity": "simple",
                "search_results": [{"id": "doc1"}],
                "response": "Test response",
                "error": None,
                "workflow_complete": False,
            }
        },
    ]
    
    async def mock_astream(state):
        for event in mock_stream_events:
            yield event
    
    with patch.object(agent.search_workflow, "astream", side_effect=mock_astream):
        events = []
        async for event in agent.stream_search("test"):
            events.append(event)
        
        # Verify event structure
        assert len(events) == 1
        event = events[0]
        
        # Check all expected fields are present
        assert "node" in event
        assert "step" in event
        assert "complexity" in event
        assert "results_count" in event
        assert "response" in event
        assert "error" in event
        assert "complete" in event
        
        # Check values
        assert event["node"] == "test_node"
        assert event["step"] == "test_step"
        assert event["complexity"] == "simple"
        assert event["results_count"] == 1
        assert event["response"] == "Test response"
        assert event["error"] is None
        assert event["complete"] is False

