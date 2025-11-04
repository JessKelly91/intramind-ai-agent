"""Tests for the search workflow using LangGraph."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from models.api import SearchResponse, SearchResult
from models.state import SearchWorkflowState
from workflows.search_workflow import (
    classify_query,
    simple_search,
    complex_search,
    synthesize_results,
    handle_error,
    route_after_classification,
    route_after_search,
    create_search_workflow,
)


# ============================================================================
# UNIT TESTS - Individual Node Functions
# ============================================================================


@pytest.mark.asyncio
async def test_classify_query_as_simple():
    """Test that simple queries are classified correctly."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="What is the weather?")],
        "user_query": "What is the weather?",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_llm:
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "simple"
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await classify_query(state)

        assert result["query_complexity"] == "simple"
        assert result["current_step"] == "classify_query"
        assert result["next_step"] == "simple_search"


@pytest.mark.asyncio
async def test_classify_query_as_complex():
    """Test that complex queries are classified correctly."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="Compare Q3 and Q4 revenue")],
        "user_query": "Compare Q3 and Q4 revenue and explain the differences",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_llm:
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "complex"
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await classify_query(state)

        assert result["query_complexity"] == "complex"
        assert result["current_step"] == "classify_query"
        assert result["next_step"] == "complex_search"


@pytest.mark.asyncio
async def test_classify_query_handles_invalid_response():
    """Test that invalid classification defaults to 'simple'."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="Test query")],
        "user_query": "Test query",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_llm:
        # Mock invalid LLM response
        mock_response = Mock()
        mock_response.content = "invalid_classification"
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await classify_query(state)

        # Should default to simple
        assert result["query_complexity"] == "simple"
        assert result["next_step"] == "simple_search"


@pytest.mark.asyncio
async def test_simple_search_success():
    """Test successful simple search execution."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="revenue projections")],
        "user_query": "What are our revenue projections?",
        "current_step": "classify_query",
        "next_step": "simple_search",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "intramind_documents"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_api_client") as mock_client:
        # Mock search response
        mock_search = AsyncMock(
            return_value=SearchResponse(
                results=[
                    SearchResult(
                        document_id="doc1",
                        content="Revenue projections show 25% growth",
                        metadata={"title": "Q4 Projections"},
                        score=0.95,
                        collection_name="intramind_documents",
                    )
                ],
                query="What are our revenue projections?",
                collection_name="intramind_documents",
                total_count=1,
            )
        )
        mock_client.return_value.search = mock_search

        result = await simple_search(state)

        assert result["current_step"] == "simple_search"
        assert result["next_step"] == "synthesize_results"
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["id"] == "doc1"
        assert result["search_results"][0]["score"] == 0.95
        assert result["error"] is None


@pytest.mark.asyncio
async def test_simple_search_handles_error():
    """Test that simple search handles errors gracefully."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="test")],
        "user_query": "test query",
        "current_step": "classify_query",
        "next_step": "simple_search",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "test_collection"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_api_client") as mock_client:
        # Mock search failure
        mock_client.return_value.search = AsyncMock(
            side_effect=Exception("Connection error")
        )

        result = await simple_search(state)

        assert result["current_step"] == "simple_search"
        assert result["next_step"] == "handle_error"
        assert result["error"] == "Connection error"


@pytest.mark.asyncio
async def test_complex_search_with_query_expansion():
    """Test complex search with query expansion."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="Compare Q3 and Q4 revenue")],
        "user_query": "Compare Q3 and Q4 revenue and explain differences",
        "current_step": "classify_query",
        "next_step": "complex_search",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "intramind_documents"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "complex",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_llm, patch(
        "workflows.search_workflow.get_api_client"
    ) as mock_client:
        # Mock LLM query expansion
        mock_expansion_response = Mock()
        mock_expansion_response.content = """1. Q3 revenue figures
2. Q4 revenue projections
3. Revenue growth comparison"""
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_expansion_response)

        # Mock search responses
        mock_search = AsyncMock(
            return_value=SearchResponse(
                results=[
                    SearchResult(
                        document_id=f"doc{i}",
                        content=f"Document {i} content",
                        metadata={"title": f"Doc {i}"},
                        score=0.9 - (i * 0.1),
                        collection_name="intramind_documents",
                    )
                    for i in range(3)
                ],
                query="test",
                collection_name="intramind_documents",
                total_count=3,
            )
        )
        mock_client.return_value.search = mock_search

        result = await complex_search(state)

        assert result["current_step"] == "complex_search"
        assert result["next_step"] == "synthesize_results"
        assert result["expanded_queries"] is not None
        assert len(result["expanded_queries"]) == 3
        assert "Q3 revenue figures" in result["expanded_queries"]
        assert result["search_results"] is not None
        # Should deduplicate results
        assert len(result["search_results"]) <= 10


@pytest.mark.asyncio
async def test_complex_search_deduplicates_results():
    """Test that complex search deduplicates results by ID."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="test")],
        "user_query": "test query",
        "current_step": "classify_query",
        "next_step": "complex_search",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 5,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "intramind_documents"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "complex",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_llm, patch(
        "workflows.search_workflow.get_api_client"
    ) as mock_client:
        # Mock LLM query expansion
        mock_expansion_response = Mock()
        mock_expansion_response.content = "1. Query 1\n2. Query 2"
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_expansion_response)

        # Mock search responses - same doc appears in both searches
        mock_search = AsyncMock(
            return_value=SearchResponse(
                results=[
                    SearchResult(
                        document_id="duplicate_doc",
                        content="Duplicate content",
                        metadata={"title": "Duplicate"},
                        score=0.95,
                        collection_name="intramind_documents",
                    ),
                    SearchResult(
                        document_id="unique_doc",
                        content="Unique content",
                        metadata={"title": "Unique"},
                        score=0.90,
                        collection_name="intramind_documents",
                    ),
                ],
                query="test",
                collection_name="intramind_documents",
                total_count=2,
            )
        )
        mock_client.return_value.search = mock_search

        result = await complex_search(state)

        # Should only have 2 unique documents (duplicate_doc and unique_doc)
        result_ids = [r["id"] for r in result["search_results"]]
        assert len(result_ids) == len(set(result_ids)), "Should deduplicate by ID"


@pytest.mark.asyncio
async def test_synthesize_results_with_documents():
    """Test result synthesis with documents."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="revenue")],
        "user_query": "What are the revenue projections?",
        "current_step": "simple_search",
        "next_step": "synthesize_results",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": [
            {
                "id": "doc1",
                "content": "Q4 revenue projections show 25% growth",
                "metadata": {"title": "Revenue Report"},
                "score": 0.95,
            },
            {
                "id": "doc2",
                "content": "Market expansion drives revenue increase",
                "metadata": {"title": "Market Analysis"},
                "score": 0.88,
            },
        ],
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_primary_llm") as mock_llm:
        # Mock LLM synthesis
        mock_response = Mock()
        mock_response.content = "According to Documents 1 and 2, revenue projections show 25% growth driven by market expansion."
        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

        result = await synthesize_results(state)

        assert result["current_step"] == "synthesize_results"
        assert result["workflow_complete"] is True
        assert result["next_step"] is None
        assert result["response"] is not None
        assert "25% growth" in result["response"]
        assert result["citations"] == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_synthesize_results_with_no_documents():
    """Test synthesis when no documents are found."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="test")],
        "user_query": "nonexistent query",
        "current_step": "simple_search",
        "next_step": "synthesize_results",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": [],  # No results
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    result = await synthesize_results(state)

    assert result["workflow_complete"] is True
    assert result["response"] == "I couldn't find any relevant documents for your query."
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_handle_error():
    """Test error handling node."""
    state: SearchWorkflowState = {
        "messages": [HumanMessage(content="test")],
        "user_query": "test query",
        "current_step": "simple_search",
        "next_step": "handle_error",
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": "Database connection failed",
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    result = await handle_error(state)

    assert result["current_step"] == "handle_error"
    assert result["workflow_complete"] is True
    assert result["next_step"] is None
    assert "Database connection failed" in result["response"]


# ============================================================================
# ROUTING TESTS
# ============================================================================


def test_route_after_classification_simple():
    """Test routing to simple search."""
    state: SearchWorkflowState = {
        "messages": [],
        "user_query": "test",
        "current_step": "classify_query",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    route = route_after_classification(state)
    assert route == "simple_search"


def test_route_after_classification_complex():
    """Test routing to complex search."""
    state: SearchWorkflowState = {
        "messages": [],
        "user_query": "test",
        "current_step": "classify_query",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "complex",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    route = route_after_classification(state)
    assert route == "complex_search"


def test_route_after_search_to_synthesize():
    """Test routing to synthesis after successful search."""
    state: SearchWorkflowState = {
        "messages": [],
        "user_query": "test",
        "current_step": "simple_search",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": [{"id": "doc1", "content": "test"}],
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    route = route_after_search(state)
    assert route == "synthesize_results"


def test_route_after_search_to_error():
    """Test routing to error handler after failed search."""
    state: SearchWorkflowState = {
        "messages": [],
        "user_query": "test",
        "current_step": "simple_search",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": "Search failed",
        "retry_count": 0,
        "query_complexity": "simple",
        "expanded_queries": None,
        "aggregated_results": None,
    }

    route = route_after_search(state)
    assert route == "handle_error"


# ============================================================================
# WORKFLOW GRAPH TESTS
# ============================================================================


def test_create_search_workflow():
    """Test that workflow graph is created correctly."""
    workflow = create_search_workflow()

    assert workflow is not None
    # Graph should be compiled and ready to use
    assert hasattr(workflow, "ainvoke")
    assert hasattr(workflow, "astream")


# ============================================================================
# INTEGRATION TESTS (require mocked dependencies)
# ============================================================================


@pytest.mark.asyncio
async def test_full_workflow_simple_query():
    """Integration test: Complete workflow for simple query."""
    workflow = create_search_workflow()

    initial_state: SearchWorkflowState = {
        "messages": [HumanMessage(content="What is the revenue?")],
        "user_query": "What is the revenue?",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "intramind_documents"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_router_llm, patch(
        "workflows.search_workflow.get_primary_llm"
    ) as mock_primary_llm, patch("workflows.search_workflow.get_api_client") as mock_client:
        # Mock classification
        mock_classify_response = Mock()
        mock_classify_response.content = "simple"
        mock_router_llm.return_value.ainvoke = AsyncMock(
            return_value=mock_classify_response
        )

        # Mock search
        mock_client.return_value.search = AsyncMock(
            return_value=SearchResponse(
                results=[
                    SearchResult(
                        document_id="doc1",
                        content="Revenue is $1M",
                        metadata={"title": "Report"},
                        score=0.95,
                        collection_name="intramind_documents",
                    )
                ],
                query="What is the revenue?",
                collection_name="intramind_documents",
                total_count=1,
            )
        )

        # Mock synthesis
        mock_synthesis_response = Mock()
        mock_synthesis_response.content = "According to Document 1, the revenue is $1M."
        mock_primary_llm.return_value.ainvoke = AsyncMock(
            return_value=mock_synthesis_response
        )

        # Execute workflow
        result = await workflow.ainvoke(initial_state)

        # Verify workflow completed successfully
        assert result["workflow_complete"] is True
        assert result["query_complexity"] == "simple"
        assert result["response"] is not None
        assert "$1M" in result["response"]
        assert result["error"] is None


@pytest.mark.asyncio
async def test_full_workflow_complex_query():
    """Integration test: Complete workflow for complex query."""
    workflow = create_search_workflow()

    initial_state: SearchWorkflowState = {
        "messages": [HumanMessage(content="Compare Q3 and Q4")],
        "user_query": "Compare Q3 and Q4 revenue",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "intramind_documents"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_router_llm, patch(
        "workflows.search_workflow.get_primary_llm"
    ) as mock_primary_llm, patch("workflows.search_workflow.get_api_client") as mock_client:
        # Mock classification as complex
        mock_classify_response = Mock()
        mock_classify_response.content = "complex"

        # Mock query expansion
        mock_expansion_response = Mock()
        mock_expansion_response.content = "1. Q3 revenue\n2. Q4 revenue\n3. Revenue comparison"

        # Set up router LLM to return different responses
        mock_router_llm.return_value.ainvoke = AsyncMock(
            side_effect=[mock_classify_response, mock_expansion_response]
        )

        # Mock search
        mock_client.return_value.search = AsyncMock(
            return_value=SearchResponse(
                results=[
                    SearchResult(
                        document_id=f"doc{i}",
                        content=f"Document {i} about revenue",
                        metadata={"title": f"Report {i}"},
                        score=0.9,
                        collection_name="intramind_documents",
                    )
                    for i in range(3)
                ],
                query="test",
                collection_name="intramind_documents",
                total_count=3,
            )
        )

        # Mock synthesis
        mock_synthesis_response = Mock()
        mock_synthesis_response.content = (
            "Q3 revenue was $800K and Q4 revenue was $1M, showing 25% growth."
        )
        mock_primary_llm.return_value.ainvoke = AsyncMock(
            return_value=mock_synthesis_response
        )

        # Execute workflow
        result = await workflow.ainvoke(initial_state)

        # Verify workflow completed successfully
        assert result["workflow_complete"] is True
        assert result["query_complexity"] == "complex"
        assert result["expanded_queries"] is not None
        assert len(result["expanded_queries"]) >= 2
        assert result["response"] is not None
        assert result["error"] is None


@pytest.mark.asyncio
async def test_workflow_handles_search_error():
    """Integration test: Workflow handles search errors gracefully."""
    workflow = create_search_workflow()

    initial_state: SearchWorkflowState = {
        "messages": [HumanMessage(content="test")],
        "user_query": "test query",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 10,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {"collection_name": "intramind_documents"},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "query_complexity": None,
        "expanded_queries": None,
        "aggregated_results": None,
    }

    with patch("workflows.search_workflow.get_router_llm") as mock_router_llm, patch(
        "workflows.search_workflow.get_api_client"
    ) as mock_client:
        # Mock classification
        mock_classify_response = Mock()
        mock_classify_response.content = "simple"
        mock_router_llm.return_value.ainvoke = AsyncMock(
            return_value=mock_classify_response
        )

        # Mock search failure
        mock_client.return_value.search = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        # Execute workflow
        result = await workflow.ainvoke(initial_state)

        # Verify error was handled
        assert result["workflow_complete"] is True
        assert result["error"] == "Connection timeout"
        assert "Connection timeout" in result["response"]

