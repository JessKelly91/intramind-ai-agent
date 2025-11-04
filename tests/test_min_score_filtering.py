"""Tests for min_score filtering in semantic search."""

import pytest
from src.tools.api_client import APIGatewayClient


@pytest.mark.asyncio
async def test_min_score_filtering():
    """Test that min_score parameter filters search results correctly."""
    
    async with APIGatewayClient() as client:
        test_query = "What are our revenue projections?"
        collection = "intramind_documents"
        
        # Test 1: No filtering (min_score=0.0) - should return all results
        response_no_filter = await client.search(
            collection_name=collection,
            query=test_query,
            limit=10,
            min_score=0.0
        )
        
        # Test 2: Medium filtering (min_score=0.5)
        response_medium = await client.search(
            collection_name=collection,
            query=test_query,
            limit=10,
            min_score=0.5
        )
        
        # Test 3: High filtering (min_score=0.7)
        response_high = await client.search(
            collection_name=collection,
            query=test_query,
            limit=10,
            min_score=0.7
        )
        
        # Assertions
        assert response_no_filter.total_count >= 0, "Should return results with min_score=0.0"
        assert len(response_no_filter.results) == response_no_filter.total_count
        
        # Medium filter should return same or fewer results
        assert response_medium.total_count <= response_no_filter.total_count
        
        # High filter should return same or fewer results than medium
        assert response_high.total_count <= response_medium.total_count
        
        # All results should have scores above their respective thresholds
        for result in response_no_filter.results:
            assert result.score is not None, "Score should not be None"
            assert result.score >= 0.0, "Score should be >= 0.0"
        
        for result in response_medium.results:
            assert result.score is not None, "Score should not be None"
            if result.score is not None:  # Double check to satisfy type checker
                assert result.score >= 0.5, f"Score {result.score} should be >= 0.5 for min_score=0.5"
        
        for result in response_high.results:
            assert result.score is not None, "Score should not be None"
            if result.score is not None:
                assert result.score >= 0.7, f"Score {result.score} should be >= 0.7 for min_score=0.7"


@pytest.mark.asyncio
async def test_min_score_returns_valid_scores():
    """Test that search returns valid similarity scores."""
    
    async with APIGatewayClient() as client:
        response = await client.search(
            collection_name="intramind_documents",
            query="revenue",
            limit=5,
            min_score=0.0
        )
        
        # Check that we get results
        assert response.total_count > 0, "Should return at least one result"
        
        # Check that all results have valid scores
        for result in response.results:
            assert result.score is not None, "Each result should have a score"
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} should be between 0.0 and 1.0"
            assert result.document_id, "Each result should have a document ID"
            assert result.content, "Each result should have content"


@pytest.mark.asyncio
async def test_high_min_score_filters_aggressively():
    """Test that a high min_score (0.9) returns fewer or no results."""
    
    async with APIGatewayClient() as client:
        # Get baseline with no filtering
        response_all = await client.search(
            collection_name="intramind_documents",
            query="remote work policy",
            limit=10,
            min_score=0.0
        )
        
        # Get results with very high threshold
        response_filtered = await client.search(
            collection_name="intramind_documents",
            query="remote work policy",
            limit=10,
            min_score=0.9
        )
        
        # High threshold should return fewer results
        assert response_filtered.total_count <= response_all.total_count
        
        # All filtered results should have very high scores
        for result in response_filtered.results:
            assert result.score is not None
            if result.score is not None:
                assert result.score >= 0.9, f"Score {result.score} should be >= 0.9"

