"""Tests for API Gateway client."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from models.api import SearchResponse, SearchResult
from tools.api_client import APIGatewayClient


@pytest.fixture
def api_client():
    """Create API client for testing."""
    return APIGatewayClient(base_url="http://test:5000", timeout=10)


@pytest.mark.asyncio
async def test_health_check(api_client):
    """Test health check endpoint."""
    with patch.object(api_client.client, "get") as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = await api_client.health_check()

        assert result["status"] == "healthy"
        mock_get.assert_called_once_with("/health")


@pytest.mark.asyncio
async def test_search(api_client):
    """Test semantic search."""
    with patch.object(api_client.client, "post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "documentId": "doc1",
                    "content": "Test content",
                    "metadata": {"title": "Test"},
                    "score": 0.95,
                    "collectionName": "test_collection",
                }
            ],
            "query": "test query",
            "collectionName": "test_collection",
            "totalCount": 1,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = await api_client.search(
            collection_name="test_collection", query="test query", limit=10
        )

        assert isinstance(result, SearchResponse)
        assert len(result.results) == 1
        assert result.results[0].document_id == "doc1"
        assert result.results[0].content == "Test content"
        assert result.total_count == 1


@pytest.mark.asyncio
async def test_insert_document(api_client):
    """Test document insertion."""
    with patch.object(api_client.client, "post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "new_doc_id",
            "content": "Test content",
            "metadata": None,
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        result = await api_client.insert_document(
            collection_name="test_collection",
            content="Test content",
            metadata={"title": "Test"},
        )

        assert result.id == "new_doc_id"
        assert result.content == "Test content"
