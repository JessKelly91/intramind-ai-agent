"""Tests for LangChain tools in agent_tools.py."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from models.api import (
    SearchResponse,
    SearchResult,
    DocumentResponse,
    CollectionResponse,
)
from tools.agent_tools import (
    search_documents,
    insert_document,
    get_document,
    list_collections,
    create_collection,
    get_api_client,
)


# ============================================================================
# API CLIENT HELPER TESTS
# ============================================================================


def test_get_api_client():
    """Test that get_api_client returns a client instance."""
    client = get_api_client()
    
    assert client is not None
    assert hasattr(client, "search")
    assert hasattr(client, "insert_document")
    
    # Should return same instance on subsequent calls
    client2 = get_api_client()
    assert client is client2


# ============================================================================
# SEARCH_DOCUMENTS TOOL TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_search_documents_success():
    """Test successful document search."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        # Mock API client and search response
        mock_client = Mock()
        mock_client.search = AsyncMock(
            return_value=SearchResponse(
                results=[
                    SearchResult(
                        document_id="doc1",
                        content="Revenue projections show growth",
                        metadata={"title": "Q4 Report", "department": "Finance"},
                        score=0.95,
                        collection_name="intramind_documents",
                    ),
                    SearchResult(
                        document_id="doc2",
                        content="Market expansion in Asia",
                        metadata={"title": "Market Analysis"},
                        score=0.88,
                        collection_name="intramind_documents",
                    ),
                ],
                query="revenue projections",
                collection_name="intramind_documents",
                total_count=2,
            )
        )
        mock_get_client.return_value = mock_client
        
        # Execute tool
        result = await search_documents.ainvoke({
            "query": "revenue projections",
            "collection_name": "intramind_documents",
            "limit": 10,
        })
        
        # Verify result structure
        assert result["success"] is True
        assert result["total_results"] == 2
        assert len(result["results"]) == 2
        
        # Verify first result
        assert result["results"][0]["id"] == "doc1"
        assert result["results"][0]["content"] == "Revenue projections show growth"
        assert result["results"][0]["score"] == 0.95
        assert result["results"][0]["metadata"]["title"] == "Q4 Report"
        
        # Verify API client was called correctly
        mock_client.search.assert_called_once_with(
            collection_name="intramind_documents",
            query="revenue projections",
            limit=10,
        )


@pytest.mark.asyncio
async def test_search_documents_with_defaults():
    """Test search_documents with default parameters."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.search = AsyncMock(
            return_value=SearchResponse(
                results=[],
                query="test",
                collection_name="intramind_documents",
                total_count=0,
            )
        )
        mock_get_client.return_value = mock_client
        
        # Execute tool with minimal parameters
        result = await search_documents.ainvoke({"query": "test query"})
        
        # Verify defaults were used
        mock_client.search.assert_called_once_with(
            collection_name="intramind_documents",  # default
            query="test query",
            limit=10,  # default
        )


@pytest.mark.asyncio
async def test_search_documents_no_results():
    """Test search_documents when no results are found."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.search = AsyncMock(
            return_value=SearchResponse(
                results=[],
                query="nonexistent",
                collection_name="intramind_documents",
                total_count=0,
            )
        )
        mock_get_client.return_value = mock_client
        
        result = await search_documents.ainvoke({"query": "nonexistent topic"})
        
        assert result["success"] is True
        assert result["total_results"] == 0
        assert len(result["results"]) == 0


@pytest.mark.asyncio
async def test_search_documents_handles_error():
    """Test that search_documents handles API errors gracefully."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.search = AsyncMock(side_effect=Exception("Connection timeout"))
        mock_get_client.return_value = mock_client
        
        result = await search_documents.ainvoke({"query": "test"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Connection timeout" in result["error"]


# ============================================================================
# INSERT_DOCUMENT TOOL TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_insert_document_success():
    """Test successful document insertion."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.insert_document = AsyncMock(
            return_value=DocumentResponse(
                id="new_doc_id_123",
                content="New document content",
                metadata={"title": "New Document", "author": "Test User"},
            )
        )
        mock_get_client.return_value = mock_client
        
        result = await insert_document.ainvoke({
            "content": "New document content",
            "collection_name": "test_collection",
            "metadata": {"title": "New Document", "author": "Test User"},
        })
        
        assert result["success"] is True
        assert result["document_id"] == "new_doc_id_123"
        assert "inserted successfully" in result["message"].lower()
        
        # Verify API client was called correctly
        mock_client.insert_document.assert_called_once_with(
            collection_name="test_collection",
            content="New document content",
            metadata={"title": "New Document", "author": "Test User"},
        )


@pytest.mark.asyncio
async def test_insert_document_with_defaults():
    """Test insert_document with default parameters."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.insert_document = AsyncMock(
            return_value=DocumentResponse(
                id="doc_id",
                content="Test content",
                metadata=None,
            )
        )
        mock_get_client.return_value = mock_client
        
        # Insert without metadata
        result = await insert_document.ainvoke({"content": "Test content"})
        
        assert result["success"] is True
        
        # Verify default collection was used
        mock_client.insert_document.assert_called_once_with(
            collection_name="intramind_documents",  # default
            content="Test content",
            metadata=None,
        )


@pytest.mark.asyncio
async def test_insert_document_handles_error():
    """Test that insert_document handles API errors gracefully."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.insert_document = AsyncMock(
            side_effect=Exception("Database write failed")
        )
        mock_get_client.return_value = mock_client
        
        result = await insert_document.ainvoke({"content": "test"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Database write failed" in result["error"]


# ============================================================================
# GET_DOCUMENT TOOL TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_get_document_success():
    """Test successful document retrieval."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.get_document = AsyncMock(
            return_value=DocumentResponse(
                id="doc123",
                content="Document content here",
                metadata={"title": "Test Document", "date": "2024-11-04"},
            )
        )
        mock_get_client.return_value = mock_client
        
        result = await get_document.ainvoke({
            "document_id": "doc123",
            "collection_name": "test_collection",
        })
        
        assert result["success"] is True
        assert result["id"] == "doc123"
        assert result["content"] == "Document content here"
        assert result["metadata"]["title"] == "Test Document"
        
        # Verify API client was called correctly
        mock_client.get_document.assert_called_once_with(
            document_id="doc123",
            collection_name="test_collection",
        )


@pytest.mark.asyncio
async def test_get_document_with_default_collection():
    """Test get_document with default collection."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.get_document = AsyncMock(
            return_value=DocumentResponse(
                id="doc456",
                content="Content",
                metadata=None,
            )
        )
        mock_get_client.return_value = mock_client
        
        result = await get_document.ainvoke({"document_id": "doc456"})
        
        assert result["success"] is True
        
        # Verify default collection was used
        mock_client.get_document.assert_called_once_with(
            document_id="doc456",
            collection_name="intramind_documents",  # default
        )


@pytest.mark.asyncio
async def test_get_document_not_found():
    """Test get_document when document doesn't exist."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.get_document = AsyncMock(
            side_effect=Exception("Document not found")
        )
        mock_get_client.return_value = mock_client
        
        result = await get_document.ainvoke({"document_id": "nonexistent"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Document not found" in result["error"]


# ============================================================================
# LIST_COLLECTIONS TOOL TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_list_collections_success():
    """Test successful collection listing."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.list_collections = AsyncMock(
            return_value=[
                CollectionResponse(
                    name="intramind_documents",
                    description="Main document collection",
                ),
                CollectionResponse(
                    name="test_collection",
                    description="Test documents",
                ),
                CollectionResponse(
                    name="archive",
                    description=None,
                ),
            ]
        )
        mock_get_client.return_value = mock_client
        
        result = await list_collections.ainvoke({})
        
        assert result["success"] is True
        assert len(result["collections"]) == 3
        
        # Verify first collection
        assert result["collections"][0]["name"] == "intramind_documents"
        assert result["collections"][0]["description"] == "Main document collection"
        
        # Verify collection with no description
        assert result["collections"][2]["name"] == "archive"
        assert result["collections"][2]["description"] is None


@pytest.mark.asyncio
async def test_list_collections_empty():
    """Test list_collections when no collections exist."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.list_collections = AsyncMock(return_value=[])
        mock_get_client.return_value = mock_client
        
        result = await list_collections.ainvoke({})
        
        assert result["success"] is True
        assert len(result["collections"]) == 0


@pytest.mark.asyncio
async def test_list_collections_handles_error():
    """Test that list_collections handles API errors gracefully."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.list_collections = AsyncMock(
            side_effect=Exception("API unavailable")
        )
        mock_get_client.return_value = mock_client
        
        result = await list_collections.ainvoke({})
        
        assert result["success"] is False
        assert "error" in result
        assert "API unavailable" in result["error"]


# ============================================================================
# CREATE_COLLECTION TOOL TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_create_collection_success():
    """Test successful collection creation."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.create_collection = AsyncMock(
            return_value=CollectionResponse(
                name="new_collection",
                description="A new collection for testing",
            )
        )
        mock_get_client.return_value = mock_client
        
        result = await create_collection.ainvoke({
            "name": "new_collection",
            "description": "A new collection for testing",
        })
        
        assert result["success"] is True
        assert result["name"] == "new_collection"
        assert "created successfully" in result["message"].lower()
        
        # Verify API client was called correctly
        mock_client.create_collection.assert_called_once_with(
            name="new_collection",
            description="A new collection for testing",
        )


@pytest.mark.asyncio
async def test_create_collection_without_description():
    """Test create_collection without description."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.create_collection = AsyncMock(
            return_value=CollectionResponse(
                name="minimal_collection",
                description=None,
            )
        )
        mock_get_client.return_value = mock_client
        
        result = await create_collection.ainvoke({"name": "minimal_collection"})
        
        assert result["success"] is True
        assert result["name"] == "minimal_collection"
        
        # Verify description was None
        mock_client.create_collection.assert_called_once_with(
            name="minimal_collection",
            description=None,
        )


@pytest.mark.asyncio
async def test_create_collection_already_exists():
    """Test create_collection when collection already exists."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.create_collection = AsyncMock(
            side_effect=Exception("Collection already exists")
        )
        mock_get_client.return_value = mock_client
        
        result = await create_collection.ainvoke({"name": "existing_collection"})
        
        assert result["success"] is False
        assert "error" in result
        assert "already exists" in result["error"].lower()


@pytest.mark.asyncio
async def test_create_collection_handles_error():
    """Test that create_collection handles API errors gracefully."""
    with patch("tools.agent_tools.get_api_client") as mock_get_client:
        mock_client = Mock()
        mock_client.create_collection = AsyncMock(
            side_effect=Exception("Permission denied")
        )
        mock_get_client.return_value = mock_client
        
        result = await create_collection.ainvoke({"name": "test"})
        
        assert result["success"] is False
        assert "error" in result
        assert "Permission denied" in result["error"]

