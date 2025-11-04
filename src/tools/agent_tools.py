"""LangChain tools for IntraMind AI Agent."""

import logging
from typing import Any

from langchain.tools import tool

from tools.api_client import APIGatewayClient

logger = logging.getLogger(__name__)

# Global API client instance
_api_client: APIGatewayClient | None = None


def get_api_client() -> APIGatewayClient:
    """Get or create API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = APIGatewayClient()
    return _api_client


@tool
async def search_documents(
    query: str, collection_name: str = "intramind_documents", limit: int = 10
) -> dict[str, Any]:
    """Search for documents using semantic search.

    Use this tool when the user wants to find documents or information.
    The search uses semantic similarity to find relevant documents.

    Args:
        query: The search query describing what to find
        collection_name: The collection to search in (default: intramind_documents)
        limit: Maximum number of results to return (default: 10)

    Returns:
        A dictionary containing search results with content and metadata
    """
    logger.info(f"Tool: search_documents - query: {query}, collection: {collection_name}")
    client = get_api_client()

    try:
        response = await client.search(
            collection_name=collection_name, query=query, limit=limit
        )

        return {
            "success": True,
            "total_results": response.total_count,
            "results": [
                {
                    "id": result.document_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                    "collection": result.collection_name,
                }
                for result in response.results
            ],
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"success": False, "error": str(e)}


@tool
async def insert_document(
    content: str,
    collection_name: str = "intramind_documents",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert a new document into the vector database.

    Use this tool when the user wants to add a new document or save information.

    Args:
        content: The document content to store
        collection_name: The collection to insert into (default: intramind_documents)
        metadata: Optional metadata about the document (e.g., title, author, date)

    Returns:
        A dictionary with the inserted document's ID and confirmation
    """
    logger.info(f"Tool: insert_document - collection: {collection_name}")
    client = get_api_client()

    try:
        response = await client.insert_document(
            collection_name=collection_name, content=content, metadata=metadata
        )

        return {
            "success": True,
            "document_id": response.id,
            "message": "Document inserted successfully",
        }
    except Exception as e:
        logger.error(f"Document insertion failed: {e}")
        return {"success": False, "error": str(e)}


@tool
async def get_document(document_id: str, collection_name: str = "intramind_documents") -> dict[str, Any]:
    """Retrieve a specific document by its ID.

    Use this tool when you need to get the full content of a specific document.

    Args:
        document_id: The unique ID of the document to retrieve
        collection_name: The collection containing the document (default: intramind_documents)

    Returns:
        A dictionary containing the document's content and metadata
    """
    logger.info(f"Tool: get_document - id: {document_id}, collection: {collection_name}")
    client = get_api_client()

    try:
        response = await client.get_document(
            document_id=document_id, collection_name=collection_name
        )

        return {
            "success": True,
            "id": response.id,
            "content": response.content,
            "metadata": response.metadata,
        }
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return {"success": False, "error": str(e)}


@tool
async def list_collections() -> dict[str, Any]:
    """List all available collections in the vector database.

    Use this tool when the user wants to know what collections exist
    or needs to choose a collection to work with.

    Returns:
        A dictionary containing a list of all collections with their details
    """
    logger.info("Tool: list_collections")
    client = get_api_client()

    try:
        collections = await client.list_collections()

        return {
            "success": True,
            "collections": [
                {
                    "name": col.name,
                    "description": col.description,
                }
                for col in collections
            ],
        }
    except Exception as e:
        logger.error(f"List collections failed: {e}")
        return {"success": False, "error": str(e)}


@tool
async def create_collection(
    name: str, description: str | None = None
) -> dict[str, Any]:
    """Create a new collection in the vector database.

    Use this tool when the user wants to create a new collection
    for organizing documents.

    Args:
        name: The name for the new collection
        description: Optional description of what the collection contains

    Returns:
        A dictionary confirming the collection was created
    """
    logger.info(f"Tool: create_collection - name: {name}")
    client = get_api_client()

    try:
        response = await client.create_collection(name=name, description=description)

        return {
            "success": True,
            "name": response.name,
            "message": f"Collection '{name}' created successfully",
        }
    except Exception as e:
        logger.error(f"Collection creation failed: {e}")
        return {"success": False, "error": str(e)}


# List of all available tools
AGENT_TOOLS = [
    search_documents,
    insert_document,
    get_document,
    list_collections,
    create_collection,
]
