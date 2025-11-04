"""API Gateway client for IntraMind."""

import logging
from typing import Any
import uuid

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from models.api import (
    CollectionCreate,
    CollectionResponse,
    DocumentBatchInsert,
    DocumentInsert,
    DocumentResponse,
    DocumentUpdate,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)


class APIGatewayClient:
    """Client for interacting with IntraMind API Gateway."""

    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        """Initialize API Gateway client.

        Args:
            base_url: Base URL for API Gateway (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
        """
        self.base_url = (base_url or settings.api_gateway_url).rstrip("/")
        self.timeout = timeout or settings.api_gateway_timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "APIGatewayClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # Health Check Methods
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def health_check(self) -> dict[str, Any]:
        """Check API Gateway health.

        Returns:
            Health status response

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.debug("Checking API Gateway health")
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()

    # Collection Methods
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def list_collections(self) -> list[CollectionResponse]:
        """List all collections.

        Returns:
            List of collections

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.debug("Listing collections")
        response = await self.client.get("/v1/collections")
        response.raise_for_status()
        data = response.json()
        return [CollectionResponse(**col) for col in data.get("collections", [])]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_collection(self, name: str) -> CollectionResponse:
        """Get collection details.

        Args:
            name: Collection name

        Returns:
            Collection details

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.debug(f"Getting collection: {name}")
        response = await self.client.get(f"/v1/collections/{name}")
        response.raise_for_status()
        return CollectionResponse(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_collection(
        self, name: str, description: str | None = None, properties: dict[str, Any] | None = None
    ) -> CollectionResponse:
        """Create a new collection.

        Args:
            name: Collection name
            description: Optional description
            properties: Optional schema properties

        Returns:
            Created collection details

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.info(f"Creating collection: {name}")
        request = CollectionCreate(name=name, description=description, properties=properties)
        response = await self.client.post("/v1/collections", json=request.model_dump())
        response.raise_for_status()
        return CollectionResponse(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def delete_collection(self, name: str) -> dict[str, Any]:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            Deletion confirmation

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.warning(f"Deleting collection: {name}")
        response = await self.client.delete(f"/v1/collections/{name}")
        response.raise_for_status()
        return response.json()

    # Document Methods
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def insert_document(
        self,
        collection_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> DocumentResponse:
        """Insert a document into a collection.

        The API Gateway expects fields: DocumentId, CollectionName, Content, Metadata.
        """
        logger.info(f"Inserting document into {collection_name}")

        doc_id = document_id or str(uuid.uuid4())
        payload = {
            "DocumentId": doc_id,
            "CollectionName": collection_name,
            "Content": content,
            "Metadata": metadata or {},
        }

        response = await self.client.post("/v1/documents", json=payload)
        response.raise_for_status()
        data = response.json()
        return DocumentResponse(
            id=data.get("documentId") or data.get("id") or doc_id,
            content=data.get("content", ""),
            metadata=data.get("metadata"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def insert_documents_batch(
        self, collection_name: str, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Insert multiple documents in batch.

        Args:
            collection_name: Target collection
            documents: List of documents to insert

        Returns:
            Batch insertion results

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.info(f"Batch inserting {len(documents)} documents into {collection_name}")
        request = DocumentBatchInsert(collection_name=collection_name, documents=documents)
        response = await self.client.post("/v1/documents/batch", json=request.model_dump())
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_document(self, document_id: str, collection_name: str) -> DocumentResponse:
        """Get a document by ID.

        Args:
            document_id: Document ID
            collection_name: Collection name

        Returns:
            Document details

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.debug(f"Getting document {document_id} from {collection_name}")
        response = await self.client.get(
            f"/v1/documents/{document_id}", params={"collection_name": collection_name}
        )
        response.raise_for_status()
        return DocumentResponse(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def update_document(
        self,
        document_id: str,
        collection_name: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentResponse:
        """Update a document.

        Args:
            document_id: Document ID
            collection_name: Collection name
            content: New content (optional)
            metadata: New metadata (optional)

        Returns:
            Updated document details

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.info(f"Updating document {document_id} in {collection_name}")
        request = DocumentUpdate(collection_name=collection_name, content=content, metadata=metadata)
        response = await self.client.put(
            f"/v1/documents/{document_id}", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return DocumentResponse(**response.json())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def delete_document(self, document_id: str, collection_name: str) -> dict[str, Any]:
        """Delete a document.

        Args:
            document_id: Document ID
            collection_name: Collection name

        Returns:
            Deletion confirmation

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.warning(f"Deleting document {document_id} from {collection_name}")
        response = await self.client.delete(
            f"/v1/documents/{document_id}", params={"collection_name": collection_name}
        )
        response.raise_for_status()
        return response.json()

    # Search Methods
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Perform semantic search.

        Args:
            collection_name: Collection to search
            query: Search query
            limit: Number of results
            min_score: Minimum similarity score threshold (0.0-1.0)
            filter: Optional metadata filters

        Returns:
            Search results

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        logger.info(f"Searching {collection_name} for: {query[:50]}...")
        request = SearchRequest(
            collection_name=collection_name, query=query, limit=limit, min_score=min_score, metadata_filters=filter
        )
        response = await self.client.post("/v1/search", json=request.model_dump(by_alias=True, exclude_none=True))
        response.raise_for_status()
        return SearchResponse(**response.json())
