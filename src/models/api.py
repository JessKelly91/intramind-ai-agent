"""API models for IntraMind API Gateway integration."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def to_camel(string: str) -> str:
    """Convert snake_case to camelCase."""
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


# Collection Models
class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str = Field(..., description="Collection name")
    description: str | None = Field(None, description="Collection description")
    properties: dict[str, Any] | None = Field(
        None, description="Schema properties for the collection"
    )


class CollectionResponse(BaseModel):
    """Response model for collection operations."""

    name: str
    description: str | None = None
    vector_config: dict[str, Any] | None = None
    properties: dict[str, Any] | None = None


# Document Models
class DocumentInsert(BaseModel):
    """Request model for inserting a document."""

    collection_name: str = Field(..., description="Target collection name")
    content: str = Field(..., description="Document content")
    metadata: dict[str, Any] | None = Field(None, description="Document metadata")
    id: str | None = Field(None, description="Optional document ID")


class DocumentBatchInsert(BaseModel):
    """Request model for batch inserting documents."""

    collection_name: str = Field(..., description="Target collection name")
    documents: list[dict[str, Any]] = Field(..., description="List of documents to insert")


class DocumentResponse(BaseModel):
    """Response model for document operations."""

    id: str
    content: str
    metadata: dict[str, Any] | None = None


class DocumentUpdate(BaseModel):
    """Request model for updating a document."""

    collection_name: str
    content: str | None = None
    metadata: dict[str, Any] | None = None


# Search Models
class SearchRequest(BaseModel):
    """Request model for semantic search."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    collection_name: str = Field(..., description="Collection to search")
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")
    metadata_filters: dict[str, Any] | None = Field(None, description="Metadata filters", alias="metadataFilters")


class SearchResult(BaseModel):
    """Individual search result."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    document_id: str = Field(alias="documentId")
    content: str
    metadata: dict[str, Any] | None = None
    score: float | None = None
    collection_name: str | None = Field(None, alias="collectionName")


class SearchResponse(BaseModel):
    """Response model for search operations."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    results: list[SearchResult]
    total_count: int = Field(alias="totalCount")
    execution_time_ms: float | None = Field(None, alias="executionTimeMs")
    searched_collections: list[str] | None = Field(None, alias="searchedCollections")
    partial_results: bool | None = Field(None, alias="partialResults")
    warnings: list[str] | None = None


# Health Check Models
class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str
    timestamp: str | None = None
    details: dict[str, Any] | None = None
