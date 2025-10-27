"""State definitions for LangGraph workflows."""

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Base state for all agent workflows."""

    # Messages history (LangGraph built-in message handling)
    messages: Annotated[list, add_messages]

    # User input
    user_query: str

    # Workflow control
    current_step: str
    next_step: str | None
    workflow_complete: bool

    # Search-related state
    search_strategy: str | None  # "simple" | "complex" | "multi_query"
    search_query: str | None
    search_results: list[dict[str, Any]] | None
    num_results: int

    # Document processing state
    document_path: str | None
    document_type: str | None  # "pdf" | "image" | "pptx" | "text"
    extracted_content: str | None
    document_metadata: dict[str, Any] | None

    # Response generation
    response: str | None
    citations: list[str] | None

    # Error handling
    error: str | None
    retry_count: int


class SearchWorkflowState(AgentState):
    """State for document search workflow."""

    query_complexity: str | None  # "simple" | "complex"
    expanded_queries: list[str] | None
    aggregated_results: list[dict[str, Any]] | None


class IngestionWorkflowState(AgentState):
    """State for document ingestion workflow."""

    file_path: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    chunks: list[dict[str, Any]] | None
    inserted_ids: list[str] | None


class MultimodalWorkflowState(AgentState):
    """State for multimodal document processing workflow."""

    file_type: str  # "pdf" | "image" | "pptx" | "text"
    ocr_enabled: bool
    extracted_text: str | None
    extracted_images: list[dict[str, Any]] | None
    processing_metadata: dict[str, Any] | None
