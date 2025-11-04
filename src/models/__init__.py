"""Models for IntraMind AI Agent."""

from .api import (
    CollectionCreate,
    CollectionResponse,
    DocumentInsert,
    DocumentResponse,
    DocumentUpdate,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .state import (
    AgentState,
    IngestionWorkflowState,
    MultimodalWorkflowState,
    SearchWorkflowState,
)

__all__ = [
    # API Models
    "CollectionCreate",
    "CollectionResponse",
    "DocumentInsert",
    "DocumentResponse",
    "DocumentUpdate",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "HealthResponse",
    # State Models
    "AgentState",
    "SearchWorkflowState",
    "IngestionWorkflowState",
    "MultimodalWorkflowState",
]
