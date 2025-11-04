"""Main agent interface for IntraMind."""

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

from models.state import IngestionWorkflowState, SearchWorkflowState
from utils.metrics import track_ingestion, track_query
from workflows.ingestion_workflow import ingestion_workflow
from workflows.search_workflow import search_workflow

logger = logging.getLogger(__name__)


class IntraMindAgent:
    """High-level interface for IntraMind AI Agent."""

    def __init__(self):
        """Initialize the IntraMind agent."""
        self.search_workflow = search_workflow
        self.ingestion_workflow = ingestion_workflow
        logger.info("IntraMind agent initialized")

    @track_query
    async def search(
        self,
        query: str,
        collection_name: str = "intramind_documents",
        num_results: int = 10,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Perform a document search.

        Args:
            query: User's search query
            collection_name: Collection to search in
            num_results: Maximum number of results
            min_score: Minimum similarity score threshold (0.0-1.0)

        Returns:
            Search response with results and synthesis
        """
        logger.info(f"Executing search: {query}")

        # Initialize state
        initial_state: SearchWorkflowState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "current_step": "start",
            "next_step": None,
            "workflow_complete": False,
            "search_strategy": None,
            "search_query": None,
            "search_results": None,
            "num_results": num_results,
            "min_score": min_score,
            "document_path": None,
            "document_type": None,
            "extracted_content": None,
            "document_metadata": {"collection_name": collection_name},
            "response": None,
            "citations": None,
            "error": None,
            "retry_count": 0,
            "query_complexity": None,
            "expanded_queries": None,
            "aggregated_results": None,
        }

        # Execute workflow
        try:
            result = await self.search_workflow.ainvoke(initial_state)

            return {
                "success": True,
                "query": query,
                "response": result.get("response"),
                "citations": result.get("citations", []),
                "results": result.get("search_results", []),
                "complexity": result.get("query_complexity"),
                "expanded_queries": result.get("expanded_queries"),
            }

        except Exception as e:
            logger.error(f"Search workflow failed: {e}", exc_info=True)
            return {
                "success": False,
                "query": query,
                "error": str(e),
            }

    async def stream_search(
        self,
        query: str,
        collection_name: str = "intramind_documents",
        num_results: int = 10,
        min_score: float = 0.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream search results as they're generated.

        Args:
            query: User's search query
            collection_name: Collection to search in
            num_results: Maximum number of results
            min_score: Minimum similarity score threshold (0.0-1.0)

        Yields:
            State updates as the workflow progresses
        """
        import time
        from utils.metrics import _load_metrics, _save_metrics
        
        logger.info(f"Streaming search: {query}")
        
        # Track metrics - start
        start_time = time.time()
        metrics = _load_metrics()
        metrics["queries_total"] += 1
        
        # Track state for metrics
        final_complexity = None
        final_expanded_queries = []
        had_error = False

        # Initialize state
        initial_state: SearchWorkflowState = {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "current_step": "start",
            "next_step": None,
            "workflow_complete": False,
            "search_strategy": None,
            "search_query": None,
            "search_results": None,
            "num_results": num_results,
            "min_score": min_score,
            "document_path": None,
            "document_type": None,
            "extracted_content": None,
            "document_metadata": {"collection_name": collection_name},
            "response": None,
            "citations": None,
            "error": None,
            "retry_count": 0,
            "query_complexity": None,
            "expanded_queries": None,
            "aggregated_results": None,
        }

        # Stream workflow execution
        try:
            async for event in self.search_workflow.astream(initial_state):
                # Extract the node name and state
                for node_name, node_state in event.items():
                    # Track complexity and expanded queries for metrics
                    if node_state.get("query_complexity"):
                        final_complexity = node_state["query_complexity"]
                    if node_state.get("expanded_queries"):
                        final_expanded_queries = node_state["expanded_queries"]
                    if node_state.get("error"):
                        had_error = True
                    
                    yield {
                        "node": node_name,
                        "step": node_state.get("current_step"),
                        "complexity": node_state.get("query_complexity"),
                        "results_count": (
                            len(node_state.get("search_results", []))
                            if node_state.get("search_results")
                            else 0
                        ),
                        "response": node_state.get("response"),
                        "error": node_state.get("error"),
                        "complete": node_state.get("workflow_complete", False),
                    }
            
            # Track metrics - end (success)
            if final_complexity == "simple":
                metrics["queries_simple"] += 1
                metrics["llm_calls_router"] += 1
            elif final_complexity == "complex":
                metrics["queries_complex"] += 1
                metrics["llm_calls_router"] += 1
                metrics["llm_calls_primary"] += 1 + len(final_expanded_queries)
            
            latency_ms = (time.time() - start_time) * 1000
            metrics["total_latency_ms"] += latency_ms
            
            if had_error:
                metrics["errors_total"] += 1
            
            _save_metrics(metrics)

        except Exception as e:
            logger.error(f"Streaming search failed: {e}", exc_info=True)
            metrics["errors_total"] += 1
            
            latency_ms = (time.time() - start_time) * 1000
            metrics["total_latency_ms"] += latency_ms
            
            _save_metrics(metrics)
            
            yield {
                "node": "error",
                "error": str(e),
                "complete": True,
            }

    @track_ingestion
    async def ingest_document(
        self,
        file_path: str,
        collection_name: str = "intramind_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        document_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingest a document into the system.

        Processes a document through the ingestion workflow:
        1. Validates the document (file exists, size, format)
        2. Extracts content (supports PDF, DOCX, PPTX, TXT, images)
        3. Chunks content intelligently (semantic boundaries)
        4. Stores chunks in vector database

        Args:
            file_path: Path to the document file
            collection_name: Target collection for storage
            chunk_size: Size of text chunks in characters (default: 1000)
            chunk_overlap: Overlap between chunks in characters (default: 200)
            document_metadata: Additional metadata to attach to all chunks

        Returns:
            Dictionary containing:
                - success: Whether ingestion succeeded
                - file_path: Path to ingested file
                - file_name: Name of the file
                - file_type: Type of file (pdf, docx, etc.)
                - chunks_created: Number of chunks created
                - chunks_stored: Number of chunks successfully stored
                - inserted_ids: List of document IDs in vector database
                - metadata: Complete document metadata
                - error: Error message (if failed)

        Example:
            >>> agent = IntraMindAgent()
            >>> result = await agent.ingest_document(
            ...     file_path="./reports/Q4_2024.pdf",
            ...     collection_name="financial_reports",
            ...     document_metadata={"year": 2024, "quarter": "Q4"}
            ... )
            >>> if result["success"]:
            ...     print(f"Stored {result['chunks_stored']} chunks")
        """
        logger.info(f"Ingesting document: {file_path}")

        # Initialize state
        initial_state: IngestionWorkflowState = {
            "messages": [],
            "user_query": "",
            "current_step": "start",
            "next_step": None,
            "workflow_complete": False,
            "search_strategy": None,
            "search_query": None,
            "search_results": None,
            "num_results": 0,
            "document_path": None,
            "document_type": None,
            "extracted_content": None,
            "document_metadata": document_metadata or {},
            "response": None,
            "citations": None,
            "error": None,
            "retry_count": 0,
            # Ingestion-specific fields
            "file_path": file_path,
            "collection_name": collection_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks": None,
            "inserted_ids": None,
        }

        # Execute workflow
        try:
            result = await self.ingestion_workflow.ainvoke(initial_state)

            # Check for errors
            if result.get("error"):
                logger.error(f"Ingestion failed: {result['error']}")
                return {
                    "success": False,
                    "file_path": file_path,
                    "error": result["error"],
                }

            # Extract results
            metadata = result.get("document_metadata", {})
            inserted_ids = result.get("inserted_ids", [])
            chunks = result.get("chunks", [])

            logger.info(
                f"Ingestion complete: {len(inserted_ids)}/{len(chunks)} chunks stored"
            )

            return {
                "success": True,
                "file_path": file_path,
                "file_name": metadata.get("filename", ""),
                "file_type": metadata.get("file_type", ""),
                "file_size_bytes": metadata.get("file_size_bytes", 0),
                "collection_name": collection_name,
                "chunks_created": len(chunks),
                "chunks_stored": len(inserted_ids),
                "inserted_ids": inserted_ids,
                "metadata": metadata,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

        except Exception as e:
            logger.error(f"Ingestion workflow failed: {e}", exc_info=True)
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
            }
