"""Main agent interface for IntraMind."""

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

from models.state import SearchWorkflowState
from workflows.search_workflow import search_workflow

logger = logging.getLogger(__name__)


class IntraMindAgent:
    """High-level interface for IntraMind AI Agent."""

    def __init__(self):
        """Initialize the IntraMind agent."""
        self.search_workflow = search_workflow
        logger.info("IntraMind agent initialized")

    async def search(
        self, query: str, collection_name: str = "intramind_documents", num_results: int = 10
    ) -> dict[str, Any]:
        """Perform a document search.

        Args:
            query: User's search query
            collection_name: Collection to search in
            num_results: Maximum number of results

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
        self, query: str, collection_name: str = "intramind_documents", num_results: int = 10
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream search results as they're generated.

        Args:
            query: User's search query
            collection_name: Collection to search in
            num_results: Maximum number of results

        Yields:
            State updates as the workflow progresses
        """
        logger.info(f"Streaming search: {query}")

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

        except Exception as e:
            logger.error(f"Streaming search failed: {e}", exc_info=True)
            yield {
                "node": "error",
                "error": str(e),
                "complete": True,
            }
