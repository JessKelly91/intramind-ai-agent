"""Document Search Workflow using LangGraph."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from models.state import SearchWorkflowState
from tools import get_api_client
from utils.llm import get_primary_llm, get_router_llm

logger = logging.getLogger(__name__)


# Node Functions
async def classify_query(state: SearchWorkflowState) -> SearchWorkflowState:
    """Classify the query complexity to determine search strategy.

    Args:
        state: Current workflow state

    Returns:
        Updated state with query classification
    """
    logger.info("Node: classify_query")

    query = state["user_query"]
    router_llm = get_router_llm()

    # Prompt for classification
    system_prompt = """You are a query classifier for a document search system.
    Classify the user's query as either 'simple' or 'complex'.

    Simple queries:
    - Direct fact lookups
    - Single-concept searches
    - Questions that can be answered with one search

    Complex queries:
    - Multi-part questions
    - Queries requiring aggregation of multiple documents
    - Comparative or analytical questions

    Respond with ONLY 'simple' or 'complex', nothing else."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Classify this query: {query}"),
    ]

    response = await router_llm.ainvoke(messages)
    complexity = response.content.strip().lower()

    # Ensure valid response
    if complexity not in ["simple", "complex"]:
        logger.warning(f"Invalid classification: {complexity}, defaulting to simple")
        complexity = "simple"

    logger.info(f"Query classified as: {complexity}")

    return {
        **state,
        "current_step": "classify_query",
        "query_complexity": complexity,
        "next_step": "simple_search" if complexity == "simple" else "complex_search",
    }


async def simple_search(state: SearchWorkflowState) -> SearchWorkflowState:
    """Perform a straightforward semantic search.

    Args:
        state: Current workflow state

    Returns:
        Updated state with search results
    """
    logger.info("Node: simple_search")

    query = state["user_query"]
    client = get_api_client()

    try:
        # Perform search
        response = await client.search(
            collection_name=state.get("document_metadata", {}).get(
                "collection_name", "intramind_documents"
            ),
            query=query,
            limit=state.get("num_results", 10),
        )

        results = [
            {
                "id": result.id,
                "content": result.content,
                "metadata": result.metadata,
                "score": result.score,
            }
            for result in response.results
        ]

        logger.info(f"Found {len(results)} results")

        return {
            **state,
            "current_step": "simple_search",
            "search_results": results,
            "next_step": "synthesize_results",
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            **state,
            "current_step": "simple_search",
            "error": str(e),
            "next_step": "handle_error",
        }


async def complex_search(state: SearchWorkflowState) -> SearchWorkflowState:
    """Perform complex search with query expansion and multi-query.

    Args:
        state: Current workflow state

    Returns:
        Updated state with aggregated search results
    """
    logger.info("Node: complex_search")

    query = state["user_query"]
    router_llm = get_router_llm()

    # Generate expanded queries
    system_prompt = """You are a query expansion expert for document search.
    Given a complex query, generate 2-3 related search queries that will help find relevant documents.
    Each query should focus on a different aspect of the original question.

    Format your response as a numbered list:
    1. [first query]
    2. [second query]
    3. [third query]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Expand this query: {query}"),
    ]

    response = await router_llm.ainvoke(messages)

    # Parse expanded queries
    expanded_queries = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove numbering
            expanded_query = line.split(".", 1)[1].strip() if "." in line else line
            expanded_queries.append(expanded_query)

    logger.info(f"Expanded into {len(expanded_queries)} queries")

    # Perform searches for each expanded query
    client = get_api_client()
    all_results = []
    seen_ids = set()

    for expanded_query in expanded_queries:
        try:
            response = await client.search(
                collection_name=state.get("document_metadata", {}).get(
                    "collection_name", "intramind_documents"
                ),
                query=expanded_query,
                limit=5,  # Fewer per query since we're doing multiple
            )

            for result in response.results:
                # Deduplicate by ID
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    all_results.append(
                        {
                            "id": result.id,
                            "content": result.content,
                            "metadata": result.metadata,
                            "score": result.score,
                        }
                    )

        except Exception as e:
            logger.warning(f"Search failed for query '{expanded_query}': {e}")

    # Sort by score (if available) and limit
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_results = all_results[: state.get("num_results", 10)]

    logger.info(f"Found {len(all_results)} unique results across all queries")

    return {
        **state,
        "current_step": "complex_search",
        "expanded_queries": expanded_queries,
        "search_results": all_results,
        "next_step": "synthesize_results",
    }


async def synthesize_results(state: SearchWorkflowState) -> SearchWorkflowState:
    """Synthesize search results into a coherent response.

    Args:
        state: Current workflow state

    Returns:
        Updated state with synthesized response
    """
    logger.info("Node: synthesize_results")

    query = state["user_query"]
    results = state.get("search_results", [])

    if not results:
        return {
            **state,
            "current_step": "synthesize_results",
            "response": "I couldn't find any relevant documents for your query.",
            "citations": [],
            "next_step": None,
            "workflow_complete": True,
        }

    primary_llm = get_primary_llm()

    # Prepare context from search results
    context_parts = []
    citations = []

    for i, result in enumerate(results[:5], 1):  # Use top 5 for context
        content = result["content"][:500]  # Truncate long content
        context_parts.append(f"[Document {i}]\n{content}\n")
        citations.append(result["id"])

    context = "\n".join(context_parts)

    # Synthesis prompt
    system_prompt = """You are a helpful assistant that answers questions based on document search results.
    Use the provided documents to answer the user's question.
    Be concise and accurate. If the documents don't contain enough information, say so.
    Mention which document numbers support your answer (e.g., "According to Documents 1 and 2...")."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Question: {query}\n\nRelevant Documents:\n{context}\n\nPlease provide a comprehensive answer."
        ),
    ]

    response = await primary_llm.ainvoke(messages)
    answer = response.content

    logger.info("Response synthesized successfully")

    return {
        **state,
        "current_step": "synthesize_results",
        "response": answer,
        "citations": citations,
        "next_step": None,
        "workflow_complete": True,
    }


async def handle_error(state: SearchWorkflowState) -> SearchWorkflowState:
    """Handle errors in the workflow.

    Args:
        state: Current workflow state

    Returns:
        Updated state with error response
    """
    logger.error(f"Node: handle_error - {state.get('error')}")

    return {
        **state,
        "current_step": "handle_error",
        "response": f"I encountered an error while processing your request: {state.get('error', 'Unknown error')}",
        "next_step": None,
        "workflow_complete": True,
    }


# Routing Functions
def route_after_classification(
    state: SearchWorkflowState,
) -> Literal["simple_search", "complex_search"]:
    """Route to appropriate search node based on classification.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    complexity = state.get("query_complexity", "simple")
    return "simple_search" if complexity == "simple" else "complex_search"


def route_after_search(state: SearchWorkflowState) -> Literal["synthesize_results", "handle_error"]:
    """Route after search based on success/failure.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("error"):
        return "handle_error"
    return "synthesize_results"


# Build the workflow graph
def create_search_workflow() -> StateGraph:
    """Create and compile the document search workflow.

    Returns:
        Compiled LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(SearchWorkflowState)

    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("simple_search", simple_search)
    workflow.add_node("complex_search", complex_search)
    workflow.add_node("synthesize_results", synthesize_results)
    workflow.add_node("handle_error", handle_error)

    # Set entry point
    workflow.set_entry_point("classify_query")

    # Add edges
    workflow.add_conditional_edges("classify_query", route_after_classification)
    workflow.add_conditional_edges("simple_search", route_after_search)
    workflow.add_conditional_edges("complex_search", route_after_search)
    workflow.add_edge("synthesize_results", END)
    workflow.add_edge("handle_error", END)

    # Compile and return
    return workflow.compile()


# Export compiled workflow
search_workflow = create_search_workflow()
