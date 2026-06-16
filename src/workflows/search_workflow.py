"""Document Search Workflow using LangGraph."""

import logging
from datetime import datetime, timezone
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from config import settings
from models.state import SearchWorkflowState
from prompts.client import get_prompt
from prompts.registry import annotate_span
from tools import get_api_client
from utils.llm import get_primary_llm, get_router_llm
from utils.metrics import record_safety_flag
from utils.safety import classify_output

logger = logging.getLogger(__name__)


def _result_score(result: dict) -> float | None:
    """Best-effort numeric score extraction from a search result."""
    score = result.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _select_synthesis_results(results: list[dict], query_complexity: str | None) -> list[dict]:
    """Select a compact context set for answer synthesis.

    Direct factual queries should not force the LLM to reason over obvious
    distractors. Keep the top hit, then include only close-scoring neighbors up
    to the configured cap. Complex queries can use the top configured cap.
    """
    if not results:
        return []

    max_contexts = max(1, settings.synthesis_max_contexts)
    if query_complexity == "complex":
        return results[:max_contexts]

    selected = [results[0]]
    top_score = _result_score(results[0])
    for result in results[1:]:
        if len(selected) >= max_contexts:
            break

        score = _result_score(result)
        if (
            top_score is None
            or score is None
            or top_score - score <= settings.synthesis_score_gap
        ):
            selected.append(result)

    return selected


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

    # Prompt for classification (runtime registry with baked-in fallback)
    prompt = get_prompt("query_classifier")
    annotate_span(prompt)
    system_prompt = prompt.template

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

    # Determine if conversation context should be used (smart cost optimization)
    use_context = state.get("use_conversation_context", False)
    if settings.smart_context_selection:
        # Only use conversation history for complex queries to save costs
        use_context = complexity == "complex"
    elif settings.enable_conversation_memory:
        # If smart selection is disabled but memory is enabled, always use context
        use_context = True

    return {
        **state,
        "current_step": "classify_query",
        "query_complexity": complexity,
        "use_conversation_context": use_context,
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
            min_score=state.get("min_score", 0.0),
        )

        results = [
            {
                "id": result.document_id,
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

    # Generate expanded queries (runtime registry with baked-in fallback)
    prompt = get_prompt("query_expansion")
    annotate_span(prompt)
    system_prompt = prompt.template

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
                min_score=state.get("min_score", 0.0),
            )

            for result in response.results:
                # Deduplicate by ID
                if result.document_id not in seen_ids:
                    seen_ids.add(result.document_id)
                    all_results.append(
                        {
                            "id": result.document_id,
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
            "synthesis_results": [],
            "next_step": "safety_check",
        }

    primary_llm = get_primary_llm()

    # Prepare context from the strongest supporting search results.
    context_parts = []
    citations = []
    selected_results = _select_synthesis_results(
        results, state.get("query_complexity")
    )

    for i, result in enumerate(selected_results, 1):
        content = result["content"][: settings.synthesis_context_chars]
        context_parts.append(f"[Document {i}]\n{content}\n")
        citations.append(result["id"])

    context = "\n".join(context_parts)

    # Build conversation messages
    messages = []
    
    # Add system prompt (runtime registry with baked-in fallback)
    prompt = get_prompt("result_synthesis")
    annotate_span(prompt)
    system_prompt = prompt.template
    messages.append(SystemMessage(content=system_prompt))
    
    # Include conversation history if enabled (smart cost optimization)
    use_context = state.get("use_conversation_context", False)
    if use_context and settings.enable_conversation_memory:
        # Get recent conversation history (excluding current query)
        history_messages = state.get("messages", [])[:-1]  # Exclude current HumanMessage
        max_history = settings.max_conversation_history * 2  # 2 messages per turn (user + assistant)
        
        if history_messages:
            recent_history = history_messages[-max_history:] if len(history_messages) > max_history else history_messages
            messages.extend(recent_history)
            logger.info(f"Including {len(recent_history)} messages from conversation history")
    
    # Add current query with document context
    messages.append(
        HumanMessage(
            content=f"Question: {query}\n\nRelevant Documents:\n{context}\n\nPlease provide a direct, supported answer."
        )
    )

    response = await primary_llm.ainvoke(messages)
    answer = response.content

    logger.info("Response synthesized successfully")

    return {
        **state,
        "current_step": "synthesize_results",
        "response": answer,
        "citations": citations,
        "synthesis_results": selected_results,
        "next_step": "safety_check",
    }


async def safety_check(state: SearchWorkflowState) -> SearchWorkflowState:
    """Screen the synthesized response with Llama Guard before returning.

    Step 4 of the Free RAI Stack. Hard-block policy: if the classifier
    flags the response as unsafe, route to ``handle_unsafe_response`` which
    discards the original answer + citations and substitutes a templated
    fallback. The original (flagged) text is never returned to the user.
    """
    logger.info("Node: safety_check")

    # Allow opting out via settings (e.g. for tests or environments without Ollama).
    if not settings.enable_safety_guard:
        return {
            **state,
            "current_step": "safety_check",
            "safety_flag": {"flagged": False, "categories": [], "checked_at": None},
            "next_step": None,
            "workflow_complete": True,
        }

    response_text = state.get("response") or ""
    if not response_text:
        # Nothing to classify - end the workflow.
        return {
            **state,
            "current_step": "safety_check",
            "safety_flag": {"flagged": False, "categories": [], "checked_at": None},
            "next_step": None,
            "workflow_complete": True,
        }

    result = await classify_output(
        prompt=state.get("user_query") or "",
        response=response_text,
        model=settings.safety_guard_model,
    )

    checked_at = datetime.now(timezone.utc).isoformat()
    flag_meta = result.to_metadata(checked_at=checked_at)

    if result.is_safe:
        return {
            **state,
            "current_step": "safety_check",
            "safety_flag": flag_meta,
            "next_step": None,
            "workflow_complete": True,
        }

    # Unsafe - record the block and route to the handler that strips the
    # original response. We do NOT keep the flagged text in state past this
    # node so it can never be exfiltrated by a downstream consumer.
    logger.warning(
        "Safety guard flagged response (categories=%s) - hard-blocking",
        result.categories,
    )
    record_safety_flag(result.categories)

    # Mark the current OTEL span (if tracing is enabled) so Phoenix shows
    # blocked traces clearly. No-op when tracing is disabled.
    try:
        from opentelemetry import trace as _otel_trace

        _span = _otel_trace.get_current_span()
        if _span is not None and _span.is_recording():
            _span.set_attribute("safety.flagged", True)
            _span.set_attribute(
                "safety.categories", ",".join(result.categories) or "UNCATEGORIZED"
            )
    except Exception:  # noqa: BLE001 - tracing must never break the request path
        pass
    return {
        **state,
        "current_step": "safety_check",
        "safety_flag": flag_meta,
        "next_step": "handle_unsafe_response",
    }


async def handle_unsafe_response(state: SearchWorkflowState) -> SearchWorkflowState:
    """Replace a flagged response with the templated fallback message.

    Hard-block policy:
      * Replace ``response`` with ``settings.safety_fallback_message``.
      * Discard ``citations`` so flagged sources aren't leaked indirectly.
      * Keep ``safety_flag`` for observability (Phoenix span attribute).
      * Do NOT preserve the original response anywhere in state.
    """
    logger.info("Node: handle_unsafe_response")
    return {
        **state,
        "current_step": "handle_unsafe_response",
        "response": settings.safety_fallback_message,
        "citations": [],
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
def create_search_workflow(checkpointer=None):
    """Create and compile the document search workflow.
    
    Args:
        checkpointer: Optional checkpointer for conversation memory persistence
        
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
    workflow.add_node("safety_check", safety_check)
    workflow.add_node("handle_unsafe_response", handle_unsafe_response)
    workflow.add_node("handle_error", handle_error)

    # Set entry point
    workflow.set_entry_point("classify_query")

    # Add edges
    workflow.add_conditional_edges("classify_query", route_after_classification)
    workflow.add_conditional_edges("simple_search", route_after_search)
    workflow.add_conditional_edges("complex_search", route_after_search)
    # Synthesis always flows into the safety screen first.
    workflow.add_edge("synthesize_results", "safety_check")
    # The safety_check node itself decides whether to end or block.
    workflow.add_conditional_edges(
        "safety_check",
        lambda state: state.get("next_step") or "end",
        {
            "handle_unsafe_response": "handle_unsafe_response",
            "end": END,
        },
    )
    workflow.add_edge("handle_unsafe_response", END)
    workflow.add_edge("handle_error", END)

    # Compile with optional checkpointer
    return workflow.compile(checkpointer=checkpointer)


# Export compiled workflow (without checkpointer for backwards compatibility)
# Use create_search_workflow() with checkpointer for conversation memory
search_workflow = create_search_workflow()
