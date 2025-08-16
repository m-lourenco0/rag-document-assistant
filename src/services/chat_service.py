import logging
import markdown2

from langgraph.graph.state import CompiledStateGraph

from src.agent.agent import run_chat

logger = logging.getLogger(__name__)


def get_agent_response(
    message: str, thread_id: str, agent_graph: CompiledStateGraph
) -> dict:
    """
    Runs the agent to get a response and references.
    Handles exceptions internally.
    """
    try:
        response_data = run_chat(message, thread_id, agent_graph)
        return response_data
    except Exception:
        logger.error(
            f"An error occurred running the agent for thread_id: {thread_id}",
            exc_info=True,
        )
        return {
            "answer": "I'm sorry, but an unexpected error occurred. Please try again.",
            "references": [],
        }


def format_response_for_htmx(response_data: dict) -> dict:
    """Formats the agent's response into HTML for the HTMX frontend."""
    html_from_markdown = markdown2.markdown(
        response_data["answer"], extras=["tables", "cuddled-lists", "breaks"]
    )
    return {
        "llm_response": html_from_markdown,
        "references": response_data["references"],
    }
