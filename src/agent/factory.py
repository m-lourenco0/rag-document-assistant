import logging

from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.storage import InMemoryStore
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.agent.tools import create_rag_retriever_tool
from src.agent.agent import build_chat_graph
from src.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)


def create_agent_graph(
    checkpointer: SqliteSaver,
    vector_store: VectorStore,
    doc_store: InMemoryStore,
    llm_provider: LLMProvider,
) -> CompiledStateGraph:
    """Assembles and builds the final RAG agent graph.

    This factory function orchestrates the creation of the agent by:
    1.  Initializing the necessary tools (e.g., the RAG retriever tool).
    2.  Configuring the primary LLM with the available tools.
    3.  Calling `build_chat_graph` to construct and compile the final
        stateful agent graph.

    Args:
        checkpointer: The checkpointer instance for persisting conversation state.
        vector_store: The vector store used for document retrieval.
        doc_store: The in-memory store for retrieving parent documents.

    Returns:
        The compiled, runnable agent graph (`CompiledStateGraph`).
    """
    rag_tool = create_rag_retriever_tool(vector_store=vector_store, doc_store=doc_store)
    tools = [rag_tool]

    llm_with_fallback = llm_provider.create_llm_with_fallback(
        settings.AGENT_LLM_PROFILE
    )
    llm_with_tools = llm_with_fallback.bind_tools(tools)

    agent_graph = build_chat_graph(
        tools=tools, llm=llm_with_tools, checkpointer=checkpointer
    )

    logger.info("RAG agent graph is built and ready.")
    return agent_graph
