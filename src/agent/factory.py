from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.agent.tools import create_rag_retriever_tool
from src.agent.agent import build_chat_graph


def create_agent_graph(
    checkpointer: SqliteSaver,
    vector_store: VectorStore,
    doc_store: InMemoryStore,
) -> CompiledStateGraph:
    """
    Factory function to assemble and build the final RAG agent graph.
    """
    rag_tool = create_rag_retriever_tool(vector_store=vector_store, doc_store=doc_store)
    tools = [rag_tool]

    llm = ChatOpenAI(
        model=settings.AGENT_LLM,
        temperature=settings.AGENT_TEMPERATURE,
    ).bind_tools(tools)

    agent_graph = build_chat_graph(tools=tools, llm=llm, checkpointer=checkpointer)

    print("RAG agent graph is built and ready.")
    return agent_graph
