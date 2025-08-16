import logging

from typing import Annotated, List

from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_cohere import CohereRerank
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import VectorStore
from langgraph.types import Command

from src.config import settings


logger = logging.getLogger(__name__)


def create_rag_retriever_tool(
    vector_store: VectorStore, doc_store: InMemoryStore
) -> BaseTool:
    """Builds and configures the complete RAG (Retrieval-Augmented Generation) tool.

    This factory function sets up a multi-stage retrieval pipeline:
    1.  It initializes a `ParentDocumentRetriever` which first fetches small,
        embedded chunks from the `vector_store` and then looks up the full
        parent documents from the `doc_store`.
    2.  It wraps this base retriever with a `ContextualCompressionRetriever` that
        uses the Cohere Rerank model to re-rank and filter the results for
        maximum relevance.
    3.  It defines and returns the final tool that the agent can invoke.

    Args:
        vector_store: The vector store containing the embedded document chunks.
        doc_store: The key-value store containing the full parent documents.

    Returns:
        A configured LangChain `BaseTool` ready to be used by an agent.
    """
    base_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        search_kwargs={"k": settings.RAG_SEARCH_K},
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE
        ),
    )

    compressor = CohereRerank(
        top_n=5,
        model=settings.COHERE_RERANK_MODEL,
        cohere_api_key=settings.COHERE_API_KEY,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    @tool
    def search_technical_documents(
        query: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Searches user-uploaded technical manuals for specific information about
        industrial equipment like electric motors and gear units. This is the
        primary tool for answering any technical questions.

        Use it to find details on topics such as: lubrication procedures, torque
        specifications, power consumption, installation steps, part dimensions,
        safety warnings, and maintenance intervals.
        """
        logger.debug("\n" + "=" * 50)
        logger.debug("TOOL 'search_technical_documents' CALLED")
        logger.debug(f"Query: '{query}'")
        logger.debug("=" * 50)

        try:
            # Step 1: Retrieve documents with the compression retriever
            retrieved_docs: List[Document] = compression_retriever.invoke(query)
            logger.debug(
                f"Retriever found {len(retrieved_docs)} documents after re-ranking."
            )
        except Exception:
            logger.error("An error occurred during document retrieval", exc_info=True)
            # Return a helpful error message to the agent if retrieval fails
            error_message = "Sorry, I was unable to search the technical documents at this time. Please try again later."
            tool_message = ToolMessage(content=error_message, tool_call_id=tool_call_id)
            return Command(update={"messages": [tool_message]})

        # Step 2: Process the retrieved documents in a single loop
        references_as_dicts = []
        formatted_chunks = []
        for doc in retrieved_docs:
            # Extract metadata once for reuse
            metadata = {
                "filetype": doc.metadata.get("filetype"),
                "filename": doc.metadata.get("filename"),
                "page_number": doc.metadata.get("page_number"),
                "element_id": doc.metadata.get("element_id"),
                "relevance_score": doc.metadata.get("relevance_score"),
            }

            # Create the reference dictionary for the UI
            references_as_dicts.append(
                {"page_content": doc.page_content, "metadata": metadata}
            )

            # Create the formatted string chunk for the LLM's context
            chunk_text = (
                f"<source file='{metadata['filename']}' page='{metadata['page_number']}' "
                f"relevance_score='{metadata['relevance_score']}'>\n"
                f"{doc.page_content}\n</source>"
            )
            formatted_chunks.append(chunk_text)

        # Step 3: Prepare the final output for the agent state
        tool_output_str = "\n\n".join(formatted_chunks)
        tool_message = ToolMessage(content=tool_output_str, tool_call_id=tool_call_id)

        return Command(
            update={
                "references": references_as_dicts,
                "messages": [tool_message],
            }
        )

    return search_technical_documents
