from pprint import pprint
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


def create_rag_retriever_tool(
    vector_store: VectorStore, doc_store: InMemoryStore
) -> BaseTool:
    """
    Builds the complete RAG tool.

    This function will:
    1. Use the provided factory to build the base ParentDocumentRetriever,
       which populates the vector_store and doc_store.
    2. Wrap the base retriever with a Cohere-powered compression retriever.
    3. Return a LangChain tool that the agent can invoke.
    """
    base_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        search_kwargs={"k": 30},
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
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
        """
        Searches user-uploaded technical manuals for specific information about
        industrial equipment like electric motors and gear units. This is the
        primary tool for answering any technical questions.

        Use it to find details on topics such as: lubrication procedures, torque
        specifications, power consumption, installation steps, part dimensions,
        safety warnings, and maintenance intervals.
        """
        print("\n" + "=" * 50)
        print("TOOL 'search_technical_documents' CALLED")
        print(f"Query: '{query}'")
        print("=" * 50)

        retrieved_docs: List[Document] = compression_retriever.invoke(query)

        print(f"\nRetriever found {len(retrieved_docs)} documents after re-ranking.")

        references_as_dicts = [
            {
                "page_content": doc.page_content,
                "metadata": {
                    "filetype": doc.metadata.get("filetype"),
                    "filename": doc.metadata.get("filename"),
                    "page_number": doc.metadata.get("page_number"),
                    "element_id": doc.metadata.get("element_id"),
                    "relevance_score": doc.metadata.get("relevance_score"),
                },
            }
            for doc in retrieved_docs
        ]

        print("\n" + "=" * 50)
        pprint(references_as_dicts)
        print("=" * 50)

        formatted_chunks = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "N/A")
            page = doc.metadata.get("page_number", "N/A")
            chunk_text = (
                f"<source file='{source}' page='{page}'>\n{doc.page_content}\n</source>"
            )
            formatted_chunks.append(chunk_text)

        tool_output_str = "\n\n".join(formatted_chunks)
        tool_message = ToolMessage(content=tool_output_str, tool_call_id=tool_call_id)

        return Command(
            update={
                "references": references_as_dicts,
                "messages": [tool_message],
            }
        )

    return search_technical_documents
