import os
import uuid
import asyncio
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata


class DocumentIndexer:
    """
    Processes and indexes documents using parallel processing for speed.
    - PDF parsing is parallelized across multiple files using threads.
    - Table summarization uses LLM batching.
    - Vector store indexing is done concurrently using asyncio.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        doc_store: InMemoryStore,
        llm_for_summarization: ChatOpenAI,
    ):
        """Initializes the indexer with shared stores and configured splitters."""
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.llm = llm_for_summarization
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)

    def _load_and_partition_single_file(self, path: str) -> List[Document]:
        """Loads and partitions a single PDF file."""
        try:
            loader = UnstructuredPDFLoader(path, mode="elements", strategy="hi_res")
            elements = loader.load()
            for element in elements:
                element.metadata["filename"] = os.path.basename(path)
            return elements
        except Exception as e:
            print(f"Error loading or partitioning {path}: {e}")
            return []

    def _load_and_partition_parallel(self, file_paths: List[str]) -> List[Document]:
        """Loads and partitions multiple PDFs in parallel using a thread pool."""
        print(f"---LOADING & PARTITIONING {len(file_paths)} PDFs IN PARALLEL---")
        with ThreadPoolExecutor() as executor:
            results = executor.map(self._load_and_partition_single_file, file_paths)

        # Flatten the list of lists into a single list of elements
        all_elements = [element for sublist in results for element in sublist]
        return all_elements

    def _summarize_and_prepare_tables(
        self, table_elements: List[Document]
    ) -> List[Document]:
        """Summarizes tables and returns a single list of enriched parent documents."""

        class TableSummary(BaseModel):
            summary: str = Field(
                description="A concise summary of the table's purpose and key information."
            )

        prompt = ChatPromptTemplate.from_template(
            "Summarize the following table to make it easily searchable. "
            "Describe its columns, rows, and the key information it contains.\n\n"
            "Table:\n{table_html}"
        )
        summarize_chain = prompt | self.llm.with_structured_output(TableSummary)

        table_htmls = [
            el.metadata.get("text_as_html", el.page_content) for el in table_elements
        ]
        print(f"Sending {len(table_htmls)} tables to LLM for summarization...")
        summaries_objs = summarize_chain.batch(
            [{"table_html": html} for html in table_htmls],
            config={"max_concurrency": 5},
        )

        enriched_table_docs = []
        for i, summary_obj in enumerate(summaries_objs):
            if summary_obj:
                table_el = table_elements[i]
                enriched_content = (
                    f"Summary of the following table: {summary_obj.summary}\n\n"
                    f"Original Table Content:\n{table_el.metadata.get('text_as_html', table_el.page_content)}"
                )
                enriched_doc = Document(
                    page_content=enriched_content, metadata=table_el.metadata
                )
                enriched_table_docs.append(enriched_doc)
        return enriched_table_docs

    async def process_files(self, file_paths: List[str]) -> dict:
        """Main method to orchestrate the indexing pipeline for new files."""
        all_elements = self._load_and_partition_parallel(file_paths)
        safe_elements = filter_complex_metadata(all_elements)
        total_chunks = 0

        text_elements, table_elements = [], []
        for el in safe_elements:
            (
                table_elements
                if el.metadata.get("category") == "Table"
                else text_elements
            ).append(el)

        parent_documents = []
        if text_elements:
            content_map: Dict[str, List[str]] = {}
            for el in text_elements:
                parent_id = el.metadata.get("parent_id")
                if parent_id:
                    if parent_id not in content_map:
                        content_map[parent_id] = []
                    content_map[parent_id].append(el.page_content)
            for el in text_elements:
                element_id = el.metadata.get("element_id")
                if element_id in content_map:
                    children_content = "\n\n".join(content_map[element_id])
                    combined_content = f"{el.page_content}\n\n{children_content}"
                    parent_documents.append(
                        Document(page_content=combined_content, metadata=el.metadata)
                    )
                elif el.metadata.get("parent_id") is None:
                    parent_documents.append(el)

        if table_elements:
            print(f"Enriching {len(table_elements)} tables with AI summaries...")
            enriched_tables = self._summarize_and_prepare_tables(table_elements)
            parent_documents.extend(enriched_tables)

        if parent_documents:
            parent_ids = [str(uuid.uuid4()) for _ in parent_documents]
            child_documents = []

            print(
                f"Creating child chunks for {len(parent_documents)} parent documents..."
            )
            for i, parent_doc in enumerate(parent_documents):
                parent_id = parent_ids[i]
                chunks = self.child_splitter.split_documents([parent_doc])
                for chunk in chunks:
                    chunk.metadata["doc_id"] = parent_id
                    child_documents.append(chunk)

            # Store the parent documents in the docstore
            print(f"Storing {len(parent_documents)} parent documents in docstore...")
            self.doc_store.mset(list(zip(parent_ids, parent_documents)))

            # Store the child documents in the vectorstore concurrenty
            batch_size = 500  # A safe batch size
            batches = [
                child_documents[i : i + batch_size]
                for i in range(0, len(child_documents), batch_size)
            ]

            total_chunks = len(child_documents)

            print(
                f"Storing {total_chunks} child documents in vectorstore "
                f"using {len(batches)} concurrent async batches..."
            )

            # Create a list of async tasks; one for each batch
            tasks = [self.vector_store.aadd_documents(batch) for batch in batches]

            # Run all tasks concurrently and wait for them to complete
            await asyncio.gather(*tasks)

            print("Finished concurrent indexing of all documents.")

        print("\n---Indexing Complete---")
        return {
            "documents_indexed": len(file_paths),
            "total_chunks": total_chunks,
        }
