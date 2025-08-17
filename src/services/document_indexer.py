import os
import uuid
import logging

from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

from src.agent.prompts import summarizer_prompt
from src.config import settings

# Get a logger for this specific module
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """A class for processing and indexing documents into a vector store.

    This class orchestrates a multi-stage pipeline that loads documents,
    partitions them into elements, summarizes tables using an LLM, splits the
    content into parent/child chunks, and indexes them into a vector store
    and an in-memory document store. It leverages parallel processing
    (threading and asyncio) for efficiency.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        doc_store: InMemoryStore,
        llm_for_summarization: BaseChatModel,
    ):
        """Initializes the DocumentIndexer."""
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.llm = llm_for_summarization
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.RAG_CHUNK_SIZE
        )

    # --- Main Public Method (Orchestrator) ---

    async def process_files(self, file_paths: List[str]) -> dict:
        """Orchestrates the end-to-end document indexing pipeline."""
        logger.info(f"Starting indexing process for {len(file_paths)} file(s).")

        all_elements = self._load_and_partition_parallel(file_paths)
        if not all_elements:
            logger.warning("No processable elements found in the provided files.")
            return {"documents_indexed": len(file_paths), "total_chunks": 0}

        text_elements, table_elements = self._separate_elements(all_elements)

        parent_docs = self._reconstruct_parent_documents(text_elements)
        if table_elements:
            logger.info(f"Enriching {len(table_elements)} tables with AI summaries...")
            enriched_tables = self._summarize_and_prepare_tables(table_elements)
            parent_docs.extend(enriched_tables)

        if not parent_docs:
            logger.warning("No parent documents could be constructed.")
            return {"documents_indexed": len(file_paths), "total_chunks": 0}

        parent_ids, safe_parents, safe_children = self._prepare_chunks_for_storage(
            parent_docs
        )

        await self._store_documents(parent_ids, safe_parents, safe_children)

        logger.info("--- Indexing Complete ---")
        return {
            "documents_indexed": len(file_paths),
            "total_chunks": len(safe_children),
        }

    # --- Private Helper Methods for Each Pipeline Stage ---

    def _load_and_partition_single_file(self, path: str) -> List[Document]:
        """Loads and partitions a single PDF file into document elements."""
        try:
            loader = UnstructuredPDFLoader(
                path, mode="elements", strategy="hi_res", infer_table_structure=True
            )
            elements = loader.load()
            for element in elements:
                element.metadata["filename"] = os.path.basename(path)
            return elements
        except Exception:
            logger.error(f"Error loading or partitioning {path}", exc_info=True)
            return []

    def _load_and_partition_parallel(self, file_paths: List[str]) -> List[Document]:
        """Loads and partitions multiple PDF files in parallel."""
        logger.info(f"Loading and partitioning {len(file_paths)} PDFs in parallel...")
        with ThreadPoolExecutor() as executor:
            results = executor.map(self._load_and_partition_single_file, file_paths)
        return [element for sublist in results for element in sublist]

    def _separate_elements(
        self, all_elements: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """Separates document elements into text and table categories."""
        text_elements, table_elements = [], []
        for el in all_elements:
            if el.metadata.get("category") == "Table":
                table_elements.append(el)
            else:
                text_elements.append(el)
        logger.info(
            f"Separated {len(text_elements)} text elements and {len(table_elements)} table elements."
        )
        return text_elements, table_elements

    def _reconstruct_parent_documents(
        self, text_elements: List[Document]
    ) -> List[Document]:
        """Reconstructs parent documents from a list of Unstructured text elements."""
        parent_documents = []
        content_map: Dict[str, List[str]] = {}
        for el in text_elements:
            parent_id = el.metadata.get("parent_id")
            if parent_id:
                content_map.setdefault(parent_id, []).append(el.page_content)

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
        return parent_documents

    def _summarize_and_prepare_tables(
        self, table_elements: List[Document]
    ) -> List[Document]:
        """Summarizes table elements using an LLM and prepares them for indexing."""

        class TableSummary(BaseModel):
            summary: str = Field(
                description="A concise summary of the table's purpose and key information."
            )

        summarize_chain = summarizer_prompt | self.llm.with_structured_output(
            TableSummary
        )

        table_htmls = [
            el.metadata.get("text_as_html", el.page_content) for el in table_elements
        ]

        logger.debug(f"Sending {len(table_htmls)} tables to LLM for summarization...")
        try:
            summaries_objs = summarize_chain.batch(
                [{"table_html": html} for html in table_htmls],
                config={"max_concurrency": 5},
            )
        except Exception:
            logger.error("Failed to summarize one or more tables.", exc_info=True)
            return []

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

    def _prepare_chunks_for_storage(
        self, parent_documents: List[Document]
    ) -> Tuple[List[str], List[Document], List[Document]]:
        """Creates child chunks, assigns IDs, and prepares docs for storage."""
        parent_ids = [str(uuid.uuid4()) for _ in parent_documents]
        child_documents = []
        logger.info(
            f"Creating child chunks for {len(parent_documents)} parent documents..."
        )
        for i, parent_doc in enumerate(parent_documents):
            parent_id = parent_ids[i]
            chunks = self.child_splitter.split_documents([parent_doc])
            for chunk in chunks:
                chunk.metadata["doc_id"] = parent_id
                child_documents.append(chunk)

        safe_parent_docs = filter_complex_metadata(parent_documents)
        safe_child_docs = filter_complex_metadata(child_documents)
        return parent_ids, safe_parent_docs, safe_child_docs

    async def _store_documents(
        self,
        parent_ids: List[str],
        parent_docs: List[Document],
        child_docs: List[Document],
    ) -> None:
        """Stores parent and child documents in their respective stores."""
        logger.info(f"Storing {len(parent_docs)} parent documents in docstore...")
        self.doc_store.mset(list(zip(parent_ids, parent_docs)))

        batch_size = settings.RAG_INDEXING_BATCH_SIZE
        batches = [
            child_docs[i : i + batch_size]
            for i in range(0, len(child_docs), batch_size)
        ]

        total_chunks = len(child_docs)
        logger.info(f"Storing {total_chunks} child documents in vectorstore ")

        # NOTE: Ideally we would run the add_documents function with
        # asyncio to improve performance, but my current OpenAI account
        # does not have enough rate limit to use that, so we are using
        # sync calls instead
        # tasks = [self.vector_store.aadd_documents(batch) for batch in batches]
        # await asyncio.gather(*tasks)

        # Process batches one by one instead of concurrently
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i + 1}/{len(batches)}...")
            await self.vector_store.aadd_documents(batch)

        logger.info("Finished storing all documents.")
