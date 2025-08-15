from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from src.config import settings


class VectorStoreManager:
    """
    A class to manage the connection to a persistent ChromaDB vector store.

    This manager handles the logic for creating or connecting to the database,
    providing a clean, reusable interface for the application.
    """

    def __init__(self):
        """Initializes the VectorStoreManager."""
        self.db_connection: VectorStore | None = None

        self.collection_name = "rag_document_collection"

    def get_connection(self) -> VectorStore:
        """
        Establishes and returns a connection to the ChromaDB vector store.

        If a connection exists, it returns it. Otherwise, it creates a new
        connection, creating the database directory if it doesn't exist.

        Returns:
            An instance of the connected Chroma vector store.
        """
        if self.db_connection:
            print("Returning existing ChromaDB connection.")
            return self.db_connection

        print("Attempting to connect to or create persistent ChromaDB vector store...")

        try:
            # Initialize the embedding model
            embeddings = OpenAIEmbeddings(model=settings.VECTORSTORE_EMBEDDING_MODEL)

            # Chroma will create the collection if it doesn't exist.
            self.db_connection = Chroma(
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )
            print("Successfully connected to ChromaDB. Vector store is ready.")
            return self.db_connection

        except Exception as e:
            print(f"Failed to connect to ChromaDB vector store: {e}")
            raise RuntimeError("Could not connect to the vector store.") from e
