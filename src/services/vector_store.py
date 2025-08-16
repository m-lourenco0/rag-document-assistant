import logging

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the connection to a persistent ChromaDB vector store.

    This class provides a singleton-like interface for accessing the ChromaDB
    instance, ensuring that only one connection is established and reused
    throughout the application's lifecycle.

    Attributes:
        db_connection (VectorStore | None): Caches the active vector store
            connection. Initially None.
        collection_name (str): The name of the collection within ChromaDB.
    """

    def __init__(self):
        """Initializes the VectorStoreManager."""
        self.db_connection: VectorStore | None = None
        self.collection_name = settings.VECTORSTORE_COLLECTION_NAME

    def get_connection(self) -> VectorStore:
        """Gets the active connection to the ChromaDB vector store.

        On the first call, it initializes a new connection using the specified
        embedding function and collection name. Subsequent calls will return the
        cached connection, avoiding redundant initializations.

        Returns:
            An instance of the active Chroma vector store.

        Raises:
            RuntimeError: If the connection to ChromaDB fails.
        """
        if self.db_connection:
            logger.info("Returning existing ChromaDB connection.")
            return self.db_connection

        logger.info(
            "Attempting to connect to or create persistent ChromaDB vector store..."
        )
        try:
            # Initialize the embedding model
            embeddings = OpenAIEmbeddings(model=settings.VECTORSTORE_EMBEDDING_MODEL)

            # Chroma will create the collection if it doesn't exist.
            self.db_connection = Chroma(
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )
            logger.info("Successfully connected to ChromaDB. Vector store is ready.")
            return self.db_connection

        except Exception as e:
            logger.error("Failed to connect to ChromaDB vector store", exc_info=True)
            raise RuntimeError("Could not connect to the vector store.") from e
