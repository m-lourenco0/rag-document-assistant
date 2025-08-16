from pathlib import Path
from pydantic import SecretStr, Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the base directory of the project for use in path configurations.
BASE_DIR = Path(__file__).resolve().parent.parent


class LLMProfile(BaseModel):
    primary_provider: str = "openai"
    primary_model_name: str = "gpt-4.1-mini"
    fallback_provider: str = "google"
    fallback_model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0


class AppSettings(BaseSettings):
    """Manages all application configuration settings.

    This class uses pydantic-settings to load configuration from environment
    variables and a `.env` file located in the project's root directory. It
    provides validation, type casting, and a centralized structure for all
    configurable parameters.
    """

    # Pydantic model configuration to load a .env file from the project root.
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Secrets ---
    OPENAI_API_KEY: SecretStr = Field(..., description="Your OpenAI API key.")
    COHERE_API_KEY: SecretStr = Field(..., description="Your Cohere API key.")
    ANTHROPIC_API_KEY: SecretStr = Field(..., description="Your Anthropic API key.")
    GOOGLE_API_KEY: SecretStr = Field(..., description="Your Google API key.")

    # --- File Paths ---
    BASE_DIR: Path = BASE_DIR
    TEMPLATES_DIR: Path = BASE_DIR / "src" / "templates"
    STATIC_DIR: Path = TEMPLATES_DIR / "static"
    CHECKPOINTER_SQLITE_PATH: str = str(BASE_DIR / "checkpoints.sqlite")

    # --- Vector Store Configuration ---
    VECTORSTORE_EMBEDDING_MODEL: str = "text-embedding-3-small"
    VECTORSTORE_COLLECTION_NAME: str = "rag_document_collection"

    # --- RAG Pipeline Configuration ---
    RAG_CHUNK_SIZE: int = 200
    RAG_INDEXING_BATCH_SIZE: int = 500
    RAG_SEARCH_K: int = 30
    RAG_RERANKER_TOP_N: int = 5

    # --- Rerank Model ---
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    # --- LLM Configurations ---
    # Each tool/agent can have its own LLM configuration for fine-tuning behavior.

    AGENT_LLM_PROFILE: LLMProfile = LLMProfile(
        primary_model_name="gpt-4.1-mini", temperature=0.0
    )

    RAG_LLM_PROFILE: LLMProfile = LLMProfile(
        primary_model_name="gpt-4.1-mini", temperature=0.0
    )


# A single, globally accessible instance of the AppSettings.
# Import this instance into other modules to access configuration.
settings = AppSettings()
