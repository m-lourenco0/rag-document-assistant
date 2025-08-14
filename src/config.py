from pathlib import Path
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    """
    Central configuration for the application, loaded from environment variables
    and/or a .env file.
    """

    # Load a .env file from the project root.
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Secrets ---
    OPENAI_API_KEY: SecretStr = Field(..., description="Your OpenAI API key.")
    COHERE_API_KEY: SecretStr = Field(..., description="Your Cohere API key.")

    # --- File Paths ---
    BASE_DIR: Path = BASE_DIR
    TEMPLATES_DIR: Path = BASE_DIR / "src" / "templates"
    STATIC_DIR: Path = TEMPLATES_DIR / "static"
    CHECKPOINTER_SQLITE_PATH: str = str(BASE_DIR / "checkpoints.sqlite")
    DOC_STORE_PATH: Path = BASE_DIR / "doc_store"

    # --- Vector Store Configuration ---
    VECTORSTORE_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # -- Rerank model ---
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    # --- LLM Configurations ---
    # Each tool/agent can have its own LLM configuration for fine-tuning behavior.

    # Main Agent
    AGENT_LLM: str = "gpt-4.1-mini"
    AGENT_TEMPERATURE: float = 0.0

    # RAG LLM
    RAG_LLM: str = "gpt-4.1-mini"
    RAG_TEMPERATURE: float = 0.0


# Create a single, importable instance of the settings
settings = AppSettings()
