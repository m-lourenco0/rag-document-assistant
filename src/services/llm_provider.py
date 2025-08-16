import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from src.config import settings, LLMProfile

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    A generic and scalable provider for creating LLM instances
    based on configuration profiles.
    """

    def __init__(self):
        self.api_keys = {
            "openai": settings.OPENAI_API_KEY.get_secret_value(),
            "anthropic": settings.ANTHROPIC_API_KEY.get_secret_value(),
            "google": settings.GOOGLE_API_KEY.get_secret_value(),
        }
        self.model_map = {
            "openai": ChatOpenAI,
            "anthropic": ChatAnthropic,
            "google": ChatGoogleGenerativeAI,
        }

    def _create_llm(
        self, provider: str, model_name: str, temperature: float
    ) -> BaseChatModel:
        """Helper to create a single LLM instance."""
        if provider not in self.model_map:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        LLMClass = self.model_map[provider]

        # Specific handling for Google's API key parameter name
        if provider == "google":
            return LLMClass(
                model=model_name,
                temperature=temperature,
                google_api_key=self.api_keys.get(provider),
            )

        return LLMClass(
            model=model_name,
            temperature=temperature,
            api_key=self.api_keys.get(provider),
        )

    def create_llm_with_fallback(self, profile: LLMProfile) -> BaseChatModel:
        """
        Creates a primary LLM with a configured fallback based on a profile.
        If the primary fails, it automatically uses the fallback.
        """
        primary_llm = self._create_llm(
            provider=profile.primary_provider,
            model_name=profile.primary_model_name,
            temperature=profile.temperature,
        )

        fallback_llm = self._create_llm(
            provider=profile.fallback_provider,
            model_name=profile.fallback_model_name,
            temperature=profile.temperature,
        )

        logger.info(
            f"Initialized LLM with profile. Primary: {profile.primary_provider} "
            f"({profile.primary_model_name}), Fallback: {profile.fallback_provider} "
            f"({profile.fallback_model_name})"
        )

        return primary_llm.with_fallbacks([fallback_llm])
