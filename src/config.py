"""Configuration management for IntraMind AI Agent."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load .env file from project root before defining Settings
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # API Gateway Configuration
    api_gateway_url: str = Field(
        default="http://localhost:5000",
        description="Base URL for the API Gateway service",
    )
    api_gateway_timeout: int = Field(
        default=30,
        description="Timeout for API Gateway requests in seconds",
    )

    # Primary LLM Configuration (for reasoning & synthesis)
    primary_llm_provider: Literal["anthropic", "openai", "ollama"] = Field(
        default="anthropic",
        description="LLM provider for primary reasoning tasks",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model to use",
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        description="OpenAI model to use",
    )

    # Router LLM Configuration (for classification & routing)
    router_llm_provider: Literal["ollama", "anthropic", "openai"] = Field(
        default="ollama",
        description="LLM provider for routing/classification tasks",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama service",
    )
    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model to use for routing",
    )

    # Agent Configuration
    agent_max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations for agent execution",
    )
    agent_verbose: bool = Field(
        default=True,
        description="Enable verbose logging for agent operations",
    )
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses",
    )

    # Multimodal Processing
    enable_ocr: bool = Field(
        default=True,
        description="Enable OCR for image processing",
    )
    tesseract_path: str | None = Field(
        default=None,
        description="Path to Tesseract executable (if not in PATH)",
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size for document processing in MB",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Logging format",
    )

    # Collection Settings
    default_collection: str = Field(
        default="intramind_documents",
        description="Default collection name for document storage",
    )
    search_limit: int = Field(
        default=10,
        description="Default number of search results to return",
    )

    # Conversation Memory Settings
    enable_conversation_memory: bool = Field(
        default=True,
        description="Enable conversation memory/context across queries",
    )
    max_conversation_history: int = Field(
        default=5,
        description="Maximum number of conversation turns to keep in context (cost optimization)",
    )
    smart_context_selection: bool = Field(
        default=True,
        description="Only include conversation history for complex queries (cost optimization)",
    )
    checkpoint_storage_path: str = Field(
        default="./data/checkpoints.db",
        description="Path to SQLite database for conversation checkpoints",
    )

    def get_primary_llm_config(self) -> dict:
        """Get configuration for primary LLM."""
        if self.primary_llm_provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
            }
        elif self.primary_llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
            }
        else:  # ollama
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            }

    def get_router_llm_config(self) -> dict:
        """Get configuration for router LLM."""
        if self.router_llm_provider == "ollama":
            return {
                "provider": "ollama",
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            }
        elif self.router_llm_provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.anthropic_model,
            }
        else:  # openai
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
            }


# Global settings instance
settings = Settings()
