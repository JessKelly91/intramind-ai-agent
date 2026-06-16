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
    synthesis_max_contexts: int = Field(
        default=3,
        description="Maximum number of retrieved chunks to include in synthesis context",
    )
    synthesis_context_chars: int = Field(
        default=1200,
        description="Maximum characters to include from each retrieved chunk during synthesis",
    )
    synthesis_score_gap: float = Field(
        default=0.12,
        description=(
            "For direct factual queries, include additional contexts only when "
            "their score is within this gap of the top result"
        ),
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

    # Observability / Tracing (Step 1: Phoenix)
    enable_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing to Phoenix (set ENABLE_TRACING=true)",
    )
    phoenix_endpoint: str = Field(
        default="http://localhost:6006",
        description="Base URL for the self-hosted Phoenix collector",
    )
    tracing_service_name: str = Field(
        default="intramind-ai-agent",
        description="Service/project name attached to OTEL spans",
    )

    # Prompt Registry (runtime-served prompts with baked-in fallback)
    prompt_registry_url: str | None = Field(
        default=None,
        description="Prompt Registry base URL. Unset keeps code-registry fallback only.",
    )
    prompt_registry_label: str = Field(
        default="production",
        description="Prompt label resolved at runtime (production, candidate, staging).",
    )
    prompt_registry_api_key: str | None = Field(
        default=None,
        description="Read/service API key for the Prompt Registry.",
    )
    prompt_registry_cache_ttl: int = Field(
        default=60,
        description="Prompt Registry in-memory cache TTL in seconds.",
    )

    # RAG Evaluation (Step 2: Ragas with Ollama judge)
    ragas_judge_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model used as the LLM judge for Ragas metrics",
    )
    ragas_enforce_thresholds: bool = Field(
        default=False,
        description=(
            "If False, Ragas threshold tests run as warning-only (xfail). "
            "Flip to True (or set RAGAS_ENFORCE_THRESHOLDS=true) to fail CI."
        ),
    )
    ragas_max_parse_failure_rate: float = Field(
        default=0.2,
        description=(
            "Maximum fraction of Ragas metric cells allowed to be NaN/null "
            "before the eval is considered failed."
        ),
    )
    ragas_threshold_faithfulness: float = Field(
        default=0.7, description="Min faithfulness score before threshold test fails"
    )
    ragas_threshold_answer_relevancy: float = Field(
        default=0.7,
        description="Min answer_relevancy score before threshold test fails",
    )
    ragas_threshold_context_precision: float = Field(
        default=0.7,
        description="Min context_precision score before threshold test fails",
    )
    ragas_threshold_context_recall: float = Field(
        default=0.7,
        description="Min context_recall score before threshold test fails",
    )

    # PII Redaction (Step 3: Presidio - redact-on-ingest)
    enable_pii_redaction: bool = Field(
        default=True,
        description="Detect and redact PII during ingestion before chunking",
    )
    pii_redaction_required: bool = Field(
        default=False,
        description=(
            "If True, ingestion fails closed when PII redaction is disabled, "
            "unavailable, or errors."
        ),
    )
    pii_entities: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IP_ADDRESS",
            "LOCATION",
        ],
        description="Presidio entity types to detect",
    )
    pii_score_threshold: float = Field(
        default=0.5,
        description="Minimum Presidio confidence score to count as a PII finding",
    )

    # Output Safety Guard (Step 4: Llama Guard via Ollama, hard-block policy)
    enable_safety_guard: bool = Field(
        default=True,
        description="Run Llama Guard on synthesized responses before returning",
    )
    safety_guard_required: bool = Field(
        default=False,
        description=(
            "If True, search responses are blocked when the safety guard is "
            "disabled, unavailable, malformed, or errors."
        ),
    )
    safety_guard_model: str = Field(
        default="llama-guard3",
        description="Ollama model tag used for output safety classification",
    )
    safety_fallback_message: str = Field(
        default=(
            "I can't provide a response to that query. "
            "If you believe this is in error, please rephrase or contact your "
            "administrator."
        ),
        description="Templated text returned in place of any flagged response",
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
