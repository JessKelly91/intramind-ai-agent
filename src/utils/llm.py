"""LLM initialization utilities."""

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)


def get_primary_llm(**kwargs: Any) -> BaseChatModel:
    """Get the primary LLM for reasoning and synthesis.

    Args:
        **kwargs: Additional arguments to pass to the LLM constructor

    Returns:
        Initialized LLM instance

    Raises:
        ValueError: If LLM provider is invalid or API key is missing
    """
    config = settings.get_primary_llm_config()
    provider = config["provider"]

    logger.info(f"Initializing primary LLM: {provider}")

    if provider == "anthropic":
        if not config["api_key"]:
            raise ValueError("Anthropic API key not set in environment")

        return ChatAnthropic(
            model=config["model"],
            anthropic_api_key=config["api_key"],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
        )

    elif provider == "openai":
        if not config["api_key"]:
            raise ValueError("OpenAI API key not set in environment")

        return ChatOpenAI(
            model=config["model"],
            openai_api_key=config["api_key"],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 4096),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
        )

    elif provider == "ollama":
        return ChatOllama(
            model=config["model"],
            base_url=config["base_url"],
            temperature=kwargs.get("temperature", 0.7),
            **{k: v for k, v in kwargs.items() if k != "temperature"},
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_router_llm(**kwargs: Any) -> BaseChatModel:
    """Get the router LLM for classification and routing decisions.

    Args:
        **kwargs: Additional arguments to pass to the LLM constructor

    Returns:
        Initialized LLM instance

    Raises:
        ValueError: If LLM provider is invalid or API key is missing
    """
    config = settings.get_router_llm_config()
    provider = config["provider"]

    logger.info(f"Initializing router LLM: {provider}")

    if provider == "ollama":
        return ChatOllama(
            model=config["model"],
            base_url=config["base_url"],
            temperature=kwargs.get("temperature", 0.0),  # Lower temp for routing
            **{k: v for k, v in kwargs.items() if k != "temperature"},
        )

    elif provider == "anthropic":
        if not config["api_key"]:
            raise ValueError("Anthropic API key not set in environment")

        return ChatAnthropic(
            model=config["model"],
            anthropic_api_key=config["api_key"],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
        )

    elif provider == "openai":
        if not config["api_key"]:
            raise ValueError("OpenAI API key not set in environment")

        return ChatOpenAI(
            model=config["model"],
            openai_api_key=config["api_key"],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]},
        )

    else:
        raise ValueError(f"Unsupported router LLM provider: {provider}")
