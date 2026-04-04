"""LLM handler supporting OpenAI GPT-4 and Ollama."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Literal, Optional

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handle LLM interactions with both OpenAI and Ollama."""

    def __init__(
        self,
        provider: Literal["openai", "ollama"] = "ollama",
        model_name: str = "mistral",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        model_base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize LLMHandler.

        Args:
            provider: LLM provider ("openai" or "ollama")
            model_name: Model name (e.g., "gpt-4o" for OpenAI, "mistral" for Ollama)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            model_base_url: Base URL for Ollama (e.g., "http://localhost:11434")
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_base_url = model_base_url

        self.llm = self._initialize_llm()
        logger.info(
            f"LLMHandler initialized with provider={provider}, "
            f"model={model_name}, temperature={temperature}"
        )

    def _initialize_llm(self):
        """Initialize LLM based on provider."""
        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "ollama":
            return self._init_ollama()
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _init_openai(self):
        """Initialize OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=True,
            )
        except ImportError:
            raise ImportError("langchain-openai is required for OpenAI LLM")

    def _init_ollama(self):
        """Initialize Ollama LLM."""
        try:
            from langchain_community.llms import Ollama

            base_url = self.model_base_url or "http://localhost:11434"
            return Ollama(
                model=self.model_name,
                base_url=base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
        except ImportError:
            raise ImportError("langchain-community is required for Ollama LLM")

    def generate_response(self, prompt: str) -> str:
        """
        Generate synchronous response.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        try:
            response = self.llm.invoke(prompt)
            # Handle both string and message responses
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response tokens asynchronously.

        Args:
            prompt: Input prompt

        Yields:
            Token strings
        """
        try:
            # For streaming, we use the sync streaming and wrap it
            if self.provider == "openai":
                # Use async streaming for OpenAI
                from langchain_openai import ChatOpenAI

                async_llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    streaming=True,
                )

                async for chunk in async_llm.astream(prompt):
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content
            else:
                # For Ollama, fall back to sync and yield all at once
                response = self.generate_response(prompt)
                # Simulate streaming by yielding chunks
                chunk_size = 10
                for i in range(0, len(response), chunk_size):
                    yield response[i : i + chunk_size]

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            raise
