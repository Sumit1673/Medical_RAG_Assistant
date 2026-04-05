"""Embedding generation with support for OpenAI and HuggingFace."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using various providers."""

    def __init__(
        self,
        provider: Literal["openai", "huggingface"] = "huggingface",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize EmbeddingGenerator.

        Args:
            provider: Embedding provider ("openai" or "huggingface")
            model_name: Model name for the provider
            device: Device to use ("cpu" or "cuda")
            normalize_embeddings: Whether to normalize embeddings
        """
        self.provider = provider
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.embeddings = self._initialize_embeddings()
        logger.info(
            f"EmbeddingGenerator initialized with provider={provider}, "
            f"model={model_name}, device={device}"
        )

    def _initialize_embeddings(self) -> Embeddings:
        """Initialize embeddings model based on provider."""
        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "huggingface":
            return self._init_huggingface()
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def _init_openai(self) -> Embeddings:
        """Initialize OpenAI embeddings."""
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=self.model_name)
        except ImportError:
            raise ImportError("langchain-openai is required for OpenAI embeddings")

    def _init_huggingface(self) -> Embeddings:
        """Initialize HuggingFace embeddings."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
            )
        except ImportError:
            raise ImportError("langchain-huggingface is required for HuggingFace embeddings")

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(query)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(documents)
