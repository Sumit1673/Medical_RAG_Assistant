"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="The heart is a vital organ that pumps blood throughout the body.",
            metadata={"source": "medical_1.txt", "format": "txt"},
        ),
        Document(
            page_content="Blood pressure is the force exerted by blood against artery walls.",
            metadata={"source": "medical_2.txt", "format": "txt"},
        ),
        Document(
            page_content="Diabetes is a metabolic disorder characterized by high blood sugar.",
            metadata={"source": "medical_3.txt", "format": "txt"},
        ),
        Document(
            page_content="The lungs are responsible for gas exchange in the respiratory system.",
            metadata={"source": "medical_4.txt", "format": "txt"},
        ),
        Document(
            page_content="Cholesterol is a lipid that is essential for cell membrane structure.",
            metadata={"source": "medical_5.txt", "format": "txt"},
        ),
    ]


@pytest.fixture
def sample_config() -> dict:
    """Create sample configuration for testing."""
    return {
        "paths": {
            "data_dir": "dataset/medical/",
            "vector_store_dir": "vector_store/",
        },
        "embedding": {
            "provider": "huggingface",
            "model_name": "BAAI/bge-small-en-v1.5",
            "device": "cpu",
            "normalize_embeddings": True,
        },
        "llm": {
            "provider": "ollama",
            "model_name": "mistral",
            "temperature": 0.2,
            "max_tokens": 1024,
        },
        "retrieval": {
            "dense_top_k": 10,
            "sparse_top_k": 10,
            "final_top_k": 5,
            "enable_query_rewriting": True,
            "enable_reranking": True,
        },
    }


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = MagicMock()
    mock.add_documents = MagicMock(return_value=["doc_1", "doc_2", "doc_3"])
    mock.similarity_search = MagicMock(return_value=[])
    mock.get_collection_info = MagicMock(
        return_value={"collection_name": "test", "document_count": 3}
    )
    return mock
