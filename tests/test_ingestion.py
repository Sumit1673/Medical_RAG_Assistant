"""Tests for document ingestion pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_assistant.pipeline.ingestion import IngestionPipeline


@pytest.fixture
def mock_vector_store():
    """Create mock vector store manager."""
    mock = MagicMock()
    mock.add_documents = MagicMock(return_value=["doc_1", "doc_2", "doc_3"])
    mock.get_collection_info = MagicMock(
        return_value={"collection_name": "test", "document_count": 3}
    )
    return mock


def test_ingestion_pipeline_initialization(mock_vector_store):
    """Test IngestionPipeline initialization."""
    pipeline = IngestionPipeline(
        vector_store_manager=mock_vector_store,
        chunk_size=512,
        chunk_overlap=64,
        chunking_strategy="recursive",
    )

    assert pipeline.vector_store_manager == mock_vector_store
    assert pipeline.splitter is not None


def test_ingestion_pipeline_ingest_directory(mock_vector_store, tmp_path):
    """Test ingesting directory of documents."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test content.")

    pipeline = IngestionPipeline(
        vector_store_manager=mock_vector_store,
    )

    with patch(
        "rag_assistant.pipeline.ingestion.DocumentLoader.load_documents"
    ) as mock_loader:
        mock_loader.return_value = [
            Document(
                page_content="Test content",
                metadata={"source": str(test_file), "format": "txt"},
            )
        ]

        result = pipeline.ingest_directory(str(tmp_path))

        assert result["status"] == "success"
        assert result["loaded_count"] >= 0
        assert result["split_count"] >= 0
        assert result["stored_count"] >= 0


def test_ingestion_pipeline_ingest_file(mock_vector_store, tmp_path):
    """Test ingesting a single file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")

    pipeline = IngestionPipeline(
        vector_store_manager=mock_vector_store,
    )

    with patch(
        "rag_assistant.pipeline.ingestion.DocumentLoader.load_file"
    ) as mock_loader:
        mock_loader.return_value = [
            Document(
                page_content="Test content",
                metadata={"source": str(test_file), "format": "txt"},
            )
        ]

        result = pipeline.ingest_file(str(test_file))

        assert result["status"] == "success"


def test_ingestion_pipeline_empty_directory(mock_vector_store):
    """Test ingesting empty directory."""
    pipeline = IngestionPipeline(
        vector_store_manager=mock_vector_store,
    )

    with patch(
        "rag_assistant.pipeline.ingestion.DocumentLoader.load_documents"
    ) as mock_loader:
        mock_loader.return_value = []

        result = pipeline.ingest_directory("/nonexistent")

        assert result["status"] == "no_documents"
        assert result["loaded_count"] == 0


def test_ingestion_pipeline_get_collection_info(mock_vector_store):
    """Test getting collection information."""
    pipeline = IngestionPipeline(
        vector_store_manager=mock_vector_store,
    )

    info = pipeline.get_collection_info()

    assert "collection_name" in info
    assert "document_count" in info


def test_ingestion_pipeline_semantic_chunking(mock_vector_store):
    """Test ingestion with semantic chunking."""
    pipeline = IngestionPipeline(
        vector_store_manager=mock_vector_store,
        chunking_strategy="semantic",
    )

    assert pipeline.splitter.strategy == "semantic"


def test_ingestion_pipeline_chunking_strategies(mock_vector_store):
    """Test different chunking strategies."""
    # Recursive strategy
    pipeline_recursive = IngestionPipeline(
        vector_store_manager=mock_vector_store,
        chunking_strategy="recursive",
    )
    assert pipeline_recursive.splitter.strategy == "recursive"

    # Semantic strategy
    pipeline_semantic = IngestionPipeline(
        vector_store_manager=mock_vector_store,
        chunking_strategy="semantic",
    )
    assert pipeline_semantic.splitter.strategy == "semantic"
