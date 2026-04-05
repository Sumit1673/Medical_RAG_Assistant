"""Tests for cross-encoder reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_assistant.core.reranker import CrossEncoderReranker


@pytest.fixture
def mock_cross_encoder():
    """Create mock cross-encoder model."""
    with patch("rag_assistant.core.reranker.CrossEncoder") as mock_class:
        mock_instance = MagicMock()
        mock_instance.predict = MagicMock(return_value=[0.8, 0.6, 0.4, 0.2, 0.1])
        mock_class.return_value = mock_instance
        yield mock_instance


def test_reranker_initialization(mock_cross_encoder):
    """Test CrossEncoderReranker initialization."""
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5,
    )

    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.top_n == 5


def test_reranker_returns_correct_count(sample_documents, mock_cross_encoder):
    """Test reranker returns correct number of documents."""
    reranker = CrossEncoderReranker(top_n=3)
    reranker.model = mock_cross_encoder

    results = reranker.rerank("medical question", sample_documents)

    assert len(results) == 3
    # Results should be (Document, score) tuples
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


def test_reranker_scores_descending(sample_documents, mock_cross_encoder):
    """Test reranker returns documents sorted by score descending."""
    mock_cross_encoder.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]

    reranker = CrossEncoderReranker(top_n=5)
    reranker.model = mock_cross_encoder

    results = reranker.rerank("question", sample_documents)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_reranker_with_metadata(sample_documents, mock_cross_encoder):
    """Test reranker attaches score to metadata."""
    mock_cross_encoder.predict.return_value = [0.8, 0.6, 0.4, 0.2, 0.1]

    reranker = CrossEncoderReranker(top_n=3)
    reranker.model = mock_cross_encoder

    results = reranker.rerank_with_metadata("question", sample_documents)

    assert len(results) == 3
    for doc in results:
        assert "rerank_score" in doc.metadata
        assert isinstance(doc.metadata["rerank_score"], float)


def test_reranker_empty_documents(mock_cross_encoder):
    """Test reranker with empty document list."""
    reranker = CrossEncoderReranker()
    reranker.model = mock_cross_encoder

    results = reranker.rerank("question", [])

    assert results == []


def test_reranker_handles_exceptions(sample_documents, mock_cross_encoder):
    """Test reranker propagates model prediction errors."""
    mock_cross_encoder.predict.side_effect = RuntimeError("Model error")

    reranker = CrossEncoderReranker()
    reranker.model = mock_cross_encoder

    with pytest.raises(RuntimeError, match="Model error"):
        reranker.rerank("question", sample_documents)
