"""Tests for hybrid retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_assistant.core.retriever import HybridRetriever


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    mock = MagicMock()
    mock.embed_query = MagicMock(return_value=[0.1] * 384)
    mock.embed_documents = MagicMock(return_value=[[0.1] * 384 for _ in range(5)])
    return mock


def test_hybrid_retriever_initialization(sample_documents, mock_embeddings):
    """Test HybridRetriever initialization."""
    retriever = HybridRetriever(
        documents=sample_documents,
        embeddings=mock_embeddings,
        dense_top_k=10,
        sparse_top_k=10,
    )

    assert retriever.documents == sample_documents
    assert retriever.dense_top_k == 10
    assert retriever.sparse_top_k == 10


def test_hybrid_retriever_dense_search(sample_documents, mock_embeddings):
    """Test dense search functionality."""
    retriever = HybridRetriever(
        documents=sample_documents,
        embeddings=mock_embeddings,
        dense_top_k=3,
    )

    # Mock sklearn import
    with patch("rag_assistant.core.retriever.cosine_similarity") as mock_cosine:
        mock_cosine.return_value = [[0.8, 0.6, 0.4, 0.2, 0.1]]
        results = retriever._dense_search("heart disease")

        assert len(results) <= 3


def test_hybrid_retriever_rrf_fusion(sample_documents):
    """Test RRF fusion logic."""
    dense_results = sample_documents[:3]
    sparse_results = sample_documents[1:4]

    fused = HybridRetriever._rrf_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=3,
        rrf_k=60,
    )

    assert len(fused) <= 3
    # Should contain unique documents
    assert len(set(doc.metadata["source"] for doc in fused)) == len(fused)


def test_hybrid_retriever_deduplication(sample_documents):
    """Test document deduplication in RRF."""
    # Same documents in both lists
    dense_results = sample_documents[:2]
    sparse_results = sample_documents[:2]

    fused = HybridRetriever._rrf_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=5,
    )

    # Should have no duplicates
    sources = [doc.metadata["source"] for doc in fused]
    assert len(sources) == len(set(sources))


def test_hybrid_retriever_retrieve_fallback_to_dense(sample_documents, mock_embeddings):
    """Test fallback to dense search when BM25 not available."""
    retriever = HybridRetriever(
        documents=sample_documents,
        embeddings=mock_embeddings,
    )
    retriever.bm25_available = False

    with patch.object(retriever, "_dense_search") as mock_dense:
        mock_dense.return_value = sample_documents[:3]

        # Should fallback to dense
        with patch.object(retriever, "_sparse_search") as mock_sparse:
            mock_sparse.return_value = []
            results = retriever.retrieve("heart", k=3)

            assert len(results) <= 3
