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


# ── Metadata filtering ────────────────────────────────────────────────────────


def test_apply_metadata_filter_single_key(sample_documents):
    """Filter by a single metadata key returns only matching documents."""
    filtered = HybridRetriever._apply_metadata_filter(
        sample_documents, {"source": "medical_1.txt"}
    )
    assert len(filtered) == 1
    assert filtered[0].metadata["source"] == "medical_1.txt"


def test_apply_metadata_filter_multiple_keys(sample_documents):
    """Filter by multiple keys applies AND logic."""
    filtered = HybridRetriever._apply_metadata_filter(
        sample_documents, {"source": "medical_1.txt", "format": "txt"}
    )
    assert len(filtered) == 1
    assert filtered[0].metadata["source"] == "medical_1.txt"


def test_apply_metadata_filter_no_match(sample_documents):
    """Filter that matches nothing returns an empty list."""
    filtered = HybridRetriever._apply_metadata_filter(
        sample_documents, {"source": "nonexistent.pdf"}
    )
    assert filtered == []


def test_apply_metadata_filter_none_returns_all(sample_documents):
    """Passing None as filter returns all documents unchanged."""
    filtered = HybridRetriever._apply_metadata_filter(sample_documents, None)
    assert filtered == sample_documents


def test_apply_metadata_filter_empty_dict_returns_all(sample_documents):
    """Passing an empty dict returns all documents (no constraints)."""
    filtered = HybridRetriever._apply_metadata_filter(sample_documents, {})
    assert filtered == sample_documents


def test_retrieve_with_metadata_filter(sample_documents, mock_embeddings):
    """retrieve() with metadata_filter only returns matching documents."""
    retriever = HybridRetriever(
        documents=sample_documents,
        embeddings=mock_embeddings,
    )
    retriever.bm25_available = False

    with patch("rag_assistant.core.retriever.cosine_similarity") as mock_cosine:
        mock_cosine.return_value = [[0.9]]
        results = retriever.retrieve(
            "heart", k=5, metadata_filter={"source": "medical_1.txt"}
        )

    assert all(doc.metadata["source"] == "medical_1.txt" for doc in results)


def test_retrieve_with_filter_no_match_returns_empty(sample_documents, mock_embeddings):
    """retrieve() returns empty list when filter matches no documents."""
    retriever = HybridRetriever(
        documents=sample_documents,
        embeddings=mock_embeddings,
    )
    retriever.bm25_available = False

    results = retriever.retrieve(
        "heart", k=5, metadata_filter={"source": "nonexistent.pdf"}
    )
    assert results == []
