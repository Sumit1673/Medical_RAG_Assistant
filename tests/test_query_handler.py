"""Tests for query handler."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_assistant.core.query_handler import QueryHandler


@pytest.fixture
def mock_retriever():
    """Create mock retriever."""
    mock = MagicMock()
    mock.retrieve = MagicMock(
        return_value=[
            Document(
                page_content="Heart disease is a leading cause of death.",
                metadata={"source": "medical_1.txt"},
            ),
            Document(
                page_content="Risk factors include high cholesterol and hypertension.",
                metadata={"source": "medical_2.txt"},
            ),
        ]
    )
    return mock


@pytest.fixture
def mock_llm_handler():
    """Create mock LLM handler."""
    mock = MagicMock()
    mock.generate_response = MagicMock(
        return_value="Heart disease is a serious medical condition with multiple risk factors."
    )
    mock.stream_response = MagicMock()
    return mock


@pytest.fixture
def mock_reranker():
    """Create mock reranker."""
    mock = MagicMock()
    mock.rerank_with_metadata = MagicMock(
        return_value=[
            Document(
                page_content="Heart disease is a leading cause of death.",
                metadata={"source": "medical_1.txt", "rerank_score": 0.95},
            ),
        ]
    )
    return mock


def test_query_handler_initialization(mock_retriever, mock_llm_handler):
    """Test QueryHandler initialization."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        enable_query_rewriting=True,
        enable_reranking=False,
    )

    assert handler.enable_query_rewriting is True
    assert handler.enable_reranking is False


def test_query_handler_answer_query(mock_retriever, mock_llm_handler, mock_reranker):
    """Test basic query answering."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        reranker=mock_reranker,
        enable_query_rewriting=False,
        enable_reranking=True,
    )

    result = handler.answer_query("What is heart disease?")

    assert "answer" in result
    assert "original_query" in result
    assert "rewritten_query" in result
    assert "source_documents" in result
    assert isinstance(result["source_documents"], list)


def test_query_handler_with_rewriting(mock_retriever, mock_llm_handler):
    """Test query handler with query rewriting enabled."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        enable_query_rewriting=True,
        enable_reranking=False,
    )

    mock_llm_handler.generate_response.side_effect = [
        "What are the causes and symptoms of heart disease?",  # rewritten
        "Heart disease is a serious condition.",  # answer
    ]

    result = handler.answer_query("heart disease")

    assert result["original_query"] == "heart disease"
    assert result["rewritten_query"] == "What are the causes and symptoms of heart disease?"


def test_query_handler_without_reranking(mock_retriever, mock_llm_handler):
    """Test query handler with reranking disabled."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        reranker=None,
        enable_reranking=False,
    )

    result = handler.answer_query("medical question")

    assert result["answer"] is not None
    mock_retriever.retrieve.assert_called_once()


def test_query_handler_rag_prompt(mock_retriever, mock_llm_handler):
    """Test RAG prompt building."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        enable_reranking=False,
    )

    docs = [
        Document(page_content="Test content 1", metadata={"source": "doc1"}),
        Document(page_content="Test content 2", metadata={"source": "doc2"}),
    ]

    prompt = handler._build_rag_prompt("test query", docs)

    assert "test query" in prompt
    assert "Test content 1" in prompt
    assert "Test content 2" in prompt


@pytest.mark.asyncio
async def test_query_handler_streaming(mock_retriever, mock_llm_handler):
    """Test streaming response."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        enable_reranking=False,
    )

    # Mock async generator
    async def mock_stream(*args, **kwargs):
        for token in ["Hello", " ", "world"]:
            yield token

    mock_llm_handler.stream_response = mock_stream

    tokens = []
    async for token in handler.answer_query_stream("test"):
        tokens.append(token)

    assert len(tokens) > 0


def test_query_handler_source_documents(mock_retriever, mock_llm_handler, mock_reranker):
    """Test source documents are included in result."""
    handler = QueryHandler(
        retriever=mock_retriever,
        llm_handler=mock_llm_handler,
        reranker=mock_reranker,
        enable_reranking=True,
    )

    result = handler.answer_query("question")

    sources = result["source_documents"]
    assert len(sources) > 0
    assert all("content" in doc for doc in sources)
    assert all("source" in doc for doc in sources)
