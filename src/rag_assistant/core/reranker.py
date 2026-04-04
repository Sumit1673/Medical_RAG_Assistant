"""Cross-encoder based reranking for retrieved documents."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Module-level import so tests can patch it
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment,misc]


class CrossEncoderReranker:
    """Rerank documents using cross-encoder models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ) -> None:
        """
        Initialize CrossEncoderReranker.

        Args:
            model_name: Name of the cross-encoder model
            top_n: Number of documents to return after reranking
        """
        self.model_name = model_name
        self.top_n = top_n

        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )

        self.model = CrossEncoder(model_name)
        logger.info(f"CrossEncoderReranker initialized with model={model_name}")

    def rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of documents to rerank

        Returns:
            List of (Document, score) tuples sorted by score descending
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []

        # Prepare pairs for scoring
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine documents with scores
        doc_scores = list(zip(documents, scores))

        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_n
        reranked = doc_scores[: self.top_n]

        logger.debug(
            f"Reranked {len(documents)} documents to {len(reranked)} "
            f"(top scores: {[f'{score:.3f}' for _, score in reranked[:3]]})"
        )

        return reranked

    def rerank_with_metadata(
        self, query: str, documents: list[Document]
    ) -> list[Document]:
        """
        Rerank documents and attach rerank scores as metadata.

        Args:
            query: Query text
            documents: List of documents to rerank

        Returns:
            List of reranked Documents with rerank_score in metadata
        """
        reranked = self.rerank(query, documents)

        # Attach scores as metadata and return
        result = []
        for doc, score in reranked:
            doc.metadata["rerank_score"] = float(score)
            result.append(doc)

        return result
