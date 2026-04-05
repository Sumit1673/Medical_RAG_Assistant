"""Cross-encoder based reranking for retrieved documents."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
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
        top_p: Optional[float] = None,
    ) -> None:
        """
        Initialize CrossEncoderReranker.

        Args:
            model_name: Name of the cross-encoder model
            top_n: Number of documents to return after reranking (used when top_p is None)
            top_p: Nucleus threshold (0-1). Applies softmax over cross-encoder scores
                   and selects the smallest set of docs whose cumulative probability
                   >= top_p. When None, falls back to hard top-n cutoff.
        """
        self.model_name = model_name
        self.top_n = top_n
        self.top_p = top_p

        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )

        self.model = CrossEncoder(model_name)
        logger.info(f"CrossEncoderReranker initialized with model={model_name}")

    def rerank(
        self, query: str, documents: list[Document]
    ) -> list[tuple[Document, float]]:
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

        # Combine documents with scores and sort descending
        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        if self.top_p is not None:
            # Nucleus filtering via softmax: keep docs until cumulative prob >= top_p
            raw_scores = np.array([s for _, s in doc_scores], dtype=float)
            exp_scores = np.exp(raw_scores - raw_scores.max())  # numerically stable
            probs = exp_scores / exp_scores.sum()

            reranked = []
            cumulative = 0.0
            for (doc, score), prob in zip(doc_scores, probs):
                reranked.append((doc, score))
                cumulative += prob
                if cumulative >= self.top_p:
                    break
            logger.debug(
                f"Reranked {len(documents)} → {len(reranked)} docs "
                f"(top_p={self.top_p}, cumulative prob={cumulative:.3f}, "
                f"top scores: {[f'{s:.3f}' for _, s in reranked[:3]]})"
            )
        else:
            reranked = doc_scores[: self.top_n]
            logger.debug(
                f"Reranked {len(documents)} → {len(reranked)} docs "
                f"(top scores: {[f'{s:.3f}' for _, s in reranked[:3]]})"
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
