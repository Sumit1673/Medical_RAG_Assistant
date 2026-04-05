"""Hybrid retriever combining BM25 sparse and dense vector search with RRF fusion."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Module-level import so tests can patch it
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever using BM25 + Dense vector search with RRF fusion."""

    def __init__(
        self,
        documents: list[Document],
        embeddings: Embeddings,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        rrf_k: int = 60,
        top_p: Optional[float] = None,
    ) -> None:
        """
        Initialize HybridRetriever.

        Args:
            documents: List of documents to index
            embeddings: Embeddings instance for dense retrieval
            dense_top_k: Number of results from dense search (initial broad retrieval)
            sparse_top_k: Number of results from sparse search (initial broad retrieval)
            rrf_k: RRF parameter (constant from the paper)
            top_p: Nucleus threshold for RRF output (0-1). If set, selects the
                   smallest set of docs whose cumulative normalised RRF score
                   reaches top_p, instead of a fixed-k cutoff.
        """
        self.documents = documents
        self.embeddings = embeddings
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k
        self.top_p = top_p

        # Initialize BM25 retriever
        try:
            from langchain_community.retrievers import BM25Retriever

            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_available = True
            logger.info("BM25Retriever initialized for sparse search")
        except Exception as e:
            logger.warning(f"BM25Retriever not available: {e}")
            self.bm25_retriever = None
            self.bm25_available = False

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Query text
            k: Maximum results to return (used when top_p is not set)

        Returns:
            List of retrieved documents
        """
        results = []

        # Dense retrieval (always top_k — broad initial net)
        dense_results = self._dense_search(query)

        # Sparse retrieval (always top_k — broad initial net)
        sparse_results = self._sparse_search(query)

        # Fuse results using RRF, applying top_p nucleus filtering if configured
        if sparse_results and self.bm25_available:
            results = self._rrf_fusion(
                dense_results, sparse_results, k, top_p=self.top_p
            )
        else:
            # Fallback to dense only — top_k still makes sense here
            logger.warning("Falling back to dense search only")
            results = dense_results[:k]

        logger.debug(f"Hybrid retrieval returned {len(results)} documents")
        return results

    def _dense_search(self, query: str) -> list[Document]:
        """Search using dense embeddings (cosine similarity)."""
        if cosine_similarity is None:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            return []
        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents(
                [doc.page_content for doc in self.documents]
            )
            query_vec = np.array(query_embedding).reshape(1, -1)
            doc_vecs = np.array(doc_embeddings)

            scores = cosine_similarity(query_vec, doc_vecs)[0]
            scored_docs = list(zip(self.documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in scored_docs[: self.dense_top_k]]
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []

    def _sparse_search(self, query: str) -> list[Document]:
        """Search using BM25."""
        if not self.bm25_available or not self.bm25_retriever:
            return []

        try:
            results = self.bm25_retriever.invoke(query)
            return results[: self.sparse_top_k]
        except Exception as e:
            logger.error(f"Error in sparse search: {e}")
            return []

    @staticmethod
    def _rrf_fusion(
        dense_results: list[Document],
        sparse_results: list[Document],
        k: int,
        rrf_k: int = 60,
        top_p: Optional[float] = None,
    ) -> list[Document]:
        """
        Fuse dense and sparse results using Reciprocal Rank Fusion.

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            k: Maximum results to return (used when top_p is None)
            rrf_k: RRF parameter (default 60 from paper)
            top_p: Nucleus threshold (0-1). Selects the smallest set of docs
                   whose cumulative normalised RRF score >= top_p. When None,
                   falls back to hard top-k cutoff.

        Returns:
            Fused results
        """
        # Create rank dictionaries
        dense_ranks = {doc.metadata.get("source", str(i)): 1.0 / (rrf_k + i + 1)
                       for i, doc in enumerate(dense_results)}
        sparse_ranks = {doc.metadata.get("source", str(i)): 1.0 / (rrf_k + i + 1)
                        for i, doc in enumerate(sparse_results)}

        # Combine scores
        combined_scores = {}
        all_docs = {doc.metadata.get("source", str(i)): doc
                    for i, doc in enumerate(dense_results + sparse_results)}

        for doc_id in all_docs:
            combined_scores[doc_id] = dense_ranks.get(doc_id, 0) + sparse_ranks.get(doc_id, 0)

        # Sort by combined score descending
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        seen = set()

        if top_p is not None:
            # Nucleus filtering: accumulate until cumulative normalised score >= top_p
            total_score = sum(score for _, score in sorted_ids)
            cumulative = 0.0
            for doc_id, score in sorted_ids:
                if doc_id not in seen:
                    result.append(all_docs[doc_id])
                    seen.add(doc_id)
                    cumulative += score / total_score if total_score > 0 else 1.0
                    if cumulative >= top_p:
                        break
            logger.debug(
                f"RRF top_p={top_p}: selected {len(result)} docs "
                f"(cumulative score mass={cumulative:.3f})"
            )
        else:
            # Hard top-k cutoff
            for doc_id, _ in sorted_ids:
                if doc_id not in seen:
                    result.append(all_docs[doc_id])
                    seen.add(doc_id)
                    if len(result) >= k:
                        break

        return result

    def retrieve_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        if cosine_similarity is None:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            return []
        try:
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents(
                [doc.page_content for doc in self.documents]
            )
            query_vec = np.array(query_embedding).reshape(1, -1)
            doc_vecs = np.array(doc_embeddings)

            scores = cosine_similarity(query_vec, doc_vecs)[0]  # type: ignore[operator]
            scored_docs = list(zip(self.documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return scored_docs[:k]
        except Exception as e:
            logger.error(f"Error in retrieve_with_scores: {e}")
            return []
