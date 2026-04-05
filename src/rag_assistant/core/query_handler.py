"""Advanced query handling with rewriting, retrieval, reranking, and generation."""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class QueryHandler:
    """Handle advanced query processing pipeline."""

    def __init__(
        self,
        retriever,
        llm_handler,
        reranker: Optional[object] = None,
        enable_query_rewriting: bool = True,
        enable_reranking: bool = True,
        final_top_k: int = 5,
    ) -> None:
        """
        Initialize QueryHandler.

        Args:
            retriever: Retriever instance
            llm_handler: LLMHandler instance
            reranker: Optional CrossEncoderReranker instance
            enable_query_rewriting: Whether to rewrite queries
            enable_reranking: Whether to rerank results
            final_top_k: Number of final results to return
        """
        self.retriever = retriever
        self.llm_handler = llm_handler
        self.reranker = reranker
        self.enable_query_rewriting = enable_query_rewriting
        self.enable_reranking = enable_reranking
        self.final_top_k = final_top_k

        logger.info(
            f"QueryHandler initialized with rewriting={enable_query_rewriting}, "
            f"reranking={enable_reranking}, final_top_k={final_top_k}"
        )

    def answer_query(self, query: str, metadata_filter: Optional[dict] = None) -> dict:
        """
        Answer a query using the full advanced pipeline.

        Args:
            query: User query
            metadata_filter: Optional dict of metadata key-value pairs to restrict
                retrieval to matching documents. Example: {"source": "diabetes.txt"}

        Returns:
            Dictionary with answer, rewritten_query, and source documents
        """
        # Step 1: Query Rewriting
        rewritten_query = query
        if self.enable_query_rewriting:
            rewritten_query = self._rewrite_query(query)
            logger.debug(f"Original: {query}")
            logger.debug(f"Rewritten: {rewritten_query}")

        # Step 2: Hybrid Retrieval
        candidates = self.retriever.retrieve(
            rewritten_query,
            k=self.final_top_k * 2,
            metadata_filter=metadata_filter,
        )
        logger.debug(f"Retrieved {len(candidates)} candidate documents")

        # Step 3: Reranking
        if self.enable_reranking and self.reranker:
            final_docs = self.reranker.rerank_with_metadata(rewritten_query, candidates)
        else:
            final_docs = candidates[: self.final_top_k]

        # Step 4: LLM Generation with RAG
        answer = self._generate_answer_with_rag(query, final_docs)

        return {
            "answer": answer,
            "original_query": query,
            "rewritten_query": rewritten_query,
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "unknown"),
                    "rerank_score": doc.metadata.get("rerank_score", None),
                }
                for doc in final_docs
            ],
        }

    async def answer_query_stream(
        self, query: str, metadata_filter: Optional[dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream answer to a query.

        Args:
            query: User query
            metadata_filter: Optional dict of metadata key-value pairs to restrict
                retrieval to matching documents.

        Yields:
            Answer tokens
        """
        # Step 1: Query Rewriting
        rewritten_query = query
        if self.enable_query_rewriting:
            rewritten_query = self._rewrite_query(query)

        # Step 2: Hybrid Retrieval
        candidates = self.retriever.retrieve(
            rewritten_query,
            k=self.final_top_k * 2,
            metadata_filter=metadata_filter,
        )

        # Step 3: Reranking
        if self.enable_reranking and self.reranker:
            final_docs = self.reranker.rerank_with_metadata(rewritten_query, candidates)
        else:
            final_docs = candidates[: self.final_top_k]

        # Step 4: Stream LLM Generation
        prompt = self._build_rag_prompt(query, final_docs)
        async for token in self.llm_handler.stream_response(prompt):
            yield token

    def _rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval.

        Args:
            query: Original query

        Returns:
            Rewritten query
        """
        rewrite_prompt = f"""Given the user question, rewrite it to be more specific and detailed
for retrieval purposes. Focus on key terms and concepts.

Original question: {query}

Rewritten question:"""

        rewritten = self.llm_handler.generate_response(rewrite_prompt)
        return rewritten.strip()

    def _build_rag_prompt(self, query: str, documents: list[Document]) -> str:
        """
        Build RAG prompt with context.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Formatted prompt
        """
        context = "\n\n".join(
            [f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(documents)]
        )

        prompt = f"""You are a helpful assistant. Use the following context to answer the question.
If the context does not contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    def _generate_answer_with_rag(self, query: str, documents: list[Document]) -> str:
        """
        Generate answer using retrieved documents.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Generated answer
        """
        prompt = self._build_rag_prompt(query, documents)
        answer = self.llm_handler.generate_response(prompt)
        return answer.strip()
