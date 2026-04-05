"""Document splitting with fixed and semantic chunking strategies."""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Split documents into chunks using various strategies."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: Literal["recursive", "semantic"] = "recursive",
    ) -> None:
        """
        Initialize DocumentSplitter.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            strategy: Chunking strategy ("recursive" or "semantic")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        logger.info(
            f"DocumentSplitter initialized with strategy={strategy}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of split Document objects
        """
        if self.strategy == "recursive":
            return self._split_recursive(documents)
        elif self.strategy == "semantic":
            return self._split_semantic(documents)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _split_recursive(self, documents: list[Document]) -> list[Document]:
        """Split using RecursiveCharacterTextSplitter."""
        split_docs = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    def _split_semantic(self, documents: list[Document]) -> list[Document]:
        """
        Split using semantic boundaries (sentences).

        This implementation uses sentence-level splitting.
        """
        split_docs = []

        for doc in documents:
            sentences = self._split_into_sentences(doc.page_content)
            current_chunk = ""
            chunk_count = 0

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        split_docs.append(
                            Document(
                                page_content=current_chunk.strip(),
                                metadata={**doc.metadata, "chunk": chunk_count},
                            )
                        )
                        chunk_count += 1

                        # Add overlap by keeping last sentence
                        overlap_text = (
                            " ".join(sentences[-2:]) if len(sentences) > 1 else sentence
                        )
                        current_chunk = overlap_text + " "

                    current_chunk += sentence + " "

            # Add remaining chunk
            if current_chunk.strip():
                split_docs.append(
                    Document(
                        page_content=current_chunk.strip(),
                        metadata={**doc.metadata, "chunk": chunk_count},
                    )
                )

        logger.info(
            f"Semantically split {len(documents)} documents into {len(split_docs)} chunks"
        )
        return split_docs

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting on period, question mark, exclamation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
