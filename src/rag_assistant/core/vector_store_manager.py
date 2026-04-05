"""Vector store management using ChromaDB."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage ChromaDB vector store for document retrieval."""

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        chroma_server_url: Optional[str] = None,
    ) -> None:
        """
        Initialize VectorStoreManager.

        Args:
            embeddings: Embeddings instance
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage (local mode)
            chroma_server_url: URL for remote ChromaDB server
        """
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chroma_server_url = chroma_server_url

        self.vector_store = self._initialize_vector_store()
        logger.info(f"VectorStoreManager initialized for collection: {collection_name}")

    def _initialize_vector_store(self) -> Chroma:
        """Initialize ChromaDB vector store."""
        if self.chroma_server_url:
            # Remote ChromaDB client
            import chromadb

            logger.info(f"Connecting to remote ChromaDB at {self.chroma_server_url}")
            host = self.chroma_server_url.split("://")[1].split(":")[0]
            port = int(self.chroma_server_url.split(":")[-1])
            client = chromadb.HttpClient(host=host, port=port)
            return Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        else:
            # Local persistent storage
            logger.info(f"Using local ChromaDB at {self.persist_directory}")
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to vector store.

        Args:
            documents: List of Document objects

        Returns:
            List of document IDs
        """
        ids = self.vector_store.add_documents(documents)
        logger.info(f"Added {len(ids)} documents to vector store")
        return ids

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of similar Document objects
        """
        results = self.vector_store.similarity_search(query, k=k)
        logger.debug(f"Similarity search found {len(results)} documents")
        return results

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results

        Returns:
            List of (Document, score) tuples
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        logger.debug(f"Similarity search with scores found {len(results)} documents")
        return results

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            count = self.vector_store._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
            }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.vector_store.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
