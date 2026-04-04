"""Document ingestion pipeline."""

from __future__ import annotations

import logging

from rag_assistant.core.document_loader import DocumentLoader
from rag_assistant.core.document_splitter import DocumentSplitter
from rag_assistant.core.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Manage document ingestion: loading, splitting, and storing."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunking_strategy: str = "recursive",
    ) -> None:
        """
        Initialize IngestionPipeline.

        Args:
            vector_store_manager: VectorStoreManager instance
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking ("recursive" or "semantic")
        """
        self.vector_store_manager = vector_store_manager
        self.splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
        )
        logger.info("IngestionPipeline initialized")

    def ingest_directory(self, directory: str) -> dict:
        """
        Ingest all documents from a directory.

        Args:
            directory: Path to directory containing documents

        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Starting ingestion from directory: {directory}")

        # Step 1: Load documents
        documents = DocumentLoader.load_documents(directory)
        if not documents:
            logger.warning(f"No documents found in {directory}")
            return {
                "status": "no_documents",
                "loaded_count": 0,
                "split_count": 0,
                "stored_count": 0,
            }

        # Step 2: Split documents
        split_documents = self.splitter.split(documents)

        # Step 3: Store in vector store
        doc_ids = self.vector_store_manager.add_documents(split_documents)

        result = {
            "status": "success",
            "loaded_count": len(documents),
            "split_count": len(split_documents),
            "stored_count": len(doc_ids),
        }

        logger.info(f"Ingestion complete: {result}")
        return result

    def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single document file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Starting ingestion from file: {file_path}")

        # Step 1: Load document
        documents = DocumentLoader.load_file(file_path)

        # Step 2: Split document
        split_documents = self.splitter.split(documents)

        # Step 3: Store in vector store
        doc_ids = self.vector_store_manager.add_documents(split_documents)

        result = {
            "status": "success",
            "loaded_count": len(documents),
            "split_count": len(split_documents),
            "stored_count": len(doc_ids),
        }

        logger.info(f"Ingestion complete: {result}")
        return result

    def get_collection_info(self) -> dict:
        """Get information about stored collection."""
        return self.vector_store_manager.get_collection_info()
