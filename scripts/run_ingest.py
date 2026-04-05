"""Script to run document ingestion."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_assistant.core.embedding_generator import EmbeddingGenerator
from rag_assistant.core.vector_store_manager import VectorStoreManager
from rag_assistant.pipeline.ingestion import IngestionPipeline
from rag_assistant.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run ingestion pipeline."""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = ConfigLoader(str(config_path))
        logger.info("Configuration loaded")

        # Initialize embeddings
        embedding_config = config.get_section("embedding")
        embeddings = EmbeddingGenerator(
            provider=embedding_config.get("provider", "huggingface"),
            model_name=embedding_config.get("model_name", "BAAI/bge-small-en-v1.5"),
            device=embedding_config.get("device", "cpu"),
            normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        )
        logger.info(f"Embeddings initialized: {embedding_config.get('model_name')}")

        # Initialize vector store
        vs_config = config.get_section("vector_store")
        vector_store = VectorStoreManager(
            embeddings=embeddings.embeddings,
            collection_name=vs_config.get("collection_name", "medical_rag"),
            persist_directory=str(Path(__file__).parent.parent / vs_config.get("vector_store_dir", "vector_store/")),
            chroma_server_url=vs_config.get("chroma_server_url"),
        )
        logger.info("Vector store initialized")

        # Initialize ingestion pipeline
        ingest_config = config.get_section("ingestion")
        pipeline = IngestionPipeline(
            vector_store_manager=vector_store,
            chunk_size=ingest_config.get("chunk_size", 512),
            chunk_overlap=ingest_config.get("chunk_overlap", 64),
            chunking_strategy=ingest_config.get("chunking_strategy", "recursive"),
        )
        logger.info("Ingestion pipeline initialized")

        # Download dataset if needed
        dataset_path = Path(__file__).parent.parent / "dataset" / "medical"
        if not dataset_path.exists() or len(list(dataset_path.glob("*.txt"))) == 0:
            logger.info("Dataset not found. Preparing sample dataset...")
            from dataset.download_dataset import download_and_prepare_dataset

            download_and_prepare_dataset(str(dataset_path))

        # Run ingestion
        result = pipeline.ingest_directory(str(dataset_path))
        logger.info(f"Ingestion result: {result}")

        # Show collection info
        info = pipeline.get_collection_info()
        logger.info(f"Collection info: {info}")

        logger.info("Ingestion complete!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
