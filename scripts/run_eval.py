"""Script to run evaluation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_assistant.core.embedding_generator import EmbeddingGenerator
from rag_assistant.core.llm_handler import LLMHandler
from rag_assistant.core.query_handler import QueryHandler
from rag_assistant.core.reranker import CrossEncoderReranker
from rag_assistant.core.retriever import HybridRetriever
from rag_assistant.core.vector_store_manager import VectorStoreManager
from rag_assistant.evaluation.ragas_eval import RAGASEvaluator
from rag_assistant.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run evaluation."""
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
        )
        logger.info(f"Embeddings initialized: {embedding_config.get('model_name')}")

        # Initialize vector store
        vs_config = config.get_section("vector_store")
        vector_store = VectorStoreManager(
            embeddings=embeddings.embeddings,
            collection_name=vs_config.get("collection_name", "medical_rag"),
            chroma_server_url=vs_config.get("chroma_server_url"),
        )
        logger.info("Vector store initialized")

        # Get documents from vector store (simplified - would need to load from storage)
        logger.info("Note: Loading documents from dataset for evaluation")
        from rag_assistant.core.document_loader import DocumentLoader

        data_dir = Path(__file__).parent.parent / "dataset" / "medical"
        documents = DocumentLoader.load_documents(str(data_dir))

        if not documents:
            logger.warning("No documents found. Please run ingestion first.")
            return

        # Initialize retriever
        retrieval_config = config.get_section("retrieval")
        retriever = HybridRetriever(
            documents=documents,
            embeddings=embeddings.embeddings,
            dense_top_k=retrieval_config.get("dense_top_k", 10),
            sparse_top_k=retrieval_config.get("sparse_top_k", 10),
        )
        logger.info("Retriever initialized")

        # Initialize LLM
        llm_config = config.get_section("llm")
        llm_handler = LLMHandler(
            provider=llm_config.get("provider", "ollama"),
            model_name=llm_config.get("model_name", "mistral"),
            temperature=llm_config.get("temperature", 0.2),
            max_tokens=llm_config.get("max_tokens", 1024),
            model_base_url=llm_config.get("model_base_url"),
        )
        logger.info(f"LLM initialized: {llm_config.get('model_name')}")

        # Initialize reranker
        reranker_config = config.get_section("reranker")
        reranker = CrossEncoderReranker(
            model_name=reranker_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_n=reranker_config.get("top_n", 5),
        )
        logger.info("Reranker initialized")

        # Initialize query handler
        query_handler = QueryHandler(
            retriever=retriever,
            llm_handler=llm_handler,
            reranker=reranker,
            enable_query_rewriting=retrieval_config.get("enable_query_rewriting", True),
            enable_reranking=retrieval_config.get("enable_reranking", True),
            final_top_k=retrieval_config.get("final_top_k", 5),
        )
        logger.info("Query handler initialized")

        # Initialize evaluator
        eval_config = config.get_section("evaluation")
        evaluator = RAGASEvaluator(
            test_data_path=str(Path(__file__).parent.parent / eval_config.get("test_data_path", "dataset/eval_samples.json")),
            metrics=eval_config.get("metrics", ["faithfulness", "answer_relevancy", "context_precision"]),
        )
        logger.info("Evaluator initialized")

        # Run evaluation
        logger.info("Starting evaluation...")
        results_df = evaluator.evaluate(query_handler)

        if not results_df.empty:
            # Save results
            output_path = Path(__file__).parent.parent / eval_config.get("output_path", "evaluation_results.json")
            evaluator.save_results(results_df, str(output_path))

            # Print summary
            evaluator.print_summary(results_df)
            logger.info("Evaluation complete!")
        else:
            logger.warning("Evaluation produced no results")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
