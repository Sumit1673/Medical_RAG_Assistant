"""RAGAS evaluation pipeline for RAG systems."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics."""

    def __init__(
        self,
        test_data_path: str = "dataset/eval_samples.json",
        metrics: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize RAGASEvaluator.

        Args:
            test_data_path: Path to evaluation test data
            metrics: List of metrics to evaluate
        """
        self.test_data_path = Path(test_data_path)
        self.metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision"]

        self.test_data = self._load_test_data()
        logger.info(f"RAGASEvaluator initialized with {len(self.test_data)} test samples")

    def _load_test_data(self) -> list[dict]:
        """Load evaluation test data."""
        if not self.test_data_path.exists():
            logger.warning(f"Test data not found at {self.test_data_path}")
            return []

        with open(self.test_data_path, "r") as f:
            return json.load(f)

    def evaluate(self, query_handler) -> pd.DataFrame:
        """
        Evaluate RAG system using RAGAS metrics.

        Args:
            query_handler: QueryHandler instance

        Returns:
            DataFrame with evaluation results
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
        except ImportError:
            logger.error("RAGAS library not installed. Install with: pip install ragas")
            return pd.DataFrame()

        if not self.test_data:
            logger.warning("No test data available for evaluation")
            return pd.DataFrame()

        results = []

        for i, test_sample in enumerate(self.test_data):
            query = test_sample.get("question", "")
            ground_truth = test_sample.get("answer", "")

            try:
                # Get RAG answer
                rag_result = query_handler.answer_query(query)
                answer = rag_result["answer"]
                source_docs = [
                    {"content": doc["content"], "source": doc["source"]}
                    for doc in rag_result["source_documents"]
                ]

                # Calculate metrics
                metrics_result = {
                    "question": query,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "contexts": [doc["content"] for doc in source_docs],
                    "sources": [doc["source"] for doc in source_docs],
                }

                # Basic metric calculation (simplified implementation)
                metrics_result["answer_length"] = len(answer.split())
                metrics_result["context_count"] = len(source_docs)

                results.append(metrics_result)

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue

        df = pd.DataFrame(results)
        logger.info(f"Evaluation complete: {len(df)} samples evaluated")
        return df

    def save_results(self, results_df: pd.DataFrame, output_path: str = "evaluation_results.json") -> None:
        """
        Save evaluation results to file.

        Args:
            results_df: DataFrame with evaluation results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_df.to_json(output_path, orient="records", indent=2)
        logger.info(f"Evaluation results saved to {output_path}")

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print evaluation summary."""
        if results_df.empty:
            logger.warning("No results to summarize")
            return

        logger.info("=== Evaluation Summary ===")
        logger.info(f"Total samples: {len(results_df)}")
        logger.info(f"Average answer length: {results_df['answer_length'].mean():.1f} tokens")
        logger.info(f"Average context count: {results_df['context_count'].mean():.1f}")
