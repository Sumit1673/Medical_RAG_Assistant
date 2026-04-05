"""RAGAS-style evaluation pipeline for RAG systems.

Metrics are computed locally without an external LLM evaluator:
- answer_similarity   : ROUGE-1 F1 between generated answer and ground truth
- faithfulness        : fraction of answer sentences supported by retrieved context
- context_precision   : fraction of retrieved contexts that overlap with ground truth
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _rouge1_f1(prediction: str, reference: str) -> float:
    """ROUGE-1 F1 between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _faithfulness_score(answer: str, contexts: list[str]) -> float:
    """
    Fraction of answer sentences that are supported by at least one context chunk.
    A sentence is considered supported if token overlap with any context >= 0.3.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sentences:
        return 0.0
    context_tokens = _tokenize(" ".join(contexts))
    supported = 0
    for sentence in sentences:
        s_tokens = _tokenize(sentence)
        if not s_tokens:
            continue
        overlap = len(s_tokens & context_tokens) / len(s_tokens)
        if overlap >= 0.3:
            supported += 1
    return supported / len(sentences)


def _context_precision_score(contexts: list[str], ground_truth: str) -> float:
    """
    Fraction of retrieved context chunks that have meaningful overlap with ground truth.
    Overlap threshold: >= 0.2 ROUGE-1 recall.
    """
    if not contexts:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    relevant = sum(
        1 for ctx in contexts if len(_tokenize(ctx) & gt_tokens) / len(gt_tokens) >= 0.2
    )
    return relevant / len(contexts)


class RAGASEvaluator:
    """Evaluate RAG system using local proxy metrics."""

    def __init__(
        self,
        test_data_path: str = "dataset/eval_samples.json",
        metrics: Optional[list[str]] = None,
    ) -> None:
        self.test_data_path = Path(test_data_path)
        self.metrics = metrics or [
            "answer_similarity",
            "faithfulness",
            "context_precision",
        ]
        self.test_data = self._load_test_data()
        logger.info(
            f"RAGASEvaluator initialized with {len(self.test_data)} test samples"
        )

    def _load_test_data(self) -> list[dict]:
        """Load evaluation test data."""
        if not self.test_data_path.exists():
            logger.warning(f"Test data not found at {self.test_data_path}")
            return []
        with open(self.test_data_path, "r") as f:
            return json.load(f)

    def evaluate(self, query_handler) -> pd.DataFrame:
        """
        Run evaluation over all test samples and compute per-sample metrics.

        Args:
            query_handler: QueryHandler instance

        Returns:
            DataFrame with one row per sample and metric columns
        """
        if not self.test_data:
            logger.warning("No test data available for evaluation")
            return pd.DataFrame()

        results = []

        for i, test_sample in enumerate(self.test_data):
            question = test_sample.get("question", "")
            ground_truth = test_sample.get("answer", "")

            try:
                logger.info(
                    f"Evaluating sample {i + 1}/{len(self.test_data)}: {question[:60]}..."
                )
                rag_result = query_handler.answer_query(question)
                answer = rag_result["answer"]
                contexts = [doc["content"] for doc in rag_result["source_documents"]]

                results.append(
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "answer_similarity": _rouge1_f1(answer, ground_truth),
                        "faithfulness": _faithfulness_score(answer, contexts),
                        "context_precision": _context_precision_score(
                            contexts, ground_truth
                        ),
                        "context_count": len(contexts),
                        "answer_length": len(answer.split()),
                    }
                )

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue

        df = pd.DataFrame(results)
        logger.info(f"Evaluation complete: {len(df)} samples evaluated")
        return df

    def save_results(
        self, results_df: pd.DataFrame, output_path: str = "evaluation_results.json"
    ) -> None:
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_json(output_path, orient="records", indent=2)
        logger.info(f"Evaluation results saved to {output_path}")

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print a formatted score summary to stdout."""
        if results_df.empty:
            logger.warning("No results to summarize")
            return

        score_cols = ["answer_similarity", "faithfulness", "context_precision"]
        available = [c for c in score_cols if c in results_df.columns]

        width = 44
        print("\n" + "═" * width)
        print("  📊 RAG Evaluation Results")
        print("═" * width)
        print(f"  {'Metric':<26} {'Score':>8}")
        print("─" * width)
        for col in available:
            mean_score = results_df[col].mean()
            bar = "█" * int(mean_score * 20)
            print(f"  {col:<26} {mean_score:>6.3f}  {bar}")
        print("─" * width)
        print(f"  {'Samples evaluated':<26} {len(results_df):>8}")
        print(
            f"  {'Avg answer length (words)':<26} {results_df['answer_length'].mean():>8.1f}"
        )
        print(
            f"  {'Avg context chunks used':<26} {results_df['context_count'].mean():>8.1f}"
        )
        print("═" * width + "\n")
