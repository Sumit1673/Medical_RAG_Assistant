"""Download and prepare medical dataset for RAG system."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_and_prepare_dataset(output_dir: str = "dataset/medical/") -> None:
    """
    Download medical dataset from HuggingFace and prepare for RAG.

    Args:
        output_dir: Output directory for dataset files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing medical dataset in {output_dir}")

    try:
        from datasets import load_dataset

        logger.info("Downloading medical dataset from HuggingFace...")

        # Try to load a medical dataset
        try:
            # Using a smaller medical dataset
            dataset = load_dataset("medalpaca/medical_meadow_wikidoc", split="train", streaming=False)
            logger.info(f"Loaded dataset with {len(dataset)} samples")

            # Save first 200 documents as text files
            for idx, sample in enumerate(dataset):
                if idx >= 200:
                    break

                # Extract text content
                content = sample.get("instruction", "") + "\n" + sample.get("input", "") + "\n" + sample.get("output", "")

                # Save to file
                file_path = output_dir / f"medical_{idx:04d}.txt"
                with open(file_path, "w") as f:
                    f.write(content)

                if (idx + 1) % 50 == 0:
                    logger.info(f"Saved {idx + 1} documents")

        except Exception as e:
            logger.warning(f"Could not load medalpaca dataset: {e}")
            logger.info("Creating sample medical documents instead...")
            _create_sample_documents(output_dir)

        # Create evaluation samples
        _create_eval_samples(output_dir)
        logger.info("Dataset preparation complete")

    except ImportError:
        logger.error("datasets library not installed. Install with: pip install datasets")
        _create_sample_documents(output_dir)
        _create_eval_samples(output_dir)


def _create_sample_documents(output_dir: Path) -> None:
    """Create sample medical documents for demo purposes."""
    sample_texts = [
        """The heart is a muscular organ that pumps blood throughout the body.
        It is divided into four chambers: two atria and two ventricles. The right atrium receives
        deoxygenated blood from the body, while the left atrium receives oxygenated blood from the lungs.""",

        """Hypertension, commonly known as high blood pressure, is a condition where the force of
        blood against artery walls is consistently too high. This can damage blood vessels and increase
        the risk of heart disease and stroke.""",

        """Type 2 diabetes is a metabolic disorder characterized by high blood glucose levels resulting
        from insulin resistance and relative insulin deficiency. It is the most common type of diabetes.""",

        """Cholesterol is a waxy substance found in the blood that is essential for various bodily functions.
        However, high levels can lead to plaque buildup in arteries, increasing risk of heart disease.""",

        """The respiratory system consists of the lungs and airways. Gas exchange occurs in the alveoli,
        where oxygen enters the bloodstream and carbon dioxide is expelled from the body.""",

        """Obesity is defined as having a body mass index (BMI) of 30 or higher. It increases the risk
        of developing various health conditions including diabetes, heart disease, and certain cancers.""",

        """Inflammation is the body's response to injury or infection. While acute inflammation is protective,
        chronic inflammation can contribute to various diseases including arthritis and cardiovascular disease.""",

        """The immune system protects the body from harmful invaders like bacteria, viruses, and cancer cells.
        It consists of white blood cells, antibodies, and various organs including the lymph nodes and spleen.""",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, text in enumerate(sample_texts):
        file_path = output_dir / f"sample_medical_{idx:04d}.txt"
        with open(file_path, "w") as f:
            f.write(text)

    logger.info(f"Created {len(sample_texts)} sample medical documents")


def _create_eval_samples(output_dir: Path) -> None:
    """Create evaluation samples for RAGAS evaluation."""
    eval_samples = [
        {
            "question": "What is the function of the heart?",
            "answer": "The heart is a muscular organ that pumps blood throughout the body.",
        },
        {
            "question": "What is hypertension?",
            "answer": "Hypertension is high blood pressure caused by the force of blood against artery walls being consistently too high.",
        },
        {
            "question": "Define Type 2 diabetes.",
            "answer": "Type 2 diabetes is a metabolic disorder characterized by high blood glucose levels due to insulin resistance.",
        },
        {
            "question": "What role does cholesterol play in the body?",
            "answer": "Cholesterol is a waxy substance essential for bodily functions but high levels can cause plaque buildup in arteries.",
        },
        {
            "question": "How does the respiratory system work?",
            "answer": "The respiratory system enables gas exchange where oxygen enters the bloodstream and carbon dioxide is expelled.",
        },
        {
            "question": "What is obesity?",
            "answer": "Obesity is defined as having a BMI of 30 or higher and increases risk of various health conditions.",
        },
        {
            "question": "What is inflammation?",
            "answer": "Inflammation is the body's response to injury or infection, but chronic inflammation can contribute to diseases.",
        },
        {
            "question": "How does the immune system protect the body?",
            "answer": "The immune system protects through white blood cells, antibodies, and organs like lymph nodes and spleen.",
        },
        {
            "question": "What are risk factors for heart disease?",
            "answer": "Risk factors include high blood pressure, high cholesterol, obesity, smoking, and diabetes.",
        },
        {
            "question": "What is the normal blood pressure range?",
            "answer": "Normal blood pressure is less than 120/80 mmHg. Elevated is 120-129/<80, and Stage 1 hypertension is 130-139/80-89.",
        },
    ]

    eval_path = output_dir.parent / "eval_samples.json"
    with open(eval_path, "w") as f:
        json.dump(eval_samples, f, indent=2)

    logger.info(f"Created evaluation samples at {eval_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    download_and_prepare_dataset()
