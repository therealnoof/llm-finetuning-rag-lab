"""
Data loading utilities for the F5 AI Technical Assistant Training Lab.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass


@dataclass
class QAPair:
    """Represents a question-answer pair."""
    question: str
    answer: str
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DataLoader:
    """Utility class for loading training and evaluation data."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory. If None, uses default.
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.f5_docs_dir = self.data_dir / "f5_docs"
        self.training_dir = self.data_dir / "training"

    def load_f5_documents(self) -> List[Dict[str, str]]:
        """
        Load all F5 documentation files.

        Returns:
            List of dictionaries with 'content' and 'source' keys.
        """
        documents = []

        if not self.f5_docs_dir.exists():
            raise FileNotFoundError(f"F5 docs directory not found: {self.f5_docs_dir}")

        for doc_file in sorted(self.f5_docs_dir.glob("*.txt")):
            content = doc_file.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "source": doc_file.name,
                "path": str(doc_file)
            })

        return documents

    def load_training_data(self) -> List[QAPair]:
        """
        Load training Q&A pairs from JSONL file.

        Returns:
            List of QAPair objects.
        """
        train_file = self.training_dir / "f5_qa_train.jsonl"
        return self._load_qa_file(train_file)

    def load_eval_data(self) -> List[QAPair]:
        """
        Load evaluation Q&A pairs from JSONL file.

        Returns:
            List of QAPair objects.
        """
        eval_file = self.training_dir / "f5_qa_eval.jsonl"
        return self._load_qa_file(eval_file)

    def _load_qa_file(self, filepath: Path) -> List[QAPair]:
        """
        Load Q&A pairs from a JSONL file.

        Args:
            filepath: Path to the JSONL file.

        Returns:
            List of QAPair objects.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        qa_pairs = []
        with jsonlines.open(filepath) as reader:
            for item in reader:
                qa_pairs.append(QAPair(
                    question=item["question"],
                    answer=item["answer"],
                    category=item.get("category"),
                    difficulty=item.get("difficulty"),
                    metadata=item.get("metadata")
                ))

        return qa_pairs

    def iter_training_data(self) -> Iterator[QAPair]:
        """
        Iterate over training data without loading all into memory.

        Yields:
            QAPair objects.
        """
        train_file = self.training_dir / "f5_qa_train.jsonl"
        with jsonlines.open(train_file) as reader:
            for item in reader:
                yield QAPair(
                    question=item["question"],
                    answer=item["answer"],
                    category=item.get("category"),
                    difficulty=item.get("difficulty"),
                    metadata=item.get("metadata")
                )

    def prepare_for_training(self, qa_pairs: List[QAPair], template: str) -> List[Dict[str, str]]:
        """
        Prepare Q&A pairs for training with a specific prompt template.

        Args:
            qa_pairs: List of QAPair objects.
            template: Chat template string with {question} and {answer} placeholders.

        Returns:
            List of formatted training examples.
        """
        formatted = []
        for qa in qa_pairs:
            text = template.format(question=qa.question, answer=qa.answer)
            formatted.append({"text": text})

        return formatted

    def get_sample_questions(self, n: int = 5) -> List[str]:
        """
        Get sample questions for testing.

        Args:
            n: Number of questions to return.

        Returns:
            List of question strings.
        """
        eval_data = self.load_eval_data()
        return [qa.question for qa in eval_data[:n]]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dictionary with data statistics.
        """
        stats = {
            "documents": {},
            "training": {},
            "evaluation": {}
        }

        # Document stats
        try:
            docs = self.load_f5_documents()
            stats["documents"] = {
                "count": len(docs),
                "total_chars": sum(len(d["content"]) for d in docs),
                "files": [d["source"] for d in docs]
            }
        except FileNotFoundError:
            stats["documents"] = {"error": "Documents not found"}

        # Training data stats
        try:
            train_data = self.load_training_data()
            categories = {}
            for qa in train_data:
                cat = qa.category or "unknown"
                categories[cat] = categories.get(cat, 0) + 1

            stats["training"] = {
                "count": len(train_data),
                "categories": categories
            }
        except FileNotFoundError:
            stats["training"] = {"error": "Training data not found"}

        # Eval data stats
        try:
            eval_data = self.load_eval_data()
            stats["evaluation"] = {
                "count": len(eval_data)
            }
        except FileNotFoundError:
            stats["evaluation"] = {"error": "Evaluation data not found"}

        return stats

    @staticmethod
    def format_for_alpaca(question: str, answer: str, instruction: Optional[str] = None) -> Dict[str, str]:
        """
        Format a Q&A pair in Alpaca instruction format.

        Args:
            question: The question/input.
            answer: The answer/output.
            instruction: Optional instruction override.

        Returns:
            Dictionary in Alpaca format.
        """
        if instruction is None:
            instruction = "Answer the following question about F5 BIG-IP and related technologies."

        return {
            "instruction": instruction,
            "input": question,
            "output": answer
        }

    @staticmethod
    def format_for_chat(question: str, answer: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format a Q&A pair in chat message format.

        Args:
            question: The user's question.
            answer: The assistant's answer.
            system_message: Optional system message.

        Returns:
            List of chat messages.
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        return messages
