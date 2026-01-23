"""
Evaluation utilities for comparing model approaches.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime


@dataclass
class EvaluationResult:
    """Container for a single evaluation result."""
    question: str
    baseline_response: str
    rag_response: str
    finetuned_response: str
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Scoring rubric for responses."""
    accuracy: float = 0.0        # Factual correctness (0-5)
    completeness: float = 0.0    # Coverage of topic (0-5)
    specificity: float = 0.0     # F5-specific detail (0-5)
    actionability: float = 0.0   # Practical usefulness (0-5)
    clarity: float = 0.0         # Clear and understandable (0-5)

    @property
    def total(self) -> float:
        """Calculate total score out of 25."""
        return self.accuracy + self.completeness + self.specificity + self.actionability + self.clarity

    @property
    def average(self) -> float:
        """Calculate average score out of 5."""
        return self.total / 5

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "specificity": self.specificity,
            "actionability": self.actionability,
            "clarity": self.clarity,
            "total": self.total,
            "average": self.average
        }


class Evaluator:
    """Handles evaluation and comparison of model responses."""

    SCORING_RUBRIC = """
## Scoring Rubric (0-5 scale for each criterion)

### Accuracy (0-5)
- 0: Completely incorrect or misleading
- 1: Mostly incorrect with minor accurate elements
- 2: Mix of correct and incorrect information
- 3: Mostly accurate with some errors
- 4: Accurate with minor issues
- 5: Completely accurate

### Completeness (0-5)
- 0: Does not address the question
- 1: Addresses only a small part
- 2: Partially addresses the question
- 3: Addresses most aspects
- 4: Comprehensive with minor gaps
- 5: Fully comprehensive

### Specificity (0-5)
- 0: No F5-specific information
- 1: Vague references to F5
- 2: Some F5-specific details
- 3: Good F5-specific content
- 4: Detailed F5-specific information
- 5: Expert-level F5-specific details

### Actionability (0-5)
- 0: No practical guidance
- 1: Vague suggestions only
- 2: Some practical elements
- 3: Reasonably actionable
- 4: Clear, actionable guidance
- 5: Immediately actionable with specifics

### Clarity (0-5)
- 0: Incomprehensible
- 1: Very difficult to understand
- 2: Somewhat unclear
- 3: Reasonably clear
- 4: Clear and well-organized
- 5: Exceptionally clear and well-structured
"""

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the evaluator.

        Args:
            results_dir: Directory for saving evaluation results.
        """
        base_dir = Path(__file__).parent.parent
        self.results_dir = results_dir or base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[EvaluationResult] = []

    def add_result(
        self,
        question: str,
        baseline_response: str,
        rag_response: str,
        finetuned_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Add an evaluation result.

        Args:
            question: The test question.
            baseline_response: Response from base model.
            rag_response: Response from RAG-enhanced model.
            finetuned_response: Response from fine-tuned model.
            metadata: Optional additional metadata.

        Returns:
            The created EvaluationResult.
        """
        result = EvaluationResult(
            question=question,
            baseline_response=baseline_response,
            rag_response=rag_response,
            finetuned_response=finetuned_response,
            metadata=metadata or {}
        )
        self.results.append(result)
        return result

    def score_response(
        self,
        result_index: int,
        approach: str,
        accuracy: float,
        completeness: float,
        specificity: float,
        actionability: float,
        clarity: float
    ):
        """
        Score a specific response.

        Args:
            result_index: Index of the result to score.
            approach: One of 'baseline', 'rag', or 'finetuned'.
            accuracy: Accuracy score (0-5).
            completeness: Completeness score (0-5).
            specificity: Specificity score (0-5).
            actionability: Actionability score (0-5).
            clarity: Clarity score (0-5).
        """
        if result_index >= len(self.results):
            raise IndexError(f"Result index {result_index} out of range")

        if approach not in ['baseline', 'rag', 'finetuned']:
            raise ValueError(f"Invalid approach: {approach}")

        metrics = EvaluationMetrics(
            accuracy=accuracy,
            completeness=completeness,
            specificity=specificity,
            actionability=actionability,
            clarity=clarity
        )

        self.results[result_index].scores[approach] = metrics.to_dict()

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all evaluations.

        Returns:
            Dictionary with summary statistics.
        """
        approaches = ['baseline', 'rag', 'finetuned']
        summary = {approach: {} for approach in approaches}

        for approach in approaches:
            scores = []
            for result in self.results:
                if approach in result.scores:
                    scores.append(result.scores[approach])

            if scores:
                # Calculate averages for each criterion
                criteria = ['accuracy', 'completeness', 'specificity', 'actionability', 'clarity', 'total', 'average']
                for criterion in criteria:
                    values = [s[criterion] for s in scores]
                    summary[approach][criterion] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
                summary[approach]['count'] = len(scores)

        return summary

    def format_comparison(self, result: EvaluationResult) -> str:
        """
        Format a side-by-side comparison of responses.

        Args:
            result: The evaluation result to format.

        Returns:
            Formatted comparison string.
        """
        output = []
        output.append("=" * 80)
        output.append(f"QUESTION: {result.question}")
        output.append("=" * 80)

        output.append("\n--- BASELINE RESPONSE ---")
        output.append(result.baseline_response)

        output.append("\n--- RAG RESPONSE ---")
        output.append(result.rag_response)

        output.append("\n--- FINE-TUNED RESPONSE ---")
        output.append(result.finetuned_response)

        if result.scores:
            output.append("\n--- SCORES ---")
            for approach, scores in result.scores.items():
                output.append(f"\n{approach.upper()}:")
                for criterion, value in scores.items():
                    if criterion not in ['total', 'average']:
                        output.append(f"  {criterion}: {value}")
                output.append(f"  TOTAL: {scores.get('total', 'N/A')}/25")

        return "\n".join(output)

    def save_results(self, filename: Optional[str] = None):
        """
        Save evaluation results to JSON.

        Args:
            filename: Optional filename. Defaults to timestamp-based name.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"

        filepath = self.results_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "question": r.question,
                    "responses": {
                        "baseline": r.baseline_response,
                        "rag": r.rag_response,
                        "finetuned": r.finetuned_response
                    },
                    "scores": r.scores,
                    "metadata": r.metadata
                }
                for r in self.results
            ],
            "summary": self.get_summary_stats()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filepath}")
        return filepath

    def load_results(self, filename: str):
        """
        Load evaluation results from JSON.

        Args:
            filename: Name of the file to load.
        """
        filepath = self.results_dir / filename

        with open(filepath, "r") as f:
            data = json.load(f)

        self.results = []
        for item in data["results"]:
            result = EvaluationResult(
                question=item["question"],
                baseline_response=item["responses"]["baseline"],
                rag_response=item["responses"]["rag"],
                finetuned_response=item["responses"]["finetuned"],
                scores=item.get("scores", {}),
                metadata=item.get("metadata", {})
            )
            self.results.append(result)

        print(f"Loaded {len(self.results)} results from {filepath}")

    def create_visualization(self, save_path: Optional[Path] = None):
        """
        Create visualization charts for evaluation results.

        Args:
            save_path: Optional path to save the figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        summary = self.get_summary_stats()

        approaches = ['baseline', 'rag', 'finetuned']
        criteria = ['accuracy', 'completeness', 'specificity', 'actionability', 'clarity']

        # Prepare data
        data = {}
        for approach in approaches:
            if approach in summary and summary[approach]:
                data[approach] = [summary[approach][c]['mean'] for c in criteria]
            else:
                data[approach] = [0] * len(criteria)

        # Create bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Chart 1: Grouped bar chart by criteria
        x = np.arange(len(criteria))
        width = 0.25

        ax1 = axes[0]
        for i, approach in enumerate(approaches):
            offset = (i - 1) * width
            bars = ax1.bar(x + offset, data[approach], width, label=approach.title())

        ax1.set_xlabel('Evaluation Criteria')
        ax1.set_ylabel('Score (0-5)')
        ax1.set_title('Comparison by Criteria')
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.title() for c in criteria], rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 5.5)
        ax1.grid(axis='y', alpha=0.3)

        # Chart 2: Overall average comparison
        ax2 = axes[1]
        averages = []
        for approach in approaches:
            if approach in summary and summary[approach]:
                averages.append(summary[approach]['average']['mean'])
            else:
                averages.append(0)

        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        bars = ax2.bar(approaches, averages, color=colors)

        ax2.set_xlabel('Approach')
        ax2.set_ylabel('Average Score (0-5)')
        ax2.set_title('Overall Performance Comparison')
        ax2.set_ylim(0, 5.5)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, averages):
            ax2.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        return fig

    def print_rubric(self):
        """Print the scoring rubric."""
        print(self.SCORING_RUBRIC)

    def generate_report(self) -> str:
        """
        Generate a text report of the evaluation.

        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 80)
        report.append("F5 AI TECHNICAL ASSISTANT EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Questions Evaluated: {len(self.results)}")

        summary = self.get_summary_stats()

        report.append("\n" + "-" * 40)
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)

        for approach in ['baseline', 'rag', 'finetuned']:
            if approach in summary and summary[approach]:
                stats = summary[approach]
                report.append(f"\n{approach.upper()}:")
                report.append(f"  Average Score: {stats['average']['mean']:.2f}/5")
                report.append(f"  Total Score: {stats['total']['mean']:.1f}/25")
                report.append("  Criteria Breakdown:")
                for criterion in ['accuracy', 'completeness', 'specificity', 'actionability', 'clarity']:
                    if criterion in stats:
                        report.append(f"    - {criterion.title()}: {stats[criterion]['mean']:.2f}")

        report.append("\n" + "-" * 40)
        report.append("KEY FINDINGS")
        report.append("-" * 40)

        # Determine winner
        if summary:
            averages = {}
            for approach in ['baseline', 'rag', 'finetuned']:
                if approach in summary and 'average' in summary[approach]:
                    averages[approach] = summary[approach]['average']['mean']

            if averages:
                winner = max(averages, key=averages.get)
                report.append(f"\nBest Overall Performer: {winner.upper()}")

                # Improvement over baseline
                if 'baseline' in averages:
                    baseline = averages['baseline']
                    for approach in ['rag', 'finetuned']:
                        if approach in averages:
                            improvement = ((averages[approach] - baseline) / baseline) * 100
                            report.append(f"{approach.upper()} improvement over baseline: {improvement:+.1f}%")

        report.append("\n" + "=" * 80)

        return "\n".join(report)
