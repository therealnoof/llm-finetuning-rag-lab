"""
Configuration settings for the F5 AI Technical Assistant Training Lab.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """Model configuration settings."""
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    max_grad_norm: float = 0.3
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = False
    seed: int = 42


@dataclass
class RAGConfig:
    """RAG system configuration."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 3
    collection_name: str = "f5_docs"


@dataclass
class PathConfig:
    """Path configuration settings."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def f5_docs_dir(self) -> Path:
        return self.data_dir / "f5_docs"

    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training"

    @property
    def train_file(self) -> Path:
        return self.training_dir / "f5_qa_train.jsonl"

    @property
    def eval_file(self) -> Path:
        return self.training_dir / "f5_qa_eval.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.base_dir / "chroma_db"

    @property
    def output_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"


class Config:
    """Main configuration class combining all settings."""

    def __init__(self, colab_mode: bool = False):
        """
        Initialize configuration.

        Args:
            colab_mode: If True, adjust paths for Google Colab environment.
        """
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.rag = RAGConfig()
        self.paths = PathConfig()
        self.colab_mode = colab_mode

        if colab_mode:
            self._setup_colab_paths()

    def _setup_colab_paths(self):
        """Adjust paths for Google Colab environment."""
        # In Colab, the repo is typically cloned to /content/
        colab_base = Path("/content/LLM-FineTuning-RAG-Lab")
        if colab_base.exists():
            self.paths = PathConfig(base_dir=colab_base)

    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    @staticmethod
    def check_gpu() -> dict:
        """Check GPU availability and memory."""
        import torch

        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "device_name": None,
            "total_memory_gb": 0,
            "free_memory_gb": 0
        }

        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
            info["free_memory_gb"] = round(
                (torch.cuda.get_device_properties(0).total_memory -
                 torch.cuda.memory_allocated(0)) / 1e9, 2
            )

        return info

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.paths.chroma_dir, self.paths.output_dir, self.paths.results_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def get_chat_template(self) -> str:
        """Get the chat template for TinyLlama."""
        return """<|system|>
{system_message}</s>
<|user|>
{user_message}</s>
<|assistant|>
"""

    def format_prompt(self, user_message: str, system_message: Optional[str] = None) -> str:
        """
        Format a prompt using TinyLlama's chat template.

        Args:
            user_message: The user's question or input.
            system_message: Optional system message for context.

        Returns:
            Formatted prompt string.
        """
        if system_message is None:
            system_message = "You are an F5 Networks technical expert. Provide accurate, detailed answers about BIG-IP, iRules, load balancing, SSL offloading, and other F5 technologies."

        return self.get_chat_template().format(
            system_message=system_message,
            user_message=user_message
        )

    def __repr__(self) -> str:
        return f"Config(colab_mode={self.colab_mode})"


# Singleton instance for easy access
_config: Optional[Config] = None


def get_config(colab_mode: Optional[bool] = None) -> Config:
    """
    Get or create the global configuration instance.

    Args:
        colab_mode: Override colab detection. If None, auto-detect.

    Returns:
        Config instance.
    """
    global _config

    if _config is None:
        if colab_mode is None:
            colab_mode = Config.is_colab()
        _config = Config(colab_mode=colab_mode)

    return _config
