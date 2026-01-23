"""
Training utilities for fine-tuning with QLoRA and Unsloth.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class TrainingResult:
    """Container for training results."""
    train_loss: float
    eval_loss: Optional[float]
    training_time: float
    steps: int
    epochs: int
    model_path: str


class TrainingHelper:
    """Helper class for model fine-tuning operations."""

    # TinyLlama chat template
    CHAT_TEMPLATE = """<|system|>
You are an F5 Networks technical expert. Provide accurate, detailed answers about BIG-IP, iRules, load balancing, SSL offloading, and other F5 technologies.</s>
<|user|>
{question}</s>
<|assistant|>
{answer}</s>"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the training helper.

        Args:
            output_dir: Directory for saving models and checkpoints.
        """
        base_dir = Path(__file__).parent.parent
        self.output_dir = output_dir or base_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_model_unsloth(
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True
    ):
        """
        Load a model using Unsloth for optimized training.

        Args:
            model_name: HuggingFace model name or path.
            max_seq_length: Maximum sequence length.
            load_in_4bit: Whether to load in 4-bit quantization.

        Returns:
            Tuple of (model, tokenizer).
        """
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit
        )

        return model, tokenizer

    @staticmethod
    def setup_lora(
        model,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None
    ):
        """
        Configure LoRA adapters for the model.

        Args:
            model: The base model.
            r: LoRA rank.
            lora_alpha: LoRA alpha scaling.
            lora_dropout: Dropout probability.
            target_modules: Modules to apply LoRA to.

        Returns:
            Model with LoRA adapters.
        """
        from unsloth import FastLanguageModel

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None
        )

        return model

    def prepare_dataset(
        self,
        train_file: Path,
        tokenizer,
        max_length: int = 2048
    ):
        """
        Prepare training dataset with proper formatting.

        Args:
            train_file: Path to training JSONL file.
            tokenizer: The tokenizer to use.
            max_length: Maximum sequence length.

        Returns:
            Prepared dataset.
        """
        from datasets import load_dataset

        # Load dataset
        dataset = load_dataset("json", data_files=str(train_file), split="train")

        def format_example(example):
            text = self.CHAT_TEMPLATE.format(
                question=example["question"],
                answer=example["answer"]
            )
            return {"text": text}

        # Format all examples
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)

        return dataset

    @staticmethod
    def get_training_args(
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_steps: int = -1
    ):
        """
        Get training arguments for the SFT trainer.

        Args:
            output_dir: Directory for outputs.
            num_epochs: Number of training epochs.
            batch_size: Per-device batch size.
            gradient_accumulation: Gradient accumulation steps.
            learning_rate: Learning rate.
            warmup_ratio: Warmup ratio.
            logging_steps: Logging frequency.
            save_steps: Checkpoint save frequency.
            max_steps: Maximum training steps (-1 for full epochs).

        Returns:
            TrainingArguments object.
        """
        from trl import SFTConfig

        return SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            max_steps=max_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            fp16=False,
            bf16=False,
            max_seq_length=2048,
            packing=False,
            dataset_text_field="text"
        )

    @staticmethod
    def create_trainer(model, tokenizer, dataset, training_args):
        """
        Create an SFT trainer.

        Args:
            model: The model to train.
            tokenizer: The tokenizer.
            dataset: The training dataset.
            training_args: Training arguments.

        Returns:
            SFTTrainer instance.
        """
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args
        )

        return trainer

    def save_model(self, model, tokenizer, save_name: str = "f5_assistant"):
        """
        Save the fine-tuned model.

        Args:
            model: The trained model.
            tokenizer: The tokenizer.
            save_name: Name for the saved model.

        Returns:
            Path to saved model.
        """
        save_path = self.output_dir / save_name
        save_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        print(f"Model saved to {save_path}")

        return save_path

    def save_merged_model(
        self,
        model,
        tokenizer,
        save_name: str = "f5_assistant_merged",
        quantization_method: str = "q4_k_m"
    ):
        """
        Save the model with LoRA weights merged.

        Args:
            model: The trained model with LoRA.
            tokenizer: The tokenizer.
            save_name: Name for the saved model.
            quantization_method: GGUF quantization method.

        Returns:
            Path to saved model.
        """
        save_path = self.output_dir / save_name

        # Save merged 16-bit model
        model.save_pretrained_merged(
            str(save_path),
            tokenizer,
            save_method="merged_16bit"
        )

        print(f"Merged model saved to {save_path}")

        return save_path

    @staticmethod
    def generate_response(
        model,
        tokenizer,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the model.

        Args:
            model: The model.
            tokenizer: The tokenizer.
            question: The input question.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.

        Returns:
            Generated response string.
        """
        from unsloth import FastLanguageModel

        # Enable inference mode
        FastLanguageModel.for_inference(model)

        # Format prompt
        prompt = f"""<|system|>
You are an F5 Networks technical expert. Provide accurate, detailed answers about BIG-IP, iRules, load balancing, SSL offloading, and other F5 technologies.</s>
<|user|>
{question}</s>
<|assistant|>
"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        return response

    def log_training_metrics(self, trainer, save_path: Optional[Path] = None):
        """
        Log and save training metrics.

        Args:
            trainer: The trainer with training history.
            save_path: Optional path to save metrics JSON.

        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "train_loss": trainer.state.log_history[-1].get("train_loss"),
            "total_steps": trainer.state.global_step,
            "epochs_completed": trainer.state.epoch
        }

        if save_path is None:
            save_path = self.output_dir / "training_metrics.json"

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory stats in GB.
        """
        import torch

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        }

    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache."""
        import torch
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
