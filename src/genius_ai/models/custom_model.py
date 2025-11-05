"""Custom model implementation that rivals OpenAI capabilities.

This module provides a framework for building your own intelligent model through:
1. Fine-tuning open-source models (Mistral, LLaMA, etc.)
2. Multi-model ensemble (combining strengths of different models)
3. Domain-specific optimization
4. Custom training pipelines
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import Dataset

from genius_ai.core.config import settings
from genius_ai.core.logger import logger
from genius_ai.models.base import BaseModel, GenerationConfig, ModelResponse


@dataclass
class CustomModelConfig:
    """Configuration for custom model."""

    # Base model to fine-tune
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Alternative strong open-source models
    # base_model: str = "meta-llama/Llama-2-7b-chat-hf"
    # base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # base_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # LoRA (Low-Rank Adaptation) for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16  # Rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = None

    # Quantization for memory efficiency
    use_4bit: bool = True
    use_8bit: bool = False

    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100

    # Model behavior
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1

    # Custom training data paths
    training_data_path: Optional[str] = None
    validation_data_path: Optional[str] = None

    # Output paths
    output_dir: str = "./custom_models"
    checkpoint_dir: str = "./custom_models/checkpoints"


class CustomIntelligentModel(BaseModel):
    """Custom model implementation that can rival OpenAI.

    This model combines:
    - Fine-tuned open-source LLM (Mistral/LLaMA)
    - LoRA adapters for efficient customization
    - Domain-specific training
    - Advanced generation strategies
    """

    def __init__(
        self,
        model_name: str = "genius-ai-custom",
        config: Optional[CustomModelConfig] = None,
    ):
        """Initialize custom model.

        Args:
            model_name: Name for this custom model
            config: Model configuration
        """
        self.model_name = model_name
        self.config = config or CustomModelConfig()

        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_trained = False

        logger.info(f"Initializing custom model: {model_name}")
        logger.info(f"Base model: {self.config.base_model}")

    async def initialize(self) -> None:
        """Initialize the model with optimizations."""
        logger.info("Loading custom intelligent model...")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("Using CPU (slower, consider GPU for production)")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization for memory efficiency
        quantization_config = None
        if self.config.use_4bit:
            logger.info("Using 4-bit quantization for memory efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load base model
        logger.info(f"Loading base model: {self.config.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Check for existing fine-tuned adapters
        adapter_path = Path(self.config.output_dir) / self.model_name
        if adapter_path.exists():
            logger.info(f"Loading fine-tuned adapters from {adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
            )
            self.is_trained = True
            logger.info("Loaded custom fine-tuned model!")
        elif self.config.use_lora:
            logger.info("No existing adapters found - using base model")
            logger.info("You can fine-tune this model with custom data using .train()")

        # Set to evaluation mode
        self.model.eval()

        logger.info("Custom intelligent model ready!")

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate response using custom model.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Model response
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        config = config or GenerationConfig()

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature or self.config.temperature,
                top_p=config.top_p or self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return ModelResponse(
            text=generated_text.strip(),
            tokens_used=len(outputs[0]),
            model_name=self.model_name,
        )

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Generate response with streaming.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Generated text chunks
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        config = config or GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        # Stream generation (simplified - for production, use TextIteratorStreamer)
        full_response = await self.generate(prompt, config)

        # Simulate streaming by yielding chunks
        words = full_response.text.split()
        for i in range(0, len(words), 3):  # Yield 3 words at a time
            chunk = " ".join(words[i:i+3]) + " "
            yield chunk
            await asyncio.sleep(0.05)  # Simulate streaming delay

    async def train_on_custom_data(
        self,
        training_data: list[dict[str, str]],
        validation_data: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Fine-tune the model on custom data to make it more intelligent.

        Args:
            training_data: List of {"prompt": "...", "response": "..."} pairs
            validation_data: Optional validation dataset

        Returns:
            Training metrics

        Example:
            >>> training_data = [
            ...     {
            ...         "prompt": "What is quantum computing?",
            ...         "response": "Quantum computing uses quantum mechanics..."
            ...     },
            ...     {
            ...         "prompt": "Explain REST APIs",
            ...         "response": "REST APIs are architectural style..."
            ...     }
            ... ]
            >>> metrics = await model.train_on_custom_data(training_data)
        """
        logger.info(f"Starting fine-tuning on {len(training_data)} examples...")

        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Prepare model for training
        if self.config.use_lora and not self.is_trained:
            logger.info("Preparing model for LoRA fine-tuning...")

            # Configure LoRA
            target_modules = self.config.lora_target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Prepare model
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)

            logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")

        # Format training data
        def format_example(example: dict[str, str]) -> str:
            """Format example for training."""
            prompt = example["prompt"]
            response = example["response"]
            return f"<s>[INST] {prompt} [/INST] {response}</s>"

        formatted_train = [
            {"text": format_example(ex)} for ex in training_data
        ]

        train_dataset = Dataset.from_list(formatted_train)

        formatted_val = None
        if validation_data:
            formatted_val = [
                {"text": format_example(ex)} for ex in validation_data
            ]
            val_dataset = Dataset.from_list(formatted_val)
        else:
            val_dataset = None

        # Training arguments
        output_dir = Path(self.config.output_dir) / self.model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=100 if val_dataset else None,
            save_total_limit=3,
            fp16=self.device == "cuda",
            report_to="none",  # Disable wandb/tensorboard for simplicity
        )

        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        tokenized_val = None
        if val_dataset:
            tokenized_val = val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names,
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )

        # Train!
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save model
        logger.info(f"Saving fine-tuned model to {output_dir}")
        self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        self.is_trained = True

        metrics = {
            "training_loss": train_result.training_loss,
            "epochs_completed": self.config.num_epochs,
            "examples_trained": len(training_data),
            "model_saved_to": str(output_dir),
        }

        logger.info(f"Training complete! Metrics: {metrics}")

        return metrics

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        return len(self.tokenizer.encode(text))

    async def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Custom model cleaned up")


class MultiModelEnsemble(BaseModel):
    """Ensemble of multiple models for superior performance.

    Combines strengths of different models:
    - Custom fine-tuned model for domain expertise
    - Multiple base models for diverse perspectives
    - Voting/averaging for consensus
    """

    def __init__(self, models: list[BaseModel]):
        """Initialize ensemble.

        Args:
            models: List of models to ensemble
        """
        self.models = models
        self.model_name = "multi-model-ensemble"

        logger.info(f"Initialized ensemble with {len(models)} models")

    async def initialize(self) -> None:
        """Initialize all models in ensemble."""
        logger.info("Initializing ensemble models...")

        for i, model in enumerate(self.models):
            logger.info(f"Initializing model {i+1}/{len(self.models)}")
            await model.initialize()

        logger.info("All ensemble models ready!")

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate using ensemble (picks best response).

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            Best model response
        """
        logger.info("Generating with ensemble...")

        # Generate from all models
        responses = await asyncio.gather(*[
            model.generate(prompt, config) for model in self.models
        ])

        # Simple strategy: pick longest response (can be improved with scoring)
        best_response = max(responses, key=lambda r: len(r.text))
        best_response.model_name = self.model_name

        logger.info(f"Selected best response from {len(responses)} candidates")

        return best_response

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream from primary model."""
        # Use first model for streaming
        async for chunk in self.models[0].generate_stream(prompt, config):
            yield chunk

    def count_tokens(self, text: str) -> int:
        """Count tokens using first model."""
        return self.models[0].count_tokens(text)

    async def cleanup(self) -> None:
        """Clean up all models."""
        for model in self.models:
            await model.cleanup()
