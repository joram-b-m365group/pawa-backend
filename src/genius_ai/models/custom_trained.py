"""Custom trained model wrapper for Genius AI."""

import torch
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from genius_ai.core.logger import logger
from genius_ai.models.base import BaseModel, GenerationConfig, ModelResponse


class CustomTrainedModel(BaseModel):
    """Wrapper for our custom trained DistilGPT-2 model."""

    def __init__(
        self,
        model_path: str = "./tiny_genius_model",
        device: str = "cpu",
    ):
        """Initialize the custom trained model.

        Args:
            model_path: Path to the trained model directory
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__(model_name="custom-distilgpt2")
        self.model_path = Path(model_path)
        self.device_str = device
        self.model = None
        self.tokenizer = None
        self._initialized = False

    async def initialize(self) -> None:
        """Load the trained model and tokenizer."""
        if self._initialized:
            return

        try:
            logger.info(f"Loading custom trained model from {self.model_path}")

            # Check if model exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Please run the training script first."
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,  # CPU mode
            )

            # Move to device
            if self.device_str == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("Model loaded on GPU")
            else:
                self.model = self.model.to("cpu")
                logger.info("Model loaded on CPU")

            self.model.eval()  # Set to evaluation mode
            self._initialized = True

            logger.info("Custom trained model initialized successfully")
            logger.info(f"Model parameters: ~82M")
            logger.info(f"Model size: ~313MB")

        except Exception as e:
            logger.error(f"Failed to initialize custom model: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate response from the custom trained model.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            ModelResponse with generated text
        """
        if not self._initialized:
            await self.initialize()

        config = config or GenerationConfig()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move inputs to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    do_sample=config.temperature > 0,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Remove the prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            # Count tokens
            prompt_tokens = len(inputs["input_ids"][0])
            completion_tokens = len(outputs[0]) - prompt_tokens

            return ModelResponse(
                text=generated_text,
                finish_reason="stop" if outputs[0][-1] == self.tokenizer.eos_token_id else "length",
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream generation from the custom model.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Yields:
            Generated text chunks
        """
        if not self._initialized:
            await self.initialize()

        config = config or GenerationConfig()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with streaming
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": config.max_tokens,
                "temperature": config.temperature,
                "do_sample": config.temperature > 0,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }

            # Run generation in separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield chunks as they're generated
            for text_chunk in streamer:
                yield text_chunk

            thread.join()

        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not self.tokenizer:
            return len(text.split())  # Rough estimate

        return len(self.tokenizer.encode(text))

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text.

        Args:
            text: Text to generate embeddings for

        Returns:
            List of floats representing the embedding
        """
        # For now, return a simple embedding based on token counts
        # In a real implementation, you'd use a proper embedding model
        if not self.tokenizer:
            await self.initialize()

        # Get last hidden state as embedding
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state as embedding
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().tolist()

        return embedding if isinstance(embedding, list) else [embedding]

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False
        logger.info("Custom model cleaned up")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "name": "Custom Trained DistilGPT-2",
            "path": str(self.model_path),
            "parameters": "82M",
            "size": "313MB",
            "device": self.device_str,
            "initialized": self._initialized,
            "base_model": "distilgpt2",
            "training": "Fine-tuned on Python programming knowledge",
        }
