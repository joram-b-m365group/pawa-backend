"""Implementation of transformer-based language models."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread
from typing import AsyncIterator

from genius_ai.core.config import settings
from genius_ai.core.logger import logger
from genius_ai.models.base import (
    BaseModel,
    GenerationConfig,
    ModelResponse,
    ModelType,
    ModelFactory,
)


class TransformerModel(BaseModel):
    """Transformer-based language model implementation."""

    def __init__(self, model_name: str, device: str = "cuda", use_quantization: bool = True):
        """Initialize transformer model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to run on
            use_quantization: Whether to use 4-bit quantization
        """
        super().__init__(model_name, device)
        self.use_quantization = use_quantization
        self._streamer = None

    async def initialize(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=settings.model_cache_dir,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Configure quantization if enabled
        if self.use_quantization and self.device == "cuda":
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            cache_dir=settings.model_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        if self.device != "cuda" and quantization_config is None:
            self._model = self._model.to(self.device)

        self._model.eval()
        logger.info("Model loaded successfully")

    async def generate(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> ModelResponse:
        """Generate text from prompt."""
        if self._model is None or self._tokenizer is None:
            await self.initialize()

        config = config or GenerationConfig()

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.memory_window_size,
        ).to(self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Count tokens
        tokens_used = outputs.shape[1]

        return ModelResponse(
            text=generated_text,
            model_name=self.model_name,
            tokens_used=tokens_used,
            finish_reason="stop",
        )

    async def generate_stream(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        if self._model is None or self._tokenizer is None:
            await self.initialize()

        config = config or GenerationConfig()

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.memory_window_size,
        ).to(self._model.device)

        # Setup streamer
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generate in thread
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output
        for text in streamer:
            yield text

        thread.join()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding (using hidden states)."""
        if self._model is None or self._tokenizer is None:
            await self.initialize()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            # Use mean of last hidden state
            embeddings = outputs.hidden_states[-1].mean(dim=1)

        return embeddings[0].cpu().tolist()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer is None:
            # Rough estimation if tokenizer not loaded
            return len(text.split()) * 1.3

        return len(self._tokenizer.encode(text))


# Register the model
ModelFactory.register(ModelType.MISTRAL, TransformerModel)
ModelFactory.register(ModelType.LLAMA, TransformerModel)
ModelFactory.register(ModelType.CUSTOM, TransformerModel)
