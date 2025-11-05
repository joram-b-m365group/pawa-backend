"""Base model interface for all LLM implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator


class ModelType(str, Enum):
    """Supported model types."""

    MISTRAL = "mistral"
    LLAMA = "llama"
    GPT = "gpt"
    CLAUDE = "claude"
    CUSTOM = "custom"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: list[str] | None = None
    stream: bool = False


@dataclass
class ModelResponse:
    """Response from model generation."""

    text: str
    model_name: str
    tokens_used: int
    finish_reason: str
    metadata: dict[str, Any] | None = None


class BaseModel(ABC):
    """Abstract base class for all language models."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the model.

        Args:
            model_name: Name or path of the model
            device: Device to run model on (cuda, mps, cpu)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize and load the model."""
        pass

    @abstractmethod
    async def generate(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> ModelResponse:
        """Generate text from prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            ModelResponse with generated text
        """
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> AsyncIterator[str]:
        """Generate text from prompt with streaming.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Yields:
            Chunks of generated text
        """
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


class ModelFactory:
    """Factory for creating model instances."""

    _registry: dict[ModelType, type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: ModelType, model_class: type[BaseModel]) -> None:
        """Register a model class.

        Args:
            model_type: Type of model
            model_class: Model class to register
        """
        cls._registry[model_type] = model_class

    @classmethod
    def create(
        cls, model_type: ModelType, model_name: str, device: str = "cuda"
    ) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Type of model to create
            model_name: Name or path of the model
            device: Device to run model on

        Returns:
            Model instance

        Raises:
            ValueError: If model type not registered
        """
        if model_type not in cls._registry:
            raise ValueError(f"Model type {model_type} not registered")

        model_class = cls._registry[model_type]
        return model_class(model_name=model_name, device=device)
