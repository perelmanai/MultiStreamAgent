"""Client ABCs and shared data types."""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Client ABCs
# ---------------------------------------------------------------------------
class LLMClient(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    def generate(self, question: str, history: list[dict]) -> str: ...

    @abstractmethod
    def load_model(self, model_key: str) -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...


class ASRClient(ABC):
    """Abstract interface for ASR providers."""

    @abstractmethod
    def transcribe(self, sr: int, audio_data: np.ndarray) -> str: ...

    @abstractmethod
    def load_model(self, model_key: str) -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...


class TTSClient(ABC):
    """Abstract interface for TTS providers."""

    @abstractmethod
    def synthesize(self, text: str) -> str: ...

    @abstractmethod
    def get_voices(self) -> list[str]: ...

    @abstractmethod
    def set_voice(self, voice: str) -> None: ...


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------
@dataclass
class QueueItem:
    id: str
    question: str
    context_summary: str
    history: list[dict]
    status: str = "queued"  # queued | processing | ready | delivered
    answer: str | None = None
    timestamp: float = field(default_factory=time.time)


class TTSSource(Enum):
    FRONTEND = "frontend"
    BACKEND = "backend"


@dataclass
class TTSQueueItem:
    id: str
    text: str
    source: TTSSource
    context: str = ""
    status: str = "queued"  # queued | processing | ready | delivered
    audio_path: str | None = None
    audio_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_immediate(self) -> bool:
        return self.source == TTSSource.FRONTEND
