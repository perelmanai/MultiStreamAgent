"""Backend package — LLM and ASR backend abstractions."""

from .asr_backend import (
    DEFAULT_ASR,
    GeminiASRBackend,
    WhisperASRBackend,
    get_asr_choices,
)
from .base import ASRBackend, BackendWorker, LLMBackend, QueueItem
from .llm_backend import GeminiBackend, LocalBackend

# Backward-compatible alias
Backend = LLMBackend

__all__ = [
    "ASRBackend",
    "Backend",
    "BackendWorker",
    "DEFAULT_ASR",
    "GeminiASRBackend",
    "GeminiBackend",
    "LLMBackend",
    "LocalBackend",
    "QueueItem",
    "WhisperASRBackend",
    "get_asr_choices",
]
