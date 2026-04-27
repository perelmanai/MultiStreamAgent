"""Backend package — LLM, ASR, and TTS backend abstractions."""

from .asr_backend import (
    DEFAULT_ASR,
    GeminiASRBackend,
    WhisperASRBackend,
    get_asr_choices,
)
from .base import ASRBackend, BackendWorker, LLMBackend, QueueItem
from .llm_backend import GeminiBackend, LocalBackend
from .tts_backend import (
    DEFAULT_TTS_VOICE,
    GEMINI_TTS_VOICES,
    GeminiTTSBackend,
    TTSBackend,
    TTSQueueItem,
    TTSQueueWorker,
)

# Backward-compatible alias
Backend = LLMBackend

__all__ = [
    "ASRBackend",
    "Backend",
    "BackendWorker",
    "DEFAULT_ASR",
    "DEFAULT_TTS_VOICE",
    "GEMINI_TTS_VOICES",
    "GeminiASRBackend",
    "GeminiBackend",
    "GeminiTTSBackend",
    "LLMBackend",
    "LocalBackend",
    "QueueItem",
    "TTSBackend",
    "TTSQueueItem",
    "TTSQueueWorker",
    "WhisperASRBackend",
    "get_asr_choices",
]
