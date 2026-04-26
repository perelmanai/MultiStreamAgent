"""LLM backend implementations — Local Qwen and Gemini API."""

import logging

import models

from .base import LLMBackend

logger = logging.getLogger(__name__)


class LocalBackend(LLMBackend):
    """Backend using a locally loaded Qwen model."""

    def __init__(self, model_key: str | None = None):
        self._model_handle = None
        self._model_key: str | None = None
        if model_key:
            self.load_model(model_key)

    def generate(self, question: str, history: list[dict]) -> str:
        if self._model_handle is None:
            raise RuntimeError("No model loaded in LocalBackend")
        return models.generate_full_response(self._model_handle, question, history)

    def load_model(self, model_key: str) -> None:
        if self._model_handle is not None:
            self.unload_model()
        logger.info("LocalBackend loading model: %s", model_key)
        self._model_handle = models.load_qwen(model_key)
        self._model_key = model_key
        logger.info("LocalBackend model loaded: %s", model_key)

    def unload_model(self) -> None:
        if self._model_handle is not None:
            logger.info("LocalBackend unloading model: %s", self._model_key)
            models.unload_model(self._model_handle)
            self._model_handle = None
            self._model_key = None

    @property
    def model_key(self) -> str | None:
        return self._model_key


class GeminiBackend(LLMBackend):
    """Backend using Gemini API."""

    def __init__(self, model_key: str | None = None):
        from gemini_client import GEMINI_DEFAULT_MODEL

        self._model_key = model_key or GEMINI_DEFAULT_MODEL

    def generate(self, question: str, history: list[dict]) -> str:
        from gemini_client import generate_gemini_response

        return generate_gemini_response(
            model_key=self._model_key,
            user_text=question,
            history=history,
            system_prompt=models.BACKEND_SYSTEM_PROMPT,
            max_tokens=1024,
        )

    def load_model(self, model_key: str) -> None:
        logger.info("GeminiBackend switching to model: %s", model_key)
        self._model_key = model_key

    def unload_model(self) -> None:
        pass

    @property
    def model_key(self) -> str | None:
        return self._model_key
