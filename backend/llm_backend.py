"""Local LLM backend — Qwen model hosting on GPU."""

import logging

import models

logger = logging.getLogger(__name__)


class LocalLLMBackend:
    """Holds a Qwen model on GPU and runs inference."""

    def __init__(self, model_key: str | None = None):
        self._model_handle = None
        self._model_key: str | None = None
        if model_key:
            self.load_model(model_key)

    def generate(self, question: str, history: list[dict]) -> str:
        if self._model_handle is None:
            raise RuntimeError("No model loaded in LocalLLMBackend")
        return models.generate_full_response(self._model_handle, question, history)

    def load_model(self, model_key: str) -> None:
        if self._model_handle is not None:
            self.unload_model()
        logger.info("LocalLLMBackend loading model: %s", model_key)
        self._model_handle = models.load_qwen(model_key)
        self._model_key = model_key
        logger.info("LocalLLMBackend model loaded: %s", model_key)

    def unload_model(self) -> None:
        if self._model_handle is not None:
            logger.info("LocalLLMBackend unloading model: %s", self._model_key)
            models.unload_model(self._model_handle)
            self._model_handle = None
            self._model_key = None

    @property
    def model_key(self) -> str | None:
        return self._model_key
