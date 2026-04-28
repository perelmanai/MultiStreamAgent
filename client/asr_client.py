"""ASR clients — Gemini API and local Whisper (via in-process server).

LocalASRClient talks through InProcessASRServer, which will be replaced
by a Thrift server/client pair for remote GPU hosting.
"""

import io
import logging

import numpy as np
import torch
import torchaudio
from google.genai import types

from backend.asr_backend import ASR_SAMPLE_RATE, WhisperASRBackend, preprocess_audio

from .base import ASRClient
from .gemini_utils import GEMINI_DEFAULT_MODEL, GEMINI_MODELS, _get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini (remote API)
# ---------------------------------------------------------------------------
class GeminiASRClient(ASRClient):

    def __init__(self, model_key: str | None = None):
        self._model_key = model_key or GEMINI_DEFAULT_MODEL

    def load_model(self, model_key: str) -> None:
        logger.info("GeminiASRClient switching to model: %s", model_key)
        self._model_key = model_key

    def unload_model(self) -> None:
        pass

    @property
    def model_key(self) -> str:
        return self._model_key

    def transcribe(self, sr: int, audio_data: np.ndarray) -> str:
        audio_np = preprocess_audio(sr, audio_data)

        buf = io.BytesIO()
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        torchaudio.save(buf, tensor, ASR_SAMPLE_RATE, format="wav")
        wav_bytes = buf.getvalue()

        client = _get_client()
        model_name = GEMINI_MODELS[self._model_key]

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="audio/wav",
                                data=wav_bytes,
                            )
                        ),
                        types.Part(text="Transcribe the audio exactly. Output ONLY the transcript text, nothing else."),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Local (in-process server → backend)
# ---------------------------------------------------------------------------
class InProcessASRServer:
    """In-process server — direct method calls to the backend.

    TODO: Replace with Thrift server/client pair.
    """

    def __init__(self, backend: WhisperASRBackend):
        self._backend = backend

    def transcribe(self, sr: int, audio_data: np.ndarray) -> str:
        return self._backend.transcribe(sr, audio_data)

    def load_model(self, model_key: str) -> None:
        self._backend.load_model(model_key)

    def unload_model(self) -> None:
        self._backend.unload_model()


class LocalASRClient(ASRClient):
    """Client that talks to an ASR server. Orchestrator uses this."""

    def __init__(self, model_path: str | None = None):
        backend = WhisperASRBackend(model_path) if model_path else WhisperASRBackend()
        self._server = InProcessASRServer(backend)

    def transcribe(self, sr: int, audio_data: np.ndarray) -> str:
        return self._server.transcribe(sr, audio_data)

    def load_model(self, model_key: str) -> None:
        self._server.load_model(model_key)

    def unload_model(self) -> None:
        self._server.unload_model()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ASR_CLIENTS: dict[str, type] = {
    "Whisper (Local)": LocalASRClient,
    "Gemini ASR": GeminiASRClient,
}

DEFAULT_ASR = "Whisper (Local)"


def get_asr_choices() -> list[str]:
    return list(ASR_CLIENTS.keys())
