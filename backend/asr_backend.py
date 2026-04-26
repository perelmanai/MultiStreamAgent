"""ASR backend implementations — Whisper (local) and Gemini API."""

import io
import logging
import os
import threading

import numpy as np
import torch
import torchaudio

from .base import ASRBackend

logger = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16000

WHISPER_MODEL_PATH = os.path.expanduser(
    "~/si_mango/tree/checkpoints/whisper/large-v3-turbo.pt"
)


def preprocess_audio(sr: int, audio_data: np.ndarray) -> np.ndarray:
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != ASR_SAMPLE_RATE:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, ASR_SAMPLE_RATE)
        audio_data = audio_tensor.squeeze(0).numpy()
    return audio_data


class WhisperASRBackend(ASRBackend):
    def __init__(self, model_path: str = WHISPER_MODEL_PATH):
        self._model_path = model_path
        self._handle = None
        self._lock = threading.Lock()

    def load_model(self, model_key: str) -> None:
        self.unload_model()
        self._model_path = model_key
        self._ensure_loaded()

    def unload_model(self) -> None:
        with self._lock:
            if self._handle is not None:
                model, fmt, device = self._handle
                if fmt == "openai":
                    del model
                else:
                    del model[0], model[1]
                self._handle = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("WhisperASRBackend: model unloaded")

    def _ensure_loaded(self):
        if self._handle is not None:
            return
        with self._lock:
            if self._handle is not None:
                return
            logger.info("Loading Whisper model from %s", self._model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._model_path.endswith((".pt", ".pth")):
                import whisper

                model = whisper.load_model(self._model_path, device=device)
                self._handle = (model, "openai", device)
            else:
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                processor = WhisperProcessor.from_pretrained(self._model_path)
                model = WhisperForConditionalGeneration.from_pretrained(
                    self._model_path, torch_dtype=torch.float16
                ).to(device)
                self._handle = ((model, processor), "hf", device)
            logger.info("Whisper model loaded")

    def transcribe(self, sr: int, audio_data: np.ndarray) -> str:
        self._ensure_loaded()
        audio_np = preprocess_audio(sr, audio_data)
        model, fmt, device = self._handle

        if fmt == "openai":
            import whisper

            result = whisper.transcribe(model, audio_np)
            return result["text"].strip()
        else:
            hf_model, processor = model
            input_features = processor(
                audio_np, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(device, torch.float16)
            with torch.no_grad():
                predicted_ids = hf_model.generate(input_features)
            return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


class GeminiASRBackend(ASRBackend):
    def __init__(self, model_key: str | None = None):
        from gemini_client import GEMINI_DEFAULT_MODEL

        self._model_key = model_key or GEMINI_DEFAULT_MODEL

    def load_model(self, model_key: str) -> None:
        logger.info("GeminiASRBackend switching to model: %s", model_key)
        self._model_key = model_key

    def unload_model(self) -> None:
        pass

    @property
    def model_key(self) -> str:
        return self._model_key

    def transcribe(self, sr: int, audio_data: np.ndarray) -> str:
        from gemini_client import GEMINI_MODELS, _get_client
        from google.genai import types

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


ASR_BACKENDS: dict[str, type[ASRBackend]] = {
    "Whisper (Local)": WhisperASRBackend,
    "Gemini ASR": GeminiASRBackend,
}

DEFAULT_ASR = "Whisper (Local)"


def get_asr_choices() -> list[str]:
    return list(ASR_BACKENDS.keys())
