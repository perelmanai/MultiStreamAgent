"""Local ASR backend — Whisper model hosting on GPU."""

import logging
import os
import threading

import numpy as np
import torch
import torchaudio

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


class WhisperASRBackend:
    """Holds a Whisper model on GPU and runs transcription."""

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
