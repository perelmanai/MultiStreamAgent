"""TTS backend abstractions and queue worker."""

import logging
import os
import queue
import tempfile
import threading
import time
import uuid
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_TTS_VOICES = [
    "Kore",
    "Zephyr",
    "Puck",
    "Charon",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
]

DEFAULT_TTS_VOICE = "Kore"
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"


def _get_genai_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Export it or use env/fb/run.sh.")
    return genai.Client(api_key=api_key)


@dataclass
class TTSQueueItem:
    id: str
    text: str
    status: str = "queued"  # queued | processing | ready | delivered
    audio_path: str | None = None
    timestamp: float = field(default_factory=time.time)


class TTSBackend(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    def synthesize(self, text: str) -> str:
        """Synthesize text to audio. Returns path to a WAV file."""
        ...

    @abstractmethod
    def get_voices(self) -> list[str]:
        """Return available voice names."""
        ...

    @abstractmethod
    def set_voice(self, voice: str) -> None:
        """Set the active voice."""
        ...


class GeminiTTSBackend(TTSBackend):
    """Gemini TTS via the genai SDK."""

    def __init__(self, voice: str = DEFAULT_TTS_VOICE):
        self._voice = voice

    def synthesize(self, text: str) -> str:
        client = _get_genai_client()
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=f"Say the following text naturally: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._voice,
                        )
                    )
                ),
            ),
        )

        pcm_bytes = response.candidates[0].content.parts[0].inline_data.data
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_bytes)
        return path

    def get_voices(self) -> list[str]:
        return list(GEMINI_TTS_VOICES)

    def set_voice(self, voice: str) -> None:
        self._voice = voice
        logger.info("TTS voice set to %s", voice)


class TTSQueueWorker:
    """Background thread that processes TTS synthesis requests."""

    def __init__(self, backend: TTSBackend):
        self._backend = backend
        self._task_queue: queue.Queue[TTSQueueItem] = queue.Queue()
        self._result_queue: queue.Queue[TTSQueueItem] = queue.Queue()
        self._items: dict[str, TTSQueueItem] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("TTSQueueWorker started")

    def stop(self) -> None:
        self._running = False

    def _run_loop(self) -> None:
        while self._running:
            try:
                item = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._lock:
                item.status = "processing"

            logger.info("TTS processing item %s: %s", item.id, item.text[:60])
            try:
                audio_path = self._backend.synthesize(item.text)
                with self._lock:
                    item.audio_path = audio_path
                    item.status = "ready"
                self._result_queue.put(item)
                logger.info("TTS item %s ready: %s", item.id, audio_path)
            except Exception:
                logger.exception("TTS error processing item %s", item.id)
                with self._lock:
                    item.status = "ready"
                    item.audio_path = None
                self._result_queue.put(item)
            finally:
                self._task_queue.task_done()

    def submit(self, text: str) -> TTSQueueItem:
        item = TTSQueueItem(
            id=str(uuid.uuid4())[:8],
            text=text,
        )
        with self._lock:
            self._items[item.id] = item
        self._task_queue.put(item)
        logger.info("TTS submitted item %s", item.id)
        return item

    def get_results(self) -> list[TTSQueueItem]:
        results = []
        while True:
            try:
                item = self._result_queue.get_nowait()
                results.append(item)
            except queue.Empty:
                break
        return results

    def get_all_items(self) -> list[TTSQueueItem]:
        with self._lock:
            return list(self._items.values())

    def mark_delivered(self, item_id: str) -> None:
        with self._lock:
            if item_id in self._items:
                self._items[item_id].status = "delivered"

    def clear_items(self) -> None:
        with self._lock:
            self._items.clear()
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except queue.Empty:
                break
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
