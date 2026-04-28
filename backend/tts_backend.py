"""TTS backend abstractions and queue worker."""

import concurrent.futures
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
from enum import Enum

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


def _wav_duration(path: str) -> float:
    """Return duration in seconds of a WAV file."""
    try:
        with wave.open(path) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


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
    """Manages TTS synthesis with a thread pool.

    All submissions (frontend and backend) are synthesized concurrently
    in a shared thread pool — no request blocks another.  Delivery is
    serialized: ``get_next_audio()`` returns one item at a time, waits
    for estimated playback to finish, and prioritizes immediate
    (frontend) results over backend ones.
    """

    def __init__(self, backend: TTSBackend, max_workers: int = 4):
        self._backend = backend
        self._max_workers = max_workers
        self._immediate_result_queue: queue.Queue[TTSQueueItem] = queue.Queue()
        self._backend_result_queue: queue.Queue[TTSQueueItem] = queue.Queue()
        self._items: dict[str, TTSQueueItem] = {}
        self._lock = threading.Lock()
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._last_delivered_at: float = 0.0
        self._last_delivered_duration: float = 0.0

    def start(self) -> None:
        if self._pool is not None:
            return
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="tts",
        )
        logger.info("TTSQueueWorker started (pool=%d)", self._max_workers)

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def _synthesize(self, item: TTSQueueItem) -> None:
        with self._lock:
            item.status = "processing"

        logger.info("TTS processing %s [%s]: %s", item.id, item.source.value, item.text[:60])
        try:
            audio_path = self._backend.synthesize(item.text)
            duration = _wav_duration(audio_path)
            with self._lock:
                item.audio_path = audio_path
                item.audio_duration = duration
                item.status = "ready"
            logger.info("TTS item %s ready (%.1fs)", item.id, duration)
        except Exception:
            logger.exception("TTS error on item %s", item.id)
            with self._lock:
                item.status = "ready"
                item.audio_path = None

        if item.is_immediate:
            self._immediate_result_queue.put(item)
        else:
            self._backend_result_queue.put(item)

    def submit(self, text: str, source: TTSSource, context: str = "") -> TTSQueueItem:
        item = TTSQueueItem(
            id=str(uuid.uuid4())[:8],
            text=text,
            source=source,
            context=context,
        )
        with self._lock:
            self._items[item.id] = item
        if self._pool is not None:
            self._pool.submit(self._synthesize, item)
        logger.info("TTS submitted %s [%s]", item.id, source.value)
        return item

    def get_next_audio(self) -> TTSQueueItem | None:
        """Return at most one completed item for audio delivery.

        Immediate results take priority.  Returns ``None`` if no results
        are ready or if the previous audio is still estimated to be
        playing (based on WAV duration).
        """
        now = time.time()
        if now < self._last_delivered_at + self._last_delivered_duration:
            return None

        item: TTSQueueItem | None = None
        try:
            item = self._immediate_result_queue.get_nowait()
        except queue.Empty:
            if not self.has_pending_immediate():
                try:
                    item = self._backend_result_queue.get_nowait()
                except queue.Empty:
                    pass

        if item is not None and item.audio_path:
            self._last_delivered_at = now
            self._last_delivered_duration = item.audio_duration
        return item

    def has_pending_immediate(self) -> bool:
        """True if any frontend-sourced item is not yet delivered."""
        with self._lock:
            return any(
                item.is_immediate and item.status != "delivered"
                for item in self._items.values()
            )

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
        self._last_delivered_at = 0.0
        self._last_delivered_duration = 0.0
        for q in (self._immediate_result_queue, self._backend_result_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
