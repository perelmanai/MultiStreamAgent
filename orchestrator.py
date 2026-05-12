"""Orchestrator — owns all business logic, workers, and model state.

No Gradio dependency. Any UI framework can drive this class by calling its
methods and mapping the returned ``OrchestratorUpdate`` objects to its own
rendering layer.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Generator

from client import (
    GEMINI_DEFAULT_MODEL,
    GeminiASRClient,
    GeminiLLMClient,
    GeminiTTSClient,
    LLMQueueWorker,
    LocalASRClient,
    LocalLLMClient,
    QueueItem,
    TTSQueueItem,
    TTSQueueWorker,
    TTSSource,
    estimate_complexity_gemini,
    estimate_intention_gemini,
    generate_gemini_response,
    generate_gemini_response_streaming,
)
from models import (
    FRONTEND_SYSTEM_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
    estimate_complexity,
    generate_response,
    generate_response_streaming,
    load_qwen,
    unload_model,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorUpdate:
    """Pure-data snapshot yielded/returned by Orchestrator methods."""

    history: list[dict]
    clear_input: bool = False
    audio_path: str | None = None
    status_message: str | None = None
    warning: str | None = None
    text_queue_count: int = 0
    speech_queue_count: int = 0
    # True when the UI should NOT overwrite the chat component on this update.
    # Used by ``poll`` while a user-message handler is mid-flight, to avoid
    # racing the streaming generator's chat writes.
    skip_chat_update: bool = False


class Orchestrator:
    """Manages the full pipeline: user input -> triage -> generate/queue -> TTS.

    Holds all worker threads, model state, and queue bookkeeping.  The caller
    (e.g. a Gradio app) passes ``history`` in on each call; the Orchestrator
    mutates it and returns it inside ``OrchestratorUpdate``.
    """

    def __init__(self) -> None:
        self.frontend_model = None
        self.frontend_lock = threading.Lock()
        self.backend_worker: LLMQueueWorker | None = None
        self.frontend_type: str = "Gemini API"
        self.frontend_gemini_model: str = GEMINI_DEFAULT_MODEL
        self.backend_insert_positions: dict[str, int] = {}
        self._ready_items: dict[str, QueueItem] = {}
        self._ready_audio: dict[str, str] = {}
        self._pending_notifications: dict[str, dict] = {}

        self.asr_engine = LocalASRClient()
        self.tts_engine: GeminiTTSClient = GeminiTTSClient()
        self.tts_queue_worker: TTSQueueWorker | None = None
        self.output_mode: str = "Text"
        self.input_mode: str = "Text"

        # Busy gate: only one user-message handler may mutate the chat at a
        # time. ``poll`` checks this and defers chat writes while busy.
        self._busy_state_lock = threading.Lock()
        self._busy_depth: int = 0
        # Notifications collected by ``poll`` while busy; flushed on next
        # non-busy poll.
        self._deferred_notifications: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("Starting backend worker with Gemini API")
        llm_client = GeminiLLMClient(model_key=GEMINI_DEFAULT_MODEL)
        self.backend_worker = LLMQueueWorker(llm_client)
        self.backend_worker.start()

        self.tts_queue_worker = TTSQueueWorker(self.tts_engine)
        self.tts_queue_worker.start()
        logger.info("Backend worker and TTS queue worker started")

    # ------------------------------------------------------------------
    # Queue accessors
    # ------------------------------------------------------------------

    def get_text_queue_items(self) -> list[QueueItem]:
        if self.backend_worker is None:
            return []
        return self.backend_worker.get_all_items()

    def get_speech_queue_items(self) -> list[TTSQueueItem]:
        if self.tts_queue_worker is None:
            return []
        return self.tts_queue_worker.get_all_items()

    @property
    def text_queue_count(self) -> int:
        return len(self.get_text_queue_items())

    @property
    def speech_queue_count(self) -> int:
        return len(self.get_speech_queue_items())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_update(self, history: list[dict], **kwargs) -> OrchestratorUpdate:
        return OrchestratorUpdate(
            history=history,
            text_queue_count=self.text_queue_count,
            speech_queue_count=self.speech_queue_count,
            **kwargs,
        )

    def _gemini_triage(self, user_text: str, threshold_n: int) -> tuple[bool, int, str]:
        raw = estimate_complexity_gemini(
            self.frontend_gemini_model, user_text, TRIAGE_SYSTEM_PROMPT
        )
        logger.info("gemini triage raw: %s", raw)
        words_match = re.search(r"ESTIMATED_WORDS:\s*(\d+)", raw)
        if not words_match:
            return False, 0, user_text[:80]
        estimated_words = int(words_match.group(1))
        return estimated_words >= threshold_n, estimated_words, user_text[:80]

    def _maybe_enqueue_tts(self, text: str, source: TTSSource) -> None:
        if self.output_mode != "Speech" or self.tts_queue_worker is None:
            return
        self.tts_queue_worker.submit(text, source=source)

    def _try_acquire_busy(self) -> bool:
        with self._busy_state_lock:
            if self._busy_depth > 0:
                return False
            self._busy_depth += 1
            return True

    def _release_busy(self) -> None:
        with self._busy_state_lock:
            self._busy_depth = max(0, self._busy_depth - 1)

    def is_busy(self) -> bool:
        with self._busy_state_lock:
            return self._busy_depth > 0

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------

    def handle_user_message(
        self,
        user_text: str,
        history: list[dict],
        threshold_n: int,
        streaming_enabled: bool,
        num_words_delay: int,
    ) -> Generator[OrchestratorUpdate, None, None]:
        if self.input_mode != "Text":
            # Silently ignore text submits when UI is in speech mode.
            return
        yield from self._process_text(
            user_text, history, threshold_n, streaming_enabled, num_words_delay
        )

    def _process_text(
        self,
        user_text: str,
        history: list[dict],
        threshold_n: int,
        streaming_enabled: bool,
        num_words_delay: int,
    ) -> Generator[OrchestratorUpdate, None, None]:
        if not user_text.strip():
            yield self._make_update(history, clear_input=True)
            return

        if (
            self.tts_queue_worker is not None
            and self.tts_queue_worker.has_pending_immediate()
        ):
            yield self._make_update(
                history,
                warning=(
                    "Please wait — the previous reply is still being spoken. "
                    "You can send a new message once TTS playback is ready."
                ),
            )
            return

        if not self._try_acquire_busy():
            yield self._make_update(
                history,
                warning=(
                    "Please wait — the previous message is still being processed."
                ),
            )
            return

        try:
            yield from self._process_text_locked(
                user_text, history, threshold_n, streaming_enabled, num_words_delay
            )
        finally:
            self._release_busy()

    def _process_text_locked(
        self,
        user_text: str,
        history: list[dict],
        threshold_n: int,
        streaming_enabled: bool,
        num_words_delay: int,
    ) -> Generator[OrchestratorUpdate, None, None]:
        using_gemini = self.frontend_type == "Gemini API"

        if not using_gemini and self.frontend_model is None:
            history.append(
                {
                    "role": "assistant",
                    "content": "Models are still loading, please wait...",
                }
            )
            yield self._make_update(history, clear_input=True)
            return

        history.append({"role": "user", "content": user_text})
        yield self._make_update(history, clear_input=True)

        final_text = None

        if using_gemini:
            ready_for_intent = self._get_ready_items_for_intent()
            if ready_for_intent:
                intention = estimate_intention_gemini(
                    model_key=self.frontend_gemini_model,
                    user_text=user_text,
                    history=history,
                    ready_items=ready_for_intent,
                )
                if intention["action"] == "SELECT":
                    answer_text, audio_path = self._deliver_ready_item(
                        intention["index"], history
                    )
                    if answer_text:
                        yield self._make_update(
                            history, clear_input=True, audio_path=audio_path
                        )
                        return

            is_complex, est_words, summary = self._gemini_triage(user_text, threshold_n)

            if is_complex:
                item = self.backend_worker.submit(user_text, summary, history[:-1])
                reply = (
                    f"That's a detailed question — I'll work on a thorough answer and get back to you. "
                    f"(Estimated ~{est_words} words needed)"
                )
                history.append({"role": "assistant", "content": reply})
                self.backend_insert_positions[item.id] = len(history)
                self._maybe_enqueue_tts(reply, TTSSource.FRONTEND)
                yield self._make_update(history, clear_input=True)
            elif streaming_enabled:
                history.append({"role": "assistant", "content": ""})
                for partial_text in generate_gemini_response_streaming(
                    model_key=self.frontend_gemini_model,
                    user_text=user_text,
                    history=history[:-2],
                    system_prompt=FRONTEND_SYSTEM_PROMPT,
                    max_tokens=2048,
                    num_words_delay=num_words_delay,
                ):
                    history[-1]["content"] = partial_text
                    yield self._make_update(history, clear_input=True)
                final_text = history[-1]["content"]
            else:
                answer = generate_gemini_response(
                    model_key=self.frontend_gemini_model,
                    user_text=user_text,
                    history=history[:-1],
                    system_prompt=FRONTEND_SYSTEM_PROMPT,
                    max_tokens=2048,
                )
                history.append({"role": "assistant", "content": answer})
                final_text = answer
                yield self._make_update(history, clear_input=True)
        else:
            with self.frontend_lock:
                is_complex, est_words, direct_answer, summary = estimate_complexity(
                    self.frontend_model,
                    user_text,
                    history[:-1],
                    threshold_n,
                    skip_answer=streaming_enabled,
                )

            if is_complex:
                item = self.backend_worker.submit(user_text, summary, history[:-1])
                reply = (
                    f"That's a detailed question — I'll work on a thorough answer and get back to you. "
                    f"(Estimated ~{est_words} words needed)"
                )
                history.append({"role": "assistant", "content": reply})
                self.backend_insert_positions[item.id] = len(history)
                self._maybe_enqueue_tts(reply, TTSSource.FRONTEND)
                yield self._make_update(history, clear_input=True)
            elif direct_answer:
                history.append({"role": "assistant", "content": direct_answer})
                final_text = direct_answer
                yield self._make_update(history, clear_input=True)
            elif streaming_enabled:
                history.append({"role": "assistant", "content": ""})
                with self.frontend_lock:
                    for partial_text in generate_response_streaming(
                        self.frontend_model,
                        user_text,
                        history[:-2],
                        num_words_delay=num_words_delay,
                    ):
                        history[-1]["content"] = partial_text
                        yield self._make_update(history, clear_input=True)
                final_text = history[-1]["content"]
            else:
                with self.frontend_lock:
                    answer = generate_response(
                        self.frontend_model, user_text, history[:-1]
                    )
                history.append({"role": "assistant", "content": answer})
                final_text = answer
                yield self._make_update(history, clear_input=True)

        if final_text:
            self._maybe_enqueue_tts(final_text, TTSSource.FRONTEND)
            yield self._make_update(history, clear_input=True)

    def _append_notification(self, history: list[dict], notification: str) -> None:
        history.append({"role": "assistant", "content": notification})

    def poll(self, history: list[dict]) -> OrchestratorUpdate:
        audio_path = None
        busy = self.is_busy()

        # When not busy, flush any notifications that were deferred during a
        # prior busy window before we process new results.
        if not busy and self._deferred_notifications:
            for text in self._deferred_notifications:
                self._append_notification(history, text)
            self._deferred_notifications.clear()

        def queue_notification(text: str) -> None:
            if busy:
                self._deferred_notifications.append(text)
            else:
                self._append_notification(history, text)

        if self.backend_worker is not None:
            results = self.backend_worker.get_results()
            for item in results:
                logger.info("poll_backend: item %s ready", item.id)
                self._ready_items[item.id] = item
                self.backend_insert_positions.pop(item.id, None)

                notification = (
                    f"I've finished working on your question about "
                    f"**{item.context_summary}**. Would you like to hear the answer?"
                )

                if self.output_mode == "Speech" and self.tts_queue_worker is not None:
                    tts_item = self.tts_queue_worker.submit(
                        item.answer, source=TTSSource.BACKEND
                    )
                    self._pending_notifications[tts_item.id] = {
                        "text": notification,
                        "item_id": item.id,
                    }
                    logger.info(
                        "Answer TTS enqueued %s for item %s (notification pending)",
                        tts_item.id,
                        item.id,
                    )
                else:
                    queue_notification(notification)

        if self.tts_queue_worker is not None:
            tts_item = self.tts_queue_worker.get_next_audio()
            if tts_item is not None:
                if tts_item.id in self._pending_notifications:
                    info = self._pending_notifications.pop(tts_item.id)
                    if tts_item.audio_path:
                        self._ready_audio[info["item_id"]] = tts_item.audio_path
                    queue_notification(info["text"])
                    self.tts_queue_worker.submit(info["text"], source=TTSSource.BACKEND)
                    logger.info(
                        "Answer speech ready — notification sent for item %s, "
                        "audio stored for SELECT",
                        info["item_id"],
                    )
                else:
                    if tts_item.audio_path:
                        audio_path = tts_item.audio_path
                self.tts_queue_worker.mark_delivered(tts_item.id)
                logger.info(
                    "TTS item %s delivered (%.1fs)",
                    tts_item.id,
                    tts_item.audio_duration,
                )

        return self._make_update(history, audio_path=audio_path, skip_chat_update=busy)

    def _get_ready_items_for_intent(self) -> list[dict]:
        """Build the ready-items list for intention estimation."""
        return [
            {
                "index": idx,
                "id": item.id,
                "question": item.question,
                "summary": item.context_summary,
            }
            for idx, item in enumerate(self._ready_items.values())
        ]

    def _deliver_ready_item(
        self, index: int, history: list[dict]
    ) -> tuple[str | None, str | None]:
        """Deliver a ready item by index. Returns (answer_text, audio_path)."""
        items_list = list(self._ready_items.values())
        if index < 0 or index >= len(items_list):
            return None, None
        item = items_list[index]

        delivery = (
            f"Regarding your question about **{item.context_summary}**:\n\n"
            f"{item.answer}"
        )
        history.append({"role": "assistant", "content": delivery})

        audio_path = self._ready_audio.pop(item.id, None)

        if self.backend_worker is not None:
            self.backend_worker.mark_delivered(item.id)
        del self._ready_items[item.id]
        logger.info(
            "Delivered ready item %s (index %d, audio=%s)",
            item.id,
            index,
            audio_path is not None,
        )
        return item.answer, audio_path

    def handle_audio_input(
        self,
        audio_data,
        history: list[dict],
        threshold_n: int,
        streaming_enabled: bool,
        num_words_delay: int,
    ) -> Generator[OrchestratorUpdate, None, None]:
        if self.input_mode != "Speech":
            # Silently ignore audio events when UI is in text mode.
            return
        if audio_data is None:
            yield self._make_update(history, status_message="Ready to record")
            return

        sr, data = audio_data
        yield self._make_update(history, status_message="Transcribing...")

        transcript = self.asr_engine.transcribe(sr, data)
        logger.info("ASR transcript: %s", transcript)

        if not transcript.strip():
            yield self._make_update(history, status_message="(no speech detected)")
            return

        # Bypass the input-mode gate on _process_text — we already validated
        # that we are in speech mode above.
        for update in self._process_text(
            transcript, history, threshold_n, streaming_enabled, num_words_delay
        ):
            update.status_message = f"Sent: {transcript[:80]}"
            yield update

    def clear(self) -> OrchestratorUpdate:
        if self.backend_worker is not None:
            self.backend_worker.clear_items()
        if self.tts_queue_worker is not None:
            self.tts_queue_worker.clear_items()
        self._ready_items.clear()
        self._ready_audio.clear()
        self._pending_notifications.clear()
        self._deferred_notifications.clear()
        return self._make_update([])

    def set_input_mode(self, mode: str) -> None:
        self.input_mode = mode
        logger.info("Input mode switched to %s", mode)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_frontend_type(
        self, fe_type: str, local_model_key: str, gemini_model_key: str
    ) -> str:
        self.frontend_type = fe_type

        if fe_type == "Gemini API":
            self.frontend_gemini_model = gemini_model_key
            if self.frontend_model is not None:
                with self.frontend_lock:
                    unload_model(self.frontend_model)
                    self.frontend_model = None
            label = f"Gemini ({gemini_model_key})"
        else:
            self.frontend_gemini_model = GEMINI_DEFAULT_MODEL
            if self.frontend_model is None:
                with self.frontend_lock:
                    self.frontend_model = load_qwen(local_model_key)
            label = f"Local ({local_model_key})"

        return f"Frontend switched to {label}."

    def set_frontend_local_model(self, model_key: str) -> str:
        if self.frontend_model is not None:
            with self.frontend_lock:
                unload_model(self.frontend_model)
                self.frontend_model = None

        logger.info("Switching frontend model to %s", model_key)
        with self.frontend_lock:
            self.frontend_model = load_qwen(model_key)
        logger.info("Frontend model switched to %s", model_key)
        return f"Frontend model switched to {model_key}."

    def set_frontend_gemini_model(self, model_key: str) -> str:
        self.frontend_gemini_model = model_key
        logger.info("Frontend Gemini model switched to %s", model_key)
        return f"Frontend Gemini model switched to {model_key}."

    def set_backend_type(
        self, be_type: str, local_model_key: str, gemini_model_key: str
    ) -> str:
        if be_type == "Gemini API":
            new_client = GeminiLLMClient(model_key=gemini_model_key)
            label = f"Gemini ({gemini_model_key})"
        else:
            new_client = LocalLLMClient(local_model_key)
            label = f"Local ({local_model_key})"

        if self.backend_worker is not None:
            self.backend_worker.swap_client(new_client)
        else:
            self.backend_worker = LLMQueueWorker(new_client)
            self.backend_worker.start()

        logger.info("Backend switched to %s", label)
        return f"Backend switched to {label}."

    def set_backend_local_model(self, model_key: str) -> str:
        if self.backend_worker is not None:
            logger.info("Switching backend model to %s", model_key)
            new_client = LocalLLMClient(model_key)
            self.backend_worker.swap_client(new_client)
        else:
            llm_client = LocalLLMClient(model_key)
            self.backend_worker = LLMQueueWorker(llm_client)
            self.backend_worker.start()

        logger.info("Backend model switched to %s", model_key)
        return f"Backend model switched to {model_key}."

    def set_backend_gemini_model(self, model_key: str) -> str:
        if self.backend_worker is not None and isinstance(
            self.backend_worker._client, GeminiLLMClient
        ):
            self.backend_worker._client.load_model(model_key)
            logger.info("Backend Gemini model switched to %s", model_key)
            return f"Backend Gemini model switched to {model_key}."
        return ""

    def set_asr(self, asr_choice: str, gemini_model_key: str) -> None:
        if asr_choice == "Gemini ASR":
            self.asr_engine = GeminiASRClient(model_key=gemini_model_key)
        else:
            self.asr_engine = LocalASRClient()
        logger.info("ASR switched to %s", asr_choice)

    def set_asr_gemini_model(self, model_key: str) -> None:
        if isinstance(self.asr_engine, GeminiASRClient):
            self.asr_engine.model_key = model_key
            logger.info("ASR Gemini model switched to %s", model_key)

    def set_output_mode(self, mode: str) -> None:
        self.output_mode = mode
        logger.info("Output mode switched to %s", mode)

    def set_tts_voice(self, voice: str) -> None:
        self.tts_engine.set_voice(voice)
