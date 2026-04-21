"""Backend abstraction and worker for async question processing."""

import copy
import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import models

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    id: str
    question: str
    context_summary: str
    history: list[dict]
    status: str = "queued"  # queued | processing | ready | delivered
    answer: str | None = None
    timestamp: float = field(default_factory=time.time)


class Backend(ABC):
    """Abstract base class for back-end inference providers."""

    @abstractmethod
    def generate(self, question: str, history: list[dict]) -> str:
        """Generate a full response for the given question + history."""
        ...

    @abstractmethod
    def load_model(self, model_key: str) -> None:
        """Load or switch the underlying model."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Free model resources."""
        ...


class LocalBackend(Backend):
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


class GeminiBackend(Backend):
    """Backend using Gemini API."""

    def __init__(self, model_key: str | None = None):
        from gemini_client import GEMINI_DEFAULT_MODEL

        self._model_key = model_key or GEMINI_DEFAULT_MODEL

    def generate(self, question: str, history: list[dict]) -> str:
        import time
        from gemini_client import generate_gemini_response

        time.sleep(10)
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


class BackendWorker:
    """Manages a background thread that processes queued questions via a Backend."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._task_queue: queue.Queue[QueueItem] = queue.Queue()
        self._result_queue: queue.Queue[QueueItem] = queue.Queue()
        self._items: dict[str, QueueItem] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("BackendWorker started")

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

            logger.info("Processing item %s: %s", item.id, item.question[:60])
            try:
                answer = self._backend.generate(item.question, item.history)
                with self._lock:
                    item.answer = answer
                    item.status = "ready"
                self._result_queue.put(item)
                logger.info("Item %s ready", item.id)
            except Exception:
                logger.exception("Error processing item %s", item.id)
                with self._lock:
                    item.answer = (
                        "Sorry, an error occurred while processing your request."
                    )
                    item.status = "ready"
                self._result_queue.put(item)
            finally:
                self._task_queue.task_done()

    def submit(
        self, question: str, context_summary: str, history: list[dict]
    ) -> QueueItem:
        item = QueueItem(
            id=str(uuid.uuid4())[:8],
            question=question,
            context_summary=context_summary,
            history=copy.deepcopy(history),
        )
        with self._lock:
            self._items[item.id] = item
        self._task_queue.put(item)
        logger.info("Submitted item %s to queue", item.id)
        return item

    def get_results(self) -> list[QueueItem]:
        results = []
        while True:
            try:
                item = self._result_queue.get_nowait()
                results.append(item)
            except queue.Empty:
                break
        return results

    def get_all_items(self) -> list[QueueItem]:
        with self._lock:
            return list(self._items.values())

    def mark_delivered(self, item_id: str) -> None:
        with self._lock:
            if item_id in self._items:
                self._items[item_id].status = "delivered"

    def clear_items(self) -> None:
        """Clear all tracked items and drain queues."""
        with self._lock:
            self._items.clear()
        # Drain both queues
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

    def swap_backend(self, new_backend: Backend) -> None:
        """Swap the backend. Waits for current processing to finish first."""
        logger.info("Swapping backend, draining queue...")
        # Wait for task queue to drain
        self._task_queue.join()
        self._backend.unload_model()
        self._backend = new_backend
        logger.info("Backend swapped")
