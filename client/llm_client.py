"""LLM clients and queue worker.

LocalLLMClient talks through InProcessLLMServer, which will be replaced
by a Thrift server/client pair for remote GPU hosting.
"""

import copy
import logging
import queue
import threading
import uuid

import models

from backend.llm_backend import LocalLLMBackend

from .base import LLMClient, QueueItem
from .gemini_utils import GEMINI_DEFAULT_MODEL, generate_gemini_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini (remote API)
# ---------------------------------------------------------------------------
class GeminiLLMClient(LLMClient):

    def __init__(self, model_key: str | None = None):
        self._model_key = model_key or GEMINI_DEFAULT_MODEL

    def generate(self, question: str, history: list[dict]) -> str:
        return generate_gemini_response(
            model_key=self._model_key,
            user_text=question,
            history=history,
            system_prompt=models.BACKEND_SYSTEM_PROMPT,
            max_tokens=1024,
        )

    def load_model(self, model_key: str) -> None:
        logger.info("GeminiLLMClient switching to model: %s", model_key)
        self._model_key = model_key

    def unload_model(self) -> None:
        pass

    @property
    def model_key(self) -> str | None:
        return self._model_key


# ---------------------------------------------------------------------------
# Local (in-process server → backend)
# ---------------------------------------------------------------------------
class InProcessLLMServer:
    """In-process server — direct method calls to the backend.

    TODO: Replace with Thrift server/client pair.
    """

    def __init__(self, backend: LocalLLMBackend):
        self._backend = backend

    def generate(self, question: str, history: list[dict]) -> str:
        return self._backend.generate(question, history)

    def load_model(self, model_key: str) -> None:
        self._backend.load_model(model_key)

    def unload_model(self) -> None:
        self._backend.unload_model()

    @property
    def model_key(self) -> str | None:
        return self._backend.model_key


class LocalLLMClient(LLMClient):
    """Client that talks to an LLM server. Orchestrator uses this."""

    def __init__(self, model_key: str | None = None):
        backend = LocalLLMBackend(model_key)
        self._server = InProcessLLMServer(backend)

    def generate(self, question: str, history: list[dict]) -> str:
        return self._server.generate(question, history)

    def load_model(self, model_key: str) -> None:
        self._server.load_model(model_key)

    def unload_model(self) -> None:
        self._server.unload_model()

    @property
    def model_key(self) -> str | None:
        return self._server.model_key


# ---------------------------------------------------------------------------
# Queue worker
# ---------------------------------------------------------------------------
class LLMQueueWorker:
    """Manages a background thread that processes queued questions via an LLMClient."""

    def __init__(self, client: LLMClient):
        self._client = client
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
        logger.info("LLMQueueWorker started")

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
                answer = self._client.generate(item.question, item.history)
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

    def swap_client(self, new_client: LLMClient) -> None:
        logger.info("Swapping LLM client, draining queue...")
        self._task_queue.join()
        self._client.unload_model()
        self._client = new_client
        logger.info("LLM client swapped")
