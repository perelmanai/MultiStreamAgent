"""Test the backend flow: delegation, processing, and result delivery.

Tests the BackendWorker + Backend abstraction without any Gradio or frontend.

Usage:
    ./run.sh python -m tests.test_backend
    ./run.sh python -m tests.test_backend --use-real-model
"""

import argparse
import logging
import sys
import time
import threading

# Ensure project root is on sys.path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import Backend, BackendWorker, LocalBackend, QueueItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock backend for fast testing (no GPU needed)
# ---------------------------------------------------------------------------
class MockBackend(Backend):
    """A fake backend that returns canned responses after a short delay."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.call_count = 0
        self._loaded_model: str | None = None

    def generate(self, question: str, history: list[dict]) -> str:
        self.call_count += 1
        time.sleep(self.delay)
        return f"[MockBackend] Answer to: {question} (call #{self.call_count})"

    def load_model(self, model_key: str) -> None:
        self._loaded_model = model_key
        logger.info("MockBackend: loaded model %s", model_key)

    def unload_model(self) -> None:
        logger.info("MockBackend: unloaded model %s", self._loaded_model)
        self._loaded_model = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_queue_lifecycle(backend: Backend):
    """Test: submit → queued → processing → ready → delivered."""
    print("\n=== Test: Queue Lifecycle ===")

    worker = BackendWorker(backend)
    worker.start()

    # Submit a question
    item = worker.submit(
        question="Explain quantum computing in detail",
        context_summary="quantum computing explanation",
        history=[],
    )
    print(f"  Submitted item {item.id}, status={item.status}")
    # Worker may pick it up instantly, so status could already be processing
    assert item.status in ("queued", "processing"), f"Expected 'queued' or 'processing', got '{item.status}'"

    # Wait for processing to start
    time.sleep(0.3)
    items = worker.get_all_items()
    current = [i for i in items if i.id == item.id][0]
    print(f"  After 0.3s: status={current.status}")
    assert current.status in ("processing", "ready"), f"Unexpected status: {current.status}"

    # Wait for completion
    max_wait = 120  # seconds (real model can take ~30-60s)
    start = time.time()
    results = []
    while time.time() - start < max_wait:
        results = worker.get_results()
        if results:
            break
        time.sleep(0.5)

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    result = results[0]
    print(f"  Result: status={result.status}, answer={result.answer[:80]}...")
    assert result.status == "ready"
    assert result.answer is not None

    # Mark delivered
    worker.mark_delivered(result.id)
    items = worker.get_all_items()
    delivered = [i for i in items if i.id == result.id][0]
    print(f"  After mark_delivered: status={delivered.status}")
    assert delivered.status == "delivered"

    worker.stop()
    print("  PASSED")


def test_multiple_questions(backend: Backend):
    """Test: multiple questions processed in order."""
    print("\n=== Test: Multiple Questions ===")

    worker = BackendWorker(backend)
    worker.start()

    questions = [
        ("What is AI?", "AI definition"),
        ("Explain neural networks", "neural networks"),
        ("History of computing", "computing history"),
    ]

    submitted = []
    for q, summary in questions:
        item = worker.submit(question=q, context_summary=summary, history=[])
        submitted.append(item)
        print(f"  Submitted: {item.id} - {q}")

    # Wait for all to complete (real models can take ~30s per question)
    max_wait = 180
    start = time.time()
    all_results = []
    while time.time() - start < max_wait and len(all_results) < len(questions):
        results = worker.get_results()
        all_results.extend(results)
        if len(all_results) < len(questions):
            time.sleep(1.0)

    print(f"  Got {len(all_results)} results")
    assert len(all_results) == len(questions), f"Expected {len(questions)}, got {len(all_results)}"

    # Verify all have answers
    for r in all_results:
        assert r.status == "ready"
        assert r.answer is not None
        print(f"  {r.id}: {r.answer[:60]}...")

    # Check queue shows all items
    all_items = worker.get_all_items()
    assert len(all_items) == len(questions)

    worker.stop()
    print("  PASSED")


def test_swap_backend():
    """Test: swapping backend while worker is running."""
    print("\n=== Test: Swap Backend ===")

    backend1 = MockBackend(delay=0.5)
    backend2 = MockBackend(delay=0.2)

    worker = BackendWorker(backend1)
    worker.start()

    # Submit with backend1
    item1 = worker.submit("Question for backend1", "q1 summary", [])

    # Wait for it to finish
    time.sleep(2)
    results1 = worker.get_results()
    assert len(results1) == 1
    assert "MockBackend" in results1[0].answer
    print(f"  Backend1 result: {results1[0].answer[:60]}")

    # Swap to backend2
    worker.swap_backend(backend2)

    # Submit with backend2
    item2 = worker.submit("Question for backend2", "q2 summary", [])

    time.sleep(2)
    results2 = worker.get_results()
    assert len(results2) == 1
    print(f"  Backend2 result: {results2[0].answer[:60]}")

    # Verify backend1 was unloaded
    assert backend1._loaded_model is None
    print("  Backend1 unloaded: OK")

    worker.stop()
    print("  PASSED")


def test_history_snapshot(backend: Backend):
    """Test: history is snapshotted at submit time, not affected by later changes."""
    print("\n=== Test: History Snapshot ===")

    worker = BackendWorker(backend)
    worker.start()

    original_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    item = worker.submit("Follow-up question", "follow-up", original_history)

    # Modify original history after submission
    original_history.append({"role": "user", "content": "This should NOT be in the snapshot"})

    # The queued item should have the original history (2 messages, not 3)
    items = worker.get_all_items()
    queued = [i for i in items if i.id == item.id][0]
    assert len(queued.history) == 2, f"Expected 2 history items, got {len(queued.history)}"
    print(f"  Snapshot has {len(queued.history)} messages (correct)")

    # Wait for completion
    time.sleep(5)
    worker.get_results()
    worker.stop()
    print("  PASSED")


def test_streaming_front_end_text_transformer(model_key: str = "Qwen3.5-2B"):
    """Test: streaming generation yields incremental text chunks."""
    print("\n=== Test: Streaming Front-End Text (transformer) ===")

    from models import (
        generate_response_streaming,
        load_qwen,
        unload_model,
    )

    print(f"  Loading model: {model_key}")
    model_handle = load_qwen(model_key)

    question = (
        "Write a detailed overview of the history of the internet, "
        "from ARPANET to the modern web. Include key milestones and dates."
    )
    history = []
    num_words_delay = 3

    print(f"  Streaming with num_words_delay={num_words_delay}...")
    print("  --- streaming output ---")

    chunks = []
    prev_len = 0
    for partial_text in generate_response_streaming(
        model_handle, question, history, num_words_delay=num_words_delay
    ):
        new_part = partial_text[prev_len:]
        print(new_part, end="", flush=True)
        prev_len = len(partial_text)
        chunks.append(partial_text)

    print("\n  --- end streaming output ---")

    # Verify properties
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    # Each chunk should be a prefix of the next
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].startswith(chunks[i]), (
            f"Chunk {i+1} is not a continuation of chunk {i}"
        )
    # Final text should be non-trivial
    final = chunks[-1]
    word_count = len(final.split())
    print(f"  Total chunks yielded: {len(chunks)}")
    print(f"  Final word count: {word_count}")
    assert word_count > 20, f"Expected substantial text, got {word_count} words"

    unload_model(model_handle)
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test backend flow")
    parser.add_argument(
        "--use-real-model",
        action="store_true",
        help="Use a real Qwen model instead of mock (requires GPU)",
    )
    parser.add_argument(
        "--model",
        default="Qwen3.5-2B",
        help="Model key to use with --use-real-model (default: Qwen3.5-2B)",
    )
    parser.add_argument(
        "--test-streaming",
        action="store_true",
        help="Run only the streaming front-end test (requires GPU)",
    )
    args = parser.parse_args()

    if args.test_streaming:
        test_streaming_front_end_text_transformer(model_key=args.model)
        print("\n=== Streaming test passed! ===")
        return

    if args.use_real_model:
        print(f"Using real model: {args.model}")
        backend = LocalBackend(args.model)
    else:
        print("Using MockBackend (pass --use-real-model for real model tests)")
        backend = MockBackend(delay=1.0)

    test_queue_lifecycle(backend)
    test_multiple_questions(backend)
    test_swap_backend()  # always uses MockBackend
    test_history_snapshot(backend)

    if args.use_real_model:
        backend.unload_model()

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
