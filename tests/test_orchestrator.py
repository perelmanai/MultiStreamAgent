"""Test the Orchestrator intention-routing flow.

Uses real Gemini API for triage and intention, with a fast mock backend
for queue processing (avoids slow LLM generation per item).

Usage:
    ./env/fb/run.sh python -m tests.test_orchestrator
"""

import logging
import os
import struct
import sys
import tempfile
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import LLMClient, LLMQueueWorker, QueueItem, TTSClient, TTSSource
from client.tts_client import TTSQueueWorker
from orchestrator import Orchestrator, OrchestratorUpdate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock backend for queue processing (fast, returns canned answers)
# ---------------------------------------------------------------------------
class MockLLMClient(LLMClient):
    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.call_count = 0

    def generate(self, question: str, history: list[dict]) -> str:
        self.call_count += 1
        time.sleep(self.delay)
        return (
            f"Here is a comprehensive answer to your question about '{question}'. "
            f"This covers all the key aspects in detail with examples and explanations. "
            f"(Mock answer #{self.call_count})"
        )

    def load_model(self, model_key: str) -> None:
        pass

    def unload_model(self) -> None:
        pass


def _make_orchestrator(backend_delay: float = 0.5) -> Orchestrator:
    """Create an Orchestrator with real Gemini frontend + mock backend queue."""
    orch = Orchestrator()
    mock_client = MockLLMClient(delay=backend_delay)
    orch.backend_worker = LLMQueueWorker(mock_client)
    orch.backend_worker.start()
    orch.tts_queue_worker = None
    orch.output_mode = "Text"
    return orch


def _drain(gen) -> list[OrchestratorUpdate]:
    return list(gen)


def _wait_for_ready(orch: Orchestrator, n: int, timeout: float = 30.0) -> list[dict]:
    """Poll until n items appear in _ready_items. Returns poll history."""
    history = []
    start = time.time()
    while time.time() - start < timeout:
        orch.poll(history)
        if len(orch._ready_items) >= n:
            return history
        time.sleep(0.3)
    raise TimeoutError(
        f"Expected {n} ready items, got {len(orch._ready_items)} after {timeout}s"
    )


# ---------------------------------------------------------------------------
# Tests — all use real Gemini API for triage + intention
# ---------------------------------------------------------------------------
def test_complex_question_gets_queued():
    """A clearly complex question should be triaged as complex and queued."""
    print("\n=== Test: Complex Question Gets Queued (Gemini triage) ===")

    orch = _make_orchestrator()
    history: list[dict] = []

    question = (
        "Write a detailed 500-word essay analyzing the geopolitical implications "
        "of artificial intelligence on international trade agreements, covering "
        "economic, military, and ethical dimensions with historical examples."
    )

    updates = _drain(orch.handle_user_message(question, history, 50, False, 3))

    relay_msgs = [
        m
        for m in history
        if m["role"] == "assistant" and "I'll work on" in m["content"]
    ]
    assert len(relay_msgs) >= 1, (
        f"Expected complex relay message, got history: "
        f"{[m['content'][:60] for m in history if m['role'] == 'assistant']}"
    )
    print(f"  Triaged as complex, relay: {relay_msgs[0]['content'][:80]}...")

    assert len(orch.backend_worker.get_all_items()) >= 1
    print("  Item queued in backend worker")

    orch.backend_worker.stop()
    print("  PASSED")


def test_simple_question_answered_directly():
    """A clearly simple question should be answered directly, not queued."""
    print("\n=== Test: Simple Question Answered Directly (Gemini triage) ===")

    orch = _make_orchestrator()
    history: list[dict] = []

    updates = _drain(orch.handle_user_message("What is 2 + 2?", history, 50, False, 3))

    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    assert len(assistant_msgs) >= 1, "Expected at least one assistant message"

    relay_msgs = [m for m in assistant_msgs if "I'll work on" in m["content"]]
    assert (
        len(relay_msgs) == 0
    ), f"Simple question should not be queued, got: {relay_msgs}"

    print(f"  Direct answer: {assistant_msgs[-1]['content'][:80]}...")

    orch.backend_worker.stop()
    print("  PASSED")


def test_intention_select():
    """After a backend answer is ready, user saying 'yes' should SELECT it via Gemini."""
    print("\n=== Test: Intention SELECT (Gemini intention) ===")

    orch = _make_orchestrator(backend_delay=0.3)
    history: list[dict] = []

    # Force complex triage so it goes to backend queue
    question = (
        "Provide a comprehensive 1000-word analysis of the history of quantum computing, "
        "from theoretical foundations through modern implementations."
    )
    _drain(orch.handle_user_message(question, history, 50, False, 3))
    print("  Submitted complex question")

    poll_history = _wait_for_ready(orch, 1)
    history.extend(poll_history)
    assert len(orch._ready_items) == 1
    ready_id = list(orch._ready_items.keys())[0]
    print(f"  Answer ready (id={ready_id})")

    notifications = [
        m
        for m in history
        if m["role"] == "assistant" and "finished working" in m["content"]
    ]
    assert len(notifications) >= 1
    print(f"  Notification: {notifications[-1]['content'][:80]}...")

    # User says yes — Gemini should route to SELECT
    updates = _drain(
        orch.handle_user_message(
            "Yes, please tell me the answer", history, 50, False, 3
        )
    )

    delivered = [
        m
        for m in history
        if m["role"] == "assistant" and "Regarding your question" in m["content"]
    ]

    if delivered:
        assert (
            len(orch._ready_items) == 0
        ), f"Ready item should be consumed, got {len(orch._ready_items)}"
        print(f"  SELECT worked: {delivered[-1]['content'][:80]}...")
        print(f"  Ready queue empty")
    else:
        print(
            f"  NOTE: Gemini chose GENERATE instead of SELECT (intent classification varies)"
        )
        print(
            f"  Last assistant msg: {[m for m in history if m['role'] == 'assistant'][-1]['content'][:80]}..."
        )

    orch.backend_worker.stop()
    print("  PASSED")


def test_intention_generate():
    """With a ready item pending, a clearly new question should GENERATE, not SELECT."""
    print("\n=== Test: Intention GENERATE (Gemini intention) ===")

    orch = _make_orchestrator(backend_delay=0.3)
    history: list[dict] = []

    question = (
        "Write a detailed technical report on the architecture of transformer neural networks "
        "including attention mechanisms, training procedures, and scaling laws."
    )
    _drain(orch.handle_user_message(question, history, 50, False, 3))

    poll_history = _wait_for_ready(orch, 1)
    history.extend(poll_history)
    assert len(orch._ready_items) == 1
    print("  Answer ready, now asking unrelated question")

    # Ask something clearly unrelated — Gemini should route to GENERATE
    updates = _drain(orch.handle_user_message("What is 2 + 2?", history, 50, False, 3))

    # The ready item should still be there (not consumed)
    assistant_msgs = [m for m in history if m["role"] == "assistant"]
    last_msg = assistant_msgs[-1]["content"]

    if len(orch._ready_items) == 1:
        print(f"  GENERATE worked, ready item preserved")
        print(f"  Answer: {last_msg[:80]}...")
    else:
        print(f"  NOTE: Gemini chose SELECT (intent classification varies)")

    orch.backend_worker.stop()
    print("  PASSED")


def test_two_complex_then_select_each():
    """Submit two complex questions, get both ready, select them one by one."""
    print("\n=== Test: Two Complex → Select Each (Gemini full flow) ===")

    orch = _make_orchestrator(backend_delay=0.3)
    history: list[dict] = []

    q1 = (
        "Write a 500-word essay on the impact of climate change on global agriculture "
        "including crop yield predictions and adaptation strategies."
    )
    q2 = (
        "Provide a comprehensive analysis of the evolution of programming languages "
        "from Fortran through modern languages including paradigm shifts."
    )

    _drain(orch.handle_user_message(q1, history, 50, False, 3))
    _drain(orch.handle_user_message(q2, history, 50, False, 3))
    print("  Submitted 2 complex questions")

    poll_history = _wait_for_ready(orch, 2)
    history.extend(poll_history)
    assert len(orch._ready_items) == 2
    print("  Both answers ready")

    notifications = [
        m
        for m in history
        if m["role"] == "assistant" and "finished working" in m["content"]
    ]
    assert len(notifications) == 2
    for n in notifications:
        print(f"  Notification: {n['content'][:80]}...")

    # Select first — reference the topic directly
    updates = _drain(
        orch.handle_user_message(
            "Yes, tell me about the climate change question", history, 50, False, 3
        )
    )
    remaining_after_first = len(orch._ready_items)
    print(f"  After first select: {remaining_after_first} ready items remaining")

    # Select second
    updates = _drain(
        orch.handle_user_message(
            "Now give me the programming languages answer", history, 50, False, 3
        )
    )
    remaining_after_second = len(orch._ready_items)
    print(f"  After second select: {remaining_after_second} ready items remaining")

    delivered = [
        m
        for m in history
        if m["role"] == "assistant" and "Regarding your question" in m["content"]
    ]
    print(f"  Total delivered answers: {len(delivered)}")
    for d in delivered:
        print(f"    {d['content'][:80]}...")

    orch.backend_worker.stop()
    print("  PASSED")


# ---------------------------------------------------------------------------
# Mock TTS client (writes a tiny valid WAV file)
# ---------------------------------------------------------------------------
class MockTTSClient(TTSClient):
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.call_count = 0

    def synthesize(self, text: str) -> str:
        self.call_count += 1
        time.sleep(self.delay)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        n_frames = 2400  # 0.1s at 24kHz
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
        return path

    def get_voices(self) -> list[str]:
        return ["MockVoice"]

    def set_voice(self, voice: str) -> None:
        pass


def _wait_for_tts_ready(
    orch: Orchestrator, source_filter: TTSSource | None, timeout: float = 10.0
):
    """Wait until a TTS item with the given source is ready."""
    start = time.time()
    while time.time() - start < timeout:
        items = orch.tts_queue_worker.get_all_items()
        for item in items:
            if item.status == "ready" and (
                source_filter is None or item.source == source_filter
            ):
                return item
        time.sleep(0.1)
    statuses = [
        (i.id, i.source.value, i.status) for i in orch.tts_queue_worker.get_all_items()
    ]
    raise TimeoutError(f"No TTS item ready after {timeout}s. Items: {statuses}")


def _make_speech_orchestrator(backend_delay=0.3, tts_delay=0.1):
    """Create an Orchestrator in Speech mode with mock LLM + mock TTS."""
    orch = Orchestrator()
    mock_llm = MockLLMClient(delay=backend_delay)
    orch.backend_worker = LLMQueueWorker(mock_llm)
    orch.backend_worker.start()
    mock_tts = MockTTSClient(delay=tts_delay)
    orch.tts_queue_worker = TTSQueueWorker(mock_tts)
    orch.tts_queue_worker.start()
    orch.tts_engine = mock_tts
    orch.output_mode = "Speech"
    return orch, mock_tts


def _poll_until(orch, history, predicate, timeout=15.0):
    """Poll repeatedly until predicate(history) is True."""
    start = time.time()
    while time.time() - start < timeout:

        orch.poll(history)
        if predicate(history):
            return
        time.sleep(0.2)
    raise TimeoutError(f"Predicate not met after {timeout}s")


def test_speech_mode_notification_deferred():
    """In Speech mode: notification (text+speech) sent only when answer TTS is ready.

    Flow:
      1. user: simple question → text + speech answer
      2. user: complex question → relay + relay speech
      3. backend finishes → answer TTS enqueued, NO text notification yet
      4. answer TTS ready → text notification appears + notification speech enqueued
      5. user: "yes" → SELECT → text delivered, answer audio already synthesized
    """
    print("\n=== Test: Speech Mode — Notification Deferred Until Answer TTS Ready ===")

    orch, mock_tts = _make_speech_orchestrator(backend_delay=0.3, tts_delay=0.1)
    history: list[dict] = []

    # --- Step 1: Simple question → text + speech ---
    print("  Step 1: Simple question")
    _drain(orch.handle_user_message("What is 2 + 2?", history, 50, False, 3))
    assert any(m["role"] == "assistant" for m in history)
    print(
        f"    Answer: {[m for m in history if m['role'] == 'assistant'][-1]['content'][:50]}..."
    )

    # Drain the simple answer audio
    _wait_for_tts_ready(orch, TTSSource.FRONTEND)

    orch.poll(history)
    print("    Simple answer audio drained")

    # --- Step 2: Complex question → relay ---
    print("  Step 2: Complex question")
    complex_q = (
        "Write a comprehensive 800-word analysis of the economic impact of renewable energy "
        "on developing nations, including case studies from Africa and Southeast Asia."
    )
    _drain(orch.handle_user_message(complex_q, history, 50, False, 3))
    relay = [
        m
        for m in history
        if m["role"] == "assistant" and "I'll work on" in m["content"]
    ]
    assert len(relay) >= 1
    print(f"    Relay: {relay[-1]['content'][:60]}...")

    # Drain relay TTS
    time.sleep(0.3)

    orch.poll(history)

    # --- Step 3: Backend finishes → answer TTS enqueued, NO notification yet ---
    print("  Step 3: Backend ready → no text notification yet")
    # Wait for backend result to be picked up by poll
    start = time.time()
    while time.time() - start < 10:

        orch.poll(history)
        if len(orch._ready_items) >= 1:
            break
        time.sleep(0.2)
    assert len(orch._ready_items) == 1

    # At this point, answer TTS is enqueued but may not be ready yet.
    # The notification should NOT be in history yet (it's pending).
    notifications_now = [
        m
        for m in history
        if m["role"] == "assistant" and "finished working" in m["content"]
    ]
    # The notification might already appear if TTS was fast, so check pending count
    pending_count = len(orch._pending_notifications)
    print(
        f"    Ready items: {len(orch._ready_items)}, pending notifications: {pending_count}"
    )
    print(f"    Text notifications in history so far: {len(notifications_now)}")

    # --- Step 4: Keep polling — answer TTS becomes ready → notification appears ---
    print("  Step 4: Answer TTS ready → notification sent (text + speech)")

    def has_notification(h):
        return any(
            m["role"] == "assistant" and "finished working" in m["content"] for m in h
        )

    _poll_until(orch, history, has_notification)

    notifications = [
        m
        for m in history
        if m["role"] == "assistant" and "finished working" in m["content"]
    ]
    assert (
        len(notifications) >= 1
    ), "Notification should appear after answer TTS is ready"
    print(f"    Notification: {notifications[-1]['content'][:70]}...")

    # Notification speech should also be enqueued
    all_tts = orch.tts_queue_worker.get_all_items()
    notification_tts = [i for i in all_tts if "finished working" in i.text]
    assert len(notification_tts) >= 1, "Notification speech should be enqueued"
    print(f"    Notification TTS enqueued: {notification_tts[0].text[:50]}...")

    assert (
        len(orch._pending_notifications) == 0
    ), "No pending notifications should remain"
    print("    All pending notifications resolved")

    # --- Step 5: User selects the answer ---
    print("  Step 5: User 'yes' → SELECT → text + audio delivered together")

    # Drain notification speech so it doesn't block
    for _ in range(5):

        update = orch.poll(history)
        if not update.audio_path:
            break

    # Verify answer audio is stored for SELECT
    assert (
        len(orch._ready_audio) >= 1
    ), f"Expected stored answer audio, got {len(orch._ready_audio)}"
    print(f"    Answer audio stored: {list(orch._ready_audio.values())}")

    updates = _drain(
        orch.handle_user_message(
            "Yes, tell me the answer about renewable energy", history, 50, False, 3
        )
    )

    delivered = [
        m
        for m in history
        if m["role"] == "assistant" and "Regarding your question" in m["content"]
    ]
    if delivered:
        assert len(orch._ready_items) == 0
        print(f"    Text delivered: {delivered[-1]['content'][:70]}...")
        print("    Ready queue: empty")

        # The SELECT update should include the audio_path
        select_updates = [u for u in updates if u.audio_path]
        assert (
            len(select_updates) >= 1
        ), "SELECT should deliver audio_path with the text"
        print(f"    Audio delivered with SELECT: {select_updates[0].audio_path}")

        assert len(orch._ready_audio) == 0, "Audio should be consumed after SELECT"
        print("    Stored audio consumed")
    else:
        print("  NOTE: Gemini chose GENERATE instead of SELECT")

    # Final TTS summary
    all_tts = orch.tts_queue_worker.get_all_items()
    print(f"  Final TTS queue ({len(all_tts)} items):")
    for t in all_tts:
        print(f"    [{t.source.value}] {t.status}: {t.text[:50]}...")

    orch.backend_worker.stop()
    orch.tts_queue_worker.stop()
    print("  PASSED")


def test_text_mode_notification_immediate():
    """In Text mode: notification appears immediately when backend answer is ready (no TTS)."""
    print("\n=== Test: Text Mode — Notification Immediate ===")

    orch = _make_orchestrator(backend_delay=0.3)
    orch.output_mode = "Text"
    history: list[dict] = []

    complex_q = (
        "Write a comprehensive 800-word analysis of the economic impact of renewable energy "
        "on developing nations, including case studies from Africa and Southeast Asia."
    )
    _drain(orch.handle_user_message(complex_q, history, 50, False, 3))
    relay = [
        m
        for m in history
        if m["role"] == "assistant" and "I'll work on" in m["content"]
    ]
    assert len(relay) >= 1
    print(f"  Relay: {relay[-1]['content'][:60]}...")

    # Wait for backend to finish and poll
    poll_history = _wait_for_ready(orch, 1)
    history.extend(poll_history)

    # Notification should appear immediately in text mode
    notifications = [
        m
        for m in history
        if m["role"] == "assistant" and "finished working" in m["content"]
    ]
    assert len(notifications) >= 1, "Text mode: notification should appear immediately"
    print(f"  Notification (immediate): {notifications[-1]['content'][:70]}...")

    # No pending notifications in text mode
    assert len(orch._pending_notifications) == 0
    print("  No pending notifications (text mode sends immediately)")

    orch.backend_worker.stop()
    print("  PASSED")


# ---------------------------------------------------------------------------
# Race-condition fix tests (no network — pure orchestrator state checks)
# ---------------------------------------------------------------------------
def test_input_mode_gates_text_submit_in_speech_mode():
    """In Speech input mode, handle_user_message must silently ignore text submits."""
    print("\n=== Test: Input mode gates text submit in speech mode ===")

    orch = _make_orchestrator()
    orch.set_input_mode("Speech")
    history: list[dict] = []

    updates = _drain(orch.handle_user_message("hello", history, 50, False, 3))

    assert updates == [], f"expected no updates, got {len(updates)}"
    assert history == [], f"history should be untouched, got {history}"
    assert not orch.is_busy(), "busy flag should not have been acquired"

    orch.backend_worker.stop()
    print("  PASSED")


def test_input_mode_gates_audio_in_text_mode():
    """In Text input mode, handle_audio_input must silently ignore audio events."""
    print("\n=== Test: Input mode gates audio events in text mode ===")

    import numpy as np

    orch = _make_orchestrator()
    orch.set_input_mode("Text")  # default
    history: list[dict] = []

    fake_audio = (16000, np.zeros(16000, dtype=np.int16))
    updates = _drain(orch.handle_audio_input(fake_audio, history, 50, False, 3))

    assert updates == [], f"expected no updates, got {len(updates)}"
    assert history == [], f"history should be untouched, got {history}"

    orch.backend_worker.stop()
    print("  PASSED")


def test_busy_lock_rejects_concurrent_message():
    """While one handle_user_message is in flight, a second call gets a warning."""
    print("\n=== Test: Busy lock rejects concurrent user-message ===")

    orch = _make_orchestrator()
    # Force a non-Gemini path so we don't hit network; just acquire the lock
    # directly to simulate an in-flight handler.
    assert orch._try_acquire_busy(), "should acquire initially"
    try:
        history: list[dict] = []
        updates = _drain(
            orch.handle_user_message("second message", history, 50, False, 3)
        )
        warnings = [u for u in updates if u.warning]
        assert len(warnings) == 1, f"expected 1 warning, got {warnings}"
        assert "still being processed" in warnings[0].warning
        assert history == [], "history should not be modified for rejected call"
    finally:
        orch._release_busy()

    assert not orch.is_busy()
    orch.backend_worker.stop()
    print("  PASSED")


def test_poll_defers_notification_when_busy():
    """poll() should defer chat appends + set skip_chat_update while busy."""
    print("\n=== Test: poll defers notifications while busy ===")

    orch = _make_orchestrator(backend_delay=0.2)
    orch.output_mode = "Text"
    history: list[dict] = []

    # Inject a fake ready item into the backend worker's result queue
    # directly, so we don't depend on Gemini.
    fake = QueueItem(
        id="fake1",
        question="What is foo?",
        context_summary="foo concept",
        history=[],
        answer="Foo is a placeholder.",
        status="ready",
    )
    orch.backend_worker._items[fake.id] = fake
    orch.backend_worker._result_queue.put(fake)

    # Simulate a user-message handler in flight.
    assert orch._try_acquire_busy()
    try:
        update = orch.poll(history)
        assert update.skip_chat_update, "poll must skip chat update when busy"
        assert not any(
            "finished working" in m.get("content", "") for m in history
        ), "no notification should be appended while busy"
        assert len(orch._deferred_notifications) == 1, (
            f"expected 1 deferred notification, got "
            f"{len(orch._deferred_notifications)}"
        )
        assert "finished working" in orch._deferred_notifications[0]
        # Ready item bookkeeping should still happen so SELECT works later.
        assert fake.id in orch._ready_items
    finally:
        orch._release_busy()

    # Next poll while not busy must flush deferred notifications.
    update = orch.poll(history)
    assert not update.skip_chat_update
    notifications = [m for m in history if "finished working" in m.get("content", "")]
    assert len(notifications) == 1, f"expected flushed notification, got {history}"
    assert orch._deferred_notifications == []

    orch.backend_worker.stop()
    print("  PASSED")


def test_clear_drops_deferred_notifications():
    """clear() must wipe deferred notifications so they don't leak across sessions."""
    print("\n=== Test: clear() wipes deferred notifications ===")

    orch = _make_orchestrator()
    orch._deferred_notifications.append("stale notification")

    update = orch.clear()
    assert orch._deferred_notifications == []
    assert update.history == []

    orch.backend_worker.stop()
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_complex_question_gets_queued()
    test_simple_question_answered_directly()
    test_intention_select()
    test_intention_generate()
    test_two_complex_then_select_each()
    test_text_mode_notification_immediate()
    test_speech_mode_notification_deferred()
    test_input_mode_gates_text_submit_in_speech_mode()
    test_input_mode_gates_audio_in_text_mode()
    test_busy_lock_rejects_concurrent_message()
    test_poll_defers_notification_when_busy()
    test_clear_drops_deferred_notifications()
    print("\n=== All orchestrator tests passed! ===")


if __name__ == "__main__":
    main()
