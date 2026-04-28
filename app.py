"""Multi-Stream Conversation — Gradio chat with front-end triage and back-end deep processing.

Usage:
    ./env/fb/run.sh python app.py

Launch client on laptop:
    ssh -L 7863:localhost:7863 bshi@<devgpu>
    http://localhost:7863
"""

import html
import logging
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import gradio as gr

from backend import (
    DEFAULT_ASR,
    DEFAULT_TTS_VOICE,
    GEMINI_TTS_VOICES,
    BackendWorker,
    GeminiASRBackend,
    GeminiBackend,
    GeminiTTSBackend,
    LocalBackend,
    TTSQueueWorker,
    TTSSource,
    WhisperASRBackend,
    get_asr_choices,
)
from gemini_client import (
    GEMINI_DEFAULT_MODEL,
    estimate_complexity_gemini,
    generate_gemini_response,
    generate_gemini_response_streaming,
    get_gemini_model_names,
)
from models import (
    BACKEND_SYSTEM_PROMPT,
    FRONTEND_SYSTEM_PROMPT,
    QWEN_DEFAULT_MODEL,
    TRIAGE_SYSTEM_PROMPT,
    estimate_complexity,
    generate_delivery_summary,
    generate_response,
    generate_response_streaming,
    get_model_names,
    load_qwen,
    unload_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
frontend_model = None
frontend_lock = threading.Lock()
backend_worker: BackendWorker | None = None
frontend_type: str = "Gemini API"
frontend_gemini_model: str = GEMINI_DEFAULT_MODEL
backend_insert_positions: dict[str, int] = {}

asr_engine = WhisperASRBackend()
tts_engine: GeminiTTSBackend = GeminiTTSBackend()
tts_queue_worker: TTSQueueWorker | None = None
output_mode: str = "Text"

STATUS_COLORS = {
    "queued": ("#999", "Queued"),
    "processing": ("#f0ad4e", "Processing"),
    "ready": ("#5cb85c", "Ready"),
    "delivered": ("#5cb85c", "Delivered"),
}

QUEUE_PANEL_CSS = """
#text-queue-panel, #speech-queue-panel {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 10000 !important;
    background: white !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    width: 620px !important;
    max-width: 90vw !important;
    max-height: 80vh !important;
    padding: 0 !important;
    border: 1px solid #ddd !important;
}
#text-queue-panel > .column-wrap, #speech-queue-panel > .column-wrap,
#text-queue-panel > div, #speech-queue-panel > div {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    max-height: 80vh;
}
#text-queue-panel .panel-header, #speech-queue-panel .panel-header {
    flex-shrink: 0;
    border-bottom: 1px solid #eee;
    padding: 8px 4px;
}
#text-queue-panel .panel-body, #speech-queue-panel .panel-body {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}
.queue-item {
    padding: 10px 12px;
    margin-bottom: 6px;
    border-radius: 6px;
    background: #f8f9fa;
    border-left: 4px solid #ccc;
}
.queue-item.status-queued { border-left-color: #999; }
.queue-item.status-processing { border-left-color: #f0ad4e; }
.queue-item.status-ready { border-left-color: #5cb85c; }
.queue-item.status-delivered { border-left-color: #5cb85c; }
.queue-item-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.status-label {
    font-size: 0.8em;
    color: #777;
    font-weight: 600;
}
.queue-item-text {
    font-size: 0.9em;
    color: #333;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.55;
}
.queue-section-title {
    font-size: 0.95em;
    font-weight: 600;
    color: #555;
    border-bottom: 1px solid #eee;
    padding-bottom: 4px;
    margin-bottom: 10px;
}
.queue-empty {
    color: #aaa;
    font-size: 0.88em;
    font-style: italic;
    padding: 6px 0;
}
"""


def _render_item_html(text: str, status: str, source_tag: str = "") -> str:
    color, label = STATUS_COLORS.get(status, ("#999", "Unknown"))
    escaped = html.escape(text)
    tag_html = ""
    if source_tag:
        tag_html = (
            f' <span style="font-size:0.75em;color:#fff;background:'
            f'{"#337ab7" if source_tag == "frontend" else "#8e44ad"};'
            f'border-radius:3px;padding:1px 5px;">{source_tag}</span>'
        )
    return (
        f'<div class="queue-item status-{status}">'
        f'<div class="queue-item-header">'
        f'<span class="status-dot" style="background:{color};"></span>'
        f'<span class="status-label">[{label}]</span>{tag_html}'
        f'</div>'
        f'<div class="queue-item-text">{escaped}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def load_models():
    global backend_worker, tts_queue_worker
    logger.info("Starting backend worker with Gemini API")
    backend = GeminiBackend(model_key=GEMINI_DEFAULT_MODEL)
    backend_worker = BackendWorker(backend)
    backend_worker.start()

    tts_queue_worker = TTSQueueWorker(tts_engine)
    tts_queue_worker.start()
    logger.info("Backend worker and TTS queue worker started")


# ---------------------------------------------------------------------------
# Queue rendering
# ---------------------------------------------------------------------------
def _count_text_queue() -> int:
    if backend_worker is None:
        return 0
    return len(backend_worker.get_all_items())


def _count_speech_queue() -> int:
    if tts_queue_worker is None:
        return 0
    return len(tts_queue_worker.get_all_items())


def render_text_queue_html() -> str:
    if backend_worker is None:
        return '<p class="queue-empty">Backend not ready</p>'

    items = backend_worker.get_all_items()
    if not items:
        return '<p class="queue-empty">No items in queue</p>'

    items_sorted = sorted(items, key=lambda x: x.timestamp)

    question_html = ""
    answer_html = ""
    for item in items_sorted:
        question_html += _render_item_html(item.question, item.status)
        if item.status in ("ready", "delivered") and item.answer:
            a_status = "delivered" if item.status == "delivered" else "ready"
            answer_html += _render_item_html(item.answer, a_status)

    return (
        f'<div style="padding:0 4px;">'
        f'<div class="queue-section-title">Question Queue</div>'
        f'{question_html or "<p class=queue-empty>Empty</p>"}'
        f'<div class="queue-section-title" style="margin-top:16px;">Answer Queue</div>'
        f'{answer_html or "<p class=queue-empty>No answers yet</p>"}'
        f'</div>'
    )


def render_speech_queue_html() -> str:
    if tts_queue_worker is None:
        return '<p class="queue-empty">TTS queue not ready</p>'

    items = tts_queue_worker.get_all_items()
    if not items:
        return '<p class="queue-empty">No TTS items</p>'

    items_sorted = sorted(items, key=lambda x: x.timestamp)
    items_html = ""
    for item in items_sorted:
        items_html += _render_item_html(item.text, item.status, item.source.value)

    return (
        f'<div style="padding:0 4px;">'
        f'<div class="queue-section-title">TTS Synthesis Queue</div>'
        f'{items_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Frontend helpers (Gemini path)
# ---------------------------------------------------------------------------
def _gemini_triage(user_text: str, threshold_n: int):
    """Triage using Gemini — returns (is_complex, estimated_words, context_summary)."""
    raw = estimate_complexity_gemini(frontend_gemini_model, user_text, TRIAGE_SYSTEM_PROMPT)
    logger.info("gemini triage raw: %s", raw)
    words_match = re.search(r"ESTIMATED_WORDS:\s*(\d+)", raw)
    if not words_match:
        return False, 0, user_text[:80]
    estimated_words = int(words_match.group(1))
    return estimated_words >= threshold_n, estimated_words, user_text[:80]


def _maybe_enqueue_tts(text: str, source: TTSSource) -> None:
    """If output mode is Speech, enqueue text for TTS synthesis."""
    if output_mode != "Speech" or tts_queue_worker is None:
        return
    tts_queue_worker.submit(text, source=source)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def on_user_message(
    user_text: str,
    history: list[dict],
    threshold_n: int,
    streaming_enabled: bool,
    num_words_delay: int,
):
    no_audio = gr.update()
    q_btn_text = gr.update(value=f"Text Queue ({_count_text_queue()})")
    q_btn_speech = gr.update(value=f"Speech Queue ({_count_speech_queue()})")

    def _q_updates():
        return (
            gr.update(value=f"Text Queue ({_count_text_queue()})"),
            gr.update(value=f"Speech Queue ({_count_speech_queue()})"),
            render_text_queue_html(),
            render_speech_queue_html(),
        )

    if not user_text.strip():
        t_btn, s_btn, t_html, s_html = _q_updates()
        yield history, "", t_btn, s_btn, t_html, s_html, no_audio
        return

    if (
        tts_queue_worker is not None
        and tts_queue_worker.has_pending_immediate()
    ):
        gr.Warning(
            "Please wait — the previous reply is still being spoken. "
            "You can send a new message once TTS playback is ready."
        )
        yield history, user_text, gr.update(), gr.update(), gr.update(), gr.update(), no_audio
        return

    using_gemini_frontend = frontend_type == "Gemini API"

    if not using_gemini_frontend and frontend_model is None:
        history.append(
            {"role": "assistant", "content": "Models are still loading, please wait..."}
        )
        t_btn, s_btn, t_html, s_html = _q_updates()
        yield history, "", t_btn, s_btn, t_html, s_html, no_audio
        return

    history.append({"role": "user", "content": user_text})
    t_btn, s_btn, t_html, s_html = _q_updates()
    yield history, "", t_btn, s_btn, t_html, s_html, no_audio

    final_text = None

    if using_gemini_frontend:
        is_complex, est_words, summary = _gemini_triage(user_text, threshold_n)

        if is_complex:
            item = backend_worker.submit(user_text, summary, history[:-1])
            reply = (
                f"That's a detailed question — I'll work on a thorough answer and get back to you. "
                f"(Estimated ~{est_words} words needed)"
            )
            history.append({"role": "assistant", "content": reply})
            backend_insert_positions[item.id] = len(history)
            _maybe_enqueue_tts(reply, TTSSource.FRONTEND)
            t_btn, s_btn, t_html, s_html = _q_updates()
            yield history, "", t_btn, s_btn, t_html, s_html, no_audio
        elif streaming_enabled:
            history.append({"role": "assistant", "content": ""})
            for partial_text in generate_gemini_response_streaming(
                model_key=frontend_gemini_model,
                user_text=user_text,
                history=history[:-2],
                system_prompt=FRONTEND_SYSTEM_PROMPT,
                max_tokens=2048,
                num_words_delay=num_words_delay,
            ):
                history[-1]["content"] = partial_text
                yield history, "", gr.update(), gr.update(), gr.update(), gr.update(), no_audio
            final_text = history[-1]["content"]
        else:
            answer = generate_gemini_response(
                model_key=frontend_gemini_model,
                user_text=user_text,
                history=history[:-1],
                system_prompt=FRONTEND_SYSTEM_PROMPT,
                max_tokens=2048,
            )
            history.append({"role": "assistant", "content": answer})
            final_text = answer
            t_btn, s_btn, t_html, s_html = _q_updates()
            yield history, "", t_btn, s_btn, t_html, s_html, no_audio
    else:
        # Local Qwen frontend
        with frontend_lock:
            is_complex, est_words, direct_answer, summary = estimate_complexity(
                frontend_model,
                user_text,
                history[:-1],
                threshold_n,
                skip_answer=streaming_enabled,
            )

        if is_complex:
            item = backend_worker.submit(user_text, summary, history[:-1])
            reply = (
                f"That's a detailed question — I'll work on a thorough answer and get back to you. "
                f"(Estimated ~{est_words} words needed)"
            )
            history.append({"role": "assistant", "content": reply})
            backend_insert_positions[item.id] = len(history)
            _maybe_enqueue_tts(reply, TTSSource.FRONTEND)
            t_btn, s_btn, t_html, s_html = _q_updates()
            yield history, "", t_btn, s_btn, t_html, s_html, no_audio
        elif direct_answer:
            history.append({"role": "assistant", "content": direct_answer})
            final_text = direct_answer
            t_btn, s_btn, t_html, s_html = _q_updates()
            yield history, "", t_btn, s_btn, t_html, s_html, no_audio
        elif streaming_enabled:
            history.append({"role": "assistant", "content": ""})
            with frontend_lock:
                for partial_text in generate_response_streaming(
                    frontend_model, user_text, history[:-2], num_words_delay=num_words_delay
                ):
                    history[-1]["content"] = partial_text
                    yield history, "", gr.update(), gr.update(), gr.update(), gr.update(), no_audio
            final_text = history[-1]["content"]
        else:
            with frontend_lock:
                answer = generate_response(frontend_model, user_text, history[:-1])
            history.append({"role": "assistant", "content": answer})
            final_text = answer
            t_btn, s_btn, t_html, s_html = _q_updates()
            yield history, "", t_btn, s_btn, t_html, s_html, no_audio

    if final_text:
        _maybe_enqueue_tts(final_text, TTSSource.FRONTEND)
        t_btn, s_btn, t_html, s_html = _q_updates()
        yield history, "", t_btn, s_btn, t_html, s_html, no_audio


def poll_backend_and_tts(history: list[dict]):
    """Poll both the LLM backend and TTS queue for completed items."""
    audio_out = gr.update()

    # --- LLM backend results ---
    if backend_worker is not None:
        results = backend_worker.get_results()
        for item in results:
            t0 = time.time()
            logger.info("poll_backend: picked up item %s", item.id)

            delivery = (
                f"Regarding your earlier question about **{item.context_summary}**:\n\n"
                f"{item.answer}"
            )

            insert_idx = backend_insert_positions.pop(item.id, None)
            if insert_idx is not None and insert_idx <= len(history):
                history.insert(insert_idx, {"role": "assistant", "content": delivery})
                for k, v in backend_insert_positions.items():
                    if v >= insert_idx:
                        backend_insert_positions[k] = v + 1
            else:
                if history and history[-1]["role"] == "assistant":
                    history.append(
                        {"role": "user", "content": f"[Completed: {item.context_summary}]"}
                    )
                history.append({"role": "assistant", "content": delivery})

            backend_worker.mark_delivered(item.id)
            _maybe_enqueue_tts(item.answer, TTSSource.BACKEND)
            logger.info("poll_backend: item %s delivered (%.2fs)", item.id, time.time() - t0)

    # --- TTS queue results (one at a time to avoid cutting off playback) ---
    if tts_queue_worker is not None:
        tts_item = tts_queue_worker.get_next_audio()
        if tts_item is not None:
            if tts_item.audio_path:
                audio_out = tts_item.audio_path
            tts_queue_worker.mark_delivered(tts_item.id)
            logger.info("TTS item %s delivered (%.1fs)", tts_item.id, tts_item.audio_duration)

    return (
        history,
        gr.update(value=f"Text Queue ({_count_text_queue()})"),
        gr.update(value=f"Speech Queue ({_count_speech_queue()})"),
        render_text_queue_html(),
        render_speech_queue_html(),
        audio_out,
    )


def on_frontend_type_change(
    fe_type: str,
    local_model_key: str,
    gemini_model_key: str,
    history: list[dict],
):
    global frontend_model, frontend_type, frontend_gemini_model

    frontend_type = fe_type

    if fe_type == "Gemini API":
        frontend_gemini_model = gemini_model_key
        if frontend_model is not None:
            with frontend_lock:
                unload_model(frontend_model)
                frontend_model = None
        label = f"Gemini ({gemini_model_key})"
    else:
        frontend_gemini_model = GEMINI_DEFAULT_MODEL
        if frontend_model is None:
            with frontend_lock:
                frontend_model = load_qwen(local_model_key)
        label = f"Local ({local_model_key})"

    history.append({"role": "assistant", "content": f"Frontend switched to {label}."})
    return (
        history,
        gr.update(visible=fe_type == "Local Qwen"),
        gr.update(visible=fe_type == "Gemini API"),
    )


def on_frontend_local_model_change(model_key: str, history: list[dict]):
    global frontend_model
    if frontend_model is not None:
        with frontend_lock:
            unload_model(frontend_model)
            frontend_model = None

    logger.info("Switching frontend model to %s", model_key)
    with frontend_lock:
        frontend_model = load_qwen(model_key)
    logger.info("Frontend model switched to %s", model_key)

    history.append(
        {"role": "assistant", "content": f"Frontend model switched to {model_key}."}
    )
    return history


def on_frontend_gemini_model_change(model_key: str, history: list[dict]):
    global frontend_gemini_model
    frontend_gemini_model = model_key
    logger.info("Frontend Gemini model switched to %s", model_key)
    history.append(
        {"role": "assistant", "content": f"Frontend Gemini model switched to {model_key}."}
    )
    return history


def on_backend_type_change(
    be_type: str,
    local_model_key: str,
    gemini_model_key: str,
    history: list[dict],
):
    global backend_worker

    if be_type == "Gemini API":
        new_backend = GeminiBackend(model_key=gemini_model_key)
        label = f"Gemini ({gemini_model_key})"
    else:
        new_backend = LocalBackend(local_model_key)
        label = f"Local ({local_model_key})"

    if backend_worker is not None:
        backend_worker.swap_backend(new_backend)
    else:
        backend_worker = BackendWorker(new_backend)
        backend_worker.start()

    logger.info("Backend switched to %s", label)
    history.append({"role": "assistant", "content": f"Backend switched to {label}."})
    return (
        history,
        gr.update(visible=be_type == "Local Qwen"),
        gr.update(visible=be_type == "Gemini API"),
    )


def on_backend_local_model_change(model_key: str, history: list[dict]):
    global backend_worker
    if backend_worker is not None:
        logger.info("Switching backend model to %s", model_key)
        new_backend = LocalBackend(model_key)
        backend_worker.swap_backend(new_backend)
    else:
        backend = LocalBackend(model_key)
        backend_worker = BackendWorker(backend)
        backend_worker.start()

    logger.info("Backend model switched to %s", model_key)
    history.append(
        {"role": "assistant", "content": f"Backend model switched to {model_key}."}
    )
    return history


def on_backend_gemini_model_change(model_key: str, history: list[dict]):
    global backend_worker
    if backend_worker is not None and isinstance(backend_worker._backend, GeminiBackend):
        backend_worker._backend.load_model(model_key)
        logger.info("Backend Gemini model switched to %s", model_key)
        history.append(
            {"role": "assistant", "content": f"Backend Gemini model switched to {model_key}."}
        )
    return history


def on_input_mode_change(mode: str):
    return (
        gr.update(visible=mode == "Text"),
        gr.update(visible=mode == "Speech"),
    )


def on_asr_change(asr_choice: str, gemini_model_key: str):
    global asr_engine
    if asr_choice == "Gemini ASR":
        asr_engine = GeminiASRBackend(model_key=gemini_model_key)
    else:
        asr_engine = WhisperASRBackend()
    logger.info("ASR switched to %s", asr_choice)
    return gr.update(visible=asr_choice == "Gemini ASR")


def on_asr_gemini_model_change(model_key: str):
    global asr_engine
    if isinstance(asr_engine, GeminiASRBackend):
        asr_engine.model_key = model_key
        logger.info("ASR Gemini model switched to %s", model_key)


def on_output_mode_change(mode: str):
    global output_mode
    output_mode = mode
    logger.info("Output mode switched to %s", mode)


def on_tts_voice_change(voice: str):
    tts_engine.set_voice(voice)


def on_audio_record(
    audio_data,
    history: list[dict],
    threshold_n: int,
    streaming_enabled: bool,
    num_words_delay: int,
):
    if audio_data is None:
        yield history, gr.update(value="Ready to record"), None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    sr, data = audio_data
    yield history, gr.update(value="Transcribing..."), None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    transcript = asr_engine.transcribe(sr, data)
    logger.info("ASR transcript: %s", transcript)

    if not transcript.strip():
        yield history, gr.update(value="(no speech detected)"), None, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    for hist_update, _, t_btn, s_btn, t_html, s_html, tts_audio in on_user_message(
        transcript, history, threshold_n, streaming_enabled, num_words_delay
    ):
        yield hist_update, gr.update(value=f"Sent: {transcript[:80]}"), None, t_btn, s_btn, t_html, s_html, tts_audio


def on_clear():
    if backend_worker is not None:
        backend_worker.clear_items()
    if tts_queue_worker is not None:
        tts_queue_worker.clear_items()
    return (
        [],
        gr.update(value=f"Text Queue (0)"),
        gr.update(value=f"Speech Queue (0)"),
        render_text_queue_html(),
        render_speech_queue_html(),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    load_thread = threading.Thread(target=load_models, daemon=True)
    load_thread.start()

    local_model_names = get_model_names()
    gemini_model_names = get_gemini_model_names()

    with gr.Blocks(title="Multi-Stream Conversation", css=QUEUE_PANEL_CSS) as demo:
        gr.Markdown("# Multi-Stream Conversation\nFront-end triage + back-end deep processing")

        with gr.Row():
            # --- Left sidebar ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Frontend Settings")
                frontend_type_radio = gr.Radio(
                    choices=["Local Qwen", "Gemini API"],
                    value="Gemini API",
                    label="Frontend Type",
                )
                with gr.Column(visible=False) as fe_local_group:
                    fe_local_dropdown = gr.Dropdown(
                        choices=local_model_names,
                        value=QWEN_DEFAULT_MODEL,
                        label="Frontend Model (Local)",
                    )
                with gr.Column(visible=True) as fe_gemini_group:
                    fe_gemini_dropdown = gr.Dropdown(
                        choices=gemini_model_names,
                        value=GEMINI_DEFAULT_MODEL,
                        label="Frontend Model (Gemini)",
                    )

                gr.Markdown("### Backend Settings")
                backend_type_radio = gr.Radio(
                    choices=["Local Qwen", "Gemini API"],
                    value="Gemini API",
                    label="Backend Type",
                )
                with gr.Column(visible=False) as be_local_group:
                    be_local_dropdown = gr.Dropdown(
                        choices=local_model_names,
                        value=QWEN_DEFAULT_MODEL,
                        label="Backend Model (Local)",
                    )
                with gr.Column(visible=True) as be_gemini_group:
                    be_gemini_dropdown = gr.Dropdown(
                        choices=gemini_model_names,
                        value=GEMINI_DEFAULT_MODEL,
                        label="Backend Model (Gemini)",
                    )

                gr.Markdown("### ASR Settings")
                asr_radio = gr.Radio(
                    choices=get_asr_choices(),
                    value=DEFAULT_ASR,
                    label="ASR Backend",
                )
                with gr.Column(visible=False) as asr_gemini_group:
                    asr_gemini_dropdown = gr.Dropdown(
                        choices=gemini_model_names,
                        value=GEMINI_DEFAULT_MODEL,
                        label="ASR Gemini Model",
                    )

                gr.Markdown("### TTS Settings")
                tts_backend_radio = gr.Radio(
                    choices=["Gemini TTS"],
                    value="Gemini TTS",
                    label="TTS Backend",
                )
                tts_voice_dropdown = gr.Dropdown(
                    choices=GEMINI_TTS_VOICES,
                    value=DEFAULT_TTS_VOICE,
                    label="TTS Voice",
                )

                gr.Markdown("### General")
                threshold_slider = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Complexity Threshold (words)",
                )
                streaming_toggle = gr.Checkbox(
                    value=True,
                    label="Stream frontend responses",
                )
                words_delay_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=3,
                    step=1,
                    label="Streaming word delay",
                )
                clear_btn = gr.Button("Clear Chat", variant="secondary")

                gr.Markdown("### Queues")
                with gr.Row():
                    text_queue_btn = gr.Button("Text Queue (0)", size="sm")
                    speech_queue_btn = gr.Button("Speech Queue (0)", size="sm")

            # --- Main chat area ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat",
                )
                with gr.Row():
                    input_mode_radio = gr.Radio(
                        choices=["Text", "Speech"],
                        value="Text",
                        label="Input Mode",
                        interactive=True,
                    )
                    output_mode_radio = gr.Radio(
                        choices=["Text", "Speech"],
                        value="Text",
                        label="Output Mode",
                        interactive=True,
                    )
                audio_output = gr.Audio(
                    label="TTS Output",
                    autoplay=True,
                )
                with gr.Row(visible=True) as text_input_group:
                    text_input = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Column(visible=False) as speech_input_group:
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record your message",
                    )
                    speech_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Click the microphone to record, click again to stop",
                    )

        # --- Floating queue panels (outside main layout) ---
        with gr.Column(visible=False, elem_id="text-queue-panel") as text_queue_panel:
            with gr.Row(elem_classes=["panel-header"]):
                gr.Markdown("### Text Queue")
                text_queue_close = gr.Button("✕", size="sm", scale=0, min_width=40)
            with gr.Column(elem_classes=["panel-body"]):
                text_queue_content = gr.HTML(value=render_text_queue_html())

        with gr.Column(visible=False, elem_id="speech-queue-panel") as speech_queue_panel:
            with gr.Row(elem_classes=["panel-header"]):
                gr.Markdown("### Speech Queue")
                speech_queue_close = gr.Button("✕", size="sm", scale=0, min_width=40)
            with gr.Column(elem_classes=["panel-body"]):
                speech_queue_content = gr.HTML(value=render_speech_queue_html())

        # --- Events ---
        msg_inputs = [
            text_input,
            chatbot,
            threshold_slider,
            streaming_toggle,
            words_delay_slider,
        ]
        msg_outputs = [
            chatbot, text_input,
            text_queue_btn, speech_queue_btn,
            text_queue_content, speech_queue_content,
            audio_output,
        ]
        send_btn.click(fn=on_user_message, inputs=msg_inputs, outputs=msg_outputs)
        text_input.submit(fn=on_user_message, inputs=msg_inputs, outputs=msg_outputs)

        # Queue panel open/close
        text_queue_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[text_queue_panel],
        )
        text_queue_close.click(
            fn=lambda: gr.update(visible=False),
            outputs=[text_queue_panel],
        )
        speech_queue_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[speech_queue_panel],
        )
        speech_queue_close.click(
            fn=lambda: gr.update(visible=False),
            outputs=[speech_queue_panel],
        )

        # Input mode switching
        input_mode_radio.change(
            fn=on_input_mode_change,
            inputs=[input_mode_radio],
            outputs=[text_input_group, speech_input_group],
        )

        # Output mode switching
        output_mode_radio.change(
            fn=on_output_mode_change,
            inputs=[output_mode_radio],
        )

        # TTS settings
        tts_voice_dropdown.change(
            fn=on_tts_voice_change,
            inputs=[tts_voice_dropdown],
        )

        # Speech: when recording stops, transcribe and auto-send
        audio_input.stop_recording(
            fn=on_audio_record,
            inputs=[audio_input, chatbot, threshold_slider, streaming_toggle, words_delay_slider],
            outputs=[
                chatbot, speech_status, audio_input,
                text_queue_btn, speech_queue_btn,
                text_queue_content, speech_queue_content,
                audio_output,
            ],
        )

        timer = gr.Timer(value=2)
        timer.tick(
            fn=poll_backend_and_tts,
            inputs=[chatbot],
            outputs=[
                chatbot,
                text_queue_btn, speech_queue_btn,
                text_queue_content, speech_queue_content,
                audio_output,
            ],
        )

        # Frontend switching
        frontend_type_radio.change(
            fn=on_frontend_type_change,
            inputs=[frontend_type_radio, fe_local_dropdown, fe_gemini_dropdown, chatbot],
            outputs=[chatbot, fe_local_group, fe_gemini_group],
        )
        fe_local_dropdown.change(
            fn=on_frontend_local_model_change,
            inputs=[fe_local_dropdown, chatbot],
            outputs=[chatbot],
        )
        fe_gemini_dropdown.change(
            fn=on_frontend_gemini_model_change,
            inputs=[fe_gemini_dropdown, chatbot],
            outputs=[chatbot],
        )

        # ASR switching
        asr_radio.change(
            fn=on_asr_change,
            inputs=[asr_radio, asr_gemini_dropdown],
            outputs=[asr_gemini_group],
        )
        asr_gemini_dropdown.change(
            fn=on_asr_gemini_model_change,
            inputs=[asr_gemini_dropdown],
        )

        # Backend switching
        backend_type_radio.change(
            fn=on_backend_type_change,
            inputs=[backend_type_radio, be_local_dropdown, be_gemini_dropdown, chatbot],
            outputs=[chatbot, be_local_group, be_gemini_group],
        )
        be_local_dropdown.change(
            fn=on_backend_local_model_change,
            inputs=[be_local_dropdown, chatbot],
            outputs=[chatbot],
        )
        be_gemini_dropdown.change(
            fn=on_backend_gemini_model_change,
            inputs=[be_gemini_dropdown, chatbot],
            outputs=[chatbot],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, text_queue_btn, speech_queue_btn, text_queue_content, speech_queue_content],
        )

    demo.launch(
        server_name="0.0.0.0", server_port=7863, share=True, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
