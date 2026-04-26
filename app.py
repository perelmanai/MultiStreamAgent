"""Multi-Stream Conversation — Gradio chat with front-end triage and back-end deep processing.

Usage:
    ./env/fb/run.sh python app.py

Launch client on laptop:
    ssh -L 7863:localhost:7863 bshi@<devgpu>
    http://localhost:7863
"""

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
    BackendWorker,
    GeminiASRBackend,
    GeminiBackend,
    LocalBackend,
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
frontend_type: str = "Local Qwen"
frontend_gemini_model: str = GEMINI_DEFAULT_MODEL
# Maps backend item ID → history index where the deferral message was placed.
# Used to insert the backend answer at the correct position in the conversation.
backend_insert_positions: dict[str, int] = {}

asr_engine = WhisperASRBackend()

STATUS_COLORS = {
    "queued": ("#999", "Queued"),
    "processing": ("#f0ad4e", "Processing"),
    "ready": ("#5cb85c", "Ready"),
    "delivered": ("#5cb85c", "Delivered"),
}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def load_models(frontend_key: str, backend_key: str):
    global frontend_model, backend_worker
    logger.info("Loading frontend model: %s", frontend_key)
    frontend_model = load_qwen(frontend_key)
    logger.info("Frontend model loaded")

    logger.info("Loading backend model: %s", backend_key)
    backend = LocalBackend(backend_key)
    backend_worker = BackendWorker(backend)
    backend_worker.start()
    logger.info("Backend worker started")


# ---------------------------------------------------------------------------
# Queue rendering
# ---------------------------------------------------------------------------
def render_queue_html() -> str:
    if backend_worker is None:
        return "<p><em>Backend not ready</em></p>"

    items = backend_worker.get_all_items()
    if not items:
        return "<p style='color:#888;font-size:0.9em;'>No items in queue</p>"

    question_html = ""
    answer_html = ""

    for item in sorted(items, key=lambda x: x.timestamp):
        color, label = STATUS_COLORS.get(item.status, ("#999", "Unknown"))
        dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin-right:6px;"></span>'
        short_q = item.question[:50] + ("..." if len(item.question) > 50 else "")

        question_html += (
            f'<div style="padding:4px 0;font-size:0.85em;">'
            f'{dot}<span style="color:#555;">[{label}]</span> {short_q}'
            f"</div>"
        )

        if item.status in ("ready", "delivered"):
            a_color = "#5cb85c" if item.status == "delivered" else "#999"
            a_label = "Delivered" if item.status == "delivered" else "Pending"
            a_dot = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{a_color};margin-right:6px;"></span>'
            answer_html += (
                f'<div style="padding:4px 0;font-size:0.85em;">'
                f'{a_dot}<span style="color:#555;">[{a_label}]</span> {short_q}'
                f"</div>"
            )

    html = (
        '<div style="margin-bottom:10px;">'
        '<h4 style="margin:0 0 6px 0;font-size:0.95em;">Question Queue</h4>'
        f'{question_html or "<p style=color:#888;font-size:0.85em>Empty</p>"}'
        "</div>"
        '<hr style="border:none;border-top:1px solid #ddd;margin:8px 0;">'
        "<div>"
        '<h4 style="margin:0 0 6px 0;font-size:0.95em;">Answer Queue</h4>'
        f'{answer_html or "<p style=color:#888;font-size:0.85em>No answers yet</p>"}'
        "</div>"
    )
    return html


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
    if not user_text.strip():
        yield history, "", render_queue_html()
        return

    using_gemini_frontend = frontend_type == "Gemini API"

    if not using_gemini_frontend and frontend_model is None:
        history.append(
            {"role": "assistant", "content": "Models are still loading, please wait..."}
        )
        yield history, "", render_queue_html()
        return

    history.append({"role": "user", "content": user_text})
    yield history, "", render_queue_html()

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
            yield history, "", render_queue_html()
        elif streaming_enabled:
            history.append({"role": "assistant", "content": ""})
            for partial_text in generate_gemini_response_streaming(
                model_key=frontend_gemini_model,
                user_text=user_text,
                history=history[:-2],
                system_prompt=FRONTEND_SYSTEM_PROMPT,
                max_tokens=256,
                num_words_delay=num_words_delay,
            ):
                history[-1]["content"] = partial_text
                yield history, "", render_queue_html()
        else:
            answer = generate_gemini_response(
                model_key=frontend_gemini_model,
                user_text=user_text,
                history=history[:-1],
                system_prompt=FRONTEND_SYSTEM_PROMPT,
                max_tokens=256,
            )
            history.append({"role": "assistant", "content": answer})
            yield history, "", render_queue_html()
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
            yield history, "", render_queue_html()
        elif direct_answer:
            history.append({"role": "assistant", "content": direct_answer})
            yield history, "", render_queue_html()
        elif streaming_enabled:
            history.append({"role": "assistant", "content": ""})
            with frontend_lock:
                for partial_text in generate_response_streaming(
                    frontend_model, user_text, history[:-2], num_words_delay=num_words_delay
                ):
                    history[-1]["content"] = partial_text
                    yield history, "", render_queue_html()
        else:
            with frontend_lock:
                answer = generate_response(frontend_model, user_text, history[:-1])
            history.append({"role": "assistant", "content": answer})
            yield history, "", render_queue_html()


def poll_backend(history: list[dict]):
    if backend_worker is None:
        return history, render_queue_html()

    results = backend_worker.get_results()
    if not results:
        return history, render_queue_html()

    for item in results:
        t0 = time.time()
        logger.info("poll_backend: picked up item %s (ready -> delivery start)", item.id)

        delivery = (
            f"Regarding your earlier question about **{item.context_summary}**:\n\n"
            f"{item.answer}"
        )

        insert_idx = backend_insert_positions.pop(item.id, None)
        if insert_idx is not None and insert_idx <= len(history):
            history.insert(insert_idx, {"role": "assistant", "content": delivery})
            # Shift all remaining insertion positions that come after this one
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
        logger.info(
            "poll_backend: item %s delivered (%.2fs total)", item.id, time.time() - t0
        )

    return history, render_queue_html()


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
        render_queue_html(),
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
    return history, render_queue_html()


def on_frontend_gemini_model_change(model_key: str, history: list[dict]):
    global frontend_gemini_model
    frontend_gemini_model = model_key
    logger.info("Frontend Gemini model switched to %s", model_key)
    history.append(
        {"role": "assistant", "content": f"Frontend Gemini model switched to {model_key}."}
    )
    return history, render_queue_html()


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
        render_queue_html(),
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
    return history, render_queue_html()


def on_backend_gemini_model_change(model_key: str, history: list[dict]):
    global backend_worker
    if backend_worker is not None and isinstance(backend_worker._backend, GeminiBackend):
        backend_worker._backend.load_model(model_key)
        logger.info("Backend Gemini model switched to %s", model_key)
        history.append(
            {"role": "assistant", "content": f"Backend Gemini model switched to {model_key}."}
        )
    return history, render_queue_html()


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


def on_audio_record(
    audio_data,
    history: list[dict],
    threshold_n: int,
    streaming_enabled: bool,
    num_words_delay: int,
):
    if audio_data is None:
        yield history, gr.update(value="Ready to record"), None, render_queue_html()
        return

    sr, data = audio_data
    yield history, gr.update(value="Transcribing..."), None, render_queue_html()

    transcript = asr_engine.transcribe(sr, data)
    logger.info("ASR transcript: %s", transcript)

    if not transcript.strip():
        yield history, gr.update(value="(no speech detected)"), None, render_queue_html()
        return

    for hist_update, _, queue_update in on_user_message(
        transcript, history, threshold_n, streaming_enabled, num_words_delay
    ):
        yield hist_update, gr.update(value=f"Sent: {transcript[:80]}"), None, queue_update


def on_clear():
    if backend_worker is not None:
        backend_worker.clear_items()
    return [], render_queue_html()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    load_thread = threading.Thread(
        target=load_models,
        args=(QWEN_DEFAULT_MODEL, QWEN_DEFAULT_MODEL),
        daemon=True,
    )
    load_thread.start()

    local_model_names = get_model_names()
    gemini_model_names = get_gemini_model_names()

    with gr.Blocks(title="Multi-Stream Conversation") as demo:
        gr.Markdown("# Multi-Stream Conversation\nFront-end triage + back-end deep processing")

        with gr.Row():
            # --- Left sidebar ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Frontend Settings")
                frontend_type_radio = gr.Radio(
                    choices=["Local Qwen", "Gemini API"],
                    value="Local Qwen",
                    label="Frontend Type",
                )
                with gr.Column(visible=True) as fe_local_group:
                    fe_local_dropdown = gr.Dropdown(
                        choices=local_model_names,
                        value=QWEN_DEFAULT_MODEL,
                        label="Frontend Model (Local)",
                    )
                with gr.Column(visible=False) as fe_gemini_group:
                    fe_gemini_dropdown = gr.Dropdown(
                        choices=gemini_model_names,
                        value=GEMINI_DEFAULT_MODEL,
                        label="Frontend Model (Gemini)",
                    )

                gr.Markdown("### Backend Settings")
                backend_type_radio = gr.Radio(
                    choices=["Local Qwen", "Gemini API"],
                    value="Local Qwen",
                    label="Backend Type",
                )
                with gr.Column(visible=True) as be_local_group:
                    be_local_dropdown = gr.Dropdown(
                        choices=local_model_names,
                        value=QWEN_DEFAULT_MODEL,
                        label="Backend Model (Local)",
                    )
                with gr.Column(visible=False) as be_gemini_group:
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

                gr.Markdown("### Queue Status")
                queue_html = gr.HTML(value=render_queue_html())

            # --- Main chat area ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="Chat",
                )
                input_mode_radio = gr.Radio(
                    choices=["Text", "Speech"],
                    value="Text",
                    label="Input Mode",
                    interactive=True,
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

        # --- Events ---
        msg_inputs = [
            text_input,
            chatbot,
            threshold_slider,
            streaming_toggle,
            words_delay_slider,
        ]
        msg_outputs = [chatbot, text_input, queue_html]
        send_btn.click(fn=on_user_message, inputs=msg_inputs, outputs=msg_outputs)
        text_input.submit(fn=on_user_message, inputs=msg_inputs, outputs=msg_outputs)

        # Input mode switching
        input_mode_radio.change(
            fn=on_input_mode_change,
            inputs=[input_mode_radio],
            outputs=[text_input_group, speech_input_group],
        )

        # Speech: when recording stops, transcribe and auto-send
        audio_input.stop_recording(
            fn=on_audio_record,
            inputs=[audio_input, chatbot, threshold_slider, streaming_toggle, words_delay_slider],
            outputs=[chatbot, speech_status, audio_input, queue_html],
        )

        timer = gr.Timer(value=2)
        timer.tick(fn=poll_backend, inputs=[chatbot], outputs=[chatbot, queue_html])

        # Frontend switching
        frontend_type_radio.change(
            fn=on_frontend_type_change,
            inputs=[frontend_type_radio, fe_local_dropdown, fe_gemini_dropdown, chatbot],
            outputs=[chatbot, queue_html, fe_local_group, fe_gemini_group],
        )
        fe_local_dropdown.change(
            fn=on_frontend_local_model_change,
            inputs=[fe_local_dropdown, chatbot],
            outputs=[chatbot, queue_html],
        )
        fe_gemini_dropdown.change(
            fn=on_frontend_gemini_model_change,
            inputs=[fe_gemini_dropdown, chatbot],
            outputs=[chatbot, queue_html],
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
            outputs=[chatbot, queue_html, be_local_group, be_gemini_group],
        )
        be_local_dropdown.change(
            fn=on_backend_local_model_change,
            inputs=[be_local_dropdown, chatbot],
            outputs=[chatbot, queue_html],
        )
        be_gemini_dropdown.change(
            fn=on_backend_gemini_model_change,
            inputs=[be_gemini_dropdown, chatbot],
            outputs=[chatbot, queue_html],
        )

        clear_btn.click(fn=on_clear, outputs=[chatbot, queue_html])

    demo.launch(
        server_name="0.0.0.0", server_port=7863, share=True, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
