"""Multi-Stream Conversation — Gradio chat with front-end triage and back-end deep processing.

Usage:
    ./run.sh python MultiStreamAgent/app.py

Launch client on laptop:
    ssh -L 7863:localhost:7863 bshi@<devgpu>
    http://localhost:7863
"""

import logging
import os
import sys
import threading
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from multi_stream_conversation.backend import BackendWorker, LocalBackend
from multi_stream_conversation.models import (
    estimate_complexity,
    generate_delivery_summary,
    generate_response,
    generate_response_streaming,
    get_model_names,
    load_qwen,
    QWEN_DEFAULT_MODEL,
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

    # Split into questions (queued/processing/ready) and answers (ready/delivered)
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
# Callbacks
# ---------------------------------------------------------------------------
def on_user_message(
    user_text: str,
    history: list[dict],
    threshold_n: int,
    streaming_enabled: bool,
    num_words_delay: int,
):
    """Handle a user message: triage and either answer directly or queue.

    Uses yield to show the user message immediately before processing.
    When streaming is enabled, yields partial assistant responses as they generate.
    """
    if not user_text.strip():
        yield history, "", render_queue_html()
        return

    if frontend_model is None:
        history.append(
            {"role": "assistant", "content": "Models are still loading, please wait..."}
        )
        yield history, "", render_queue_html()
        return

    # Show user message immediately
    history.append({"role": "user", "content": user_text})
    yield history, "", render_queue_html()

    # Triage — when streaming, skip generating the answer in triage
    # so we can stream it token-by-token below
    with frontend_lock:
        is_complex, est_words, direct_answer, summary = estimate_complexity(
            frontend_model,
            user_text,
            history[:-1],
            threshold_n,
            skip_answer=streaming_enabled,
        )

    if is_complex:
        # Queue for backend
        backend_worker.submit(user_text, summary, history[:-1])
        reply = (
            f"That's a detailed question — I'll work on a thorough answer and get back to you. "
            f"(Estimated ~{est_words} words needed)"
        )
        history.append({"role": "assistant", "content": reply})
        yield history, "", render_queue_html()
    elif direct_answer:
        # Triage already generated the full answer (non-streaming path)
        history.append({"role": "assistant", "content": direct_answer})
        yield history, "", render_queue_html()
    elif streaming_enabled:
        # Stream partial responses
        history.append({"role": "assistant", "content": ""})
        with frontend_lock:
            for partial_text in generate_response_streaming(
                frontend_model, user_text, history[:-2], num_words_delay=num_words_delay
            ):
                history[-1]["content"] = partial_text
                logger.info(
                    "stream chunk (%d chars): ...%s",
                    len(partial_text),
                    partial_text[-80:],
                )
                yield history, "", render_queue_html()
    else:
        with frontend_lock:
            answer = generate_response(frontend_model, user_text, history[:-1])
        history.append({"role": "assistant", "content": answer})
        yield history, "", render_queue_html()


def poll_backend(history: list[dict]):
    """Timer callback: check for completed backend results and deliver them."""
    if backend_worker is None:
        return history, render_queue_html()

    results = backend_worker.get_results()
    if not results:
        # Still update queue HTML for status changes (queued -> processing)
        return history, render_queue_html()

    for item in results:
        t0 = time.time()
        logger.info("poll_backend: picked up item %s (ready → delivery start)", item.id)

        # Deliver the backend answer directly with a brief intro prefix.
        # (Previously used generate_delivery_summary via the frontend LLM,
        # but that added ~60s of latency on larger models.)
        delivery = (
            f"Regarding your earlier question about **{item.context_summary}**:\n\n"
            f"{item.answer}"
        )

        # Insert a user-role separator so Gradio renders each answer as a
        # separate bubble (consecutive assistant messages get merged in Gradio 6.x)
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


def on_frontend_model_change(model_key: str, history: list[dict]):
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


def on_backend_model_change(model_key: str, history: list[dict]):
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


def on_clear():
    if backend_worker is not None:
        backend_worker.clear_items()
    return [], render_queue_html()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    # Start model loading in background
    load_thread = threading.Thread(
        target=load_models,
        args=(QWEN_DEFAULT_MODEL, QWEN_DEFAULT_MODEL),
        daemon=True,
    )
    load_thread.start()

    model_names = get_model_names()

    with gr.Blocks(title="Multi-Stream Conversation") as demo:
        gr.Markdown(
            "# Multi-Stream Conversation\nFront-end triage + back-end deep processing"
        )

        with gr.Row():
            # --- Left sidebar ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Settings")
                frontend_dropdown = gr.Dropdown(
                    choices=model_names,
                    value=QWEN_DEFAULT_MODEL,
                    label="Frontend Model",
                )
                backend_dropdown = gr.Dropdown(
                    choices=model_names,
                    value=QWEN_DEFAULT_MODEL,
                    label="Backend Model",
                )
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
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

        # --- State ---
        # chatbot component IS the history (type="messages" uses list[dict] format)

        # --- Events ---
        msg_inputs = [
            text_input,
            chatbot,
            threshold_slider,
            streaming_toggle,
            words_delay_slider,
        ]
        msg_outputs = [chatbot, text_input, queue_html]
        send_btn.click(
            fn=on_user_message,
            inputs=msg_inputs,
            outputs=msg_outputs,
        )
        text_input.submit(
            fn=on_user_message,
            inputs=msg_inputs,
            outputs=msg_outputs,
        )

        # Timer to poll backend results
        timer = gr.Timer(value=2)
        timer.tick(
            fn=poll_backend,
            inputs=[chatbot],
            outputs=[chatbot, queue_html],
        )

        # Model switching
        frontend_dropdown.change(
            fn=on_frontend_model_change,
            inputs=[frontend_dropdown, chatbot],
            outputs=[chatbot, queue_html],
        )
        backend_dropdown.change(
            fn=on_backend_model_change,
            inputs=[backend_dropdown, chatbot],
            outputs=[chatbot, queue_html],
        )

        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, queue_html],
        )

    demo.launch(
        server_name="0.0.0.0", server_port=7863, share=True, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
