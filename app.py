"""Multi-Stream Conversation — Gradio UI layer.

All business logic lives in ``orchestrator.py``.  This file handles only
Gradio layout, HTML rendering, and thin wrappers that map
``OrchestratorUpdate`` objects to Gradio outputs.

Usage:
    ./env/fb/run.sh python app.py

Launch client on laptop:
    ssh -L 7863:localhost:7863 bshi@<devgpu>
    http://localhost:7863
"""

import html
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import gradio as gr
from client import (
    DEFAULT_ASR,
    DEFAULT_TTS_VOICE,
    GEMINI_DEFAULT_MODEL,
    GEMINI_TTS_VOICES,
    get_asr_choices,
    get_gemini_model_names,
)
from models import get_model_names, QWEN_DEFAULT_MODEL
from orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_log_dir = (
    Path.home()
    / "logs"
    / "MultiStreamAgent"
    / "session"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_log_dir / "logs.txt")
_file_handler.setFormatter(logging.Formatter(_log_fmt))

logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[logging.StreamHandler(), _file_handler],
)
logger = logging.getLogger(__name__)
logger.info("Log file: %s", _log_dir / "logs.txt")

# ---------------------------------------------------------------------------
# Orchestrator (single instance)
# ---------------------------------------------------------------------------
orch = Orchestrator()

# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------
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
        f"</div>"
        f'<div class="queue-item-text">{escaped}</div>'
        f"</div>"
    )


def render_text_queue_html() -> str:
    items = orch.get_text_queue_items()
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
        f"</div>"
    )


def render_speech_queue_html() -> str:
    items = orch.get_speech_queue_items()
    if not items:
        return '<p class="queue-empty">No TTS items</p>'

    items_sorted = sorted(items, key=lambda x: x.timestamp)
    items_html = ""
    for item in items_sorted:
        items_html += _render_item_html(item.text, item.status, item.source.value)

    return (
        f'<div style="padding:0 4px;">'
        f'<div class="queue-section-title">TTS Synthesis Queue</div>'
        f"{items_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Gradio callback wrappers
# ---------------------------------------------------------------------------
def on_user_message(
    user_text: str,
    history: list[dict],
    threshold_n: int,
    streaming_enabled: bool,
    num_words_delay: int,
):
    for update in orch.handle_user_message(
        user_text, history, threshold_n, streaming_enabled, num_words_delay
    ):
        if update.warning:
            gr.Warning(update.warning)
        yield (
            update.history,
            "" if update.clear_input else gr.update(),
            gr.update(value=f"Text Queue ({update.text_queue_count})"),
            gr.update(value=f"Speech Queue ({update.speech_queue_count})"),
            render_text_queue_html(),
            render_speech_queue_html(),
            update.audio_path if update.audio_path else gr.update(),
        )


def poll_backend_and_tts(history: list[dict], audio_flag: str):
    update = orch.poll(history)
    is_playing = audio_flag == "playing"
    audio_out = gr.update()
    if update.audio_path and not is_playing:
        audio_out = update.audio_path
    # When a user-message handler is mid-flight, avoid overwriting the chat
    # component — its streaming yields are the source of truth.
    chat_out = gr.update() if update.skip_chat_update else update.history
    return (
        chat_out,
        gr.update(value=f"Text Queue ({update.text_queue_count})"),
        gr.update(value=f"Speech Queue ({update.speech_queue_count})"),
        render_text_queue_html(),
        render_speech_queue_html(),
        audio_out,
    )


def on_audio_record(
    audio_data,
    history: list[dict],
    threshold_n: int,
    streaming_enabled: bool,
    num_words_delay: int,
):
    for update in orch.handle_audio_input(
        audio_data, history, threshold_n, streaming_enabled, num_words_delay
    ):
        yield (
            update.history,
            (
                gr.update(value=update.status_message)
                if update.status_message
                else gr.update()
            ),
            None,
            gr.update(value=f"Text Queue ({update.text_queue_count})"),
            gr.update(value=f"Speech Queue ({update.speech_queue_count})"),
            render_text_queue_html(),
            render_speech_queue_html(),
            update.audio_path if update.audio_path else gr.update(),
        )


def on_frontend_type_change(
    fe_type: str,
    local_model_key: str,
    gemini_model_key: str,
    history: list[dict],
):
    msg = orch.set_frontend_type(fe_type, local_model_key, gemini_model_key)
    history.append({"role": "assistant", "content": msg})
    return (
        history,
        gr.update(visible=fe_type == "Local Qwen"),
        gr.update(visible=fe_type == "Gemini API"),
    )


def on_frontend_local_model_change(model_key: str, history: list[dict]):
    msg = orch.set_frontend_local_model(model_key)
    history.append({"role": "assistant", "content": msg})
    return history


def on_frontend_gemini_model_change(model_key: str, history: list[dict]):
    msg = orch.set_frontend_gemini_model(model_key)
    history.append({"role": "assistant", "content": msg})
    return history


def on_backend_type_change(
    be_type: str,
    local_model_key: str,
    gemini_model_key: str,
    history: list[dict],
):
    msg = orch.set_backend_type(be_type, local_model_key, gemini_model_key)
    history.append({"role": "assistant", "content": msg})
    return (
        history,
        gr.update(visible=be_type == "Local Qwen"),
        gr.update(visible=be_type == "Gemini API"),
    )


def on_backend_local_model_change(model_key: str, history: list[dict]):
    msg = orch.set_backend_local_model(model_key)
    history.append({"role": "assistant", "content": msg})
    return history


def on_backend_gemini_model_change(model_key: str, history: list[dict]):
    msg = orch.set_backend_gemini_model(model_key)
    if msg:
        history.append({"role": "assistant", "content": msg})
    return history


def on_asr_change(asr_choice: str, gemini_model_key: str):
    orch.set_asr(asr_choice, gemini_model_key)
    return gr.update(visible=asr_choice == "Gemini ASR")


def on_asr_gemini_model_change(model_key: str):
    orch.set_asr_gemini_model(model_key)


def on_input_mode_change(mode: str):
    orch.set_input_mode(mode)
    return (
        gr.update(visible=mode == "Text"),
        gr.update(visible=mode == "Speech"),
    )


def on_output_mode_change(mode: str):
    orch.set_output_mode(mode)


def on_tts_voice_change(voice: str):
    orch.set_tts_voice(voice)


def on_clear():
    update = orch.clear()
    return (
        update.history,
        gr.update(value=f"Text Queue ({update.text_queue_count})"),
        gr.update(value=f"Speech Queue ({update.speech_queue_count})"),
        render_text_queue_html(),
        render_speech_queue_html(),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    load_thread = threading.Thread(target=orch.start, daemon=True)
    load_thread.start()

    local_model_names = get_model_names()
    gemini_model_names = get_gemini_model_names()

    audio_tracking_js = """
    () => {
        function setFlag(val) {
            const flag = document.querySelector('#audio-playing-flag textarea');
            if (flag && flag.value !== val) {
                flag.value = val;
                flag.dispatchEvent(new Event('input', {bubbles: true}));
            }
        }
        function hookAudio(audio) {
            if (!audio || audio._tts_hooked) return;
            audio._tts_hooked = true;
            audio.addEventListener('play', () => setFlag('playing'));
            audio.addEventListener('pause', () => setFlag('idle'));
            audio.addEventListener('ended', () => setFlag('idle'));
        }
        // Re-hook whenever Gradio replaces the <audio> element
        const observer = new MutationObserver(() => {
            const container = document.getElementById('tts-audio-output');
            if (!container) return;
            const audio = container.querySelector('audio');
            hookAudio(audio);
        });
        const waitForContainer = setInterval(() => {
            const container = document.getElementById('tts-audio-output');
            if (!container) return;
            clearInterval(waitForContainer);
            observer.observe(container, {childList: true, subtree: true});
            hookAudio(container.querySelector('audio'));
        }, 300);
    }
    """

    with gr.Blocks(
        title="Multi-Stream Conversation", css=QUEUE_PANEL_CSS, js=audio_tracking_js
    ) as demo:
        gr.Markdown(
            "# Multi-Stream Conversation\nFront-end triage + back-end deep processing"
        )

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
                    elem_id="tts-audio-output",
                )
                audio_playing_flag = gr.Textbox(
                    value="idle",
                    visible=False,
                    elem_id="audio-playing-flag",
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

        with gr.Column(
            visible=False, elem_id="speech-queue-panel"
        ) as speech_queue_panel:
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
            chatbot,
            text_input,
            text_queue_btn,
            speech_queue_btn,
            text_queue_content,
            speech_queue_content,
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
            inputs=[
                audio_input,
                chatbot,
                threshold_slider,
                streaming_toggle,
                words_delay_slider,
            ],
            outputs=[
                chatbot,
                speech_status,
                audio_input,
                text_queue_btn,
                speech_queue_btn,
                text_queue_content,
                speech_queue_content,
                audio_output,
            ],
        )

        timer = gr.Timer(value=2)
        timer.tick(
            fn=poll_backend_and_tts,
            inputs=[chatbot, audio_playing_flag],
            outputs=[
                chatbot,
                text_queue_btn,
                speech_queue_btn,
                text_queue_content,
                speech_queue_content,
                audio_output,
            ],
        )

        # Frontend switching
        frontend_type_radio.change(
            fn=on_frontend_type_change,
            inputs=[
                frontend_type_radio,
                fe_local_dropdown,
                fe_gemini_dropdown,
                chatbot,
            ],
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
            outputs=[
                chatbot,
                text_queue_btn,
                speech_queue_btn,
                text_queue_content,
                speech_queue_content,
            ],
        )

    demo.launch(
        server_name="0.0.0.0", server_port=7863, share=True, theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
