## Architecture

- **Frontend model** — triages each user message by estimating response complexity. Simple questions are answered directly (optionally streamed). Complex questions (exceeding the word threshold) are delegated to the backend queue. Defaults to Gemini API.
- **Backend model** — processes queued questions in a background thread. When an answer is ready, it is inserted into the conversation at the correct position (right after the deferral message). Defaults to Gemini API.
- **Frontend/Backend types** — each can be independently set to "Local Qwen" (GPU-based transformer) or "Gemini API".
- **ASR (speech input)** — the UI supports text or speech input modes. In speech mode, the user records audio via the browser microphone; on stop, the audio is transcribed and sent as a chat message. ASR backend is configurable:
  - **Whisper (Local)** — runs OpenAI Whisper (large-v3-turbo) on GPU. Supports both `.pt` checkpoints and HuggingFace models. Audio is preprocessed (int→float, stereo→mono, resampled to 16kHz).
  - **Gemini ASR** — sends WAV audio to the Gemini API for transcription. No local GPU required for ASR.
- **TTS (speech output)** — the UI supports text or speech output modes. When output mode is "Speech", assistant responses are enqueued for TTS synthesis in a background thread. Completed audio auto-plays via the browser.
  - **Gemini TTS** — uses `gemini-3.1-flash-tts-preview` to synthesize speech from text. Returns PCM audio at 24kHz, saved as WAV files. Supports multiple voice presets (Kore, Zephyr, Puck, Charon, Fenrir, Leda, Orus, Aoede).
  - **TTSQueueWorker** — background thread that processes TTS synthesis requests sequentially. Each item tracks status (queued → processing → ready → delivered). Completed audio is picked up by the polling timer and delivered to the audio player.

## Flow Diagram

```
                          User Input
                          (Text or Speech)
                               |
                  +------------+------------+
                  |                         |
              [Text mode]            [Speech mode]
                  |                         |
                  |                    +----------+
                  |                    | Record   |
                  |                    | audio    |
                  |                    +----------+
                  |                         |
                  |                    +----------+
                  |                    | ASR      |
                  |                    | Backend  |
                  |                    | (Whisper |
                  |                    |  or      |
                  |                    |  Gemini) |
                  |                    +----------+
                  |                         |
                  +------------+------------+
                               |
                          transcript / text
                               |
                               v
                     +-------------------+
                     |  Frontend Model   |
                     |  (Triage)         |
                     |                   |
                     |  ESTIMATED_WORDS  |
                     +-------------------+
                               |
                  +------------+------------+
                  |                         |
          words < threshold          words >= threshold
                  |                         |
                  v                         v
        +----------------+        +------------------+
        | Frontend Model |        | "I'll get back   |
        | (Direct Reply) |        |  to you..."      |
        |                |        +------------------+
        | Stream or      |                |
        | full response  |                v
        +----------------+        +------------------+
                  |               | Backend Queue    |
                  v               |  (FIFO)          |
             +----------+        +------------------+
             |          |                |
             |          |                v
             |          |        +------------------+
             |          |        | Backend Model    |
             |          |        | (Deep Response)  |
             |          |        +------------------+
             |          |                |
             |          |                v
             |          |        +------------------+
             |          +<-------| Insert answer at |
             |                   | original position|
             |                   | in history       |
             |                   +------------------+
             |                           |
             v                           v
     +-------+---------------------------+-------+
     |            Output Mode?                    |
     +--------------------+-----------------------+
                  |                         |
              [Text mode]            [Speech mode]
                  |                         |
                  v                         v
             +----------+        +------------------+
             |  Chat UI |        | TTS Queue        |
             +----------+        | (FIFO)           |
                                 +------------------+
                                         |
                                         v
                                 +------------------+
                                 | TTS Backend      |
                                 | (Gemini TTS)     |
                                 +------------------+
                                         |
                                         v
                                 +------------------+
                                 | Audio Player     |
                                 | (autoplay)       |
                                 +------------------+
```

## Sequence Diagram

```
User     ASR Backend   Frontend      Backend Queue   Backend Model   TTS Queue   TTS Backend   Chat UI
 |           |            |               |               |              |            |           |
 |-- text ---|----------->|               |               |              |            |           |
 |           |            |               |               |              |            |           |
 |-- audio ->|            |               |               |              |            |           |
 |           |-- xscribe->|               |               |              |            |           |
 |           |            |               |               |              |            |           |
 |           |            |-- triage ---->|               |              |            |           |
 |           |            |               |               |              |            |           |
 |  [simple] |            |               |               |              |            |           |
 |           |            |-- stream/reply|---------------|--------------|------------|---------->|
 |           |            |               |               |              |            |           |
 |           |            |  [Speech output mode]         |              |            |           |
 |           |            |-- enqueue ----|---------------|------------->|            |           |
 |           |            |               |               |              |-- synth -->|           |
 |           |            |               |               |              |            |-- WAV --->|
 |           |            |               |               |              |            |  autoplay |
 |           |            |               |               |              |            |           |
 |  [complex]|            |               |               |              |            |           |
 |           |            |-- "I'll get back"-------------|--------------|------------|---------->|
 |           |            |-- submit ---->|               |              |            |           |
 |           |            |               |-- dequeue --->|              |            |           |
 |           |            |               |               |-- generate   |            |           |
 |           |            |               |<-- answer ----|              |            |           |
 |           |            |               |-- insert at --+--------------|------------|---------->|
 |           |            |               |   original pos|              |            |           |
 |           |            |               |               |              |            |           |
 |           |            |  [Speech output mode]         |              |            |           |
 |           |            |               |-- enqueue ----|------------->|            |           |
 |           |            |               |               |              |-- synth -->|           |
 |           |            |               |               |              |            |-- WAV --->|
```

## Queue UI

The queue system is exposed via two buttons in the sidebar:

- **Text Queue (N)** — opens a floating panel showing:
  - **Question Queue**: all submitted questions with status dots (gray=queued, orange=processing, green=ready/delivered)
  - **Answer Queue**: completed backend answers with delivery status
- **Speech Queue (N)** — opens a floating panel showing:
  - **TTS Synthesis Queue**: all enqueued TTS items with status dots

Each panel is a fixed-position overlay (centered, max 80vh, scrollable body, pinned header with close button). Panels are native Gradio components (`gr.Column` with `elem_id` + CSS positioning) to avoid Gradio's HTML sanitization of `onclick`/`<script>`.

The 2-second polling timer updates button badges and panel content without affecting panel open/close state.

## Project Structure

```
app.py              # Gradio UI and callbacks
backend/            # Backend package
  __init__.py       # Re-exports all backend classes
  base.py           # LLMBackend, ASRBackend ABCs, QueueItem, BackendWorker
  llm_backend.py    # LocalBackend (Qwen), GeminiBackend (Gemini API)
  asr_backend.py    # WhisperASRBackend (local), GeminiASRBackend (API)
  tts_backend.py    # TTSBackend ABC, GeminiTTSBackend, TTSQueueItem, TTSQueueWorker
models.py           # Local Qwen model loading and generation utilities
gemini_client.py    # Gemini API client (triage, generate, streaming)
tests/              # Backend integration tests
env/
  public/           # Public environment
```
