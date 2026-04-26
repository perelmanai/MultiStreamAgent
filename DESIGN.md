## Architecture

- **Frontend model** — triages each user message by estimating response complexity. Simple questions are answered directly (optionally streamed). Complex questions (exceeding the word threshold) are delegated to the backend queue.
- **Backend model** — processes queued questions in a background thread. When an answer is ready, it is inserted into the conversation at the correct position (right after the deferral message).
- **Frontend/Backend types** — each can be independently set to "Local Qwen" (GPU-based transformer) or "Gemini API".
- **ASR (speech input)** — the UI supports text or speech input modes. In speech mode, the user records audio via the browser microphone; on stop, the audio is transcribed and sent as a chat message. ASR backend is configurable:
  - **Whisper (Local)** — runs OpenAI Whisper (large-v3-turbo) on GPU. Supports both `.pt` checkpoints and HuggingFace models. Audio is preprocessed (int→float, stereo→mono, resampled to 16kHz).
  - **Gemini ASR** — sends WAV audio to the Gemini API for transcription. No local GPU required for ASR.

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
             |  Chat UI |                |
             +----------+                v
                  ^               +------------------+
                  |               | Backend Model    |
                  |               | (Deep Response)  |
                  |               +------------------+
                  |                       |
                  |                       v
                  |               +------------------+
                  +<--------------| Insert answer at |
                                  | original position|
                                  | in history       |
                                  +------------------+
```

## Sequence Diagram

```
User        ASR Backend      Frontend         Backend Queue       Backend Model       Chat UI
 |              |                |                  |                   |                 |
 |-- text ------|--------------->|                  |                   |                 |
 |              |                |                  |                   |                 |
 |-- audio ---->|                |                  |                   |                 |
 |              |-- transcribe ->|                  |                   |                 |
 |              |                |                  |                   |                 |
 |              |                |-- triage ------->|                   |                 |
 |              |                |                  |                   |                 |
 |   [simple]   |                |                  |                   |                 |
 |              |                |-- stream/reply --|-------------------|---------------->|
 |              |                |                  |                   |                 |
 |   [complex]  |                |                  |                   |                 |
 |              |                |-- "I'll get back"|-------------------|---------------->|
 |              |                |-- submit ------->|                   |                 |
 |              |                |                  |-- dequeue ------->|                 |
 |              |                |                  |                   |-- generate -->  |
 |              |                |                  |                   |                 |
 |-- message -->|--------------->|                  |                   |                 |
 |              |                |-- stream/reply --|-------------------|---------------->|
 |              |                |                  |                   |                 |
 |              |                |                  |<-- answer --------|                 |
 |              |                |                  |-- insert at ------+---------------->|
 |              |                |                  |   original pos    |                 |
```

## Project Structure

```
app.py              # Gradio UI and callbacks
backend/            # Backend package
  __init__.py       # Re-exports all backend classes
  base.py           # LLMBackend, ASRBackend ABCs, QueueItem, BackendWorker
  llm_backend.py    # LocalBackend (Qwen), GeminiBackend (Gemini API)
  asr_backend.py    # WhisperASRBackend (local), GeminiASRBackend (API)
models.py           # Local Qwen model loading and generation utilities
gemini_client.py    # Gemini API client (triage, generate, streaming)
tests/              # Backend integration tests
env/
  public/           # Public environment
```
