## Architecture

### Layers

```
app.py (Gradio UI)  ──>  orchestrator.py (business logic)  ──>  client/ (request interface)
                                                                     │
                                                            ┌────────┴────────┐
                                                      Gemini API        InProcess Server
                                                      (remote)          (wraps backend/)
                                                                             │
                                                                        backend/
                                                                    (local GPU models)
```

- **app.py** — Gradio UI layer. Thin wrappers that map `OrchestratorUpdate` objects to Gradio outputs. Owns HTML rendering, CSS, layout, and event wiring. No business logic.
- **orchestrator.py** — Owns all business logic, worker lifecycle, model state, and queue bookkeeping. Zero Gradio dependency. Yields `OrchestratorUpdate` dataclass objects.
- **client/** — Request interface. ABCs (`LLMClient`, `ASRClient`, `TTSClient`), Gemini API clients, local clients with in-process server wrappers, and queue workers (`LLMQueueWorker`, `TTSQueueWorker`).
- **backend/** — Local GPU model hosting only. `LocalLLMBackend` (Qwen) and `WhisperASRBackend`. No ABCs, no queues, no Gemini code. Will be hosted behind Thrift servers.

### Communication

```
Gemini:   GeminiLLMClient ────> Google API (remote)
          GeminiASRClient ────> Google API (remote)
          GeminiTTSClient ────> Google API (remote)

Local:    LocalLLMClient ────> InProcessLLMServer ────> LocalLLMBackend (GPU)
          LocalASRClient ────> InProcessASRServer ────> WhisperASRBackend (GPU)
                                      ^
                                      └── TODO: replace with Thrift server/client
```

### Components

- **Frontend model** — triages each user message by estimating response complexity. Simple questions are answered directly (optionally streamed). Complex questions are delegated to the backend queue. Defaults to Gemini API.
- **Backend model** — processes queued questions via `LLMQueueWorker` in a background thread. When an answer is ready, the user is notified and can choose to hear it.
- **Intention routing** — when backend answers are ready, `estimate_intention_gemini` classifies the user's next message as SELECT (retrieve a ready answer) or GENERATE (new question). This runs before triage.
- **ASR (speech input)** — text or speech input modes. In speech mode, audio is transcribed and sent as a chat message.
  - **Whisper (Local)** — runs OpenAI Whisper (large-v3-turbo) on GPU via `LocalASRClient` → `InProcessASRServer` → `WhisperASRBackend`.
  - **Gemini ASR** — sends WAV audio to the Gemini API via `GeminiASRClient`.
- **TTS (speech output)** — text or speech output modes. When output mode is "Speech", assistant responses are enqueued for TTS synthesis via `TTSQueueWorker`.
  - **Gemini TTS** — uses `gemini-3.1-flash-tts-preview` via `GeminiTTSClient`. Returns PCM audio at 24kHz. Supports voice presets (Kore, Zephyr, Puck, etc.).
  - **TTSQueueWorker** — thread pool that synthesizes TTS items concurrently. Immediate (frontend) items take priority over backend items.

### Notification flow

- **Text mode**: when a backend answer is ready, a text notification appears immediately in chat.
- **Speech mode**: the answer is enqueued for TTS first. The text notification only appears once the answer audio is synthesized. The notification itself is then also spoken. On SELECT, the pre-synthesized answer audio is delivered alongside the text.

### Audio playback blocking

Audio delivery is gated by browser playback state, not a duration timer. JavaScript hooks on the `<audio>` element's `play`, `pause`, and `ended` events write to a hidden flag. The poll function reads this flag and only delivers new audio when the previous clip has finished or been stopped.

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
                  |                    | ASR      |
                  |                    | Client   |
                  |                    +----------+
                  |                         |
                  +------------+------------+
                               |
                          transcript / text
                               |
                               v
                   +------------------------+
                   | Ready answers pending? |
                   +------------------------+
                      |                |
                     yes               no
                      |                |
                      v                v
              +--------------+  +--------------+
              | Intention    |  | Triage       |
              | (SELECT or   |  | (simple or   |
              |  GENERATE)   |  |  complex)    |
              +--------------+  +--------------+
                 |        |        |         |
              SELECT   GENERATE  simple   complex
                 |        |        |         |
                 v        v        v         v
           +---------+ +------+ +------+ +--------+
           | Deliver | | Fall | | Direct| | Queue  |
           | ready   | | thru | | reply | | to     |
           | answer  | | to   | |       | | backend|
           | + audio | | triage|        | +--------+
           +---------+ +------+ +------+      |
                                               v
                                        +-----------+
                                        | Backend   |
                                        | generates |
                                        +-----------+
                                               |
                                               v
                                  +------------------------+
                                  | Text mode: notify now  |
                                  | Speech mode: wait for  |
                                  |   answer TTS, then     |
                                  |   notify (text+speech) |
                                  +------------------------+
                                               |
                                               v
                                      User says "yes" → SELECT
```

## Project Structure

```
app.py                  # Gradio UI — thin adapter over Orchestrator
orchestrator.py         # Business logic, worker lifecycle, queue bookkeeping
models.py               # Local Qwen model loading and generation utilities

client/                 # Request interface (no GPU models)
  __init__.py           # Re-exports all public symbols
  base.py               # ABCs (LLMClient, ASRClient, TTSClient) + data types
  gemini_utils.py       # Gemini API helpers (generate, stream, triage, intention)
  llm_client.py         # GeminiLLMClient, LocalLLMClient, InProcessLLMServer, LLMQueueWorker
  asr_client.py         # GeminiASRClient, LocalASRClient, InProcessASRServer, registry
  tts_client.py         # GeminiTTSClient, TTSQueueWorker, TTS constants

backend/                # Local GPU model hosting
  __init__.py           # Exports LocalLLMBackend, WhisperASRBackend
  llm_backend.py        # LocalLLMBackend (Qwen)
  asr_backend.py        # WhisperASRBackend (Whisper)

tests/
  test_backend.py       # LLMQueueWorker + mock backend tests
  test_orchestrator.py  # Full orchestrator flow tests (Gemini-backed)

env/
  public/               # Public environment setup
  fb/                   # Internal environment setup
```
