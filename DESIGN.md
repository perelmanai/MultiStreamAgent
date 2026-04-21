## Architecture

- **Frontend model** — triages each user message by estimating response complexity. Simple questions are answered directly (optionally streamed). Complex questions (exceeding the word threshold) are delegated to the backend queue.
- **Backend model** — processes queued questions in a background thread. When an answer is ready, it is inserted into the conversation at the correct position (right after the deferral message).
- **Frontend/Backend types** — each can be independently set to "Local Qwen" (GPU-based transformer) or "Gemini API".

## Flow Diagram

```
                          User Message
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
User        Frontend         Backend Queue       Backend Model       Chat UI
 |              |                  |                   |                 |
 |-- message -->|                  |                   |                 |
 |              |-- triage ------->|                   |                 |
 |              |                  |                   |                 |
 |   [simple]   |                  |                   |                 |
 |              |-- stream/reply --|-------------------|---------------->|
 |              |                  |                   |                 |
 |   [complex]  |                  |                   |                 |
 |              |-- "I'll get back"|-------------------|---------------->|
 |              |-- submit ------->|                   |                 |
 |              |                  |-- dequeue ------->|                 |
 |              |                  |                   |-- generate -->  |
 |              |                  |                   |                 |
 |-- message -->|                  |                   |                 |
 |              |-- stream/reply --|-------------------|---------------->|
 |              |                  |                   |                 |
 |              |                  |<-- answer --------|                 |
 |              |                  |-- insert at ------+---------------->|
 |              |                  |   original pos    |                 |
```

## Project Structure

```
app.py              # Gradio UI and callbacks
backend.py          # Backend abstraction, LocalBackend, GeminiBackend, BackendWorker
models.py           # Local Qwen model loading and generation utilities
gemini_client.py    # Gemini API client (triage, generate, streaming)
tests/              # Backend integration tests
env/
  public/           # Public environment
```
