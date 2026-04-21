# MultiStreamAgent

A multi-stream conversation system with front-end triage and back-end deep processing. Simple questions are answered immediately (with streaming); complex questions are queued for a backend model and the answer is inserted back into the conversation when ready.

Supports both **local Qwen models** (GPU) and **Gemini API** as frontend and backend independently.

## Setup

### Public

```bash
bash env/public/setup_conda.sh
export GEMINI_API_KEY="your-api-key"
```

## Running


```bash
./env/public/run.sh python app.py
```


Then SSH tunnel from your laptop and open in browser:

```bash
ssh -L 7863:localhost:7863 <user>@<devgpu>
# Open http://localhost:7863
```

