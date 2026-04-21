"""Model loading and generation utilities for multi-stream conversation."""

import gc
import logging
import os
import re
import threading
from collections.abc import Generator
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

QWEN_MODELS = {
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.5-35B-A3B": "Qwen/Qwen3.5-35B-A3B",
    "Qwen3.5-122B-A10B": "Qwen/Qwen3.5-122B-A10B",
}

QWEN_DEFAULT_MODEL = "Qwen3.5-2B"

CHECKPOINTS_DIR = Path(os.path.expanduser("~/si_mango/tree/users/bshi/checkpoints"))

# TRIAGE_SYSTEM_PROMPT = """\
# You are a message triage assistant. For each user message:
# 1. Estimate how many words a thorough answer would need.
# 2. Write a one-line summary of what the user is asking about.
# 3. If the answer would be short (under {threshold} words), provide the answer directly.

# You MUST respond in this exact format (no extra text before or after):
# ESTIMATED_WORDS: <number>
# SUMMARY: <one line summary of the question>
# ANSWER: <your answer if simple, or DEFERRED if complex>"""

TRIAGE_SYSTEM_PROMPT = """
You are a message triage assistant. Estimate how many words a thorough answer would need.
Your response MUST be in this format.
NOTE: you should estimate the number of words only and don't output any other text.
ESTIMATED_WORDS: <number>
"""

BACKEND_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide thorough, detailed, and well-structured answers. "
    "Take your time to give a comprehensive response."
)

FRONTEND_SYSTEM_PROMPT = (
    "You are a helpful chat assistant. Keep your responses concise and conversational."
)


def get_model_names() -> list[str]:
    return list(QWEN_MODELS.keys())


def _resolve_model_path(model_key: str) -> str:
    hf_id = QWEN_MODELS[model_key]
    local_path = CHECKPOINTS_DIR / model_key
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path)
    return hf_id


def download_model(model_key: str) -> str:
    from huggingface_hub import snapshot_download

    hf_id = QWEN_MODELS[model_key]
    local_path = CHECKPOINTS_DIR / model_key
    if local_path.exists() and any(local_path.iterdir()):
        logger.info("Model %s already downloaded at %s", model_key, local_path)
        return str(local_path)

    local_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s (%s) to %s ...", model_key, hf_id, local_path)
    snapshot_download(repo_id=hf_id, local_dir=str(local_path))
    logger.info("Download complete: %s", local_path)
    return str(local_path)


def load_qwen(model_key: str = QWEN_DEFAULT_MODEL):
    """Load Qwen model and tokenizer. Returns (model, tokenizer, device)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = download_model(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Qwen model from %s ...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    return model, tokenizer, device


def unload_model(model_handle):
    """Free GPU memory for a model handle."""
    if model_handle is None:
        return
    model, tokenizer, device = model_handle
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate(model_handle, messages: list[dict], max_new_tokens: int = 256) -> str:
    model, tokenizer, device = model_handle
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_streaming(
    model_handle,
    messages: list[dict],
    max_new_tokens: int = 256,
    num_words_delay: int = 3,
) -> Generator[str, None, None]:
    """Generate text token-by-token, yielding accumulated text every num_words_delay words."""
    from transformers import TextIteratorStreamer

    model, tokenizer, device = model_handle
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    words_since_yield = 0
    for chunk in streamer:
        accumulated += chunk
        # Count new words in this chunk
        words_since_yield += len(chunk.split())
        if words_since_yield >= num_words_delay:
            words_since_yield = 0
            yield accumulated.strip()

    thread.join()
    # Final yield with complete text
    yield accumulated.strip()


def generate_response_streaming(
    model_handle,
    user_text: str,
    history: list[dict],
    num_words_delay: int = 3,
) -> Generator[str, None, None]:
    """Streaming version of generate_response — yields partial text."""
    messages = [{"role": "system", "content": FRONTEND_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    yield from _generate_streaming(
        model_handle, messages, max_new_tokens=256, num_words_delay=num_words_delay
    )


def estimate_complexity(
    model_handle,
    user_text: str,
    history: list[dict],
    threshold_n: int = 50,
    skip_answer: bool = False,
) -> tuple[bool, int, str | None, str]:
    """Estimate whether a question needs deep processing.

    Args:
        skip_answer: If True, don't generate the answer for simple questions
            (caller will handle it, e.g. via streaming).

    Returns:
        (is_complex, estimated_words, direct_answer_or_None, context_summary)
    """
    messages = [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"The user said: {user_text}"})

    raw = _generate(model_handle, messages, max_new_tokens=64)
    logger.info("triage raw output: %s", raw)

    words_match = re.search(r"ESTIMATED_WORDS:\s*(\d+)", raw)
    if not words_match:
        logger.warning("Triage parsing failed, falling back to direct answer")
        if skip_answer:
            return False, 0, None, user_text[:80]
        direct = generate_response(model_handle, user_text, history)
        return False, 0, direct, user_text[:80]

    estimated_words = int(words_match.group(1))
    is_complex = estimated_words >= threshold_n
    context_summary = user_text[:80]

    if is_complex or skip_answer:
        return is_complex, estimated_words, None, context_summary

    answer = generate_response(model_handle, user_text, history)
    return False, estimated_words, answer, context_summary


def generate_response(
    model_handle,
    user_text: str,
    history: list[dict],
) -> str:
    """Generate a concise front-end response."""
    messages = [{"role": "system", "content": FRONTEND_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return _generate(model_handle, messages, max_new_tokens=256)


def generate_full_response(
    model_handle,
    user_text: str,
    history: list[dict],
) -> str:
    """Generate a thorough back-end response for complex questions."""
    messages = [{"role": "system", "content": BACKEND_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return _generate(model_handle, messages, max_new_tokens=1024)


def generate_delivery_summary(
    model_handle,
    context_summary: str,
    answer: str,
) -> str:
    """Generate a delivery message for a completed back-end answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a chat assistant delivering a previously researched answer. "
                "Start your message with a brief reference to the original question, "
                "then present the answer. Be conversational."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The user previously asked about: {context_summary}\n\n"
                f"The detailed answer is:\n{answer}\n\n"
                "Please deliver this to the user conversationally, starting with "
                "'On your previous request about...'."
            ),
        },
    ]
    return _generate(model_handle, messages, max_new_tokens=1024)
