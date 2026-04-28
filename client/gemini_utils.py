"""Gemini API utilities for multi-stream conversation."""

import logging
import os
from collections.abc import Generator

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GEMINI_MODELS = {
    "Gemini-3-Flash": "gemini-3-flash-preview",
    "Gemini-3-Pro": "gemini-3-pro-preview",
}

GEMINI_DEFAULT_MODEL = "Gemini-3-Flash"


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Export it or use env/fb/run.sh.")
    return genai.Client(api_key=api_key)


def get_gemini_model_names() -> list[str]:
    return list(GEMINI_MODELS.keys())


def _extract_text(content) -> str:
    """Extract plain text from a message content field.

    Content can be a plain string or a list of dicts like
    [{"text": "...", "type": "text"}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _build_contents(
    history: list[dict],
    user_text: str,
) -> list[types.Content]:
    """Build Gemini contents from chat history.

    Gemini requires strictly alternating user/model turns.
    Consecutive same-role messages are merged, and empty messages are skipped.
    """
    contents: list[types.Content] = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        text = _extract_text(msg["content"])
        if not text.strip():
            continue
        if contents and contents[-1].role == role:
            existing_text = contents[-1].parts[0].text
            contents[-1] = types.Content(
                role=role,
                parts=[types.Part(text=existing_text + "\n\n" + text)],
            )
        else:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    if contents and contents[-1].role == "user":
        existing_text = contents[-1].parts[0].text
        contents[-1] = types.Content(
            role="user",
            parts=[types.Part(text=existing_text + "\n\n" + user_text)],
        )
    else:
        contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    return contents


def generate_gemini_response(
    model_key: str,
    user_text: str,
    history: list[dict],
    system_prompt: str,
    max_tokens: int = 1024,
) -> str:
    client = _get_client()
    model_name = GEMINI_MODELS[model_key]
    contents = _build_contents(history, user_text)

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.7,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def generate_gemini_response_streaming(
    model_key: str,
    user_text: str,
    history: list[dict],
    system_prompt: str,
    max_tokens: int = 2048,
    num_words_delay: int = 3,
) -> Generator[str, None, None]:
    client = _get_client()
    model_name = GEMINI_MODELS[model_key]
    contents = _build_contents(history, user_text)

    response = client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.7,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    accumulated = ""
    words_since_yield = 0
    for chunk in response:
        if chunk.text:
            accumulated += chunk.text
            words_since_yield += len(chunk.text.split())
            if words_since_yield >= num_words_delay:
                words_since_yield = 0
                yield accumulated

    yield accumulated


def estimate_complexity_gemini(
    model_key: str,
    user_text: str,
    triage_prompt: str,
) -> str:
    client = _get_client()
    model_name = GEMINI_MODELS[model_key]

    response = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part(text=user_text)])],
        config=types.GenerateContentConfig(
            system_instruction="Estimate the word count needed for a thorough answer to the user message. Reply with ONLY a single line: ESTIMATED_WORDS: <number>. Nothing else.",
            max_output_tokens=256,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def estimate_intention_gemini(
    model_key: str,
    user_text: str,
    history: list[dict],
    ready_items: list[dict],
) -> dict:
    """Decide whether the user wants to SELECT a ready answer or GENERATE a new one.

    Args:
        model_key: Gemini model key.
        user_text: The latest user message.
        history: Conversation history.
        ready_items: List of dicts with keys "index", "question", "summary"
                     representing backend answers waiting for delivery.

    Returns:
        {"action": "SELECT", "index": <int>} or {"action": "GENERATE"}
    """
    items_desc = "\n".join(
        f"  [{i['index']}] \"{i['question'][:120]}\" (topic: {i['summary']})"
        for i in ready_items
    )

    system_instruction = (
        "You are an intent classifier for a voice assistant. "
        "The user is chatting with an assistant that processes complex questions in the background. "
        "Some answers are now ready and waiting to be delivered.\n\n"
        "Ready answers:\n"
        f"{items_desc}\n\n"
        "Based on the user's latest message and conversation context, decide:\n"
        "- If the user is asking to hear, see, read, or retrieve one of the ready answers "
        "(e.g. 'yes', 'sure', 'tell me', 'what about that question', 'read it', referencing a topic), "
        "reply: SELECT <index>\n"
        "- If the user is asking a new question or continuing the conversation on a different topic, "
        "reply: GENERATE\n\n"
        "Reply with ONLY one line: either 'SELECT <index>' or 'GENERATE'. Nothing else."
    )

    recent_history = history[-6:] if len(history) > 6 else history
    contents = _build_contents(recent_history, user_text)

    client = _get_client()
    model_name = GEMINI_MODELS[model_key]

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=32,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    raw = response.text.strip()
    logger.info("estimate_intention raw: %s | ready_items: %s | user: %s", raw, items_desc, user_text)

    if raw.upper().startswith("SELECT"):
        parts = raw.split()
        if len(parts) >= 2:
            try:
                idx = int(parts[1])
                valid_indices = [i["index"] for i in ready_items]
                if idx in valid_indices:
                    return {"action": "SELECT", "index": idx}
            except ValueError:
                pass
        if ready_items:
            return {"action": "SELECT", "index": ready_items[0]["index"]}

    return {"action": "GENERATE"}
