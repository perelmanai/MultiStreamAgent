"""Gemini API client for multi-stream conversation."""

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
        raise RuntimeError(
            "GEMINI_API_KEY not set. Export it or use env/fb/run.sh."
        )
    return genai.Client(api_key=api_key)


def get_gemini_model_names() -> list[str]:
    return list(GEMINI_MODELS.keys())


def _extract_text(content) -> str:
    """Extract plain text from a Gradio message content field.

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

    # Gemini needs the conversation to end with a user turn
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
    max_tokens: int = 256,
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
