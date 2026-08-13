"""
Thin wrapper around an OpenAI-compatible chat completions endpoint.
Works with OpenAI directly, or any provider exposing the same API
shape (Groq, Together, local servers) by overriding OPENAI_BASE_URL.
"""
import os
import json
from openai import OpenAI

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
    return _client


def chat(system: str, user: str, temperature: float = 0.0, json_mode: bool = False) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    return response.choices[0].message.content


def chat_json(system: str, user: str, temperature: float = 0.0) -> dict:
    raw = chat(system, user, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # best-effort fallback if the model wraps JSON in prose/fences
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise
