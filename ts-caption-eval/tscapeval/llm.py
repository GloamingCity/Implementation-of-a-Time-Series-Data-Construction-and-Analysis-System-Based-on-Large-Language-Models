"""OpenAI-compatible LLM client.

Credentials are read from environment variables (standard OpenAI SDK names):
- OPENAI_API_KEY
- OPENAI_BASE_URL  (optional, defaults to real OpenAI)

No URLs or keys are hardcoded. Any compatible endpoint works: real OpenAI,
Azure, a local vLLM, or an instructor-provided reverse proxy.
"""

from __future__ import annotations

import json
import os
import time
import threading
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional, env may come from shell
    pass

from openai import OpenAI


_CLIENT: OpenAI | None = None
_api_lock = threading.Lock()


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    _CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format_json: bool = False,
    disable_thinking: bool = False,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Single chat-completion round-trip with retry. Returns the string content."""
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    base_url = os.environ.get("OPENAI_BASE_URL") or ""
    if response_format_json and "zhipuai" not in base_url and "bigmodel" not in base_url:
        kwargs["response_format"] = {"type": "json_object"}
    
    if disable_thinking and ("zhipuai" in base_url or "bigmodel" in base_url):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with _api_lock:
                resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == retries - 1:
                break
            time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def chat_json(model: str, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
    """Like `chat` but demand JSON back and parse it (lenient fallback)."""
    content = chat(model, messages, response_format_json=True, **kwargs)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise
