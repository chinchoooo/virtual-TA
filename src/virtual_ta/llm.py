"""Minimal client for OpenAI-compatible HTTP APIs (OpenAI, AI Pipe, vLLM, Ollama, ...)."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .settings import Settings

log = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(self, settings: Settings, max_retries: int = 3) -> None:
        self.base_url = settings.api_base_url
        self.timeout = settings.request_timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {settings.require_api_key()}",
                "Content-Type": "application/json",
            }
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path}"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise LLMError(f"Request to {url} failed: {e}") from e
            else:
                if resp.ok:
                    return resp.json()
                if resp.status_code not in _RETRY_STATUS or attempt == self.max_retries:
                    raise LLMError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
            delay = 2**attempt
            log.warning("Retrying %s in %ss (attempt %d)", path, delay, attempt + 1)
            time.sleep(delay)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        data = self._post("embeddings", {"model": model, "input": texts})["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]

    def chat(self, messages: list[dict[str, Any]], model: str, **params: Any) -> str:
        data = self._post("chat/completions", {"model": model, "messages": messages, **params})
        return (data["choices"][0]["message"]["content"] or "").strip()


def image_part(image_b64: str) -> dict[str, Any]:
    """Build a chat ``image_url`` content part from raw base64 or a data URI."""
    url = image_b64 if image_b64.startswith("data:") else f"data:image/png;base64,{image_b64}"
    return {"type": "image_url", "image_url": {"url": url}}
