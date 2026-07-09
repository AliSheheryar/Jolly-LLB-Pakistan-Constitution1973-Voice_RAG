"""LLM client abstraction.

`AnthropicLLM` is the production client. `StubLLM` is a deterministic offline
client so the pipeline can be exercised end-to-end without a network or API key
(useful for tests and the demo in pipeline.__main__).
"""
from __future__ import annotations

import os
import re
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class OpenRouterLLM:
    """Answer generation via OpenRouter's OpenAI-compatible API.

    Defaults to Llama 3.1 8B Instruct. Set OPENROUTER_API_KEY (get one at
    https://openrouter.ai/keys). Temperature is low: this is grounded citation
    over retrieved statute text, not creative writing."""

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct",
                 max_tokens: int = 600, temperature: float = 0.1,
                 api_key: str | None = None,
                 base_url: str = "https://openrouter.ai/api/v1"):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Get a key at "
                "https://openrouter.ai/keys and `export OPENROUTER_API_KEY=...`")
        self._client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_headers={  # optional OpenRouter attribution
                "HTTP-Referer": "https://github.com/legal-rag",
                "X-Title": "Legal RAG (Pakistani law)",
            },
        )
        return resp.choices[0].message.content or ""


class AnthropicLLM:
    def __init__(self, model: str = "claude-sonnet-5",
                 max_tokens: int = 600, api_key: str | None = None):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class StubLLM:
    """Offline stand-in. Grounds its reply in the first provided source and
    emits a correct [[CIT:...]] marker, so verification passes deterministically.
    Not for production — it does no reasoning."""

    _CIT = re.compile(r"\[\[ID:(?P<id>[^\]]+)\]\]")

    def complete(self, system: str, user: str) -> str:
        ids = self._CIT.findall(user)
        if not ids:
            return ("I don't have a specific provision on that in the sources "
                    "I was given.")
        first = ids[0]
        return (f"Based on the law provided, this is addressed under "
                f"[[CIT:{first}]]. I'm giving you the relevant provision; "
                f"for how it applies you should consult a lawyer.")
