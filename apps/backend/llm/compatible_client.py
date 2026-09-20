"""Chat Completions adapter for Qwen, DeepSeek, and explicit compatible endpoints."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from llm.base import BaseLLMClient, StructuredResponseT
from llm.providers import (
    ProviderConfigurationError,
    UnsupportedProviderCapability,
    provider_connection,
)


class CompatibleLLMClient(BaseLLMClient):
    """Retain local schema validation; never pretend an unsupported search succeeded."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        key, url = provider_connection(provider, api_key=api_key, base_url=base_url)
        if not model.strip():
            raise ProviderConfigurationError("Set LLM_MODEL for the selected provider.")
        try:
            self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
            if self.max_tokens < 1 or not 0 < timeout <= 300:
                raise ValueError
        except ValueError as exc:
            raise ProviderConfigurationError(
                "Invalid LLM_MAX_TOKENS or LLM_TIMEOUT_SECONDS."
            ) from exc
        self.provider = provider
        self.model = model
        # Explicit base_url prevents OPENAI_BASE_URL from rerouting vendor credentials.
        self.client = AsyncOpenAI(
            api_key=key, base_url=url, timeout=timeout, max_retries=1
        )
        # Low latency, bounded costs, and no reasoning-history replay dependency.
        self.extra_body: dict[str, Any] = (
            {"enable_thinking": False}
            if provider == "qwen"
            else {"thinking": {"type": "disabled"}}
            if provider == "deepseek"
            else {}
        )

    def _request(
        self, *, prompt: str, system_instruction: str | None, use_search: bool = False
    ) -> dict[str, Any]:
        extra = dict(self.extra_body)
        if use_search:
            if (
                self.provider != "qwen"
                or os.getenv("QWEN_ENABLE_SEARCH", "false").lower() != "true"
            ):
                raise UnsupportedProviderCapability(
                    "Verified web search is not configured for this provider. "
                    "Qwen native search requires QWEN_ENABLE_SEARCH=true."
                )
            extra["enable_search"] = True
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            extra_body=extra,
        )

    @staticmethod
    def _text(response: Any) -> str:
        if not response.choices:
            raise ValueError("Provider returned no completion choices.")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(
                f"Provider completion did not finish: {choice.finish_reason}."
            )
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provider returned an empty or refused completion.")
        return text

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        response = await self.client.chat.completions.create(
            **self._request(
                prompt=prompt,
                system_instruction=system_instruction,
                use_search=use_search,
            )
        )
        return self._text(response)

    async def generate_text_stream(
        self, *, prompt: str, system_instruction: str | None = None
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            **self._request(prompt=prompt, system_instruction=system_instruction),
            stream=True,
            stream_options={"include_usage": True},
        )
        finished = False
        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue  # Providers may emit an extra usage-only frame.
                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    if choice.finish_reason != "stop":
                        raise ValueError(
                            f"Provider stream did not finish: {choice.finish_reason}."
                        )
                    finished = True
                if choice.delta.content:
                    yield choice.delta.content  # Never expose reasoning_content.
        if not finished:
            raise ValueError("Provider stream closed before its terminal event.")

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        instruction = (system_instruction or "") + (
            "\nReturn ONLY a JSON object matching this JSON Schema:\n"
            + json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
        )
        request = self._request(
            prompt=prompt, system_instruction=instruction, use_search=use_search
        )
        request["response_format"] = {"type": "json_object"}
        # One bounded format repair, not an unbounded agent loop or safety downgrade.
        for attempt in range(2):
            response = await self.client.chat.completions.create(**request)
            text = self._text(response)
            try:
                return response_schema.model_validate_json(text)
            except ValidationError:
                if attempt:
                    raise ValueError(
                        "Provider output failed local JSON Schema validation."
                    ) from None
                request["messages"].append(
                    {
                        "role": "user",
                        "content": "The previous output did not match the JSON Schema. Return a corrected JSON object.",
                    }
                )
        raise AssertionError("Unreachable")
