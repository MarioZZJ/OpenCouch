"""Experimental Qwen3.5 Omni WebRTC signaling; DashScope keys stay server-side.

Only SDP travels through this broker. Media travels directly over WebRTC and the
existing application endpoints retain tool, safety, and persistence ownership.
Tickets are single-use, bounded, and local to OpenCouch's single API worker.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from agent.voice.policy import build_voice_instructions
from agent.voice.tools import build_voice_realtime_tools
from llm.providers import ProviderConfigurationError

SUPPORTED_MODELS = frozenset(
    {
        "qwen3.5-omni-flash-realtime",
        "qwen3.5-omni-plus-realtime",
    }
)
DEFAULT_MODEL = "qwen3.5-omni-flash-realtime"
DEFAULT_VOICE = "Tina"


def voice_options() -> list[str]:
    """A small documented selection; extra approved voice IDs may be configured."""
    voices = [DEFAULT_VOICE, "Ethan", os.getenv("QWEN_REALTIME_VOICE", DEFAULT_VOICE)]
    voices.extend(os.getenv("QWEN_REALTIME_EXTRA_VOICES", "").split(","))
    return list(dict.fromkeys(voice.strip() for voice in voices if voice.strip()))


def selected_model() -> str:
    model = os.getenv("QWEN_REALTIME_MODEL", DEFAULT_MODEL).strip()
    if model not in SUPPORTED_MODELS:
        raise ProviderConfigurationError(
            "This voice adapter supports qwen3.5-omni-flash-realtime and "
            "qwen3.5-omni-plus-realtime; older models have different tool capabilities."
        )
    return model


def build_session_config(
    *,
    thread_id: str,
    user_id: str | None,
    memory_mode: str,
    memory_context: str | None = None,
    assistant_voice: str | None = None,
) -> dict[str, Any]:
    if os.getenv("OPENCOUCH_ENABLE_EXPERIMENTAL_QWEN_VOICE", "").lower() != "true":
        raise ProviderConfigurationError(
            "Set OPENCOUCH_ENABLE_EXPERIMENTAL_QWEN_VOICE=true after reviewing "
            "docs/provider-adaptation.md. Live voice validation is required."
        )
    voice = assistant_voice or os.getenv("QWEN_REALTIME_VOICE", DEFAULT_VOICE)
    if voice not in voice_options():
        raise ProviderConfigurationError(
            "Unsupported Qwen Realtime voice; refresh voice settings."
        )
    vad = os.getenv("QWEN_REALTIME_VAD", "semantic_vad")
    if vad not in {"server_vad", "semantic_vad"}:
        raise ProviderConfigurationError(
            "Qwen WebRTC requires server_vad or semantic_vad."
        )
    tools = []
    for tool in build_voice_realtime_tools(memory_mode=memory_mode):
        # Qwen expects Chat Completions-style nested function definitions.
        tools.append(
            {
                "type": "function",
                "function": {
                    key: tool[key]
                    for key in ("name", "description", "parameters")
                    if key in tool
                },
            }
        )
    return {
        "model": selected_model(),
        "modalities": ["text", "audio"],
        "enable_input_audio_transcription": True,
        "voice": voice,
        "turn_detection": {"type": vad, "threshold": 0.5, "silence_duration_ms": 800},
        "instructions": build_voice_instructions(
            thread_id=thread_id,
            user_id=user_id,
            memory_mode=memory_mode,
            memory_context=memory_context,
        ),
        "tools": tools,
        # No OpenAI GA audio transcription/reasoning fields, tool_choice, or
        # parallel_tool_calls. Qwen native search cannot be combined with tools.
    }


def signaling_url() -> str:
    url = os.getenv("QWEN_REALTIME_URL", "").strip()
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    allowed = host.endswith(
        (".cn-beijing.maas.aliyuncs.com", ".ap-southeast-1.maas.aliyuncs.com")
    )
    if (
        parsed.scheme != "https"
        or not allowed
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/api/v1/webrtc/realtime"
        or "{" in url
        or "}" in url
    ):
        raise ProviderConfigurationError(
            "Set QWEN_REALTIME_URL to your workspace's HTTPS /api/v1/webrtc/realtime "
            "endpoint in Beijing or Singapore (without a model query parameter)."
        )
    return url


@dataclass(frozen=True)
class SignalingTicket:
    expires_at: float
    url: str
    model: str


class TicketStore:
    """Bounded one-time tickets, consumed atomically before a network await."""

    def __init__(self, *, ttl_seconds: float = 120, capacity: int = 128) -> None:
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self._tickets: dict[str, SignalingTicket] = {}

    def issue(self, *, url: str, model: str) -> str:
        now = time.monotonic()
        self._tickets = {
            key: value for key, value in self._tickets.items() if value.expires_at > now
        }
        if len(self._tickets) >= self.capacity:
            raise ValueError(
                "Too many pending voice connections. Retry after ticket expiry."
            )
        token = secrets.token_urlsafe(32)
        self._tickets[token] = SignalingTicket(now + self.ttl_seconds, url, model)
        return token

    def consume(self, token: str) -> SignalingTicket:
        ticket = self._tickets.pop(token, None)
        if ticket is None or ticket.expires_at <= time.monotonic():
            raise ValueError(
                "Voice connection ticket expired or was already used. Start a new session."
            )
        return ticket


TICKETS = TicketStore()


def create_signaling_ticket(session_config: dict[str, Any]) -> str:
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        raise ProviderConfigurationError("Set DASHSCOPE_API_KEY for Qwen Realtime.")
    return TICKETS.issue(url=signaling_url(), model=str(session_config["model"]))


async def exchange_sdp(ticket: SignalingTicket, offer: str) -> str:
    """Never forward upstream error bodies (they may contain credentials or SDP)."""
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ProviderConfigurationError("Set DASHSCOPE_API_KEY for Qwen Realtime.")
    if not offer.startswith("v=0") or "m=audio " not in offer:
        raise ValueError("A WebRTC audio SDP offer is required.")
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            response = await client.post(
                ticket.url,
                params={"model": ticket.model},
                content=offer.encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/sdp",
                },
            )
    except httpx.HTTPError:
        raise RuntimeError(
            "Qwen Realtime signaling could not connect to the configured region."
        ) from None
    if response.status_code != 200:
        raise RuntimeError(
            f"Qwen Realtime signaling failed (HTTP {response.status_code})."
        )
    answer = response.text
    if len(answer) > 131072 or not answer.startswith("v=0") or "m=audio " not in answer:
        raise RuntimeError("Qwen Realtime returned an invalid SDP answer.")
    return answer
