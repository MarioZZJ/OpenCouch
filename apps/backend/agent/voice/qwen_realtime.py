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


def _simplify_schema_types(node: Any) -> Any:
    """Drop JSON-Schema union types that Qwen's realtime parser rejects.

    OpenAI accepts a parameter declared as ``{"type": ["string", "null"]}``.
    Qwen's ``session.update`` parser does not: a tool schema containing a
    nullable union makes the whole session.update fail with a generic
    ``InternalError: Parse RealtimeEvent error``, so the voice session never
    reaches ``session.updated``.

    Every affected parameter in the voice tool surface is optional and is not
    listed in ``required``, so collapsing ``["string", "null"]`` to ``"string"``
    keeps the same meaning: omitting the property is still valid, and the
    runtime already treats an absent value as None.
    """

    if isinstance(node, dict):
        simplified = {key: _simplify_schema_types(value) for key, value in node.items()}
        node_type = simplified.get("type")
        if isinstance(node_type, list):
            non_null = [entry for entry in node_type if entry != "null"]
            if len(non_null) == 1 and len(non_null) != len(node_type):
                simplified["type"] = non_null[0]
        return simplified
    if isinstance(node, list):
        return [_simplify_schema_types(value) for value in node]
    return node


def _split_env_list(name: str) -> list[str]:
    """Parse a comma-separated environment variable into stripped entries.

    Empty entries are dropped so a trailing comma does not shift a positional
    pairing (see voice_labels).
    """

    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


def voice_options() -> list[str]:
    """A small documented selection; extra approved voice IDs may be configured."""
    voices = [DEFAULT_VOICE, "Ethan", os.getenv("QWEN_REALTIME_VOICE", DEFAULT_VOICE)]
    voices.extend(_split_env_list("QWEN_REALTIME_EXTRA_VOICES"))
    return list(dict.fromkeys(voice.strip() for voice in voices if voice.strip()))


def voice_labels() -> dict[str, str]:
    """Optional display names for the configured voices.

    ``QWEN_REALTIME_VOICE_LABELS`` is positional against
    ``QWEN_REALTIME_EXTRA_VOICES``: the Nth label names the Nth extra voice.
    Positional pairing keeps the provider's voice IDs authoritative, so a
    renamed or removed voice can never be sent to Qwen by mistake - it simply
    falls back to showing its ID. Labels are display-only and are never sent
    as the ``voice`` parameter.
    """

    extra = _split_env_list("QWEN_REALTIME_EXTRA_VOICES")
    labels = _split_env_list("QWEN_REALTIME_VOICE_LABELS")
    mapping: dict[str, str] = {}
    for index, voice in enumerate(extra):
        if index < len(labels):
            mapping[voice] = labels[index]
    return mapping


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
        # Parameter schemas also have to avoid nullable unions - see
        # _simplify_schema_types.
        tools.append(
            {
                "type": "function",
                "function": {
                    key: (
                        _simplify_schema_types(tool[key])
                        if key == "parameters"
                        else tool[key]
                    )
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
        # WebRTC delivers Opus, but Qwen still needs the *decoded* input format
        # declared, and it only accepts 16 kHz PCM here. Without this the server
        # accepts the session and counts every inbound RTP packet (verified via
        # remote-inbound-rtp: packetsReceived == packetsSent, no loss) yet never
        # runs VAD, so the session reports connected and then stays silent:
        # no speech_started, no transcript, no reply. Declaring 16 kHz PCM makes
        # the same audio produce speech_started -> committed -> transcription ->
        # response. 24000 is the documented default output rate.
        "audio": {
            "input": {"format": {"type": "pcm", "sample_rate": 16000}},
            "output": {"format": {"type": "pcm", "sample_rate": 24000}},
        },
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
