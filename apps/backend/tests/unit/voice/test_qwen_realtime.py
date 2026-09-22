"""Qwen WebRTC contract, credential boundary and one-time signaling tests."""

from __future__ import annotations

import os

import httpx
import pytest
from agent.voice import qwen_realtime as qwen
from agent.voice import realtime
from api.routes.voice import router
from fastapi import FastAPI
from fastapi.testclient import TestClient
from llm.providers import ProviderConfigurationError

SDP = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
URL = "https://unitworkspace.cn-beijing.maas.aliyuncs.com/api/v1/webrtc/realtime"


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("OPENCOUCH_VOICE_PROVIDER", "qwen")
    monkeypatch.setenv("OPENCOUCH_ENABLE_EXPERIMENTAL_QWEN_VOICE", "true")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-server-only-key")
    monkeypatch.setenv("QWEN_REALTIME_URL", URL)
    monkeypatch.setattr(qwen, "TICKETS", qwen.TicketStore())


def test_session_retains_policy_memory_tools_without_openai_fields():
    config = realtime.build_realtime_session_config(
        thread_id="t",
        user_id="u",
        memory_mode="persistent",
        memory_context="User prefers brief responses.",
        assistant_voice="Tina",
    )
    assert config["model"] == "qwen3.5-omni-flash-realtime"
    assert config["voice"] == "Tina"
    assert "brief responses" in config["instructions"]
    assert {tool["function"]["name"] for tool in config["tools"]} >= {
        "show_saved_memory",
        "lookup_crisis_resources",
    }
    assert (
        not {"type", "reasoning", "tool_choice", "parallel_tool_calls"} & config.keys()
    )
    assert "create_response" not in config["turn_detection"]


def test_voice_opt_in_is_required(monkeypatch):
    monkeypatch.delenv("OPENCOUCH_ENABLE_EXPERIMENTAL_QWEN_VOICE")
    with pytest.raises(ProviderConfigurationError, match="EXPERIMENTAL"):
        realtime.build_realtime_session_config(
            thread_id="t", user_id=None, memory_mode="incognito"
        )


def test_rejects_old_model_that_cannot_execute_current_tools(monkeypatch):
    monkeypatch.setenv("QWEN_REALTIME_MODEL", "qwen3-omni-flash-realtime")
    with pytest.raises(ProviderConfigurationError, match="older models"):
        qwen.selected_model()


@pytest.mark.parametrize(
    "url",
    [
        "http://unitworkspace.cn-beijing.maas.aliyuncs.com/api/v1/webrtc/realtime",
        "https://attacker.invalid/api/v1/webrtc/realtime",
        URL + "?model=override",
        URL + "#fragment",
        "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/webrtc/realtime",
    ],
)
def test_signaling_target_is_server_configured_and_allowlisted(monkeypatch, url):
    monkeypatch.setenv("QWEN_REALTIME_URL", url)
    with pytest.raises(ProviderConfigurationError):
        qwen.signaling_url()


def test_tickets_are_one_use_expiring_and_capacity_bounded(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(qwen.time, "monotonic", lambda: clock[0])
    tickets = qwen.TicketStore(ttl_seconds=2, capacity=1)
    token = tickets.issue(url=URL, model=qwen.DEFAULT_MODEL)
    with pytest.raises(ValueError, match="Too many"):
        tickets.issue(url=URL, model=qwen.DEFAULT_MODEL)
    assert tickets.consume(token).model == qwen.DEFAULT_MODEL
    with pytest.raises(ValueError, match="already used"):
        tickets.consume(token)
    token = tickets.issue(url=URL, model=qwen.DEFAULT_MODEL)
    clock[0] = 13
    with pytest.raises(ValueError, match="expired"):
        tickets.consume(token)


async def test_session_credential_is_not_dashscope_key():
    ticket = await realtime.create_realtime_client_secret(
        session_config={"model": qwen.DEFAULT_MODEL}, safety_identifier=None
    )
    assert ticket != os.getenv("DASHSCOPE_API_KEY")
    assert len(ticket) >= 32
    assert qwen.TICKETS.consume(ticket).url == URL


async def test_sdp_exchange_uses_provider_model_and_does_not_follow_redirects(
    monkeypatch,
):
    real_client = httpx.AsyncClient
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=SDP)

    def client_factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(qwen.httpx, "AsyncClient", client_factory)
    ticket = qwen.SignalingTicket(999, URL, qwen.DEFAULT_MODEL)
    assert await qwen.exchange_sdp(ticket, SDP) == SDP
    assert requests[0].headers["authorization"] == "Bearer fake-server-only-key"
    assert requests[0].url.params["model"] == qwen.DEFAULT_MODEL
    assert requests[0].content.decode() == SDP


async def test_upstream_failure_never_exposes_sensitive_error_body(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        qwen.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(401, text="fake-server-only-key sensitive SDP")
            ),
            **kwargs,
        ),
    )
    with pytest.raises(RuntimeError) as error:
        await qwen.exchange_sdp(qwen.SignalingTicket(999, URL, qwen.DEFAULT_MODEL), SDP)
    assert str(error.value) == "Qwen Realtime signaling failed (HTTP 401)."


def test_signaling_route_rejects_replay_and_exposes_no_key_in_capabilities(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    async def exchange(ticket, offer):
        return SDP

    monkeypatch.setattr(qwen, "exchange_sdp", exchange)
    with TestClient(app) as client:
        config = client.get("/voice/realtime/config")
        assert config.status_code == 200
        assert "fake-server-only-key" not in config.text
        assert "unitworkspace" not in config.text
        token = qwen.TICKETS.issue(url=URL, model=qwen.DEFAULT_MODEL)
        assert (
            client.post(
                "/voice/realtime/qwen/sdp", json={"ticket": token, "sdp": SDP}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/voice/realtime/qwen/sdp", json={"ticket": token, "sdp": SDP}
            ).status_code
            == 410
        )
