"""Offline provider contract tests: exercise real SDK serialization over HTTP mocks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from agents import Agent, RunConfig, Runner, function_tool
from llm.compatible_client import CompatibleLLMClient, resolve_reasoning_effort
from llm.providers import ProviderConfigurationError, UnsupportedProviderCapability
from llm.sdk_models import CompatibleChatModel, sdk_run_kwargs
from pydantic import BaseModel


class Decision(BaseModel):
    route: str
    safe: bool


def completion(content: str | None = "Hello", *, finish: str = "stop", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return {
        "id": "chat-test",
        "object": "chat.completion",
        "created": 1,
        "model": "mock-model",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def make_client(monkeypatch, handler, *, provider="qwen", model="qwen-flash"):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-qwen-test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-test-key")
    client = CompatibleLLMClient(provider=provider, model=model)
    # Replace just the HTTP transport: retain the real AsyncOpenAI SDK serializer.
    from openai import AsyncOpenAI

    client.client = AsyncOpenAI(
        api_key="fake-wire-test-key",
        base_url="https://unit.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return client


@pytest.mark.parametrize(
    "provider,model,extra",
    [
        ("qwen", "qwen-flash", {"enable_thinking": True, "reasoning_effort": "max"}),
        (
            "deepseek",
            "deepseek-flash",
            {"thinking": {"type": "enabled"}, "reasoning_effort": "max"},
        ),
    ],
)
async def test_text_wire_protocol(monkeypatch, provider, model, extra):
    # Reasoning is on by default; LLM_REASONING_EFFORT drives the level.
    monkeypatch.setenv("LLM_REASONING_EFFORT", "max")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion("你好"))

    client = make_client(monkeypatch, handler, provider=provider, model=model)
    assert await client.generate_text(prompt="test") == "你好"
    payload = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert payload["model"] == model
    assert all(payload[key] == value for key, value in extra.items())
    assert "input" not in payload
    await client.client.close()


@pytest.mark.parametrize("provider", ["qwen", "deepseek"])
async def test_reasoning_off_is_the_explicit_disable_path(monkeypatch, provider):
    """LLM_REASONING_EFFORT=none must send no reasoning fields at all."""
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion("你好"))

    client = make_client(monkeypatch, handler, provider=provider, model="x-model")
    assert await client.generate_text(prompt="test") == "你好"
    payload = json.loads(requests[0].content)
    assert "reasoning_effort" not in payload
    assert "enable_thinking" not in payload
    assert "thinking" not in payload
    await client.client.close()


async def test_default_reasoning_effort_is_highest_level(monkeypatch):
    """With the env var unset, reasoning defaults on at the highest level."""
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    assert resolve_reasoning_effort() == "max"
    assert resolve_reasoning_effort("max") == "max"
    assert resolve_reasoning_effort("none") is None
    assert resolve_reasoning_effort("") is None


async def test_unsupported_reasoning_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "turbo")

    def handler(request):
        return httpx.Response(200, json=completion("你好"))

    with pytest.raises(ProviderConfigurationError):
        make_client(monkeypatch, handler, provider="deepseek", model="x-model")


async def test_structured_repair_is_bounded_and_validates_locally(monkeypatch):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        content = (
            '{"route":"support"}'
            if len(requests) == 1
            else '{"route":"support","safe":true}'
        )
        return httpx.Response(200, json=completion(content))

    client = make_client(monkeypatch, handler)
    result = await client.generate_structured(prompt="test", response_schema=Decision)
    assert result.safe is True
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in requests[0]["messages"][0]["content"]
    await client.client.close()


async def test_invalid_structured_output_does_not_become_a_safe_assessment(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion("not JSON"))

    client = make_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="local JSON Schema"):
        await client.generate_structured(prompt="test", response_schema=Decision)
    assert len(calls) == 2
    await client.client.close()


@pytest.mark.parametrize("finish", ["length", "content_filter"])
async def test_incomplete_response_is_not_accepted(monkeypatch, finish):
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(200, json=completion("partial", finish=finish)),
    )
    with pytest.raises(ValueError, match="did not finish"):
        await client.generate_text(prompt="test")
    await client.client.close()


async def test_search_is_explicit_and_never_faked(monkeypatch):
    client = make_client(
        monkeypatch,
        lambda _: pytest.fail("No HTTP request expected"),
        provider="deepseek",
    )
    with pytest.raises(UnsupportedProviderCapability):
        await client.generate_text(prompt="official resources", use_search=True)
    await client.client.close()


async def test_sdk_structured_agent_wire_uses_json_mode(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"route":"support","safe":true}'))

    client = make_client(
        monkeypatch, handler, provider="deepseek", model="deepseek-flash"
    )
    agent = Agent(name="Test triage", instructions="Classify.", output_type=Decision)
    result = await Runner.run(
        agent,
        "test",
        run_config=RunConfig(model=CompatibleChatModel(client), tracing_disabled=True),
    )
    assert isinstance(result.final_output, Decision)
    body = bodies[0]
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "enabled"}
    assert "JSON Schema" in body["messages"][0]["content"]
    assert body["reasoning_effort"] == "max"
    await client.client.close()


async def test_sdk_tools_roundtrip_preserves_local_execution(monkeypatch):
    bodies, executed = [], []

    @function_tool
    def read_memory(topic: str) -> str:
        """Read a memory. Args: topic: Memory topic."""
        executed.append(topic)
        return "Verified memory"

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(
                200,
                json=completion(
                    None,
                    finish="tool_calls",
                    calls=[
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_memory",
                                "arguments": '{"topic":"sleep"}',
                            },
                        }
                    ],
                ),
            )
        return httpx.Response(200, json=completion("I remember."))

    client = make_client(monkeypatch, handler)
    result = await Runner.run(
        Agent(name="Test", tools=[read_memory]),
        "test",
        run_config=RunConfig(model=CompatibleChatModel(client), tracing_disabled=True),
    )
    assert result.final_output == "I remember."
    assert executed == ["sleep"]
    assert any(
        item["role"] == "tool" and "Verified memory" in item["content"]
        for item in bodies[1]["messages"]
    )
    assert bodies[0]["tools"][0]["function"].get("strict") is not True
    await client.client.close()


def test_per_run_provider_selection_does_not_mutate_global_defaults(monkeypatch):
    control = make_client(monkeypatch, lambda _: None)
    response = make_client(
        monkeypatch, lambda _: None, provider="deepseek", model="deepseek-flash"
    )
    context = SimpleNamespace(
        workflow_context=SimpleNamespace(llm_client=control, response_llm=response)
    )
    assert (
        sdk_run_kwargs(context, triage=True)["run_config"].model.model == "qwen-flash"
    )
    assert sdk_run_kwargs(context)["run_config"].model.model == "deepseek-flash"
    assert sdk_run_kwargs(context)["run_config"].tracing_disabled


def test_settings_keep_providers_and_model_tiers_independent(monkeypatch):
    import config

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("OPENAI_MODEL", "must-not-leak")
    monkeypatch.setenv("RESPONSE_FAST_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("RESPONSE_FAST_LLM_MODEL", "deepseek-flash")
    settings = config.get_settings()
    assert settings.openai_model == "qwen-flash"
    assert settings.response_fast_openai_model == "deepseek-flash"
    assert settings.response_quality_openai_model == "qwen-plus"


def test_missing_third_party_key_never_uses_openai_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    with pytest.raises(ProviderConfigurationError, match="DASHSCOPE_API_KEY"):
        CompatibleLLMClient(provider="qwen", model="qwen-flash")


async def test_stream_skips_usage_and_reasoning_frames(monkeypatch):
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "private"},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": "你好"}, "finish_reason": None}
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    ]

    def handler(request):
        assert json.loads(request.content)["stream"]
        data = (
            "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "chunk",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "test",
                        **chunk,
                    }
                )
                + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, text=data, headers={"Content-Type": "text/event-stream"}
        )

    client = make_client(monkeypatch, handler)
    assert [part async for part in client.generate_text_stream(prompt="test")] == [
        "你好"
    ]
    await client.client.close()


async def test_sdk_streamed_structured_output_is_validated(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["response_format"] == {"type": "json_object"}
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": '{"route":"support","safe":true}',
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        data = (
            "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "c",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "test",
                        **chunk,
                    }
                )
                + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, text=data, headers={"Content-Type": "text/event-stream"}
        )

    client = make_client(monkeypatch, handler)
    result = Runner.run_streamed(
        Agent(name="Structured stream", output_type=Decision),
        "test",
        run_config=RunConfig(model=CompatibleChatModel(client), tracing_disabled=True),
    )
    events = [event async for event in result.stream_events()]
    assert events
    assert isinstance(result.final_output, Decision)
    assert result.final_output.safe is True
    await client.client.close()


def test_qwen_embedding_configuration_and_incognito_boundary(monkeypatch):
    from agent.memory.modes import MemoryMode
    from agent.memory.providers.embeddings import (
        NullEmbeddingProvider,
        create_configured_embedding_provider,
    )
    from agent.runtime.backends import create_embedding_provider

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-embedding-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "qwen")
    provider = create_configured_embedding_provider()
    assert provider.model_name == "text-embedding-v4"
    assert provider.dimension == 1024
    assert "dashscope.aliyuncs.com" in str(provider._client.base_url)
    assert isinstance(
        create_embedding_provider(
            memory_mode=MemoryMode.INCOGNITO, embedding_provider=None
        ),
        NullEmbeddingProvider,
    )
    monkeypatch.setenv("EMBEDDING_PROVIDER", "deepseek")
    with pytest.raises(ProviderConfigurationError, match="EMBEDDING_PROVIDER"):
        create_configured_embedding_provider()


async def test_qwen_embedding_wire_sends_dimensions(monkeypatch):
    from agent.memory.providers.embeddings import create_configured_embedding_provider
    from openai import AsyncOpenAI

    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-embedding-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "qwen")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    provider = create_configured_embedding_provider()

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-v4"
        assert body["dimensions"] == 1024
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-v4",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1] * 1024}
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    provider._client = AsyncOpenAI(
        api_key="fake-key",
        base_url="https://unit.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert len((await provider.aembed(["test"]))[0]) == 1024
    await provider._client.close()


@pytest.mark.parametrize("provider", ["openai", "openai_compatible"])
def test_generic_model_configuration_inherits_to_default_fast_tier(
    monkeypatch, provider
):
    import config

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    for name in (
        "RESPONSE_FAST_LLM_PROVIDER",
        "RESPONSE_FAST_LLM_MODEL",
        "RESPONSE_FAST_OPENAI_MODEL",
        "RESPONSE_QUALITY_LLM_PROVIDER",
        "RESPONSE_QUALITY_LLM_MODEL",
        "RESPONSE_QUALITY_OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = config.get_settings()
    assert settings.response_fast_openai_model == "custom-model"
    if provider == "openai_compatible":
        assert settings.response_quality_openai_model == "custom-model"
