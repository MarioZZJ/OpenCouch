"""Per-run provider injection; no process-global SDK client or API switching."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from agents import (
    FunctionTool,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunConfig,
)

from llm.compatible_client import CompatibleLLMClient
from llm.openai_client import OpenAILLMClient


class CompatibleChatModel(OpenAIChatCompletionsModel):
    """Translate structured output requests to JSON mode; Runner still validates.

    Qwen/DeepSeek compatibility does not imply OpenAI strict json_schema support.
    Only the wire format is changed, not Agent.output_type or its validation.
    """

    def __init__(self, client: CompatibleLLMClient) -> None:
        super().__init__(model=client.model, openai_client=client.client)
        self._provider_client = client

    def _prepare(
        self, system_instructions: str | None, model_settings: Any, output_schema: Any
    ) -> tuple[str | None, Any]:
        body = {**(model_settings.extra_body or {}), **self._provider_client.extra_body}
        if output_schema is not None and not output_schema.is_plain_text():
            system_instructions = (system_instructions or "") + (
                "\nReturn ONLY a JSON object matching this JSON Schema:\n"
                + json.dumps(output_schema.json_schema(), ensure_ascii=False)
            )
            # extra_body merges into the serialized HTTP body after the SDK's
            # omitted response_format. Covered by an actual HTTP mock test.
            body["response_format"] = {"type": "json_object"}
        return system_instructions, replace(
            model_settings,
            extra_body=body,
            max_tokens=model_settings.max_tokens or self._provider_client.max_tokens,
            reasoning=None,
            verbosity=None,
            store=None,
            metadata=None,
            prompt_cache_retention=None,
            parallel_tool_calls=None,
        )

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> Any:
        instructions, settings = self._prepare(
            system_instructions, model_settings, output_schema
        )
        return await super().get_response(
            system_instructions=instructions,
            input=input,
            model_settings=settings,
            tools=[
                replace(tool, strict_json_schema=False)
                if isinstance(tool, FunctionTool)
                else tool
                for tool in tools
            ],
            output_schema=None,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> Any:
        instructions, settings = self._prepare(
            system_instructions, model_settings, output_schema
        )
        async for event in super().stream_response(
            system_instructions=instructions,
            input=input,
            model_settings=settings,
            tools=[
                replace(tool, strict_json_schema=False)
                if isinstance(tool, FunctionTool)
                else tool
                for tool in tools
            ],
            output_schema=None,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        ):
            yield event


def sdk_run_kwargs(context: Any, *, triage: bool = False) -> dict[str, Any]:
    """Use control for triage, selected response tier for specialist execution."""
    workflow = context.workflow_context
    client = (
        workflow.llm_client if triage else workflow.response_llm or workflow.llm_client
    )
    if isinstance(client, CompatibleLLMClient):
        return {
            "run_config": RunConfig(
                model=CompatibleChatModel(client),
                tracing_disabled=True,
                trace_include_sensitive_data=False,
            )
        }
    if isinstance(client, OpenAILLMClient):
        return {
            "run_config": RunConfig(
                model=OpenAIResponsesModel(
                    model=client.model, openai_client=client.client
                ),
                trace_include_sensitive_data=False,
            )
        }
    # Deterministic clients and injected test runners keep the existing contract.
    return {}
