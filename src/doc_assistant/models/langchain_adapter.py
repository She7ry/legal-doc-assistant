"""LangChain BaseChatModel 适配器：把项目自研 HTTP LLM 客户端接入 LangGraph。

项目内 ``OpenAICompatibleChatModel`` 已含重试、熔断、thinking 模式等；
本模块将其包装为 LangChain ``BaseChatModel``，以便 ``bind_tools`` 与
LangGraph 节点直接使用。
"""

from __future__ import annotations
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------

def _lc_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """LangChain BaseMessage -> OpenAI-compatible message dict."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}

    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}

    if isinstance(message, AIMessage):
        d: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                }
                for tc in message.tool_calls
            ]
        elif message.additional_kwargs.get("tool_calls"):
            d["tool_calls"] = message.additional_kwargs["tool_calls"]
        return d

    if isinstance(message, ToolMessage):
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        d = {
            "role": "tool",
            "content": content,
            "tool_call_id": message.tool_call_id,
        }
        if message.name:
            d["name"] = message.name
        return d

    return {"role": "user", "content": str(message.content)}


def _parse_tool_calls(raw_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-format tool_calls -> LangChain ToolCall dicts."""
    parsed: list[dict[str, Any]] = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}
        parsed.append(
            {
                "name": func.get("name", ""),
                "args": args,
                "id": tc.get("id", ""),
                "type": "tool_call",
            }
        )
    return parsed


def _response_dict_to_ai_message(response: dict[str, Any]) -> AIMessage:
    """OpenAI-compatible response message dict -> LangChain AIMessage."""
    content = response.get("content") or ""
    raw_tool_calls = response.get("tool_calls")

    kwargs: dict[str, Any] = {}
    if raw_tool_calls:
        kwargs["tool_calls"] = _parse_tool_calls(raw_tool_calls)
        kwargs["additional_kwargs"] = {"tool_calls": raw_tool_calls}

    return AIMessage(content=content, **kwargs)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ChatOpenAICompatible(BaseChatModel):
    """LangChain 聊天模型适配层，底层委托给项目的 HTTP LLM 客户端。

    用途：ToolCalling / LangGraph 需要 ``bind_tools``、``invoke([HumanMessage(...)])``
    等 LangChain 接口；本类负责消息格式互转，业务逻辑仍在 OpenAICompatibleChatModel。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any  # OpenAICompatibleChatModel (Any avoids Pydantic issues with frozen dataclass)

    # -- BaseChatModel required property --

    @property
    def _llm_type(self) -> str:
        return "openai-compatible-adapter"

    # -- Sync --

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_lc_message_to_dict(m) for m in messages]
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice", "auto" if tools else None)

        response = self.client.invoke_messages(
            message_dicts,
            tools=tools or None,
            tool_choice=tool_choice,
        )

        ai_msg = _response_dict_to_ai_message(response)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # -- Async --

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_lc_message_to_dict(m) for m in messages]
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice", "auto" if tools else None)

        response = await self.client.ainvoke_messages(
            message_dicts,
            tools=tools or None,
            tool_choice=tool_choice,
        )

        ai_msg = _response_dict_to_ai_message(response)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # -- Streaming (content only; tool-call responses fall back to full invoke) --

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if kwargs.get("tools"):
            result = self._generate(messages, stop, run_manager, **kwargs)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=result.generations[0].message.content,
                    tool_calls=getattr(result.generations[0].message, "tool_calls", []),
                )
            )
            yield chunk
            return

        message_dicts = [_lc_message_to_dict(m) for m in messages]
        for token in self.client.stream(messages=message_dicts):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if kwargs.get("tools"):
            result = await self._agenerate(messages, stop, run_manager, **kwargs)
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=result.generations[0].message.content,
                    tool_calls=getattr(result.generations[0].message, "tool_calls", []),
                )
            )
            yield chunk
            return

        message_dicts = [_lc_message_to_dict(m) for m in messages]
        async for token in self.client.astream(messages=message_dicts):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                await run_manager.on_llm_new_token(token)
            yield chunk


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_langchain_chat_model() -> ChatOpenAICompatible:
    """Build a LangChain-compatible chat model reusing the project's LLM client.

    The returned object supports ``invoke``, ``ainvoke``, ``stream``, ``astream``,
    ``bind_tools``, and all standard LangChain Runnable operations.
    """
    from doc_assistant.models.language_model import build_chat_model

    return ChatOpenAICompatible(client=build_chat_model())
