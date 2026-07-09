"""Conversation history helpers shared by chat services."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import convert_to_messages


def is_conversation_summary_context(content: str) -> bool:
    return content.strip().casefold().startswith("conversation summary:")


def format_chat_history(messages: list[dict[str, object]], max_messages: int = 12) -> str:
    clean_messages = list(_history_messages(messages))
    history_parts = [
        f"Session summary: {_message_content(message)}"
        for message in clean_messages
        if isinstance(message, SystemMessage)
    ]
    history_parts.extend(
        f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: {_message_content(message)}"
        for message in _trim_chat_messages(clean_messages, max_messages)
    )
    return "\n".join(history_parts) if history_parts else "No previous messages."


def merge_chat_history(
    persisted_history: list[dict[str, object]],
    incoming_history: list[dict[str, object]],
    *,
    max_messages: int,
) -> list[dict[str, object]]:
    system_context: list[BaseMessage] = []
    merged: list[BaseMessage] = []
    seen: set[tuple[str, str]] = set()
    for message in _history_messages([*persisted_history, *incoming_history]):
        content = _message_content(message)
        key = (message.type, content)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(message, SystemMessage):
            system_context.append(message)
        else:
            merged.append(message)
    recent_messages = _trim_chat_messages(merged, max_messages)
    return [_message_dict(message) for message in [*system_context, *recent_messages]]


def _history_messages(messages: Iterable[dict[str, object]]) -> Iterable[BaseMessage]:
    for message in convert_to_messages(messages):
        content = _message_content(message)
        if not content:
            continue
        if isinstance(message, SystemMessage):
            if is_conversation_summary_context(content):
                yield message
            continue
        if isinstance(message, (HumanMessage, AIMessage)):
            yield message


def _trim_chat_messages(messages: list[BaseMessage], max_messages: int) -> list[BaseMessage]:
    chat_messages = [
        message for message in messages if isinstance(message, (HumanMessage, AIMessage))
    ]
    return trim_messages(
        chat_messages,
        max_tokens=max(0, max_messages),
        token_counter=len,
        strategy="last",
    )


def _message_content(message: BaseMessage) -> str:
    return str(message.content or "").strip()


def _message_dict(message: BaseMessage) -> dict[str, object]:
    role = "assistant" if isinstance(message, AIMessage) else message.type
    return {
        "role": "user" if isinstance(message, HumanMessage) else role,
        "content": _message_content(message),
    }
