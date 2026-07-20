"""Conversation history helpers shared by chat services."""

from __future__ import annotations

from collections.abc import Iterable


def is_conversation_summary_context(content: str) -> bool:
    normalized = content.strip().casefold()
    return normalized.startswith(("conversation summary:", "会话摘要：", "会话摘要:"))


def format_chat_history(messages: list[dict[str, object]], max_messages: int = 12) -> str:
    clean = list(_history_messages(messages))
    parts = [
        f"会话摘要：{message['content']}"
        for message in clean
        if message["role"] == "system"
    ]
    parts.extend(
        f"{'用户' if message['role'] == 'user' else '助手'}：{message['content']}"
        for message in _recent_chat_messages(clean, max_messages)
    )
    return "\n".join(parts) if parts else "没有历史消息。"


def merge_chat_history(
    persisted_history: list[dict[str, object]],
    incoming_history: list[dict[str, object]],
    *,
    max_messages: int,
) -> list[dict[str, object]]:
    system_context: list[dict[str, str]] = []
    chat_messages: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for message in _history_messages([*persisted_history, *incoming_history]):
        key = (message["role"], message["content"])
        if key in seen:
            continue
        seen.add(key)
        (system_context if message["role"] == "system" else chat_messages).append(message)
    return [*system_context, *_recent_chat_messages(chat_messages, max_messages)]


def _history_messages(messages: Iterable[dict[str, object]]) -> Iterable[dict[str, str]]:
    for message in messages:
        role = str(message.get("role") or "").casefold()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role in {"human", "user"}:
            yield {"role": "user", "content": content}
        elif role in {"ai", "assistant"}:
            yield {"role": "assistant", "content": content}
        elif role == "system" and is_conversation_summary_context(content):
            yield {"role": "system", "content": content}


def _recent_chat_messages(
    messages: list[dict[str, str]],
    max_messages: int,
) -> list[dict[str, str]]:
    chat_messages = [message for message in messages if message["role"] != "system"]
    return chat_messages[-max(0, max_messages) :] if max_messages > 0 else []
