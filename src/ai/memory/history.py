"""Conversation history helpers shared by chat services."""

from __future__ import annotations

from collections.abc import Iterable

from ai.utils.tokens import count_message_tokens, truncate_text_tokens


def is_conversation_summary_context(content: str) -> bool:
    normalized = content.strip().casefold()
    return normalized.startswith(("conversation summary:", "会话摘要：", "会话摘要:"))


def format_chat_history(
    messages: list[dict[str, object]],
    max_messages: int = 12,
    *,
    max_tokens: int | None = None,
) -> str:
    clean = list(_history_messages(messages))
    summaries = [message for message in clean if message["role"] == "system"][-1:]
    recent = _recent_chat_messages(clean, max_messages)
    if max_tokens is not None:
        summaries, recent = _fit_summary_and_recent(summaries, recent, max_tokens=max_tokens)
    parts = [message["content"] for message in summaries]
    parts.extend(
        f"{'用户' if message['role'] == 'user' else '助手'}：{message['content']}"
        for message in recent
    )
    return "\n".join(parts) if parts else "没有历史消息。"


def merge_chat_history(
    persisted_history: list[dict[str, object]],
    incoming_history: list[dict[str, object]],
    *,
    max_messages: int,
) -> list[dict[str, object]]:
    persisted = list(_history_messages(persisted_history))
    incoming = list(_history_messages(incoming_history))
    summaries = [message for message in [*persisted, *incoming] if message["role"] == "system"]
    persisted_chat = [message for message in persisted if message["role"] != "system"]
    incoming_chat = [message for message in incoming if message["role"] != "system"]
    merged = _merge_overlapping_history(persisted_chat, incoming_chat)
    return [*summaries[-1:], *_recent_chat_messages(merged, max_messages)]


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


def _merge_overlapping_history(
    persisted: list[dict[str, str]],
    incoming: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not persisted:
        return incoming
    if not incoming:
        return persisted
    if _contains_sequence(incoming, persisted):
        return incoming
    if _contains_sequence(persisted, incoming):
        return persisted
    for overlap in range(min(len(persisted), len(incoming)), 0, -1):
        if persisted[-overlap:] == incoming[:overlap]:
            return [*persisted, *incoming[overlap:]]
        if incoming[-overlap:] == persisted[:overlap]:
            return [*incoming, *persisted[overlap:]]
    return incoming


def _contains_sequence(
    messages: list[dict[str, str]],
    candidate: list[dict[str, str]],
) -> bool:
    width = len(candidate)
    return any(
        messages[index : index + width] == candidate for index in range(len(messages) - width + 1)
    )


def _fit_summary_and_recent(
    summaries: list[dict[str, str]],
    recent: list[dict[str, str]],
    *,
    max_tokens: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    remaining = max(0, max_tokens)
    selected_summary: list[dict[str, str]] = []
    if summaries and remaining:
        summary_budget = max(1, remaining // 3)
        summary = _bounded_message(summaries[-1], summary_budget)
        if count_message_tokens([summary]) <= remaining:
            selected_summary = [summary]
            remaining -= count_message_tokens(selected_summary)

    selected_recent: list[dict[str, str]] = []
    for message in reversed(recent):
        size = count_message_tokens([message])
        if size <= remaining:
            selected_recent.append(message)
            remaining -= size
            continue
        if remaining > 8:
            selected_recent.append(_bounded_message(message, remaining))
        break
    return selected_summary, list(reversed(selected_recent))


def _bounded_message(message: dict[str, str], max_tokens: int) -> dict[str, str]:
    empty = {**message, "content": ""}
    content_budget = max(0, max_tokens - count_message_tokens([empty]))
    return {
        **message,
        "content": truncate_text_tokens(message["content"], content_budget),
    }
