from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryWriteIntent
from doc_assistant.models.language_model import build_chat_model

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """You extract durable memory for a legal document assistant.
Only return stable user preferences or durable user/org context that will help future answers.
Do not store one-off questions, document excerpts, legal conclusions, secrets, credentials, or facts that only belong in the current retrieved document.
Keep memory content in the user's language when possible.

For legal document review, capture durable context such as:
- the user's role, organization type, practice area, or counterparty/client side
- commonly reviewed contract types, for example SaaS MSAs, NDAs, DPAs, patent licenses, procurement contracts, employment agreements, or leases
- recurring clause focus areas, for example indemnity, liability cap, termination, renewal, payment, audit, confidentiality, governing law, jurisdiction, privacy, or data processing
- preferred negotiation positions, risk tolerance, jurisdiction/law preferences, output style, or citation/detail preferences
- counterpart industries or client industries that are stable across future matters

Do not store facts about a specific uploaded contract unless the user frames them as durable preferences or business context.

Return strict JSON only:
{
  "memories": [
    {
      "type": "preference" | "fact",
      "key": "short_snake_case_key",
      "content": "one concise memory",
      "confidence": 0.60
    }
  ]
}

Return {"memories": []} when nothing should be remembered."""


class LLMMemoryExtractor:
    """Optional LLM-backed fallback for memory writes after rules miss."""

    def __init__(
        self,
        chat_model: object | None = None,
        *,
        model_factory: Callable[[], object] = build_chat_model,
        max_items: int | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._model_factory = model_factory
        self.max_items = max(1, max_items or settings.memory_llm_extraction_max_items)
        self.min_confidence = _clamp_confidence(
            min_confidence
            if min_confidence is not None
            else settings.memory_llm_extraction_min_confidence
        )

    def __call__(self, user_text: str) -> list[MemoryWriteIntent]:
        return self.extract(user_text)

    def extract(self, user_text: str) -> list[MemoryWriteIntent]:
        text = " ".join(user_text.split())
        if not _worth_llm_extraction(text):
            return []

        try:
            response = self._invoke_model(text)
        except Exception:
            logger.debug("LLM memory extraction failed.", exc_info=True)
            return []

        return self._intents_from_response(response)

    def _invoke_model(self, user_text: str) -> str:
        model = self._chat_model or self._model_factory()
        self._chat_model = model
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        invoke_messages = getattr(model, "invoke_messages", None)
        if callable(invoke_messages):
            response = invoke_messages(messages)
            if isinstance(response, dict):
                return str(response.get("content") or "")
            return str(response)

        invoke = getattr(model, "invoke", None)
        if callable(invoke):
            try:
                response = invoke(messages=messages)
            except TypeError:
                response = invoke(messages)
            return str(getattr(response, "content", response))

        raise ValueError("The configured chat model does not support memory extraction.")

    def _intents_from_response(self, response: str) -> list[MemoryWriteIntent]:
        payload = _extract_json_object(response)
        raw_items = payload.get("memories") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            return []

        intents: list[MemoryWriteIntent] = []
        seen_keys: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            intent = _intent_from_item(item, min_confidence=self.min_confidence)
            if intent is None or intent.key in seen_keys:
                continue
            intents.append(intent)
            seen_keys.add(intent.key)
            if len(intents) >= self.max_items:
                break
        return intents


def build_memory_extractor() -> LLMMemoryExtractor | None:
    if not settings.memory_llm_extraction_enabled:
        return None
    return LLMMemoryExtractor()


def _intent_from_item(item: dict[str, Any], *, min_confidence: float) -> MemoryWriteIntent | None:
    memory_type = str(item.get("type") or "").strip().casefold()
    if memory_type not in {"preference", "fact"}:
        return None

    content = " ".join(str(item.get("content") or "").split())
    if len(content) < 4:
        return None

    key = _normalize_key(str(item.get("key") or "") or _infer_key(content, memory_type))
    confidence = _coerce_confidence(item.get("confidence"), default=max(min_confidence, 0.6))
    if confidence < min_confidence:
        return None

    return MemoryWriteIntent(
        type=memory_type,  # type: ignore[arg-type]
        key=key,
        content=content,
        value_json={"text": content, "extracted_by": "llm"},
        source="inferred",
        confidence=confidence,
    )


def _worth_llm_extraction(text: str) -> bool:
    if len(text) < 8 or len(text) > 800:
        return False
    if _looks_like_question(text):
        return False

    normalized = text.casefold()
    english_signals = (
        " i ",
        "i am",
        "i'm",
        "my ",
        "we ",
        "our ",
        "our company",
        "my company",
        "prefer",
        "usually",
        "client",
        "customer",
        "team",
        "role",
        "business",
        "practice",
        "contract",
        "agreement",
        "clause",
        "license",
        "patent",
        "trademark",
        "privacy",
        "saas",
        "msa",
        "nda",
        "dpa",
        "indemnity",
        "liability",
        "jurisdiction",
    )
    cjk_signals = (
        "我",
        "我们",
        "我司",
        "本公司",
        "客户",
        "偏好",
        "希望",
        "常用",
        "主营",
        "负责",
        "岗位",
        "职位",
        "团队",
        "业务",
        "甲方是我们",
        "乙方是我们",
    )
    legal_cjk_signals = (
        "\u77e5\u8bc6\u4ea7\u6743",
        "\u4e13\u5229",
        "\u5546\u6807",
        "\u8bb8\u53ef\u5408\u540c",
        "\u5408\u540c\u5ba1\u67e5",
        "\u5e38\u5ba1",
        "\u6761\u6b3e",
        "\u8d54\u507f",
        "\u8d23\u4efb\u9650\u5236",
        "\u9002\u7528\u6cd5",
        "\u7ba1\u8f96",
    )
    padded = f" {normalized} "
    return any(signal in padded for signal in english_signals) or any(
        signal in text for signal in (*cjk_signals, *legal_cjk_signals)
    )


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith(("?", "？")):
        return True
    normalized = stripped.casefold()
    if re.search(r"^(?:what|how|why|when|where|who|can you|could you|please explain)\b", normalized):
        return True
    return any(term in stripped for term in ("什么", "如何", "怎么", "是否", "吗"))


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        return {}

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced_match.group(1)] if fenced_match else []
    candidates.append(text)
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if 0 <= first_brace < last_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _infer_key(content: str, memory_type: str) -> str:
    normalized = content.casefold()
    if memory_type == "preference":
        if any(term in normalized for term in ("answer", "reply", "response", "回答", "回复")):
            return "answer_style"
        return "user_preference"
    if any(term in normalized for term in ("company", "business", "公司", "业务", "主营", "客户")):
        return "business_context"
    return "user_fact"


def _infer_key(content: str, memory_type: str) -> str:
    normalized = content.casefold()
    if memory_type == "preference":
        if any(term in normalized for term in ("answer", "reply", "response")):
            return "answer_style"
        if _contains_clause_focus(normalized):
            return "clause_review_focus"
        return "user_preference"
    if _contains_review_profile(normalized):
        return "review_profile"
    if any(term in normalized for term in ("company", "business")):
        return "business_context"
    return "user_fact"


def _contains_clause_focus(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "indemnity",
            "liability",
            "termination",
            "renewal",
            "governing law",
            "jurisdiction",
            "clause",
            "\u8d54\u507f",
            "\u8d23\u4efb",
            "\u7ec8\u6b62",
            "\u7eed\u7ea6",
            "\u9002\u7528\u6cd5",
            "\u7ba1\u8f96",
            "\u6761\u6b3e",
        )
    )


def _contains_review_profile(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "contract",
            "agreement",
            "license",
            "patent",
            "trademark",
            "practice",
            "review",
            "\u5408\u540c",
            "\u534f\u8bae",
            "\u8bb8\u53ef",
            "\u4e13\u5229",
            "\u5546\u6807",
            "\u5ba1\u67e5",
            "\u5e38\u5ba1",
        )
    )


def _normalize_key(key: str) -> str:
    words = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", key.casefold())
    normalized = "_".join(words)[:80].strip("_")
    return normalized or "user_memory"


def _coerce_confidence(value: object, *, default: float) -> float:
    try:
        return _clamp_confidence(float(value))
    except (TypeError, ValueError):
        return _clamp_confidence(default)


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))
