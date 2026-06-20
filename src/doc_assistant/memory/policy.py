"""记忆写入策略：从用户消息中识别「请记住」等显式/隐式写入意图。

规则引擎 ``extract_memory_write_intents`` 与 LLM 抽取（``extraction.py``）配合，
过滤一次性指令与敏感内容，输出 ``MemoryWriteIntent`` 列表。
"""

from __future__ import annotations

import re
from hashlib import sha1

from doc_assistant.memory.schemas import MemoryWriteIntent

_EXPLICIT_WRITE_MARKERS = (
    "remember",
    "remember that",
    "remember my",
    "from now on",
    "going forward",
    "always answer",
    "以后",
    "以后都",
    "以后请",
    "记住",
    "请记住",
    "帮我记住",
    "我的偏好",
    "偏好是",
)

_TEMPORARY_MARKERS = (
    "just this time",
    "for this question",
    "for this task",
    "本次",
    "这次",
    "当前问题",
    "临时",
)

_IMPLICIT_PROFILE_MARKERS = (
    "i am ",
    "i'm ",
    "i work as",
    "my role is",
    "我是",
    "我从事",
    "我做",
    "我的职位是",
    "我的岗位是",
)

_ANSWER_STYLE_TERMS = (
    "answer",
    "reply",
    "response",
    "style",
    "concise",
    "detailed",
    "language",
    "中文",
    "英文",
    "回答",
    "回复",
    "风格",
    "简洁",
    "详细",
    "实现",
)

_LEGAL_REVIEW_TERMS = (
    "contract",
    "contracts",
    "agreement",
    "agreements",
    "clause",
    "clauses",
    "review",
    "negotiate",
    "license",
    "licensing",
    "patent",
    "trademark",
    "copyright",
    "intellectual property",
    "ip",
    "privacy",
    "data processing",
    "dpa",
    "saas",
    "msa",
    "nda",
    "employment",
    "procurement",
    "lease",
    "compliance",
    "indemnity",
    "liability",
    "termination",
    "renewal",
    "governing law",
    "jurisdiction",
    "\u77e5\u8bc6\u4ea7\u6743",
    "\u4e13\u5229",
    "\u5546\u6807",
    "\u8457\u4f5c\u6743",
    "\u8bb8\u53ef\u5408\u540c",
    "\u5408\u540c",
    "\u534f\u8bae",
    "\u5ba1\u67e5",
    "\u5e38\u5ba1",
    "\u6761\u6b3e",
    "\u8d54\u507f",
    "\u8d23\u4efb\u9650\u5236",
    "\u7ec8\u6b62",
    "\u7eed\u7ea6",
    "\u9002\u7528\u6cd5",
    "\u7ba1\u8f96",
    "\u5408\u89c4",
)

_LEGAL_PROFILE_SIGNALS = (
    "we review",
    "we mainly review",
    "we primarily review",
    "we often review",
    "we usually review",
    "we negotiate",
    "we usually negotiate",
    "we focus on",
    "we care about",
    "we are in",
    "we work on",
    "our clients",
    "our practice",
    "our standard position",
    "standard position",
    "\u6211\u4eec\u4e3b\u8981",
    "\u6211\u4eec\u901a\u5e38",
    "\u6211\u4eec\u5e38",
    "\u6211\u4eec\u91cd\u70b9",
    "\u6211\u4eec\u5173\u6ce8",
    "\u6211\u65b9\u901a\u5e38",
    "\u6211\u65b9\u7acb\u573a",
    "\u6807\u51c6\u7acb\u573a",
    "\u5e38\u5ba1",
    "\u7ecf\u5e38\u5ba1",
    "\u4e3b\u8981\u5ba1\u67e5",
)


def extract_memory_write_intents(user_text: str) -> list[MemoryWriteIntent]:
    """从用户消息中识别「请记住…」等显式长期记忆写入意图。

    策略偏保守：普通聊天不会自动入库，除非用户明确表达要记住，
    或命中隐式 profile 规则（如稳定角色/偏好描述）。
    """

    text = " ".join(user_text.split())
    if not text or _looks_temporary(text):
        return []

    normalized = text.casefold()
    explicit = any(marker.casefold() in normalized for marker in _EXPLICIT_WRITE_MARKERS)
    implicit_profile = _looks_like_implicit_profile(text)
    implicit_intent = _implicit_memory_intent(text)
    if not explicit and not implicit_profile and implicit_intent is None:
        return []
    if not explicit and implicit_intent is not None:
        return [implicit_intent]

    content = _strip_write_marker(text) if explicit else text
    intents = []
    for part in _split_memory_content(content):
        clean_part = _strip_write_marker(part)
        if len(clean_part) < 4:
            continue
        key = _infer_key(clean_part)
        memory_type = "preference" if _looks_like_preference(clean_part) else "fact"
        intents.append(
            MemoryWriteIntent(
                type=memory_type,  # type: ignore[arg-type]
                key=key,
                content=clean_part,
                value_json={"text": clean_part},
                source="explicit" if explicit else "inferred",
                confidence=0.95 if explicit else 0.75,
            )
        )
    return intents


def _implicit_memory_intent(text: str) -> MemoryWriteIntent | None:
    if _looks_like_future_answer_preference(text):
        content = _strip_future_preference_prefix(text)
        return MemoryWriteIntent(
            type="preference",
            key="answer_style",
            content=content,
            value_json={"text": content},
            source="inferred",
            confidence=0.8,
        )
    if _looks_like_legal_review_profile_statement(text):
        return MemoryWriteIntent(
            type="fact",
            key="review_profile",
            content=text,
            value_json={"text": text},
            source="inferred",
            confidence=0.76,
        )
    if _looks_like_business_context_statement(text):
        return MemoryWriteIntent(
            type="fact",
            key="business_context",
            content=text,
            value_json={"text": text},
            source="inferred",
            confidence=0.72,
        )
    return None


def extract_task_memory_write_intents(
    assistant_text: str,
    *,
    task_id: str,
) -> list[MemoryWriteIntent]:
    """从 Agent 回答中提取任务级事实（scope=task），任务结束后可标记 stale。"""

    text = " ".join(assistant_text.split())
    if not text or not task_id:
        return []

    intents: list[MemoryWriteIntent] = []
    seen: set[str] = set()
    for sentence in _task_fact_sentences(text):
        clean = _clean_task_fact(sentence)
        if not _looks_like_task_fact(clean):
            continue
        key = _task_fact_key(clean)
        if key in seen:
            continue
        seen.add(key)
        intents.append(
            MemoryWriteIntent(
                type="fact",
                key=key,
                content=clean,
                value_json={"text": clean, "task_id": task_id},
                scope="task",
                source="system_generated",
                confidence=_task_fact_confidence(clean),
                task_id=task_id,
            )
        )
        if len(intents) >= 5:
            break
    return intents


def _looks_temporary(text: str) -> bool:
    normalized = text.casefold()
    return any(marker.casefold() in normalized for marker in _TEMPORARY_MARKERS)


def _looks_like_business_context_statement(text: str) -> bool:
    if _looks_like_question(text):
        return False
    normalized = text.casefold()
    english_signals = (
        "our company",
        "my company",
        "we mainly",
        "we primarily",
        "our business",
        "we provide",
        "we focus on",
        "we specialize in",
        "we are in",
        "we work on",
        "our practice",
        "our clients",
        "we represent",
    )
    chinese_patterns = (
        r"(?:\u6211\u4eec\u516c\u53f8|\u6211\u53f8|\u672c\u516c\u53f8).*(?:\u4e3b\u8425|\u4e3b\u8981\u4e1a\u52a1|\u4ece\u4e8b|\u63d0\u4f9b|\u4e13\u6ce8)",
        r"(?:\u4e3b\u8425|\u4e3b\u8981\u4e1a\u52a1).*(?:\u516c\u53f8|\u4e1a\u52a1|\u670d\u52a1)",
        r"(?:\u6211\u4eec|\u6211\u53f8|\u672c\u516c\u53f8|\u6211\u4eec\u56e2\u961f).*(?:\u884c\u4e1a|\u5ba2\u6237|\u77e5\u8bc6\u4ea7\u6743|\u4e13\u5229|\u5546\u6807|\u8457\u4f5c\u6743|\u5408\u540c|\u534f\u8bae)",
    )
    legal_domain_signal = any(term.casefold() in normalized for term in _LEGAL_REVIEW_TERMS)
    return any(signal in normalized for signal in english_signals) or _looks_like_legal_review_profile_statement(text) or any(
        re.search(pattern, text) for pattern in chinese_patterns
    ) or (legal_domain_signal and any(signal in normalized for signal in ("we ", "our ", "my company")))


def _looks_like_legal_review_profile_statement(text: str) -> bool:
    if _looks_like_question(text):
        return False
    normalized = text.casefold()
    has_profile_signal = any(signal.casefold() in normalized for signal in _LEGAL_PROFILE_SIGNALS)
    has_legal_term = any(term.casefold() in normalized for term in _LEGAL_REVIEW_TERMS)
    chinese_patterns = (
        r"(?:\u5e38\u5ba1|\u7ecf\u5e38\u5ba1|\u4e3b\u8981\u5ba1\u67e5|\u4e3b\u8981\u770b).*(?:\u5408\u540c|\u534f\u8bae|\u6761\u6b3e)",
        r"(?:\u6211\u4eec|\u6211\u65b9|\u6211\u53f8).*(?:\u91cd\u70b9\u5173\u6ce8|\u901a\u5e38\u5173\u6ce8|\u5e38\u770b).*(?:\u6761\u6b3e|\u8d54\u507f|\u8d23\u4efb|\u7ec8\u6b62|\u7eed\u7ea6|\u7ba1\u8f96)",
        r"(?:\u6211\u4eec|\u6211\u53f8).*(?:\u77e5\u8bc6\u4ea7\u6743|\u4e13\u5229|\u5546\u6807|\u8457\u4f5c\u6743).*(?:\u5408\u540c|\u534f\u8bae|\u8bb8\u53ef)",
    )
    return (has_profile_signal and has_legal_term) or any(
        re.search(pattern, text) for pattern in chinese_patterns
    )


def _looks_like_future_answer_preference(text: str) -> bool:
    if _looks_like_question(text):
        return False
    normalized = text.casefold()
    english_future = ("from now on", "going forward", "in future", "next time", "always")
    english_reply = ("answer", "reply", "response", "include", "use", "write")
    english_style = (
        "concise",
        "detailed",
        "english",
        "chinese",
        "bilingual",
        "side-by-side",
        "citations",
        "format",
    )
    has_english_signal = (
        any(term in normalized for term in english_future)
        and any(term in normalized for term in english_reply)
        and any(term in normalized for term in english_style)
    )
    chinese_patterns = (
        r"(?:\u4ee5\u540e|\u540e\u7eed|\u4eca\u540e|\u4e4b\u540e).*(?:\u56de\u7b54|\u56de\u590d|\u7b54\u590d).*(?:\u4e2d\u6587|\u82f1\u6587|\u4e2d\u82f1|\u5bf9\u7167|\u7b80\u6d01|\u8be6\u7ec6|\u683c\u5f0f|\u98ce\u683c|\u9644\u4e0a)",
        r"(?:\u90fd|\u8bf7|\u5e2e\u6211).*(?:\u9644\u4e0a|\u52a0\u4e0a).*(?:\u82f1\u6587|\u4e2d\u82f1|\u5bf9\u7167)",
    )
    return has_english_signal or any(re.search(pattern, text) for pattern in chinese_patterns)


def _strip_future_preference_prefix(text: str) -> str:
    content = _strip_write_marker(text)
    replacements = (
        r"^\s*(?:in\s+future|next\s+time)[:,]?\s*",
        r"^\s*(?:\u4ee5\u540e|\u540e\u7eed|\u4eca\u540e|\u4e4b\u540e)[\uff0c,:\s]*",
    )
    for pattern in replacements:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content.strip(" \uff0c,.\u3002")


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith(("?", "\uff1f")):
        return True
    normalized = stripped.casefold()
    if re.search(r"^(?:what|how|why|when|where|who|can you|could you|please explain)\b", normalized):
        return True
    return any(term in stripped for term in ("\u4ec0\u4e48", "\u5982\u4f55", "\u600e\u4e48", "\u662f\u5426", "\u5417"))


def _looks_like_preference(text: str) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in _ANSWER_STYLE_TERMS) or any(
        marker in normalized for marker in ("prefer", "喜欢", "偏好", "希望")
    )


def _looks_like_implicit_profile(text: str) -> bool:
    normalized = text.casefold().strip()
    has_profile_marker = any(marker.casefold() in normalized for marker in _IMPLICIT_PROFILE_MARKERS)
    has_legal_profile_marker = any(signal.casefold() in normalized for signal in _LEGAL_PROFILE_SIGNALS)
    if not has_profile_marker and not has_legal_profile_marker:
        return False
    if any(term.casefold() in normalized for term in _LEGAL_REVIEW_TERMS):
        return True
    return any(
        term in normalized
        for term in (
            "role",
            "job",
            "company",
            "legal",
            "law",
            "ip",
            "知识产权",
            "法务",
            "律师",
            "公司",
            "岗位",
            "职位",
            "行业",
        )
    )


def _infer_key(content: str) -> str:
    normalized = content.casefold()
    if any(term.casefold() in normalized for term in _ANSWER_STYLE_TERMS):
        return "answer_style"
    if any(
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
    ):
        return "clause_review_focus"
    if _looks_like_legal_review_profile_statement(content):
        return "review_profile"
    if any(term in normalized for term in ("name", "call me", "称呼", "叫我")):
        return "form_of_address"
    if any(
        term in normalized
        for term in (
            "role",
            "job",
            "company",
            "business",
            "industry",
            "legal",
            "law",
            "业务",
            "背景",
            "职位",
            "岗位",
            "公司",
            "行业",
            "法务",
            "律师",
            "知识产权",
        )
    ):
        return "business_context"
    if any(term in normalized for term in ("prefer", "喜欢", "偏好", "希望")):
        return "user_preference"
    words = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", content.casefold())
    canonical = "_".join(words[:8])[:48]
    digest = sha1(" ".join(words).encode("utf-8")).hexdigest()[:10] if words else ""
    if canonical and digest:
        return f"{canonical}_{digest}"[:80]
    return "user_memory"


def _strip_write_marker(text: str) -> str:
    replacements = [
        r"^\s*(please\s+)?remember\s+(that\s+)?",
        r"^\s*(please\s+)?remember\s+my\s+",
        r"^\s*from\s+now\s+on[:,]?\s*",
        r"^\s*going\s+forward[:,]?\s*",
        r"^\s*always\s+answer\s*[:,]?\s*",
        r"^\s*请?帮?我?记住[：:\s]*",
        r"^\s*以后(都|请)?[：:\s]*",
        r"^\s*我的偏好是[：:\s]*",
        r"^\s*偏好是[：:\s]*",
    ]
    content = text.strip()
    for pattern in replacements:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content.strip(" ：:。.")


def _task_fact_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+|(?:^|\s)[-*]\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _clean_task_fact(sentence: str) -> str:
    clean = re.sub(r"^#+\s*", "", sentence.strip())
    clean = re.sub(r"\s+", " ", clean)
    return clean[:500].strip(" -")


def _looks_like_task_fact(sentence: str) -> bool:
    if len(sentence) < 12 or len(sentence) > 500:
        return False
    normalized = sentence.casefold()
    has_source_ref = bool(re.search(r"\[[SCDPW]\d+\]", sentence, flags=re.IGNORECASE))
    fact_terms = (
        " is ",
        " are ",
        " was ",
        " were ",
        " must ",
        " requires ",
        " provides ",
        " states ",
        " governs",
        " due ",
        " within ",
        "party",
        "parties",
        "effective date",
        "termination",
        "governing law",
        "jurisdiction",
    )
    has_structured_signal = any(term in normalized for term in fact_terms)
    has_cjk_fact = bool(re.search(r"[\u4e00-\u9fff].*(是|为|应|须|必须|约定|适用|管辖)", sentence))
    has_number_or_date = bool(
        re.search(
            r"\b\d{1,4}(?:[-/.]\d{1,2}){0,2}\b|\b\d+\s+(?:days?|months?|years?)\b",
            normalized,
        )
    )
    if has_source_ref:
        return has_structured_signal or has_cjk_fact or has_number_or_date
    return _looks_like_structured_legal_task_fact(sentence)


def _task_fact_confidence(sentence: str) -> float:
    if re.search(r"\[[SCDPW]\d+\]", sentence, flags=re.IGNORECASE):
        return 0.65
    return 0.56


def _looks_like_structured_legal_task_fact(sentence: str) -> bool:
    normalized = sentence.casefold()
    english_patterns = (
        r"\bpart(?:y|ies)\b.{0,80}\b(?:is|are|include|between)\b",
        r"\bbetween\s+[A-Z][A-Za-z0-9&'., -]{1,80}\s+and\s+[A-Z][A-Za-z0-9&'., -]{1,80}",
        r"\beffective\s+date\b.{0,80}\b(?:is|was|:)\b",
        r"\b(?:term|duration)\b.{0,80}\b(?:\d+\s+(?:days?|months?|years?)|until|expires?)\b",
        r"\bgoverning\s+law\b.{0,80}\b(?:is|was|:|governs?)\b",
        r"\bjurisdiction\b.{0,80}\b(?:is|was|:|courts?|venue)\b",
        r"\b(?:notice|deadline)\b.{0,80}\b\d+\s+(?:business\s+)?(?:days?|months?|years?)\b",
    )
    if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in english_patterns):
        return True
    cjk_patterns = (
        r"(?:\u7532\u65b9|\u4e59\u65b9|\u5f53\u4e8b\u65b9).{0,40}(?:\u4e3a|\u662f)",
        r"(?:\u751f\u6548\u65e5|\u6709\u6548\u671f|\u5408\u540c\u671f\u9650).{0,40}(?:\u4e3a|\u662f|\d)",
        r"(?:\u9002\u7528\u6cd5|\u7ba1\u8f96).{0,40}(?:\u4e3a|\u662f|\u6cd5|\u6cd5\u9662)",
        r"(?:\u901a\u77e5|\u622a\u6b62\u65e5|\u671f\u9650).{0,40}\d+.{0,8}(?:\u65e5|\u5929|\u4e2a\u6708|\u5e74)",
    )
    return any(re.search(pattern, sentence) for pattern in cjk_patterns) or (
        "governing law" in normalized and len(sentence) >= 12
    )


def _task_fact_key(sentence: str) -> str:
    words = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", sentence.casefold())
    canonical = "_".join(words[:8])[:48] or "assistant_fact"
    digest_source = " ".join(words) if words else sentence
    digest = sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"task_fact_{canonical}_{digest}"[:100]


def _split_memory_content(content: str) -> list[str]:
    normalized = re.sub(
        r"\s+(?:and also|also|plus|and my)\s+",
        "；",
        content,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"(并且|而且|另外|同时)", "；", normalized)
    parts = re.split(r"[;；。]\s*|[，,]\s*(?=(?:我|my|i\s|i'm|以后|always|from now on))", normalized)
    return [part.strip(" ：:。.") for part in parts if part.strip(" ：:。.")]
