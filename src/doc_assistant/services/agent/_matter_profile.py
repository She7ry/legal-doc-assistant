"""Matter Profile 构建：从执行步骤抽取当事方、法域、日期等案件画像。"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from doc_assistant.services.agent._helpers import _clean_text, _dedupe_texts
from doc_assistant.services.agent.schemas import AgentStepResult, MatterProfile


def _build_matter_profile(
    *,
    matter_id: str,
    objective: str,
    review_scope: list[str],
    steps: list[AgentStepResult],
    missing_information: list[str],
) -> MatterProfile:
    profile_step = next((step for step in steps if step.step_id == "profile"), None)
    source_text = _profile_source_text(objective, profile_step)
    parties = _extract_parties(source_text)
    governing_law = _extract_governing_law(source_text)
    jurisdiction = _extract_jurisdiction(source_text, governing_law)
    citations = [citation.source_id for citation in profile_step.citations] if profile_step else []

    profile = MatterProfile(
        matter_id=matter_id,
        document_type=_infer_document_type(source_text),
        parties=parties,
        user_side=_extract_user_side(source_text),
        governing_law=governing_law,
        jurisdiction=jurisdiction,
        key_dates=_extract_key_dates(source_text, citations),
        review_scope=_dedupe_texts(review_scope),
        open_questions=[],
        confidence=_profile_confidence(
            citations=citations,
            document_type=_infer_document_type(source_text),
            parties=parties,
            governing_law=governing_law,
        ),
        citations=citations,
        source_step_id=profile_step.step_id if profile_step else "",
    )
    return replace(
        profile,
        open_questions=_matter_open_questions(profile, missing_information),
    )


def _profile_source_text(objective: str, profile_step: AgentStepResult | None) -> str:
    parts = [objective]
    if profile_step:
        parts.append(profile_step.summary)
        for citation in profile_step.citations:
            parts.append(citation.exact_quote or citation.preview)
    return "\n".join(part for part in parts if part)


def _infer_document_type(text: str) -> str:
    lowered = text.casefold()
    rules = [
        ("SaaS MSA", ("saas", "msa")),
        ("SaaS agreement", ("saas", "agreement")),
        ("Master services agreement", ("master services agreement",)),
        ("Mutual NDA", ("mutual nda", "mutual non-disclosure")),
        ("Non-disclosure agreement", ("non-disclosure agreement", "nda")),
        ("Data processing addendum", ("data processing addendum", "dpa")),
        ("Supply agreement", ("supply agreement", "purchase agreement")),
        ("Employment document", ("employee handbook", "employment agreement")),
        ("Policy document", ("policy", "procedure")),
        ("Agreement", ("agreement", "contract")),
    ]
    for label, keywords in rules:
        if all(keyword in lowered for keyword in keywords):
            return label
    return "Unknown"


def _extract_parties(text: str) -> list[str]:
    patterns = [
        re.compile(
            r"\bby\s+and\s+between\s+(.{2,90}?)\s+and\s+(.{2,90}?)(?=\.|,|;|\n|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbetween\s+(.{2,90}?)\s+and\s+(.{2,90}?)(?=\.|,|;|\n|$)",
            re.IGNORECASE,
        ),
    ]
    parties: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            for group in match.groups():
                party = _clean_party_name(group)
                if party and party.casefold() not in {item.casefold() for item in parties}:
                    parties.append(party)
            if parties:
                return parties[:6]
    return parties


def _clean_party_name(value: str) -> str:
    text = _clean_text(value).strip(" .,:;()[]")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.split(
        r"\s+(?:under|pursuant|whereas|whose|which|that)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text[:120].strip(" .,:;()[]")


def _extract_governing_law(text: str) -> str:
    patterns = [
        r"\b([A-Z][A-Za-z .-]{2,60}?)\s+law\s+governs\b",
        r"\bgoverned\s+by\s+(?:the\s+)?laws?\s+of\s+(?:the\s+State\s+of\s+)?"
        r"([A-Z][A-Za-z .-]{2,60})",
        r"\bgoverning\s+law\s*[:;-]\s*([A-Z][A-Za-z .-]{2,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_law_name(match.group(1))
    return ""


def _clean_law_name(value: str) -> str:
    text = _clean_text(value).strip(" .,:;()[]")
    sentence_parts = [part.strip(" .,:;()[]") for part in re.split(r"[.。]", text) if part.strip()]
    if sentence_parts:
        text = sentence_parts[-1]
    text = re.sub(r"^(?:and|the|state\s+of)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+laws?$", "", text, flags=re.IGNORECASE)
    return text.title() if text.islower() else text


def _extract_jurisdiction(text: str, governing_law: str) -> str:
    if governing_law:
        return governing_law
    patterns = [
        r"\bjurisdiction\s+(?:of|in)\s+([A-Z][A-Za-z .-]{2,60})",
        r"\bvenue\s+(?:is\s+)?(?:in|of)\s+([A-Z][A-Za-z .-]{2,60})",
        r"\bcourts?\s+of\s+([A-Z][A-Za-z .-]{2,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_law_name(match.group(1))
    return ""


def _extract_user_side(text: str) -> str:
    patterns = [
        r"\bI\s+represent\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,40})",
        r"\bwe\s+represent\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,40})",
        r"\bon\s+behalf\s+of\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,40})",
        (
            r"\bfor\s+(?:the\s+)?"
            r"(buyer|seller|customer|vendor|supplier|employee|employer|tenant|landlord)\b"
        ),
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            side = _clean_text(match.group(1)).strip(" .,:;")
            return side[:1].upper() + side[1:] if side else ""
    return ""


def _extract_key_dates(text: str, citations: list[str]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
        r"\b\d+\s+(?:business\s+days?|calendar\s+days?|days?|hours?|weeks?|months?)\b",
        re.IGNORECASE,
    )
    dates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in pattern.finditer(text):
        value = match.group(0)
        context = _sentence_around(text, match.start(), match.end())
        key = (value.casefold(), context.casefold())
        if key in seen:
            continue
        seen.add(key)
        dates.append(
            {
                "label": _date_label(context),
                "value": value,
                "description": context,
                "citations": citations[:2],
            }
        )
        if len(dates) >= 8:
            break
    return dates


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), text.rfind(";", 0, start))
    right_candidates = [
        index
        for index in (text.find(".", end), text.find("\n", end), text.find(";", end))
        if index >= 0
    ]
    right = min(right_candidates) if right_candidates else min(len(text), end + 160)
    return _clean_text(text[left + 1 : right + 1])


def _date_label(context: str) -> str:
    lowered = context.casefold()
    if "notice" in lowered:
        return "Notice period"
    if "renew" in lowered:
        return "Renewal date"
    if "terminat" in lowered:
        return "Termination deadline"
    if "pay" in lowered or "invoice" in lowered:
        return "Payment deadline"
    if "effective" in lowered:
        return "Effective date"
    return "Date or deadline"


def _profile_confidence(
    *,
    citations: list[str],
    document_type: str,
    parties: list[str],
    governing_law: str,
) -> str:
    signals = sum(
        [
            bool(citations),
            document_type != "Unknown",
            bool(parties),
            bool(governing_law),
        ]
    )
    if signals >= 4:
        return "High"
    if signals >= 2:
        return "Medium"
    return "Low"


def _matter_open_questions(
    profile: MatterProfile,
    missing_information: list[str],
) -> list[str]:
    questions: list[str] = []
    if not profile.parties:
        questions.append("Confirm the exact parties and their roles in this matter.")
    if not profile.governing_law:
        questions.append("Confirm the governing law or jurisdiction for this matter.")
    if not profile.user_side:
        questions.append("Confirm which side the user represents or wants optimized.")
    if not profile.review_scope:
        questions.append("Confirm the intended review scope.")
    questions.extend(missing_information)
    return _dedupe_texts(questions)[:12]
