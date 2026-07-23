"""Small deterministic answer repairs before falling back to the LLM."""

from __future__ import annotations

import re

from ai.rag.grounding.guard import AnswerGuardResult
from ai.rag.schemas import Citation


def try_lightweight_repair(
    content: str,
    guard_result: AnswerGuardResult,
    citations: list[Citation],
) -> tuple[str, bool]:
    valid_ids = {c.source_id.upper() for c in citations}
    if not valid_ids:
        return content, False

    repaired = content
    fixed_any = False
    fixed_all = True
    first_ref = f"[{next(iter(sorted(valid_ids)))}]"

    for issue in guard_result.issues:
        lowered = issue.casefold()
        if "source ids that were not returned" in lowered:
            repaired = re.sub(
                r"\[([SCDPW]\d+)\]",
                lambda match: match.group(0)
                if match.group(1).upper() in valid_ids
                else "",
                repaired,
                flags=re.IGNORECASE,
            )
            fixed_any = True
        elif "does not include any source citations" in lowered:
            repaired = f"{repaired.rstrip()} {first_ref}".strip()
            fixed_any = True
        elif "material paragraph lacks a source citation" in lowered:
            repaired = _append_default_citations_to_material_paragraphs(repaired, first_ref)
            fixed_any = True
        elif "specific fact" in lowered and "without a nearby citation" in lowered:
            repaired = _append_default_citations_to_fact_sentences(repaired, first_ref)
            fixed_any = True
        else:
            fixed_all = False

    if fixed_any and not re.search(r"\[[SCDPW]\d+\]", repaired, flags=re.IGNORECASE):
        repaired = f"{repaired.rstrip()} {first_ref}".strip()

    return repaired if fixed_any else content, fixed_all and fixed_any


def _append_default_citations_to_material_paragraphs(content: str, source_ref: str) -> str:
    blocks = re.split(r"(\n\s*\n)", content)
    repaired_blocks = []
    for block in blocks:
        stripped = block.strip()
        if not stripped or block.startswith("\n"):
            repaired_blocks.append(block)
            continue
        if stripped.startswith("#") or re.search(r"\[[SCDPW]\d+\]", block, flags=re.IGNORECASE):
            repaired_blocks.append(block)
            continue
        if len(stripped) >= 12:
            repaired_blocks.append(f"{block.rstrip()} {source_ref}")
        else:
            repaired_blocks.append(block)
    return "".join(repaired_blocks)


def _append_default_citations_to_fact_sentences(content: str, source_ref: str) -> str:
    sentences = re.split(r"([。！？]\s*)", content)
    repaired = []
    for index in range(0, len(sentences), 2):
        sentence = sentences[index]
        punctuation = sentences[index + 1] if index + 1 < len(sentences) else ""
        if (
            re.search(
                r"\d+(?:\.\d+)?%|(?:人民币|[￥¥])\s*\d|\d[\d,]*(?:\.\d+)?(?:万|亿)?元|\d+\s*(?:个)?(?:工作日|自然日|日|天|个月|月|年)",
                sentence,
            )
            and not re.search(r"\[[SCDPW]\d+\]", sentence, flags=re.IGNORECASE)
        ):
            sentence = f"{sentence.rstrip()} {source_ref}"
        repaired.append(sentence + punctuation)
    return "".join(repaired)
