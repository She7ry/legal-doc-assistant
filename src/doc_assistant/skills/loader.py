"""Bounded, read-only loading for selected skill instructions and references."""

from __future__ import annotations

import re
from pathlib import Path

from doc_assistant.skills.catalog import parse_skill_document
from doc_assistant.skills.models import (
    LoadedSkill,
    SkillLimitError,
    SkillMetadata,
    SkillValidationError,
)

_PROHIBITED_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|system|developer)\s+instructions\b",
        r"\b(?:reveal|print|expose)\b.{0,30}\bsystem\s+prompt\b",
        r"\b(?:disable|bypass|override)\b.{0,40}\b(?:security|tenant|authorization)\b",
        r"</?runtime_skills\b",
        r"</?skill\b",
    )
)


def estimate_tokens(text: str) -> int:
    """Return a conservative, dependency-free prompt token estimate."""
    return max(len(text), (len(text.encode("utf-8")) + 3) // 4)


def _validate_instruction_safety(text: str, *, label: str) -> None:
    for pattern in _PROHIBITED_INSTRUCTION_PATTERNS:
        if pattern.search(text):
            raise SkillValidationError(f"Skill content contains a prohibited directive: {label}")


class SkillLoader:
    """Load only catalog-approved files; never execute skill-provided content."""

    def __init__(
        self,
        root: Path,
        *,
        max_reference_files: int = 16,
        max_reference_bytes: int = 131_072,
        max_total_tokens: int = 4_000,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.max_reference_files = max_reference_files
        self.max_reference_bytes = max_reference_bytes
        self.max_total_tokens = max_total_tokens

    def load(
        self,
        metadata: SkillMetadata,
        *,
        reference_names: tuple[str, ...] = (),
    ) -> LoadedSkill:
        skill_root = metadata.path.resolve(strict=True)
        if metadata.path.is_symlink() or not skill_root.is_relative_to(self.root):
            raise SkillValidationError(f"Skill path is outside the catalog: {metadata.name}")
        skill_file = skill_root / "SKILL.md"
        if skill_file.is_symlink():
            raise SkillValidationError(f"SKILL.md must not be a symlink: {metadata.name}")
        _, instructions = parse_skill_document(skill_file.read_text(encoding="utf-8"))
        _validate_instruction_safety(instructions, label=f"{metadata.name}/SKILL.md")

        if len(reference_names) > self.max_reference_files:
            raise SkillLimitError("Too many skill references requested.")
        references: dict[str, str] = {}
        total_reference_bytes = 0
        references_root = skill_root / "references"
        for name in reference_names:
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix.lower() != ".md":
                raise SkillValidationError(f"Invalid skill reference path: {name!r}")
            path = references_root / candidate
            if path.is_symlink() or not path.is_file():
                raise SkillValidationError(f"Skill reference does not exist or is unsafe: {name!r}")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(references_root.resolve(strict=True)):
                raise SkillValidationError(f"Skill reference escapes its directory: {name!r}")
            total_reference_bytes += resolved.stat().st_size
            if total_reference_bytes > self.max_reference_bytes:
                raise SkillLimitError("Selected skill references exceed the byte limit.")
            content = resolved.read_text(encoding="utf-8")
            _validate_instruction_safety(content, label=f"{metadata.name}/references/{name}")
            references[candidate.as_posix()] = content.strip()

        combined = instructions + "\n" + "\n".join(references.values())
        token_cost = estimate_tokens(combined)
        if token_cost > self.max_total_tokens:
            raise SkillLimitError(
                f"Loaded skill {metadata.name!r} costs about {token_cost} tokens; "
                f"limit is {self.max_total_tokens}."
            )
        return LoadedSkill(
            metadata=metadata,
            instructions=instructions,
            references=references,
            token_cost=token_cost,
        )

