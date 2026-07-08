"""Runtime skill metadata and immutable execution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class SkillError(ValueError):
    """Base error for invalid or unsafe runtime skills."""


class SkillValidationError(SkillError):
    """Raised when a skill does not satisfy the runtime contract."""


class SkillLimitError(SkillError):
    """Raised when a skill exceeds a configured resource limit."""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path
    version: str
    file_size: int


@dataclass(frozen=True)
class LoadedSkill:
    metadata: SkillMetadata
    instructions: str
    references: dict[str, str] = field(default_factory=dict)
    token_cost: int = 0


@dataclass(frozen=True)
class SkillSelection:
    skills: tuple[SkillMetadata, ...]
    reason: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillRuntimeContext:
    selected_skills: tuple[str, ...] = ()
    skill_versions: dict[str, str] = field(default_factory=dict)
    selection_reason: str = "Skill runtime disabled or no skill matched."
    skill_token_cost: int = 0
    rendered_instructions: str = ""

    def metadata_payload(self) -> dict[str, object]:
        return {
            "selected_skills": list(self.selected_skills),
            "skill_versions": dict(self.skill_versions),
            "selection_reason": self.selection_reason,
            "skill_token_cost": self.skill_token_cost,
        }


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    sufficient: bool
    status: str
    reasons: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "sufficient": self.sufficient,
            "status": self.status,
            "reasons": list(self.reasons),
            "missing_information": list(self.missing_information),
            "conflicts": list(self.conflicts),
        }

