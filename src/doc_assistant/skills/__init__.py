"""Safe runtime skill engine: metadata selection, bounded loading, and prompt rendering."""

from __future__ import annotations

from pathlib import Path

from doc_assistant.skills.catalog import SkillCatalog
from doc_assistant.skills.loader import SkillLoader
from doc_assistant.skills.models import SkillLimitError, SkillRuntimeContext
from doc_assistant.skills.renderer import SkillRenderer
from doc_assistant.skills.selector import SkillSelector


class SkillEngine:
    def __init__(
        self,
        root: Path,
        *,
        enabled_names: tuple[str, ...] = (),
        max_catalog_skills: int = 32,
        max_file_bytes: int = 65_536,
        max_reference_files: int = 16,
        max_reference_bytes: int = 131_072,
        max_loaded_tokens: int = 4_000,
        max_selected_skills: int = 4,
    ) -> None:
        self.catalog = SkillCatalog(
            root,
            enabled_names=enabled_names,
            max_skills=max_catalog_skills,
            max_file_bytes=max_file_bytes,
            max_files_per_skill=max_reference_files + 1,
        )
        self.skills = self.catalog.discover()
        self.selector = SkillSelector(self.skills)
        self.loader = SkillLoader(
            root,
            max_reference_files=max_reference_files,
            max_reference_bytes=max_reference_bytes,
            max_total_tokens=max_loaded_tokens,
        )
        self.renderer = SkillRenderer()
        self.max_selected_skills = max_selected_skills
        self.max_loaded_tokens = max_loaded_tokens

    def prepare(
        self,
        task: str,
        *,
        phase: str,
        purpose: str = "",
        required_names: tuple[str, ...] = (),
        selected_names: tuple[str, ...] | None = None,
        references: dict[str, tuple[str, ...]] | None = None,
    ) -> SkillRuntimeContext:
        selection_task = f"{task}\n{purpose}".strip()
        forced = selected_names or ()
        selection = self.selector.select(
            selection_task,
            max_skills=self.max_selected_skills,
            required_names=tuple(dict.fromkeys((*required_names, *forced))),
        )
        if selected_names is not None:
            allowed = set(selected_names)
            selected_metadata = tuple(skill for skill in selection.skills if skill.name in allowed)
        else:
            selected_metadata = selection.skills
        reference_map = references or {}
        loaded = tuple(
            self.loader.load(skill, reference_names=reference_map.get(skill.name, ()))
            for skill in selected_metadata
        )
        total_token_cost = sum(skill.token_cost for skill in loaded)
        if total_token_cost > self.max_loaded_tokens:
            raise SkillLimitError(
                f"Selected skills cost about {total_token_cost} tokens; "
                f"limit is {self.max_loaded_tokens}."
            )
        return SkillRuntimeContext(
            selected_skills=tuple(skill.metadata.name for skill in loaded),
            skill_versions={skill.metadata.name: skill.metadata.version for skill in loaded},
            selection_reason=selection.reason,
            skill_token_cost=total_token_cost,
            rendered_instructions=self.renderer.render(loaded, phase=phase),
        )


def build_skill_engine_from_settings() -> SkillEngine | None:
    """Build and validate the configured catalog when an application service starts."""
    from doc_assistant.config.settings import settings

    if not settings.skills_enabled:
        return None
    return SkillEngine(
        settings.skills_root,
        enabled_names=settings.skills_allowlist,
        max_catalog_skills=settings.skill_max_catalog_size,
        max_file_bytes=settings.skill_max_file_bytes,
        max_reference_files=settings.skill_max_reference_files,
        max_reference_bytes=settings.skill_max_reference_bytes,
        max_loaded_tokens=settings.skill_max_loaded_tokens,
        max_selected_skills=settings.skill_max_selected,
    )


__all__ = ["SkillEngine", "SkillRuntimeContext", "build_skill_engine_from_settings"]
