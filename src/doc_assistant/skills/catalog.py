"""Discover and validate portable Markdown skills without loading their bodies."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from doc_assistant.skills.models import (
    SkillLimitError,
    SkillMetadata,
    SkillValidationError,
)

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def parse_skill_document(text: str) -> tuple[dict[str, str], str]:
    """Parse the strict ``name + description`` frontmatter contract."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillValidationError("SKILL.md must start with YAML frontmatter.")
    try:
        closing = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillValidationError("SKILL.md frontmatter is not closed.") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        if not line.strip():
            continue
        if ":" not in line:
            raise SkillValidationError(f"Invalid frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"\'')
        if key in metadata:
            raise SkillValidationError(f"Duplicate frontmatter field: {key}")
        metadata[key] = value

    if set(metadata) != {"name", "description"}:
        raise SkillValidationError("SKILL.md frontmatter must contain only name and description.")
    if not metadata["description"]:
        raise SkillValidationError("Skill description must not be empty.")
    body = "\n".join(lines[closing + 1 :]).strip()
    if not body:
        raise SkillValidationError("Skill instructions must not be empty.")
    return metadata, body


class SkillCatalog:
    """Read skill metadata at startup and reject ambiguous or unsafe layouts."""

    def __init__(
        self,
        root: Path,
        *,
        enabled_names: tuple[str, ...] = (),
        max_skills: int = 32,
        max_file_bytes: int = 65_536,
        max_files_per_skill: int = 17,
    ) -> None:
        self.root = root
        self.enabled_names = frozenset(enabled_names)
        self.max_skills = max_skills
        self.max_file_bytes = max_file_bytes
        self.max_files_per_skill = max_files_per_skill

    def discover(self) -> tuple[SkillMetadata, ...]:
        if not self.root.exists():
            return ()
        if self.root.is_symlink() or not self.root.is_dir():
            raise SkillValidationError("Skill root must be a real directory, not a symlink.")

        root = self.root.resolve(strict=True)
        skill_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if len(skill_dirs) > self.max_skills:
            raise SkillLimitError(
                f"Skill catalog contains {len(skill_dirs)} entries; limit is {self.max_skills}."
            )

        discovered: list[SkillMetadata] = []
        names: set[str] = set()
        for skill_dir in skill_dirs:
            if skill_dir.is_symlink():
                raise SkillValidationError(f"Skill directory must not be a symlink: {skill_dir.name}")
            self._validate_layout(skill_dir, root)
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            if skill_file.is_symlink() or not skill_file.is_file():
                raise SkillValidationError(f"SKILL.md must be a regular file: {skill_dir.name}")
            resolved_file = skill_file.resolve(strict=True)
            if not resolved_file.is_relative_to(root):
                raise SkillValidationError(f"Skill escapes configured root: {skill_dir.name}")
            size = resolved_file.stat().st_size
            if size > self.max_file_bytes:
                raise SkillLimitError(f"Skill file exceeds {self.max_file_bytes} bytes: {skill_dir.name}")

            raw = resolved_file.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SkillValidationError(f"SKILL.md must be UTF-8: {skill_dir.name}") from exc
            fields, _ = parse_skill_document(text)
            name = fields["name"]
            if not _NAME_PATTERN.fullmatch(name) or len(name) > 64:
                raise SkillValidationError(f"Invalid skill name: {name!r}")
            if name != skill_dir.name:
                raise SkillValidationError(
                    f"Skill name {name!r} must match directory {skill_dir.name!r}."
                )
            if name in names:
                raise SkillValidationError(f"Duplicate skill name: {name}")
            names.add(name)
            if self.enabled_names and name not in self.enabled_names:
                continue
            discovered.append(
                SkillMetadata(
                    name=name,
                    description=fields["description"],
                    path=skill_dir,
                    version=hashlib.sha256(raw).hexdigest(),
                    file_size=size,
                )
            )

        missing = self.enabled_names - names
        if missing:
            raise SkillValidationError(
                "Enabled skills were not found: " + ", ".join(sorted(missing))
            )
        return tuple(discovered)

    def _validate_layout(self, skill_dir: Path, root: Path) -> None:
        entries = list(skill_dir.rglob("*"))
        files = [entry for entry in entries if entry.is_file()]
        if len(files) > self.max_files_per_skill:
            raise SkillLimitError(
                f"Skill {skill_dir.name!r} contains {len(files)} files; "
                f"limit is {self.max_files_per_skill}."
            )
        for entry in entries:
            if entry.is_symlink():
                raise SkillValidationError(f"Skill entries must not be symlinks: {entry}")
            resolved = entry.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise SkillValidationError(f"Skill entry escapes configured root: {entry}")
            relative = entry.relative_to(skill_dir)
            if entry.is_dir():
                if relative != Path("references"):
                    raise SkillValidationError(
                        f"Only the references directory is allowed in read-only skills: {relative}"
                    )
                continue
            if relative == Path("SKILL.md"):
                continue
            if (
                len(relative.parts) != 2
                or relative.parts[0] != "references"
                or entry.suffix.lower() != ".md"
            ):
                raise SkillValidationError(
                    f"Only Markdown reference files are allowed in read-only skills: {relative}"
                )
