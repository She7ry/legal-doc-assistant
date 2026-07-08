"""Pure matter-domain transitions and synchronization rules."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from doc_assistant.matter._utils import (
    _as_dict,
    _as_dict_list,
    _as_text_list,
    _clean_text,
    _dedupe_texts,
)
from doc_assistant.matter.schemas import MatterArtifactRecord, MatterFindingRecord


def _matter_title(result: dict[str, Any], profile: dict[str, Any]) -> str:
    document_type = _clean_text(profile.get("document_type"))
    objective = _clean_text(result.get("objective"))
    if document_type and document_type != "Unknown":
        return document_type
    return objective[:120] or _clean_text(profile.get("matter_id")) or "Untitled matter"


def _matter_status(profile: dict[str, Any]) -> str:
    if _as_text_list(profile.get("open_questions")) or _unresolved_required_gate_ids(profile):
        return "needs_input"
    return "active"


def _merge_agent_profile(
    incoming: dict[str, Any],
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge generated profile data without discarding explicit human decisions."""
    merged = deepcopy(incoming)
    if not existing:
        return merged

    old_gates = {
        _clean_text(gate.get("gate_id")): gate
        for gate in _as_dict_list(existing.get("confirmation_gates"))
        if _clean_text(gate.get("gate_id"))
    }
    merged_gates: list[dict[str, Any]] = []
    for generated_gate in _as_dict_list(merged.get("confirmation_gates")):
        gate = deepcopy(generated_gate)
        old_gate = old_gates.get(_clean_text(gate.get("gate_id")))
        old_metadata = _as_dict(old_gate.get("metadata")) if old_gate else {}
        if old_gate and old_metadata.get("last_decision"):
            gate["status"] = old_gate.get("status", gate.get("status", "pending"))
            gate["metadata"] = deepcopy(old_metadata)
            for key in ("updated_at", "decided_by", "decided_at"):
                if key in old_gate:
                    gate[key] = old_gate[key]
                else:
                    gate.pop(key, None)
        merged_gates.append(gate)
    merged["confirmation_gates"] = merged_gates

    confirmed_facts = _merge_confirmed_facts(
        _as_dict_list(existing.get("confirmed_facts")),
        _as_dict_list(merged.get("confirmed_facts")),
    )
    if confirmed_facts:
        merged["confirmed_facts"] = confirmed_facts
        for fact in confirmed_facts:
            field = _clean_text(fact.get("field"))
            value = _clean_text(fact.get("value"))
            if field in {"user_side", "governing_law", "jurisdiction", "document_type"} and value:
                merged[field] = value
    return merged


def _merge_agent_artifact(
    incoming: dict[str, Any],
    existing: MatterArtifactRecord | None,
) -> dict[str, Any]:
    """Keep counsel-edited artifact fields authoritative across generated refreshes."""
    merged = deepcopy(incoming)
    if existing is None or not existing.metadata.get("last_edit"):
        return merged
    merged.update(
        title=existing.title,
        summary=existing.summary,
        items=deepcopy(existing.items),
        status=existing.status,
    )
    merged["metadata"] = {
        **_as_dict(merged.get("metadata")),
        **deepcopy(existing.metadata),
    }
    return merged


def _merge_finding_metadata(
    incoming: dict[str, Any],
    existing: MatterFindingRecord | None,
) -> dict[str, Any]:
    metadata = deepcopy(_as_dict(incoming.get("metadata")))
    if incoming.get("evidence"):
        metadata["evidence"] = deepcopy(incoming["evidence"])
    if existing is None:
        return metadata
    for key in ("decisions", "last_decision"):
        if key in existing.metadata:
            metadata[key] = deepcopy(existing.metadata[key])
    return metadata


def _apply_confirmation_gate_decision(
    profile: dict[str, Any],
    *,
    gate_id: str,
    status: str,
    note: str | None,
    confirmed_value: str | None,
    decided_by: str,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str], dict[str, Any]]:
    updated_profile = deepcopy(profile)
    gates = _as_dict_list(updated_profile.get("confirmation_gates"))
    gate = next(
        (item for item in gates if _clean_text(item.get("gate_id")) == gate_id),
        None,
    )
    if gate is None:
        raise KeyError(gate_id)
    old_gate = deepcopy(gate)
    decision = {
        "status": status,
        "note": _clean_text(note),
        "decided_by": decided_by,
        "decided_at": now.isoformat(),
    }
    metadata = deepcopy(_as_dict(gate.get("metadata")))
    decisions = _as_dict_list(metadata.get("decisions"))
    metadata["decisions"] = [*decisions[-19:], decision]
    metadata["last_decision"] = decision
    gate["status"] = status
    gate["metadata"] = metadata
    gate["updated_at"] = now.isoformat()
    if status in {"approved", "waived"}:
        gate["decided_by"] = decided_by
        gate["decided_at"] = decision["decided_at"]
    else:
        gate.pop("decided_by", None)
        gate.pop("decided_at", None)
    updated_profile["confirmation_gates"] = gates
    _apply_gate_profile_value(
        updated_profile,
        gate=gate,
        status=status,
        confirmed_value=confirmed_value,
        decision=decision,
    )
    return (
        updated_profile,
        old_gate,
        deepcopy(gate),
        _as_text_list(gate.get("related_finding_ids")),
        decision,
    )


def _build_artifact_update(
    artifact: MatterArtifactRecord,
    *,
    title: str | None,
    summary: str | None,
    items: list[dict[str, Any]] | None,
    status: str | None,
    note: str | None,
    actor: str,
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    next_title = _clean_text(title) if title is not None else artifact.title
    next_summary = _clean_text(summary) if summary is not None else artifact.summary
    next_items = _as_dict_list(items) if items is not None else artifact.items
    next_status = _clean_text(status) or artifact.status
    next_version = artifact.version + 1
    metadata = deepcopy(artifact.metadata)
    edit = {
        "status": next_status,
        "note": _clean_text(note),
        "updated_by": actor,
        "updated_at": now.isoformat(),
        "version": next_version,
    }
    edits = _as_dict_list(metadata.get("edits"))
    metadata["edits"] = [*edits[-19:], edit]
    metadata["last_edit"] = edit
    values = {
        "title": next_title or artifact.title,
        "summary": next_summary,
        "items": deepcopy(next_items),
        "metadata": metadata,
        "version": next_version,
        "status": next_status,
    }
    old_value = _artifact_snapshot(artifact)
    new_value = {**values, "note": _clean_text(note)}
    return values, old_value, new_value


def _formal_report_blockers(
    profile: dict[str, Any],
    findings: list[MatterFindingRecord] | None = None,
) -> list[str]:
    blockers: list[str] = []
    unresolved_gate_ids = _unresolved_required_gate_ids(profile)
    if unresolved_gate_ids:
        blockers.append(
            "Matter still has unresolved required confirmation gates: "
            + ", ".join(unresolved_gate_ids)
        )
    for finding in findings or []:
        missing = _formal_finding_missing_fields(finding)
        if missing:
            blockers.append(
                f"Finding {finding.finding_id} is not formal-report ready: " + ", ".join(missing)
            )
    return blockers


def _build_formal_report_artifact(
    *,
    matter_id: str,
    profile: dict[str, Any],
    findings: list[MatterFindingRecord],
    artifacts: list[MatterArtifactRecord],
    source_task_id: str,
    generated_by: str,
    generated_at: str | None,
    note: str | None,
) -> dict[str, Any]:
    gates = _as_dict_list(profile.get("confirmation_gates"))
    source_artifact_ids = [item.artifact_id for item in artifacts if item.artifact_id != "formal_report"]
    return {
        "artifact_id": "formal_report",
        "artifact_type": "formal_report",
        "title": "Formal report record",
        "summary": (
            "Confirmation gates were resolved and the current matter artifacts "
            "were approved for formal use."
        ),
        "items": [
            {
                "item_id": "formal-report-1",
                "matter_id": matter_id,
                "document_type": _clean_text(profile.get("document_type")) or "Unknown",
                "status": "approved_for_formal_use",
                "generated_by": generated_by,
                "generated_at": generated_at,
                "source_task_id": source_task_id,
                "source_artifact_ids": source_artifact_ids,
                "finding_count": len(findings),
                "gate_count": len(gates),
                "note": _clean_text(note),
            }
        ],
        "source_finding_ids": [finding.finding_id for finding in findings],
        "citations": _dedupe_texts(
            [source_id for gate in gates for source_id in _as_text_list(gate.get("citations"))]
            + [source_id for finding in findings for source_id in finding.citations]
        ),
        "metadata": {
            "matter_id": matter_id,
            "generated_by": generated_by,
            "generated_at": generated_at,
            "source_task_id": source_task_id,
            "source_artifact_ids": source_artifact_ids,
            "finding_statuses": [
                {
                    "finding_id": finding.finding_id,
                    "status": finding.status,
                    "human_review_status": finding.human_review_status,
                    "support_level": finding.support_level,
                    "evidence_coverage": finding.evidence_coverage,
                }
                for finding in findings
            ],
            "gate_statuses": [
                {
                    "gate_id": _clean_text(gate.get("gate_id")),
                    "status": _clean_text(gate.get("status")) or "pending",
                }
                for gate in gates
            ],
            "note": _clean_text(note),
        },
        "status": "approved",
    }


def _formal_finding_missing_fields(finding: MatterFindingRecord) -> list[str]:
    missing: list[str] = []
    if not finding.citations:
        missing.append("source citation")
    if not finding.source_quote:
        missing.append("exact quote")
    if not finding.location_label:
        missing.append("location")
    if not finding.support_level:
        missing.append("support level")
    if finding.support_level != "direct" and not finding.unsupported_reason:
        missing.append("unsupported reason")
    if finding.needs_human_review and finding.human_review_status not in {
        "approved", "waived", "resolved", "not_required",
    }:
        missing.append("human review status")
    return missing


def _unresolved_required_gate_ids(profile: dict[str, Any]) -> list[str]:
    return [
        _clean_text(gate.get("gate_id")) or "unknown_gate"
        for gate in _as_dict_list(profile.get("confirmation_gates"))
        if gate.get("required", True)
        and _clean_text(gate.get("status")) not in {"approved", "waived"}
    ]


def _apply_gate_profile_value(
    profile: dict[str, Any],
    *,
    gate: dict[str, Any],
    status: str,
    confirmed_value: str | None,
    decision: dict[str, Any],
) -> None:
    if status != "approved":
        return
    profile_field = _clean_text(_as_dict(gate.get("metadata")).get("profile_field"))
    value = _clean_text(confirmed_value)
    if profile_field not in {"user_side", "governing_law", "jurisdiction", "document_type"} or not value:
        return
    profile[profile_field] = value
    confirmed_facts = _as_dict_list(profile.get("confirmed_facts"))
    confirmed_facts.append(
        {
            "field": profile_field,
            "value": value,
            "source": "confirmation_gate",
            "gate_id": _clean_text(gate.get("gate_id")),
            "decided_by": decision.get("decided_by", ""),
            "decided_at": decision.get("decided_at", ""),
        }
    )
    profile["confirmed_facts"] = confirmed_facts[-50:]


def _human_review_status_for_gate_status(status: str) -> str:
    return {"approved": "approved", "waived": "waived", "needs_info": "needs_info"}.get(
        status, "pending",
    )


def _finding_status(
    *, needs_human_review: bool, human_review_status: str, evidence_coverage: str,
) -> str:
    if human_review_status == "needs_info":
        return "needs_info"
    if needs_human_review and human_review_status not in {
        "approved", "waived", "resolved", "not_required",
    }:
        return "needs_human_review"
    if evidence_coverage == "direct":
        return "resolved"
    if human_review_status in {"approved", "waived", "resolved"}:
        return "resolved_with_evidence_gap"
    return "open"


def _merge_confirmed_facts(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    positions: dict[tuple[str, str], int] = {}
    for fact in [*incoming, *existing]:
        key = (_clean_text(fact.get("field")), _clean_text(fact.get("gate_id")))
        if not key[0]:
            continue
        if key in positions:
            result[positions[key]] = deepcopy(fact)
        else:
            positions[key] = len(result)
            result.append(deepcopy(fact))
    return result[-50:]


def _artifact_snapshot(artifact: MatterArtifactRecord) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "title": artifact.title,
        "summary": artifact.summary,
        "items": deepcopy(artifact.items),
        "source_finding_ids": list(artifact.source_finding_ids),
        "citations": list(artifact.citations),
        "metadata": deepcopy(artifact.metadata),
        "source_task_id": artifact.source_task_id,
        "version": artifact.version,
        "status": artifact.status,
    }
