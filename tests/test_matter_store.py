from __future__ import annotations

from doc_assistant.matter.export import render_artifact_pdf
from doc_assistant.matter.store import MatterStore


def test_matter_store_upserts_profile_and_artifact_versions(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    result = {
        "task_id": "task-1",
        "objective": "Review termination.",
        "matter_profile": {
            "matter_id": "matter-1",
            "document_type": "SaaS agreement",
            "parties": ["VendorCo", "CustomerCo"],
            "open_questions": [],
        },
        "artifacts": [
            {
                "artifact_id": "risk_matrix",
                "artifact_type": "risk_matrix",
                "title": "Risk matrix",
                "summary": "Structured risks.",
                "items": [{"item_id": "risk-1", "category": "termination"}],
                "source_finding_ids": ["f1"],
                "citations": ["S1"],
                "metadata": {},
            }
        ],
    }

    first = store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result=result,
    )
    second = store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={**result, "task_id": "task-2"},
    )

    loaded = store.get("matter-1", "tenant-a", "user-a", include_artifacts=True)
    assert loaded is not None
    assert first.matter_id == "matter-1"
    assert second.latest_task_id == "task-2"
    assert loaded.title == "SaaS agreement"
    assert loaded.matter_profile["parties"] == ["VendorCo", "CustomerCo"]
    assert loaded.artifacts
    assert loaded.artifacts[0].artifact_type == "risk_matrix"
    assert loaded.artifacts[0].version == 2
    assert loaded.artifacts[0].items[0]["category"] == "termination"

    events = store.list_events("matter-1", "tenant-a", "user-a")
    assert events is not None
    assert {event.event_type for event in events} >= {
        "matter_profile_upserted",
        "artifact_upserted",
    }


def test_matter_store_updates_artifact_with_new_version_and_event(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {"matter_id": "matter-1", "open_questions": []},
            "artifacts": [
                {
                    "artifact_id": "risk_matrix",
                    "artifact_type": "risk_matrix",
                    "title": "Risk matrix",
                    "summary": "Original risks.",
                    "items": [{"item_id": "risk-1", "category": "payment"}],
                    "source_finding_ids": ["f1"],
                    "citations": ["S1"],
                    "metadata": {"source": "agent"},
                }
            ],
        },
    )

    updated = store.update_artifact(
        matter_id="matter-1",
        tenant_id="tenant-a",
        user_id="user-a",
        artifact_id="risk_matrix",
        title="Reviewed risk matrix",
        summary="Reviewed risks.",
        items=[{"item_id": "risk-1", "category": "payment", "status": "approved"}],
        status="approved",
        note="Approved for client discussion.",
        updated_by="lawyer-1",
    )

    assert updated is not None
    assert updated.artifacts
    artifact = updated.artifacts[0]
    assert artifact.title == "Reviewed risk matrix"
    assert artifact.summary == "Reviewed risks."
    assert artifact.version == 2
    assert artifact.status == "approved"
    assert artifact.items[0]["status"] == "approved"
    assert artifact.source_finding_ids == ["f1"]
    assert artifact.citations == ["S1"]
    assert artifact.metadata["source"] == "agent"
    assert artifact.metadata["last_edit"]["note"] == "Approved for client discussion."
    assert artifact.metadata["last_edit"]["updated_by"] == "lawyer-1"

    events = store.list_events("matter-1", "tenant-a", "user-a")
    assert events is not None
    assert any(
        event.event_type == "artifact_updated"
        and event.entity_id == "risk_matrix"
        and event.actor == "lawyer-1"
        for event in events
    )


def test_matter_store_isolates_tenants_and_users(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={"task_id": "task-1", "matter_profile": {"matter_id": "matter-1"}},
    )

    assert store.get("matter-1", "tenant-a", "user-a") is not None
    assert store.get("matter-1", "tenant-b", "user-a") is None
    assert store.get("matter-1", "tenant-a", "user-b") is None


def test_matter_store_updates_confirmation_gate_decision(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {
                "matter_id": "matter-1",
                "document_type": "SaaS agreement",
                "open_questions": [],
                "confirmation_gates": [
                    {
                        "gate_id": "approve_report_use",
                        "gate_type": "delivery",
                        "title": "Approve report use",
                        "question": "Approve before external reliance.",
                        "status": "pending",
                        "priority": "high",
                        "required": True,
                        "metadata": {},
                    }
                ],
            },
        },
    )

    loaded = store.get("matter-1", "tenant-a", "user-a")
    assert loaded is not None
    assert loaded.status == "needs_input"

    updated = store.update_confirmation_gate(
        matter_id="matter-1",
        tenant_id="tenant-a",
        user_id="user-a",
        gate_id="approve_report_use",
        status="approved",
        note="Reviewed by legal.",
        decided_by="lawyer-1",
    )

    assert updated is not None
    assert updated.status == "active"
    gate = updated.matter_profile["confirmation_gates"][0]
    assert gate["status"] == "approved"
    assert gate["decided_by"] == "lawyer-1"
    assert gate["metadata"]["last_decision"]["note"] == "Reviewed by legal."


def test_matter_store_writes_confirmed_gate_value_to_profile(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {
                "matter_id": "matter-1",
                "document_type": "SaaS agreement",
                "user_side": "",
                "open_questions": [],
                "confirmation_gates": [
                    {
                        "gate_id": "confirm_user_side",
                        "gate_type": "matter_fact",
                        "title": "Confirm represented side",
                        "question": "Confirm side.",
                        "status": "pending",
                        "required": True,
                        "metadata": {"profile_field": "user_side"},
                    }
                ],
            },
        },
    )

    updated = store.update_confirmation_gate(
        matter_id="matter-1",
        tenant_id="tenant-a",
        user_id="user-a",
        gate_id="confirm_user_side",
        status="approved",
        note="Confirmed by client.",
        confirmed_value="Customer",
        decided_by="lawyer-1",
    )

    assert updated is not None
    assert updated.matter_profile["user_side"] == "Customer"
    assert updated.matter_profile["confirmed_facts"][0]["field"] == "user_side"
    assert updated.matter_profile["confirmed_facts"][0]["value"] == "Customer"


def test_matter_store_persists_and_reviews_findings(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {
                "matter_id": "matter-1",
                "open_questions": [],
                "confirmation_gates": [
                    {
                        "gate_id": "review_high_risk_findings",
                        "status": "pending",
                        "required": True,
                        "related_finding_ids": ["f1"],
                    }
                ],
            },
            "findings": [
                {
                    "finding_id": "f1",
                    "category": "termination",
                    "severity": "High",
                    "summary": "Termination right is one-sided.",
                    "recommended_action": "Seek mutual termination rights.",
                    "citations": ["S1"],
                    "source_step_id": "review_1",
                    "evidence_coverage": "direct",
                    "support_level": "direct",
                    "source_quote": "Customer may terminate on 30 days notice.",
                    "location_label": "page 2, chunk 4",
                    "needs_human_review": True,
                    "human_review_status": "pending",
                }
            ],
        },
    )

    loaded = store.get(
        "matter-1",
        "tenant-a",
        "user-a",
        include_artifacts=True,
        include_findings=True,
    )
    assert loaded is not None
    assert loaded.findings
    assert loaded.findings[0].finding_id == "f1"
    assert loaded.findings[0].human_review_status == "pending"

    updated = store.update_confirmation_gate(
        matter_id="matter-1",
        tenant_id="tenant-a",
        user_id="user-a",
        gate_id="review_high_risk_findings",
        status="approved",
        decided_by="lawyer-1",
    )

    assert updated is not None
    assert updated.findings
    assert updated.findings[0].human_review_status == "approved"
    assert updated.findings[0].status == "resolved"


def test_matter_store_creates_formal_report_after_gates_are_resolved(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {
                "matter_id": "matter-1",
                "document_type": "SaaS agreement",
                "open_questions": [],
                "confirmation_gates": [
                    {
                        "gate_id": "approve_report_use",
                        "gate_type": "delivery",
                        "title": "Approve report use",
                        "question": "Approve before external reliance.",
                        "status": "approved",
                        "priority": "high",
                        "required": True,
                        "citations": ["S1"],
                        "metadata": {},
                    }
                ],
            },
            "artifacts": [
                {
                    "artifact_id": "risk_matrix",
                    "artifact_type": "risk_matrix",
                    "title": "Risk matrix",
                    "summary": "Structured risks.",
                    "items": [],
                    "source_finding_ids": [],
                    "citations": ["S1"],
                    "metadata": {},
                }
            ],
        },
    )

    first = store.create_formal_report_artifact(
        matter_id="matter-1",
        tenant_id="tenant-a",
        user_id="user-a",
        requested_by="lawyer-1",
        note="Ready for delivery.",
    )
    second = store.create_formal_report_artifact(
        matter_id="matter-1",
        tenant_id="tenant-a",
        user_id="user-a",
        requested_by="lawyer-1",
    )

    assert first is not None
    assert second is not None
    formal_reports = [
        artifact for artifact in second.artifacts or [] if artifact.artifact_id == "formal_report"
    ]
    assert len(formal_reports) == 1
    formal_report = formal_reports[0]
    assert formal_report.version == 2
    assert formal_report.status == "approved"
    assert formal_report.items[0]["generated_by"] == "lawyer-1"
    assert formal_report.items[0]["source_artifact_ids"] == ["risk_matrix"]
    assert formal_report.metadata["note"] == ""

    pdf = render_artifact_pdf(matter=second, artifact=formal_report)
    assert pdf.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf


def test_matter_store_rejects_formal_report_with_unresolved_gates(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {
                "matter_id": "matter-1",
                "open_questions": [],
                "confirmation_gates": [
                    {
                        "gate_id": "approve_report_use",
                        "status": "pending",
                        "required": True,
                    }
                ],
            },
        },
    )

    try:
        store.create_formal_report_artifact(
            matter_id="matter-1",
            tenant_id="tenant-a",
            user_id="user-a",
        )
    except ValueError as exc:
        assert "approve_report_use" in str(exc)
    else:
        raise AssertionError("Expected unresolved gate to block formal report creation.")


def test_matter_store_rejects_formal_report_with_unreviewed_finding(tmp_path) -> None:
    store = MatterStore(tmp_path / "matters.sqlite3")
    store.upsert_from_agent_result(
        tenant_id="tenant-a",
        user_id="user-a",
        matter_id="matter-1",
        result={
            "task_id": "task-1",
            "matter_profile": {
                "matter_id": "matter-1",
                "open_questions": [],
                "confirmation_gates": [],
            },
            "findings": [
                {
                    "finding_id": "f1",
                    "category": "payment",
                    "severity": "Medium",
                    "summary": "Payment timing is unclear.",
                    "citations": ["S1"],
                    "evidence_coverage": "direct",
                    "support_level": "direct",
                    "source_quote": "Payment is due after invoice approval.",
                    "location_label": "page 1, chunk 2",
                    "needs_human_review": True,
                    "human_review_status": "pending",
                }
            ],
        },
    )

    try:
        store.create_formal_report_artifact(
            matter_id="matter-1",
            tenant_id="tenant-a",
            user_id="user-a",
        )
    except ValueError as exc:
        assert "human review status" in str(exc)
    else:
        raise AssertionError("Expected unreviewed finding to block formal report creation.")
