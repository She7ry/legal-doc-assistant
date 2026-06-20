from __future__ import annotations

from io import BytesIO
import time
from zipfile import ZipFile

from fastapi.testclient import TestClient

from api.agent_tasks import AgentTaskStore
from api import dependencies
from api.main import app
from doc_assistant.matter.store import MatterStore
from doc_assistant.services.agent.schemas import (
    AgentArtifact,
    AgentConfirmationGate,
    AgentTaskResult,
    MatterProfile,
)


class FastAgentService:
    def run_task(self, **kwargs) -> AgentTaskResult:
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(
                event_type="plan_created",
                stage="planning",
                progress=10,
                message="Created a test plan.",
                payload={"plan": []},
            )
        return AgentTaskResult(
            task_id=kwargs["task_id"],
            status="completed",
            objective=kwargs["objective"],
            plan=[],
            steps=[],
            findings=[],
            missing_information=[],
            human_review_required=False,
            report="Test report.",
            citations=[],
            confidence="High",
        )


class RichAgentService:
    def run_task(self, **kwargs) -> AgentTaskResult:
        matter_id = kwargs["matter_id"]
        gate = AgentConfirmationGate(
            gate_id="confirm_user_side",
            gate_type="matter_fact",
            title="Confirm represented side",
            question="Confirm which side the user represents.",
            priority="high",
            reason="The fake task leaves user_side unconfirmed.",
        )
        return AgentTaskResult(
            task_id=kwargs["task_id"],
            status="needs_human_review",
            objective=kwargs["objective"],
            plan=[],
            steps=[],
            findings=[],
            missing_information=[],
            human_review_required=True,
            report="Matter report.",
            citations=[],
            confidence="Medium",
            matter_profile=MatterProfile(
                matter_id=matter_id,
                document_type="SaaS agreement",
                parties=["VendorCo", "CustomerCo"],
                governing_law="New York",
                review_scope=["termination"],
                open_questions=["Confirm user side."],
                confirmation_gates=[
                    {
                        "gate_id": gate.gate_id,
                        "gate_type": gate.gate_type,
                        "title": gate.title,
                        "question": gate.question,
                        "status": gate.status,
                        "priority": gate.priority,
                        "required": gate.required,
                        "reason": gate.reason,
                        "related_finding_ids": gate.related_finding_ids,
                        "related_artifact_ids": gate.related_artifact_ids,
                        "citations": gate.citations,
                        "metadata": gate.metadata,
                    }
                ],
            ),
            artifacts=[
                AgentArtifact(
                    artifact_id="risk_matrix",
                    artifact_type="risk_matrix",
                    title="Risk matrix",
                    summary="Structured risk rows.",
                    items=[{"item_id": "risk-1", "category": "termination"}],
                    source_finding_ids=["f1"],
                    citations=["S1"],
                )
            ],
            confirmation_gates=[gate],
        )


class ExplodingAgentService:
    def run_task(self, **_kwargs) -> AgentTaskResult:
        raise AssertionError("Agent service should not run when clarification is required.")


def _wait_for_agent_task(client: TestClient, task_id: str, *, user_id: str = "api-test-user") -> dict:
    deadline = time.monotonic() + 3
    last_data = None
    while time.monotonic() < deadline:
        loaded = client.get(
            f"/api/v1/agent/tasks/{task_id}",
            headers={"X-User-Id": user_id},
        )
        assert loaded.status_code == 200
        last_data = loaded.json()
        if last_data["status"] in {"succeeded", "failed", "needs_input"}:
            return last_data
        time.sleep(0.02)
    raise AssertionError(f"Agent task did not finish in time: {last_data}")


def test_agent_task_api_creates_gets_and_streams_events(tmp_path) -> None:
    store = AgentTaskStore()
    matter_store = MatterStore(tmp_path / "matters.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_matter_store] = lambda: matter_store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: FastAgentService()
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "Review payment terms.",
                "focus_areas": ["payment"],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        data = _wait_for_agent_task(client, task_id)
        assert data["status"] == "succeeded"
        assert data["result"]["report"] == "Test report."

        events = client.get(
            f"/api/v1/agent/tasks/{task_id}/events",
            headers={"X-User-Id": "api-test-user"},
        )
        assert events.status_code == 200
        assert "event: queued" in events.text
        assert "event: succeeded" in events.text
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_marks_underspecified_tasks_as_needing_input(tmp_path) -> None:
    store = AgentTaskStore()
    matter_store = MatterStore(tmp_path / "matters.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_matter_store] = lambda: matter_store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: ExplodingAgentService()
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "帮我看看",
                "focus_areas": [],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        data = created.json()
        assert data["status"] == "needs_input"
        assert data["stage"] == "needs_input"
        assert data["result"] is None
        assert data["events"][-1]["event_type"] == "needs_input"
        assert data["events"][-1]["payload"]["questions"]

        events = client.get(
            f"/api/v1/agent/tasks/{data['task_id']}/events",
            headers={"X-User-Id": "api-test-user"},
        )
        assert events.status_code == 200
        assert "event: needs_input" in events.text
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_resumes_task_after_supplemental_input(tmp_path) -> None:
    store = AgentTaskStore()
    matter_store = MatterStore(tmp_path / "matters.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_matter_store] = lambda: matter_store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: FastAgentService()
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "review this",
                "focus_areas": [],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        assert created.json()["status"] == "needs_input"

        resumed = client.post(
            f"/api/v1/agent/tasks/{task_id}/resume",
            json={
                "clarification_answers": [
                    (
                        "Review payment and termination risk. "
                        "I represent the customer."
                    )
                ],
                "focus_areas": ["payment", "termination"],
                "user_role": "lawyer",
                "max_steps": 4,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert resumed.status_code == 202
        assert resumed.json()["status"] == "queued"
        assert resumed.json()["events"][-2]["event_type"] == "input_received"

        data = _wait_for_agent_task(client, task_id)
        assert data["status"] == "succeeded"
        assert data["user_role"] == "lawyer"
        assert data["focus_areas"] == ["payment", "termination"]
        assert data["result"]["objective"].startswith("review this")
        assert "Review payment and termination risk" in data["result"]["objective"]
        event_types = [event["event_type"] for event in data["events"]]
        assert "input_received" in event_types
        assert event_types[-4:] == ["queued", "running", "plan_created", "succeeded"]
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_rejects_resume_for_non_needs_input_task(tmp_path) -> None:
    store = AgentTaskStore()
    matter_store = MatterStore(tmp_path / "matters.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_matter_store] = lambda: matter_store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: FastAgentService()
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "Review payment terms.",
                "focus_areas": ["payment"],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        task_id = created.json()["task_id"]

        rejected = client.post(
            f"/api/v1/agent/tasks/{task_id}/resume",
            json={"clarification_answers": ["I represent the customer."]},
            headers={"X-User-Id": "api-test-user"},
        )

        assert rejected.status_code == 409
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_persists_matter_profile_and_artifacts(tmp_path) -> None:
    store = AgentTaskStore()
    matter_store = MatterStore(tmp_path / "matters.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_matter_store] = lambda: matter_store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: RichAgentService()
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "Review termination risk.",
                "focus_areas": ["termination"],
                "user_role": "lawyer",
                "max_steps": 3,
                "matter_id": "matter-saas-1",
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        loaded_data = _wait_for_agent_task(client, task_id)
        assert loaded_data["status"] == "succeeded"
        assert loaded_data["matter_id"] == "matter-saas-1"

        matter = client.get(
            "/api/v1/matters/matter-saas-1",
            headers={"X-User-Id": "api-test-user"},
        )
        assert matter.status_code == 200
        data = matter.json()
        assert data["matter_id"] == "matter-saas-1"
        assert data["matter_profile"]["document_type"] == "SaaS agreement"
        assert data["matter_profile"]["governing_law"] == "New York"
        assert data["matter_profile"]["confirmation_gates"][0]["gate_id"] == "confirm_user_side"
        assert data["artifacts"][0]["artifact_type"] == "risk_matrix"
        assert data["artifacts"][0]["version"] == 1

        exported = client.get(
            "/api/v1/matters/matter-saas-1/artifacts/risk_matrix/export",
            headers={"X-User-Id": "api-test-user"},
        )
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("text/markdown")
        assert "matter-saas-1-risk_matrix-v1.md" in exported.headers["content-disposition"]
        assert "# Risk matrix" in exported.text
        assert "termination" in exported.text

        exported_docx = client.get(
            "/api/v1/matters/matter-saas-1/artifacts/risk_matrix/export?format=docx",
            headers={"X-User-Id": "api-test-user"},
        )
        assert exported_docx.status_code == 200
        assert exported_docx.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert "matter-saas-1-risk_matrix-v1.docx" in exported_docx.headers[
            "content-disposition"
        ]
        assert exported_docx.content.startswith(b"PK")
        with ZipFile(BytesIO(exported_docx.content)) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")
        assert "Risk matrix" in document_xml
        assert "termination" in document_xml

        exported_pdf = client.get(
            "/api/v1/matters/matter-saas-1/artifacts/risk_matrix/export?format=pdf",
            headers={"X-User-Id": "api-test-user"},
        )
        assert exported_pdf.status_code == 200
        assert exported_pdf.headers["content-type"].startswith("application/pdf")
        assert "matter-saas-1-risk_matrix-v1.pdf" in exported_pdf.headers[
            "content-disposition"
        ]
        assert exported_pdf.content.startswith(b"%PDF-1.4")

        exported_zip = client.get(
            "/api/v1/matters/matter-saas-1/artifacts/export?format=docx",
            headers={"X-User-Id": "api-test-user"},
        )
        assert exported_zip.status_code == 200
        assert exported_zip.headers["content-type"].startswith("application/zip")
        assert "matter-saas-1-artifacts-docx.zip" in exported_zip.headers[
            "content-disposition"
        ]
        with ZipFile(BytesIO(exported_zip.content)) as archive:
            names = archive.namelist()
            assert "manifest.json" in names
            assert "matter-saas-1-risk_matrix-v1.docx" in names
            bundled_docx = archive.read("matter-saas-1-risk_matrix-v1.docx")
        assert bundled_docx.startswith(b"PK")

        exported_both_zip = client.get(
            "/api/v1/matters/matter-saas-1/artifacts/export?format=both",
            headers={"X-User-Id": "api-test-user"},
        )
        assert exported_both_zip.status_code == 200
        with ZipFile(BytesIO(exported_both_zip.content)) as archive:
            names = archive.namelist()
        assert "markdown/matter-saas-1-risk_matrix-v1.md" in names
        assert "docx/matter-saas-1-risk_matrix-v1.docx" in names

        missing_export = client.get(
            "/api/v1/matters/matter-saas-1/artifacts/not-an-artifact/export",
            headers={"X-User-Id": "api-test-user"},
        )
        assert missing_export.status_code == 404

        blocked_report = client.post(
            "/api/v1/matters/matter-saas-1/formal-report",
            json={"note": "Ready."},
            headers={"X-User-Id": "api-test-user"},
        )
        assert blocked_report.status_code == 409

        updated = client.patch(
            "/api/v1/matters/matter-saas-1/confirmation-gates/confirm_user_side",
            json={"status": "waived", "note": "Business accepted this gap."},
            headers={"X-User-Id": "api-test-user"},
        )
        assert updated.status_code == 200
        updated_gate = updated.json()["matter_profile"]["confirmation_gates"][0]
        assert updated_gate["status"] == "waived"
        assert updated_gate["metadata"]["last_decision"]["note"] == "Business accepted this gap."

        missing_gate = client.patch(
            "/api/v1/matters/matter-saas-1/confirmation-gates/not-a-gate",
            json={"status": "approved"},
            headers={"X-User-Id": "api-test-user"},
        )
        assert missing_gate.status_code == 404

        formal_report = client.post(
            "/api/v1/matters/matter-saas-1/formal-report",
            json={"note": "Ready for formal use."},
            headers={"X-User-Id": "api-test-user"},
        )
        assert formal_report.status_code == 200
        formal_artifacts = [
            artifact
            for artifact in formal_report.json()["artifacts"]
            if artifact["artifact_id"] == "formal_report"
        ]
        assert len(formal_artifacts) == 1
        assert formal_artifacts[0]["version"] == 1
        assert formal_artifacts[0]["metadata"]["note"] == "Ready for formal use."

        listed = client.get(
            "/api/v1/matters",
            headers={"X-User-Id": "api-test-user"},
        )
        assert listed.status_code == 200
        assert listed.json()["total"] == 1
    finally:
        app.dependency_overrides.clear()
