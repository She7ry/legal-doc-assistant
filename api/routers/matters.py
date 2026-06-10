from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from api.dependencies import MatterStoreDep, TenantIdDep, UserIdDep, require_api_key
from api.schemas.requests import (
    MatterConfirmationGateUpdateRequest,
    MatterFindingUpdateRequest,
    MatterFormalReportCreateRequest,
)
from api.schemas.responses import (
    MatterArtifactRecordOut,
    MatterFindingRecordOut,
    MatterListResponse,
    MatterRecordOut,
)
from doc_assistant.matter.export import (
    artifact_bundle_filename,
    artifact_docx_filename,
    artifact_markdown_filename,
    render_artifacts_zip,
    render_artifact_docx,
    render_artifact_markdown,
)

router = APIRouter(prefix="/matters", tags=["matters"], dependencies=[Depends(require_api_key)])


@router.get(
    "",
    response_model=MatterListResponse,
    summary="List persisted legal matters for the current user",
)
def list_matters(
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> MatterListResponse:
    matters = matter_store.list(tenant_id, user_id, limit=limit)
    return MatterListResponse(
        matters=[MatterRecordOut.from_record(matter) for matter in matters],
        total=len(matters),
    )


@router.get(
    "/{matter_id}",
    response_model=MatterRecordOut,
    summary="Get a persisted matter profile and generated artifacts",
)
def get_matter(
    matter_id: str,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MatterRecordOut:
    matter = matter_store.get(
        matter_id,
        tenant_id,
        user_id,
        include_artifacts=True,
        include_findings=True,
    )
    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")
    return MatterRecordOut.from_record(matter)


@router.patch(
    "/{matter_id}/confirmation-gates/{gate_id}",
    response_model=MatterRecordOut,
    summary="Update a persisted matter confirmation gate decision",
)
def update_matter_confirmation_gate(
    matter_id: str,
    gate_id: str,
    body: MatterConfirmationGateUpdateRequest,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MatterRecordOut:
    try:
        matter = matter_store.update_confirmation_gate(
            matter_id=matter_id,
            tenant_id=tenant_id,
            user_id=user_id,
            gate_id=gate_id,
            status=body.status,
            note=body.note,
            confirmed_value=body.confirmed_value,
            decided_by=user_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Confirmation gate not found.",
        ) from exc

    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")
    return MatterRecordOut.from_record(matter)


@router.post(
    "/{matter_id}/formal-report",
    response_model=MatterRecordOut,
    summary="Create a versioned formal report artifact after confirmation gates are resolved",
)
def create_matter_formal_report(
    matter_id: str,
    body: MatterFormalReportCreateRequest,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MatterRecordOut:
    try:
        matter = matter_store.create_formal_report_artifact(
            matter_id=matter_id,
            tenant_id=tenant_id,
            user_id=user_id,
            requested_by=user_id,
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")
    return MatterRecordOut.from_record(matter)


@router.get(
    "/{matter_id}/artifacts",
    response_model=list[MatterArtifactRecordOut],
    summary="List generated artifacts for a persisted matter",
)
def list_matter_artifacts(
    matter_id: str,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> list[MatterArtifactRecordOut]:
    artifacts = matter_store.list_artifacts(matter_id, tenant_id, user_id)
    if artifacts is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")
    return [MatterArtifactRecordOut.from_record(artifact) for artifact in artifacts]


@router.get(
    "/{matter_id}/findings",
    response_model=list[MatterFindingRecordOut],
    summary="List persisted review findings for a matter",
)
def list_matter_findings(
    matter_id: str,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> list[MatterFindingRecordOut]:
    findings = matter_store.list_findings(matter_id, tenant_id, user_id)
    if findings is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")
    return [MatterFindingRecordOut.from_record(finding) for finding in findings]


@router.patch(
    "/{matter_id}/findings/{finding_id}",
    response_model=MatterRecordOut,
    summary="Update human review status for a persisted review finding",
)
def update_matter_finding(
    matter_id: str,
    finding_id: str,
    body: MatterFindingUpdateRequest,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MatterRecordOut:
    try:
        matter = matter_store.update_finding_decision(
            matter_id=matter_id,
            tenant_id=tenant_id,
            user_id=user_id,
            finding_id=finding_id,
            human_review_status=body.human_review_status,
            note=body.note,
            decided_by=user_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Finding not found.",
        ) from exc

    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")
    return MatterRecordOut.from_record(matter)


@router.get(
    "/{matter_id}/artifacts/export",
    summary="Export all generated matter artifacts as a ZIP archive",
)
def export_matter_artifacts(
    matter_id: str,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    format: str = Query(default="docx", pattern="^(markdown|docx|both)$"),
) -> Response:
    matter = matter_store.get(
        matter_id,
        tenant_id,
        user_id,
        include_artifacts=True,
        include_findings=True,
    )
    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")

    artifacts = matter.artifacts or []
    if not artifacts:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Matter has no generated artifacts to export.",
        )

    content = render_artifacts_zip(
        matter=matter,
        artifacts=artifacts,
        export_format=format,
    )
    filename = artifact_bundle_filename(matter_id, format)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{matter_id}/artifacts/{artifact_id}/export",
    summary="Export a generated matter artifact as Markdown",
)
def export_matter_artifact(
    matter_id: str,
    artifact_id: str,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    format: str = Query(default="markdown", pattern="^(markdown|docx)$"),
) -> Response:
    matter = matter_store.get(
        matter_id,
        tenant_id,
        user_id,
        include_artifacts=True,
        include_findings=True,
    )
    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found.")

    artifact = next(
        (
            item
            for item in matter.artifacts or []
            if item.artifact_id == artifact_id
        ),
        None,
    )
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")

    if format == "docx":
        content = render_artifact_docx(matter=matter, artifact=artifact)
        filename = artifact_docx_filename(matter_id, artifact_id, artifact.version)
        return Response(
            content=content,
            media_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    content = render_artifact_markdown(matter=matter, artifact=artifact)
    filename = artifact_markdown_filename(matter_id, artifact_id, artifact.version)
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
