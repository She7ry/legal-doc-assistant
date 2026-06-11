import { apiRequest } from "./http";
import type {
  MatterArtifactRecord,
  MatterConfirmationGateUpdateRequest,
  MatterFindingRecord,
  MatterFindingUpdateRequest,
  MatterFormalReportCreateRequest,
  MatterListResponse,
  MatterRecord,
} from "./types";

export function listMatters(limit = 50): Promise<MatterListResponse> {
  return apiRequest<MatterListResponse>({
    method: "GET",
    url: "/api/v1/matters",
    params: { limit },
  });
}

export function getMatter(matterId: string): Promise<MatterRecord> {
  return apiRequest<MatterRecord>({
    method: "GET",
    url: `/api/v1/matters/${matterId}`,
  });
}

export function listMatterArtifacts(matterId: string): Promise<MatterArtifactRecord[]> {
  return apiRequest<MatterArtifactRecord[]>({
    method: "GET",
    url: `/api/v1/matters/${matterId}/artifacts`,
  });
}

export function listMatterFindings(matterId: string): Promise<MatterFindingRecord[]> {
  return apiRequest<MatterFindingRecord[]>({
    method: "GET",
    url: `/api/v1/matters/${matterId}/findings`,
  });
}

export function updateMatterConfirmationGate(
  matterId: string,
  gateId: string,
  body: MatterConfirmationGateUpdateRequest,
): Promise<MatterRecord> {
  return apiRequest<MatterRecord>({
    method: "PATCH",
    url: `/api/v1/matters/${matterId}/confirmation-gates/${gateId}`,
    data: body,
  });
}

export function updateMatterFinding(
  matterId: string,
  findingId: string,
  body: MatterFindingUpdateRequest,
): Promise<MatterRecord> {
  return apiRequest<MatterRecord>({
    method: "PATCH",
    url: `/api/v1/matters/${matterId}/findings/${findingId}`,
    data: body,
  });
}

export function generateMatterFormalReport(
  matterId: string,
  body: MatterFormalReportCreateRequest = {},
): Promise<MatterRecord> {
  return apiRequest<MatterRecord>({
    method: "POST",
    url: `/api/v1/matters/${matterId}/formal-report`,
    data: body,
  });
}

export function exportMatterArtifactMarkdown(
  matterId: string,
  artifactId: string,
): Promise<string> {
  return apiRequest<string>({
    method: "GET",
    url: `/api/v1/matters/${matterId}/artifacts/${artifactId}/export`,
    params: { format: "markdown" },
    responseType: "text",
  });
}

export function exportMatterArtifactDocx(
  matterId: string,
  artifactId: string,
): Promise<Blob> {
  return apiRequest<Blob>({
    method: "GET",
    url: `/api/v1/matters/${matterId}/artifacts/${artifactId}/export`,
    params: { format: "docx" },
    responseType: "blob",
  });
}

export function exportMatterArtifactsZip(
  matterId: string,
  format: "markdown" | "docx" | "both" = "docx",
): Promise<Blob> {
  return apiRequest<Blob>({
    method: "GET",
    url: `/api/v1/matters/${matterId}/artifacts/export`,
    params: { format },
    responseType: "blob",
  });
}
