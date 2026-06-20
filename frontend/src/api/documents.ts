import { apiRequest } from "./http";
import type { DocumentListResponse, DocumentTextResponse, IngestJobResponse } from "./types";

export function ingestDocument(file: File): Promise<IngestJobResponse> {
  const body = new FormData();
  body.append("file", file);

  return apiRequest<IngestJobResponse>({
    method: "POST",
    url: "/api/v1/documents/ingest",
    data: body,
  });
}

export function getIngestJob(jobId: string): Promise<IngestJobResponse> {
  return apiRequest<IngestJobResponse>({
    method: "GET",
    url: `/api/v1/documents/jobs/${jobId}`,
  });
}

export function listDocuments(): Promise<DocumentListResponse> {
  return apiRequest<DocumentListResponse>({
    method: "GET",
    url: "/api/v1/documents",
  });
}

export function getDocumentText(params: {
  document_key?: string | null;
  file_id?: string | null;
  document_version?: number | null;
}): Promise<DocumentTextResponse> {
  return apiRequest<DocumentTextResponse>({
    method: "GET",
    url: "/api/v1/documents/text",
    params: {
      document_key: params.document_key || undefined,
      file_id: params.file_id || undefined,
      document_version: params.document_version || undefined,
    },
  });
}
