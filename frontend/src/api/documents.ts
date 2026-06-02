import { apiRequest } from "./http";
import type { DocumentListResponse, IngestJobResponse } from "./types";

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
