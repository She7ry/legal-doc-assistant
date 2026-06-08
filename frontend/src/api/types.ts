export interface Citation {
  source_id: string;
  file_name: string;
  page: number | null;
  chunk_id: number | null;
  preview: string;
  location_label: string;
}

export interface AskRequest {
  question: string;
  chat_history: ChatHistoryMessage[];
  conversation_id?: string;
  task_id?: string | null;
}

export interface ChatHistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface AnswerResponse {
  content: string;
  citations: Citation[];
  memories_used?: MemoryUsage[];
  confidence?: string | null;
  guard_warnings?: string[];
}

export interface MemoryUsage {
  memory_id: string;
  type: string;
  key: string;
  content: string;
  source: string;
  confidence: number;
  scope: string;
  score: number | null;
}

export interface ClauseReviewRequest {
  clause_type: string;
  top_k: number;
}

export interface ConflictCheckRequest {
  contract_query: string;
  policy_query: string;
  top_k: number;
}

export interface IngestResult {
  file_id: string;
  file_name: string;
  document_count: number;
  chunk_count: number;
  document_key: string;
  document_version: number;
  file_extension: string;
  page_count: number | null;
  skipped: boolean;
  warnings: string[];
}

export type IngestJobStatus = "queued" | "running" | "succeeded" | "failed";

export interface IngestJobResponse {
  job_id: string;
  status: IngestJobStatus;
  file_name: string;
  stage: string;
  progress: number;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  result: IngestResult | null;
  error: string | null;
  warnings: string[];
}

export interface DocumentInfo {
  file_name: string;
  file_id: string;
  document_key: string;
  document_version: number;
  file_extension: string;
  document_count: number;
  chunk_count: number;
  page_count: number | null;
  indexed_at: string | null;
  warning_count: number;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
}

export interface MemoryRecord {
  memory_id: string;
  scope: string;
  type: string;
  key: string;
  content: string;
  value: Record<string, unknown> | null;
  source: string;
  confidence: number;
  created_at: string;
  updated_at: string;
  expires_at: string | null;
  visibility: string;
  permissions: string[];
  embedding_id: string | null;
  supersedes_id: string | null;
  status: string;
  source_message_id: string | null;
  conversation_id: string | null;
  task_id: string | null;
}

export interface MemoryListResponse {
  memories: MemoryRecord[];
  total: number;
}

export interface MemoryCreateRequest {
  scope: string;
  type: string;
  key: string;
  content: string;
  value?: Record<string, unknown> | null;
  source: string;
  confidence: number;
  expires_at?: string | null;
  visibility: string;
}

export interface MemoryUpdateRequest {
  key?: string;
  content?: string;
  value?: Record<string, unknown> | null;
  source?: string;
  confidence?: number;
  expires_at?: string | null;
  visibility?: string;
  status?: string;
}

export interface ApiErrorPayload {
  code?: string;
  detail?: unknown;
  request_id?: string;
}
