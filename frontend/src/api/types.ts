export interface Citation {
  source_id: string;
  file_name: string;
  page: number | null;
  chunk_id: number | null;
  preview: string;
  location_label: string;
  source_type?: string;
  file_id?: string | null;
  document_key?: string | null;
  document_version?: number | null;
  page_label?: string | null;
  section_heading?: string | null;
  exact_quote?: string | null;
  char_start?: number | null;
  char_end?: number | null;
  retrieval_score?: number | null;
  retrieval_relevance?: number | null;
}

export interface EvidenceItem {
  source_id: string;
  source_type: string;
  file_name: string;
  file_id: string | null;
  document_key: string | null;
  document_version: number | null;
  page: number | null;
  page_label: string | null;
  chunk_id: number | null;
  section_heading: string | null;
  quote: string;
  location_label: string;
  char_start: number | null;
  char_end: number | null;
  retrieval_score: number | null;
  retrieval_relevance: number | null;
}

export interface EvidenceClaim {
  claim_id: string;
  text: string;
  citations: string[];
  support_level: "direct" | "partial" | "missing" | string;
  evidence: EvidenceItem[];
  uncertainty: string;
  needs_human_review: boolean;
}

export interface EvidenceProfile {
  claims: EvidenceClaim[];
  unsupported_claims: string[];
  missing_evidence: string[];
  possibly_conflicting_clauses: string[];
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
  evidence?: EvidenceProfile | null;
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

export interface ClauseRiskReason {
  reason: string;
  citation: string | null;
}

export interface ClauseReviewResponse extends AnswerResponse {
  clause_type: string;
  normalized_clause_type: string;
  found: boolean | null;
  summary: string;
  risk_level: string;
  risk_reasons: ClauseRiskReason[];
  affected_party: string | null;
  plain_language_explanation: string;
  questions_for_lawyer: string[];
  missing_information: string[];
  needs_human_review: boolean;
  guard_warnings: string[];
}

export interface ConflictCheckRequest {
  contract_query: string;
  policy_query: string;
  top_k: number;
}

export interface ConflictItem {
  topic: string;
  conflict_type: string;
  severity: string;
  contract_position: string;
  policy_position: string;
  why_conflict: string;
  recommended_action: string;
  contract_citations: string[];
  policy_citations: string[];
  needs_human_review: boolean;
  confidence: string | null;
}

export interface ConflictCheckResponse extends AnswerResponse {
  overall_status: string;
  conflicts: ConflictItem[];
  needs_human_review: boolean;
  guard_warnings: string[];
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
