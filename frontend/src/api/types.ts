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
  support_score?: number;
  unsupported_facts?: string[];
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

export interface ConversationMessagesResponse {
  conversation_id: string;
  messages: ChatHistoryMessage[];
}

export interface ConversationRecord {
  conversation_id: string;
  title: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationListResponse {
  conversations: ConversationRecord[];
  total: number;
  offset: number;
  limit: number | null;
}

export interface ConversationCreateRequest {
  conversation_id?: string | null;
  title?: string | null;
}

export interface ConversationUpdateRequest {
  title?: string | null;
  status?: "active" | "archived";
}

export interface AnswerResponse {
  content: string;
  citations: Citation[];
  memories_used?: MemoryUsage[];
  confidence?: string | null;
  guard_warnings?: string[];
  evidence?: EvidenceProfile | null;
}

export interface AgentTaskRequest {
  objective: string;
  focus_areas: string[];
  user_role: "ordinary" | "lawyer";
  max_steps: number;
  conversation_id?: string | null;
  matter_id?: string | null;
}

export interface AgentTaskResumeRequest {
  objective?: string | null;
  clarification_answers: string[];
  focus_areas?: string[] | null;
  user_role?: "ordinary" | "lawyer" | null;
  max_steps?: number | null;
  conversation_id?: string | null;
  matter_id?: string | null;
}

export interface AgentPlanStep {
  step_id: string;
  title: string;
  purpose: string;
  tool: string;
  arguments: Record<string, unknown>;
  requires_confirmation: boolean;
}

export interface AgentStepResult {
  step_id: string;
  title: string;
  tool: string;
  status: string;
  summary: string;
  citations: Citation[];
  evidence: EvidenceProfile | null;
  guard_warnings: string[];
  output: Record<string, unknown>;
}

export interface AgentFinding {
  finding_id: string;
  category: string;
  severity: string;
  summary: string;
  citations: string[];
  recommended_action: string;
  needs_human_review: boolean;
  source_step_id: string;
  clause_reference: string;
  evidence_coverage: string;
  support_level: string;
  unsupported_reason: string;
  source_quote: string;
  location_label: string;
  human_review_status: string;
  status: string;
  evidence: Record<string, unknown>[];
}

export interface MatterProfile {
  matter_id: string;
  document_type: string;
  parties: string[];
  user_side: string;
  governing_law: string;
  jurisdiction: string;
  key_dates: Record<string, unknown>[];
  review_scope: string[];
  open_questions: string[];
  confidence: string;
  citations: string[];
  source_step_id: string;
  confirmation_gates: AgentConfirmationGate[];
}

export interface AgentArtifact {
  artifact_id: string;
  artifact_type: string;
  title: string;
  summary: string;
  items: Record<string, unknown>[];
  source_finding_ids: string[];
  citations: string[];
  metadata: Record<string, unknown>;
}

export interface AgentConfirmationGate {
  gate_id: string;
  gate_type: string;
  title: string;
  question: string;
  status: string;
  priority: string;
  required: boolean;
  reason: string;
  related_finding_ids: string[];
  related_artifact_ids: string[];
  citations: string[];
  metadata: Record<string, unknown>;
}

export interface AgentTaskResponse {
  task_id: string;
  status: string;
  objective: string;
  plan: AgentPlanStep[];
  steps: AgentStepResult[];
  findings: AgentFinding[];
  missing_information: string[];
  human_review_required: boolean;
  report: string;
  citations: Citation[];
  confidence: string | null;
  guard_warnings: string[];
  evidence: EvidenceProfile | null;
  matter_profile: MatterProfile | null;
  artifacts: AgentArtifact[];
  confirmation_gates: AgentConfirmationGate[];
  metadata: Record<string, unknown>;
}

export interface AgentTaskEvent {
  event_id: number;
  task_id: string;
  event_type: string;
  stage: string;
  progress: number;
  message: string;
  created_at: string;
  step_id: string | null;
  payload: Record<string, unknown>;
}

export interface AgentTaskRecordResponse {
  task_id: string;
  status: "queued" | "running" | "needs_input" | "succeeded" | "failed" | string;
  objective: string;
  focus_areas: string[];
  user_role: "ordinary" | "lawyer" | string;
  max_steps: number;
  conversation_id: string | null;
  matter_id: string | null;
  stage: string;
  progress: number;
  submitted_at: string;
  started_at: string | null;
  completed_at: string | null;
  result: AgentTaskResponse | null;
  error: string | null;
  events: AgentTaskEvent[];
}

export interface MatterArtifactRecord {
  artifact_id: string;
  matter_id: string;
  artifact_type: string;
  title: string;
  summary: string;
  items: Record<string, unknown>[];
  source_finding_ids: string[];
  citations: string[];
  metadata: Record<string, unknown>;
  source_task_id: string;
  version: number;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface MatterFindingRecord {
  finding_id: string;
  matter_id: string;
  category: string;
  severity: string;
  summary: string;
  recommended_action: string;
  citations: string[];
  source_step_id: string;
  clause_reference: string;
  evidence_coverage: string;
  support_level: string;
  unsupported_reason: string;
  source_quote: string;
  location_label: string;
  needs_human_review: boolean;
  human_review_status: "pending" | "approved" | "waived" | "needs_info" | "resolved" | "not_required" | string;
  status: string;
  metadata: Record<string, unknown>;
  source_task_id: string;
  created_at: string;
  updated_at: string;
}

export interface MatterRecord {
  matter_id: string;
  title: string;
  status: string;
  matter_profile: Record<string, unknown>;
  source_task_id: string;
  latest_task_id: string;
  created_at: string;
  updated_at: string;
  artifacts: MatterArtifactRecord[];
  findings: MatterFindingRecord[];
}

export interface MatterListResponse {
  matters: MatterRecord[];
  total: number;
}

export type MatterConfirmationGateStatus = "pending" | "approved" | "waived" | "needs_info";

export interface MatterConfirmationGateUpdateRequest {
  status: MatterConfirmationGateStatus;
  note?: string | null;
  confirmed_value?: string | null;
}

export interface MatterFormalReportCreateRequest {
  note?: string | null;
}

export interface MatterFindingUpdateRequest {
  human_review_status: "pending" | "approved" | "waived" | "needs_info" | "resolved";
  note?: string | null;
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
  last_accessed_at?: string | null;
  access_count?: number;
  superseded_conflicting?: boolean;
  superseded_from_content?: string | null;
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
  last_accessed_at: string | null;
  access_count: number;
  superseded_conflicting: boolean;
  superseded_from_content: string | null;
}

export interface MemoryListResponse {
  memories: MemoryRecord[];
  total: number;
}

export interface MemoryMaintenanceResponse {
  expired_stale: number;
  limit_stale: number;
  vector_deleted: number;
  vector_upserted: number;
}

export interface MemoryAccessStats {
  tracked_memories: number;
  never_accessed: number;
  accessed: number;
  accessed_last_7d: number;
  accessed_last_30d: number;
  total_access_count: number;
  average_access_count: number;
  max_access_count: number;
}

export interface MemoryRetrievalStats {
  total: number;
  with_memory: number;
  last_7d: number;
  last_30d: number;
  hit_rate: number;
  average_memory_count: number;
  average_document_count: number;
  last_retrieval_at: string | null;
  selected_memory_source_counts: Record<string, number>;
  selected_memory_source_ratios: Record<string, number>;
}

export interface MemoryStatsResponse {
  tenant_id: string;
  user_id: string;
  generated_at: string;
  total_memories: number;
  active_memories: number;
  stale_memories: number;
  deleted_memories: number;
  expired_active_memories: number;
  status_counts: Record<string, number>;
  scope_counts: Record<string, number>;
  type_counts: Record<string, number>;
  average_confidence: number;
  average_active_confidence: number;
  access: MemoryAccessStats;
  retrievals: MemoryRetrievalStats;
}

export interface MemoryConversationSummaryRequest {
  conversation_id: string;
  limit?: number;
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

export type FeedbackRating = "positive" | "negative" | 1 | -1;

export interface FeedbackCreateRequest {
  rating: FeedbackRating;
  conversation_id?: string | null;
  message_id?: string | null;
  memory_ids?: string[];
  comment?: string | null;
}

export interface FeedbackMemoryAdjustment {
  memory_id: string;
  status: string;
  previous_confidence: number | null;
  new_confidence: number | null;
  memory: MemoryRecord | null;
}

export interface FeedbackResponse {
  feedback_id: string;
  tenant_id: string;
  user_id: string;
  rating: number;
  created_at: string;
  conversation_id: string | null;
  message_id: string | null;
  memory_ids: string[];
  comment: string | null;
  adjusted_memories: FeedbackMemoryAdjustment[];
}

export interface ApiErrorPayload {
  code?: string;
  detail?: unknown;
  request_id?: string;
}

export type HealthState = "ok" | "degraded" | "error" | string;

export interface HealthCheck {
  name: string;
  status: HealthState;
  detail: string;
}

export interface HealthResponse {
  status: HealthState;
  version: string;
  auth_required: boolean;
  default_tenant_id: string;
  providers: {
    chat?: {
      provider?: string;
      api?: string;
      model?: string;
      api_key_configured?: boolean;
    };
    embedding?: {
      provider?: string;
      model?: string;
      api_key_configured?: boolean;
    };
    [key: string]: unknown;
  };
  features: Record<string, boolean>;
  limits: {
    max_upload_bytes?: number;
    supported_extensions?: string[];
    [key: string]: unknown;
  };
  checks: HealthCheck[];
}
