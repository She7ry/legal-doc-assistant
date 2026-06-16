# Legal Document Assistant

A citation-first legal document analysis and legal information assistant for contracts, policies, leases, privacy policies, school rules, and compliance documents.

It helps users understand document content, spot risks, organize questions, and prepare checklists for discussion with a qualified lawyer. It does not replace a lawyer, provide final legal advice, litigation strategy, or case-specific determinations.

## Features

- Upload PDF, DOCX, TXT, or Markdown documents via REST API.
- Index documents into a local Chroma vector store with legal-section-aware chunking
  and hybrid vector + BM25 retrieval.
- Ask questions grounded in retrieved excerpts with source citations.
- Maintain a separate user memory system for preferences, conversation state,
  task context, and feedback without mixing it into the document RAG index.
- Run persistent task-oriented agent reviews that plan document work, stream
  progress events, call controlled review tools, track evidence, identify
  missing information, and produce a human-review-ready report.
- Persist review findings as first-class matter records with severity, status,
  evidence coverage, support level, source quote/location, and human review
  state.
- Generate matter artifacts including risk matrices, lawyer questions,
  negotiation checklists, obligation calendars, and gated formal report records.
- Clause review: assess risk level for specific clause types.
- Conflict detection: compare contract and policy excerpts for conflicts.
- API key authentication, configurable CORS, upload size limits, and tenant-isolated indexes.
- Persistent background document ingestion with stage progress, warnings, and job status polling.
- Same-name re-uploads create a new active document version; older versions are retained but excluded from retrieval.

## Project Layout

```text
legal_doc_assistant/
  api/
    main.py            # FastAPI application entry point
    dependencies.py    # Singleton DI (vector store, QA service)
    jobs.py            # Persistent ingest job store
    agent_tasks.py     # Persistent agent task store
    task_queue.py      # Background task submission
    routers/
      documents.py     # POST /api/v1/documents/ingest, GET /api/v1/documents
      chat.py          # POST /api/v1/chat/ask
      agent.py         # Agent task lifecycle APIs
      matters.py       # Matter CRUD and exports
      memories.py      # CRUD for user memories
      review.py        # POST /api/v1/review/clause, POST /api/v1/review/conflict
    middleware/
      rate_limit.py    # Sliding-window API rate limiting
    schemas/
      requests.py      # Pydantic request models
      responses.py     # Pydantic response models

  src/doc_assistant/
    config/settings.py
    evaluation/        # RAG metric helpers and CLI entry points
    ingestion/         # File loading, hashing, upload persistence
    matter/            # Matter storage and exports
    memory/            # User memory policy, storage, retrieval
    models/            # Chat and embedding provider adapters
    retrieval/         # Chroma/BM25 hybrid retrieval and chunking
    services/          # QA, review, evidence, tool calling, agent workflows
    services/agent/    # Planner/executor schemas and helpers
    tools/             # Optional external tools such as web search
    prompts/           # Layered prompt templates
    schemas/           # Shared API/domain schemas
    utils/

  data/
    uploads/
    vector_store/
    eval/

  tests/
```

## Setup

```powershell
cd E:\project\legal_doc_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

Then edit `.env` and set the chat and embedding provider keys.

By default, chat uses DashScope's OpenAI-compatible endpoint. The generic
`DOC_ASSISTANT_CHAT_API_KEY` is preferred, while `DASHSCOPE_API_KEY` is still
accepted for backward compatibility:

```env
DOC_ASSISTANT_CHAT_PROVIDER=dashscope
DOC_ASSISTANT_CHAT_API_KEY=<your-dashscope-key>
DOC_ASSISTANT_CHAT_MODEL=qwen3.5-flash
DOC_ASSISTANT_CHAT_API=compatible
DOC_ASSISTANT_CHAT_BASE_URL=
DOC_ASSISTANT_ENABLE_THINKING=false

DOC_ASSISTANT_EMBEDDING_PROVIDER=dashscope
DOC_ASSISTANT_EMBEDDING_API_KEY=<your-dashscope-key>
DOC_ASSISTANT_EMBEDDING_MODEL=text-embedding-v3
```

This keeps Qwen3.5 models on the supported `chat/completions` path instead of
DashScope's older text-generation endpoint, which returns `url error` for
Qwen3.5 models. Set `DOC_ASSISTANT_ENABLE_THINKING=true` if you want to use
thinking mode and are comfortable with the extra token usage.

To switch chat generation to DeepSeek, change only the chat provider, key, and
model. DeepSeek's official OpenAI-compatible base URL is
`https://api.deepseek.com`; leaving `DOC_ASSISTANT_CHAT_BASE_URL` empty uses
that provider default. Check the
[DeepSeek API docs](https://api-docs.deepseek.com/) for current model names.

```env
DOC_ASSISTANT_CHAT_PROVIDER=deepseek
DOC_ASSISTANT_CHAT_API_KEY=<your-deepseek-key>
DOC_ASSISTANT_CHAT_MODEL=deepseek-v4-flash
DOC_ASSISTANT_CHAT_API=compatible
DOC_ASSISTANT_CHAT_BASE_URL=

# Keep retrieval embeddings separate. DeepSeek chat can run while embeddings
# remain on DashScope.
DOC_ASSISTANT_EMBEDDING_PROVIDER=dashscope
DOC_ASSISTANT_EMBEDDING_API_KEY=<your-dashscope-key>
DOC_ASSISTANT_EMBEDDING_MODEL=text-embedding-v3
```

For another OpenAI-compatible provider, use:

```env
DOC_ASSISTANT_CHAT_PROVIDER=openai-compatible
DOC_ASSISTANT_CHAT_API_KEY=<provider-key>
DOC_ASSISTANT_CHAT_MODEL=<provider-model>
DOC_ASSISTANT_CHAT_BASE_URL=https://provider.example/v1
```

Provider-specific request fields can be passed as JSON with
`DOC_ASSISTANT_CHAT_EXTRA_BODY`, for example:

```env
DOC_ASSISTANT_CHAT_EXTRA_BODY={"reasoning_effort":"high"}
```

Tool calling settings:

```env
DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS=6

# Disabled by default so sensitive document text is not sent to public search.
DOC_ASSISTANT_WEB_SEARCH_ENABLED=false
DOC_ASSISTANT_WEB_SEARCH_PROVIDER=duckduckgo
DOC_ASSISTANT_WEB_SEARCH_API_KEY=
DOC_ASSISTANT_WEB_SEARCH_BASE_URL=
DOC_ASSISTANT_WEB_SEARCH_MAX_RESULTS=5
DOC_ASSISTANT_WEB_SEARCH_TIMEOUT_SECONDS=10
```

Agent execution settings:

```env
DOC_ASSISTANT_AGENT_MAX_PARALLEL_STEPS=3
DOC_ASSISTANT_AGENT_STEP_MAX_RETRIES=2
DOC_ASSISTANT_AGENT_STEP_RETRY_BACKOFF_SECONDS=2,5
DOC_ASSISTANT_AGENT_LLM_PLANNER_ENABLED=true
DOC_ASSISTANT_AGENT_REACT_ENABLED=true
DOC_ASSISTANT_AGENT_REACT_MAX_ITERATIONS=2
```

When `DOC_ASSISTANT_AGENT_REACT_ENABLED=true`, eligible Agent review and
deliverable steps run a small controlled ReAct evidence loop after the planned
tool call. The loop observes missing citations, guard warnings, and weak
evidence, then uses whitelisted document-only actions such as `document_qa` or
`build_evidence_profile` to repair evidence before the final report is compiled.

Retrieval settings:

```env
DOC_ASSISTANT_TOP_K=5
DOC_ASSISTANT_RETRIEVAL_MODE=hybrid
DOC_ASSISTANT_RETRIEVAL_FETCH_K=40
DOC_ASSISTANT_RETRIEVAL_MIN_RELEVANCE=0
DOC_ASSISTANT_RETRIEVAL_RRF_K=60
DOC_ASSISTANT_RETRIEVAL_DENSE_WEIGHT=1
DOC_ASSISTANT_RETRIEVAL_BM25_WEIGHT=1
DOC_ASSISTANT_RETRIEVAL_RERANK_MODE=lexical
DOC_ASSISTANT_RETRIEVAL_RERANK_WEIGHT=0.25
DOC_ASSISTANT_RETRIEVAL_MMR_LAMBDA=0.85
DOC_ASSISTANT_CHUNK_SIZE=900
DOC_ASSISTANT_CHUNK_OVERLAP=120
```

`DOC_ASSISTANT_RETRIEVAL_MODE` supports `hybrid`, `dense`, and `bm25`.
Hybrid mode combines Chroma vector results with in-process BM25 using reciprocal
rank fusion, applies a lightweight local lexical rerank, then uses MMR selection
to reduce near-duplicate chunks. `DOC_ASSISTANT_RETRIEVAL_MIN_RELEVANCE`
defaults to `0` to preserve recall until you have enough evaluation coverage to
tune a stricter cutoff.

`POST /api/v1/chat/tools` lets the model call controlled tools while answering:

- `search_documents`: searches uploaded/indexed documents and returns `[D#]` sources.
- `web_search`: searches public web pages and returns `[W#]` sources.

Web search requires both `DOC_ASSISTANT_WEB_SEARCH_ENABLED=true` and
`enable_web_search=true` in the request body. Supported web providers are
`duckduckgo`, `brave`, and `bing`; Brave and Bing require
`DOC_ASSISTANT_WEB_SEARCH_API_KEY`.

Example tool-calling request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/v1/chat/tools" `
  -ContentType "application/json" `
  -Body '{
    "question": "结合最近公开新闻和已上传合同，分析供应商履约风险。",
    "enable_web_search": true,
    "max_tool_iterations": 6
  }'
```

Memory settings:

```env
DOC_ASSISTANT_MEMORY_DB_PATH=
DOC_ASSISTANT_MEMORY_COLLECTION=user_memories
DOC_ASSISTANT_MEMORY_TOP_K=5
DOC_ASSISTANT_MEMORY_MIN_CONFIDENCE=0.55
DOC_ASSISTANT_MEMORY_SEMANTIC_DEDUP_MIN_SCORE=0.88
DOC_ASSISTANT_CHAT_HISTORY_WINDOW=12
DOC_ASSISTANT_MEMORY_SESSION_TTL_HOURS=24
DOC_ASSISTANT_MEMORY_TASK_TTL_HOURS=168
DOC_ASSISTANT_MEMORY_MAX_ACTIVE_PER_USER=500
DOC_ASSISTANT_MEMORY_DECAY_HALF_LIFE_DAYS=90
DOC_ASSISTANT_MEMORY_MAINTENANCE_ENABLED=true
DOC_ASSISTANT_MEMORY_MAINTENANCE_COOLDOWN_SECONDS=300
DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_THRESHOLD=12
DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_INTERVAL=5
DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_WINDOW=40
DOC_ASSISTANT_MEMORY_PROMPT_MAX_TOKENS=800
DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_ENABLED=true
DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_MAX_ITEMS=3
DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_MIN_CONFIDENCE=0.6
```

Security and isolation settings:

```env
# If this is empty, local API authentication is disabled.
DOC_ASSISTANT_API_KEYS=
DOC_ASSISTANT_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
DOC_ASSISTANT_CORS_ALLOW_CREDENTIALS=false
DOC_ASSISTANT_DEFAULT_TENANT_ID=default
DOC_ASSISTANT_MAX_UPLOAD_BYTES=20971520
```

Document ingestion settings:

```env
DOC_ASSISTANT_INGEST_JOBS_DB_PATH=
DOC_ASSISTANT_AGENT_TASKS_DB_PATH=
DOC_ASSISTANT_MATTER_DB_PATH=
DOC_ASSISTANT_PDF_OCR_ENABLED=false
DOC_ASSISTANT_PDF_OCR_LANG=eng
```

If `DOC_ASSISTANT_INGEST_JOBS_DB_PATH` is empty, ingest jobs are persisted at
`data/ingest_jobs.sqlite3`. PDF OCR is optional and disabled by default. When it
is enabled, install and configure `pdf2image`, `pytesseract`, and the local OCR
runtime; otherwise scanned PDF pages are reported as ingest warnings instead of
being silently indexed as empty text.

If `DOC_ASSISTANT_AGENT_TASKS_DB_PATH` is empty, persistent agent task records
and event streams are stored at `data/agent_tasks.sqlite3`.
Agent task status can be `queued`, `running`, `needs_input`, `succeeded`, or
`failed`; `needs_input` means the task was not executed because required
context such as objective, deadline, jurisdiction, or party role is missing.

If `DOC_ASSISTANT_MATTER_DB_PATH` is empty, persistent matter profiles and
generated review artifacts/findings are stored at `data/matters.sqlite3`.
Completed Agent tasks write their `matter_profile`, generated artifacts, and
evidence-audited `findings` into this matter store, keyed by the task's
`matter_id`. Confirmation gates can write approved matter facts such as
`user_side` or `governing_law` back into the Matter Profile and record them in
`confirmed_facts`.

When `DOC_ASSISTANT_API_KEYS` is set, call protected endpoints with either
`X-API-Key: <key>` or `Authorization: Bearer <key>`. Use `X-Tenant-Id` to route
requests to a tenant-specific upload directory and Chroma collection. If the
header is omitted, the API uses `DOC_ASSISTANT_DEFAULT_TENANT_ID`.

Use `X-User-Id` to scope user memories and conversations. If it is omitted, the
API uses `local-user`.

## Memory System

The memory system is intentionally separate from the document RAG index:

- Document RAG stores uploaded document chunks and returns cited source excerpts.
- Memory stores user preferences, stable context, conversation messages, task
  state, retrieval logs, and feedback metadata.
- Long-term memory is written conservatively. By default, only explicit requests
  such as "remember this" or "以后..." are promoted into active user memory.
- Each memory includes scope, type, source, confidence, status, timestamps,
  visibility, permissions, optional expiry, and `supersedes_id` for conflict
  handling.
- Retrieved memories are compressed into a `<user_memory>` prompt section and
  are treated as data, not instructions. They never count as document evidence
  and are not cited as `[S1]` document sources.

Structured memory data is stored in SQLite at `data/memory.sqlite3` by default.
Memory embeddings use a separate Chroma collection named by
`DOC_ASSISTANT_MEMORY_COLLECTION`.

## Run

```powershell
uvicorn api.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

## Frontend

The Vue frontend lives in `frontend/` and uses Vite, Vue 3, TypeScript,
Element Plus, Vue Router, and Pinia.

```powershell
cd E:\project\legal_doc_assistant\frontend
npm.cmd install
npm.cmd run dev
```

Open http://127.0.0.1:5173 after the API is running. If your PowerShell
execution policy blocks `npm`, use `npm.cmd` as shown above.

Optional frontend environment file:

```powershell
Copy-Item .env.example .env.local
```

```env
VITE_API_BASE_URL=http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/documents/ingest` | Upload a document and queue an ingest job (PDF/DOCX/TXT/MD/Markdown) |
| GET | `/api/v1/documents/jobs/{job_id}` | Get document ingest job status |
| GET | `/api/v1/documents` | List indexed documents |
| POST | `/api/v1/chat/ask` | Ask a question with optional chat history |
| POST | `/api/v1/chat/tools` | Ask with model-driven `search_documents` and optional `web_search` tools |
| GET | `/api/v1/chat/conversations/{conversation_id}/messages` | Restore persisted chat messages for a conversation |
| POST | `/api/v1/agent/tasks` | Create a persistent Agent task and queue background execution |
| GET | `/api/v1/agent/tasks/{task_id}` | Get Agent task status, events, and final result |
| POST | `/api/v1/agent/tasks/{task_id}/resume` | Resume a `needs_input` Agent task with supplemental context |
| GET | `/api/v1/agent/tasks/{task_id}/events` | Stream Agent task progress as server-sent events |
| GET | `/api/v1/matters` | List persisted matter profiles |
| GET | `/api/v1/matters/{matter_id}` | Get a matter profile with generated artifacts and findings |
| GET | `/api/v1/matters/{matter_id}/artifacts` | List generated artifacts for a matter |
| GET | `/api/v1/matters/{matter_id}/findings` | List persisted review findings for a matter |
| PATCH | `/api/v1/matters/{matter_id}/findings/{finding_id}` | Update a finding's human review status |
| PATCH | `/api/v1/matters/{matter_id}/confirmation-gates/{gate_id}` | Approve, waive, or request information for a confirmation gate |
| POST | `/api/v1/matters/{matter_id}/formal-report` | Create a gated formal report artifact |
| GET | `/api/v1/memories` | List active user memories |
| GET | `/api/v1/memories/stats` | Get memory health and retrieval statistics |
| POST | `/api/v1/memories` | Create a user memory |
| POST | `/api/v1/memories/maintenance` | Run expiry, pruning, and vector index maintenance |
| POST | `/api/v1/memories/summarize-conversation` | Compress a conversation into session memory |
| PATCH | `/api/v1/memories/{memory_id}` | Update a user memory |
| DELETE | `/api/v1/memories/{memory_id}` | Soft-delete a user memory |
| POST | `/api/v1/feedback` | Record answer feedback and adjust linked memory confidence |
| POST | `/api/v1/review/clause` | Review a specific clause type with risk assessment |
| POST | `/api/v1/review/conflict` | Detect conflicts between contract and policy excerpts |

Example upload flow:

```powershell
$headers = @{ "X-Tenant-Id" = "acme" }
$job = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/v1/documents/ingest" `
  -Headers $headers `
  -Form @{ file = Get-Item ".\contract.pdf" }

Invoke-RestMethod -Method Get `
  -Uri "http://localhost:8000/api/v1/documents/jobs/$($job.job_id)" `
  -Headers $headers
```

## Starter Evaluation

Generate the starter legal PDF fixture and eval dataset:

```powershell
generate-eval-fixtures
```

Run the RAG evaluation after the configured chat and embedding provider keys
are set:

```powershell
run-rag-eval --clean --concurrency 4
```

The script writes `data/eval/latest_report.json`. Common CI usage:

```powershell
run-rag-eval --skip-ingest `
  --baseline data/eval/baseline_report.json `
  --fail-on-regression `
  --min-score retrieval.at_5.recall=0.8 `
  --min-score generation.citation_accuracy=0.9
```

The starter dataset lives at `data/eval/eval_dataset.json`. It includes answerable,
unanswerable, Chinese-query, and cross-document cases. `default_refusal_terms`
defines shared refusal language, while per-case `required_refusal_terms` is only
needed for overrides. The dataset also records a chunking config hash so eval runs
can detect stale `chunk_id` expectations after chunking changes.

Retrieval metrics:

- `recall`: fraction of gold sources found in the top-k retrieved chunks.
- `hit`: `1` when at least one gold source is found in top-k.
- `precision`: fraction of the top-k retrieved chunks that match a gold source.
- `mrr`: reciprocal rank of the first matching source.
- `ndcg`: ranking quality with earlier matching sources weighted more heavily.

Generation metrics:

- `answer_correctness`: required terms are present and forbidden terms are absent.
- `faithfulness`: answer numbers and required answer terms are supported by cited context.
- `citation_accuracy`: cited source ids map back to gold sources.
- `refusal_accuracy`: unanswerable questions include expected refusal language.

Interpret the report by checking `summary` first, then opening any `records[*]`
with `status: "error"` or a low metric. Per-case failures are recorded without
stopping the full evaluation run.

## Layered Prompts

Prompts live in `src/doc_assistant/prompts/` and are composed as layered system + task messages:

- `base_legal_assistant.txt`: global identity, safety boundaries, evidence rules, jurisdiction awareness, and user-mode guidance
- `document_qa.txt`: structured document Q&A output
- `clause_review.txt`: clause review with explicit risk rubric
- `conflict_check.txt`: contract/policy conflict detection with conflict types
- `tool_calling_system.txt`: tool-use policy for document search vs web search
- `answer_repair.txt`: second-pass repair when citation guard checks fail

Generated answers also pass through `answer_guard.py`, which checks citation validity, unsupported strong legal conclusions, and missing-evidence refusal behavior before returning low-confidence warnings to the API.

## Roadmap

Near term:

1. Document original-text side-by-side review and editable artifact lifecycle.
2. CI-published RAG baseline reports with regression gates.
3. Memory evaluation dashboard for precision, staleness, conflicts, and leakage.

Mid term:

1. Deeper workflow policies for contract review, version comparison, dispute fact
   organization, compliance checks, and negotiation preparation.
2. External reranker (cross-encoder or provider rerank API) for two-stage retrieval.

Long term:

1. JWT-based authentication and tenant administration.
2. Expanded evaluation labels by document type, language, and workflow category.
