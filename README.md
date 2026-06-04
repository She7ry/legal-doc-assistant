# Legal Document Assistant

A citation-first RAG assistant for contracts, policies, leases, privacy policies, school rules, and compliance documents.

This tool is for document review assistance only. It does not provide legal advice.

## Features

- Upload PDF, TXT, or Markdown documents via REST API.
- Index documents into a local Chroma vector store.
- Ask questions grounded in retrieved excerpts with source citations.
- Maintain a separate user memory system for preferences, conversation state,
  task context, and feedback without mixing it into the document RAG index.
- Clause review: assess risk level for specific clause types.
- Conflict detection: compare contract and policy excerpts for conflicts.
- API key authentication, configurable CORS, upload size limits, and tenant-isolated indexes.
- Background document ingestion with job status polling.

## Project Layout

```text
legal_doc_assistant/
  api/
    main.py            # FastAPI application entry point
    dependencies.py    # Singleton DI (vector store, QA service)
    routers/
      documents.py     # POST /api/v1/documents/ingest, GET /api/v1/documents
      chat.py          # POST /api/v1/chat/ask
      memories.py      # CRUD for user memories
      review.py        # POST /api/v1/review/clause, POST /api/v1/review/conflict
    schemas/
      requests.py      # Pydantic request models
      responses.py     # Pydantic response models

  src/doc_assistant/
    config/
    models/
    memory/
    ingestion/
    retrieval/
    services/
    prompts/
    schemas/
    utils/

  data/
    uploads/
    vector_store/

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
| POST | `/api/v1/documents/ingest` | Upload a document and queue an ingest job (PDF/TXT/MD) |
| GET | `/api/v1/documents/jobs/{job_id}` | Get document ingest job status |
| GET | `/api/v1/documents` | List indexed documents |
| POST | `/api/v1/chat/ask` | Ask a question with optional chat history |
| POST | `/api/v1/chat/tools` | Ask with model-driven `search_documents` and optional `web_search` tools |
| GET | `/api/v1/memories` | List active user memories |
| POST | `/api/v1/memories` | Create a user memory |
| PATCH | `/api/v1/memories/{memory_id}` | Update a user memory |
| DELETE | `/api/v1/memories/{memory_id}` | Soft-delete a user memory |
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
.\.venv\Scripts\python.exe scripts\generate_eval_fixtures.py
```

Run the RAG evaluation after the configured chat and embedding provider keys
are set:

```powershell
.\.venv\Scripts\python.exe scripts\run_rag_eval.py
```

The starter dataset lives at `data/eval/eval_dataset.json`. It currently includes:

- one answerable question for `Recall@5`, `Recall@10`, `Hit Rate`, `Precision`, and `MRR`
- one unanswerable question for refusal behavior
- lightweight generation checks for answer correctness, faithfulness, citation accuracy, and refusal accuracy

## Roadmap

1. Hybrid search (vector + BM25) with RRF fusion.
2. Reranker (cross-encoder) for two-stage retrieval.
3. Authentication (JWT) and multi-tenant collection isolation.
4. Async document ingestion via task queue.
5. Memory evaluation dashboard for precision, staleness, conflicts, and leakage.
