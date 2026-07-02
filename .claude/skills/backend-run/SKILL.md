---
name: backend-run
description: Start the FastAPI backend dev server for the legal document assistant. Use when the user wants to launch, run, or restart the backend API.
user-invocable: true
argument-hint: [port]
---

# Backend Run

Start the Legal Document Assistant's FastAPI backend.

## Prerequisites

1. `.env` file configured with LLM/Embedding API keys
2. Python virtual environment activated

## Steps

1. Check `.env` exists and has required keys:

```bash
python -c "from doc_assistant.config.settings import settings; print(f'chat_api_key={bool(settings.chat_api_key)} embedding_api_key={bool(settings.embedding_api_key)}')"
```

2. If keys missing, tell user to copy `.env.example` → `.env` and fill in values.

3. Start the server:

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Verify:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
