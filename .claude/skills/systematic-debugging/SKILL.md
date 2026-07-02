---
name: systematic-debugging
description: Methodical step-by-step debugging process. Use when encountering a bug, test failure, or unexpected behavior in the Python/FastAPI backend or Vue frontend.
user-invocable: true
argument-hint: [bug description]
---

# Systematic Debugging

A methodical debugging process tailored for this Python/FastAPI + Vue project.

## Phase 1: Reproduce

Before touching any code, reproduce the issue:

```bash
# For backend — check the health endpoint first
curl -s http://localhost:8000/health | python -m json.tool

# For test failures — run with full traceback
python -m pytest tests/<failing_test>.py -v --tb=long -x

# For import errors — verify the module loads
python -c "from doc_assistant.<module> import <Class>; print('OK')"
```

## Phase 2: Isolate

Narrow down the failure point:

1. **Stack trace analysis** — Start from the bottom (root cause), not the top (where it crashed)
2. **Binary search** — Comment out half the code. Still broken? Bug is in remaining half. Repeat.
3. **Check recent changes** — `git diff HEAD~1 --name-only` — what files changed?

## Phase 3: Hypothesize

Form at least TWO competing hypotheses about the cause:

1. Hypothesis A: _______________
2. Hypothesis B: _______________

Then design a test that would distinguish between them.

## Phase 4: Verify (not assume)

```bash
# Don't assume — run the actual code:
python -c "
import sys
sys.path.insert(0, 'src')
# ... minimal reproduction code ...
print(f'Result: {result}')
print(f'Expected: {expected}')
"
```

## Project-Specific Debugging

### LLM calls not working
```bash
python -c "
from doc_assistant.config.settings import settings
print(f'Provider: {settings.chat_provider}')
print(f'Model: {settings.chat_model_name}')
print(f'API Key configured: {bool(settings.chat_api_key)}')
print(f'Base URL: {settings.chat_base_url}')
"
```

### Vector store issues
```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='data/vector_store')
for col in client.list_collections():
    print(f'{col.name}: {col.count()} docs')
"
```

### Tenant isolation issues
- Check `X-Tenant-Id` header in requests
- Verify Chroma collection name includes tenant ID
- Check BM25 index path includes tenant ID

## Phase 5: Fix + Regression Guard

1. Apply minimal fix
2. Run the specific test: `python -m pytest tests/<test>.py -v`
3. Run related tests (at minimum, the same module)
4. If the bug is user-facing, add a regression test that specifically covers it
