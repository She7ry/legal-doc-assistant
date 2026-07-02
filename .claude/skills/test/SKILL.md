---
name: test
description: Run pytest tests for the legal document assistant. Use when the user wants to run tests, check test results, or measure coverage.
user-invocable: true
argument-hint: [module name or test function]
---

# Run Tests

Execute the pytest suite for the legal document assistant.

## Prerequisites

Dev dependencies installed: `pip install -e ".[dev]"`

## Commands

### All tests

```bash
python -m pytest tests/ -v
```

### Single module

```bash
python -m pytest tests/test_qa_service.py -v
```

### Single function

```bash
python -m pytest tests/test_qa_service.py::test_function_name -v
```

### With coverage

```bash
python -m pytest tests/ -v --cov=src/doc_assistant --cov-report=term-missing
```

### Only last failed

```bash
python -m pytest tests/ -v --lf
```

### Full traceback

```bash
python -m pytest tests/ -v --tb=long
```

## Test Files

| File | Coverage |
|------|----------|
| `test_qa_service.py` | Q&A pipeline |
| `test_tool_calling_service.py` | Tool calling |
| `test_agent_service.py` / `test_agent_api.py` / `test_agent_tasks.py` | Agent system |
| `test_answer_guard.py` / `test_evidence_profile.py` | Answer quality |
| `test_memory_service.py` | User memory |
| `test_vector_store_ingestion.py` | Retrieval engine |
| `test_document_loader.py` | Document loading |
| `test_api_dependencies.py` | DI system |
| `test_settings.py` | Configuration |
| `test_ingest_jobs.py` / `test_matter_store.py` | Storage |
| `test_eval_metrics.py` | Evaluation |
| `test_web_search.py` | Web search |
| `test_review_taxonomy.py` | Review taxonomy |
