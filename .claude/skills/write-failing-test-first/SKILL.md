---
name: write-failing-test-first
description: Write a failing test before implementing any fix or feature. Red-Green-Refactor for the legal document assistant. Use when implementing new features or fixing bugs.
user-invocable: true
argument-hint: [feature or bug description]
---

# Write Failing Test First (TDD)

Follow the Red-Green-Refactor cycle for every code change in this project.

## The Cycle

```
RED   → Write a test that FAILS (proves the bug/feature gap exists)
GREEN → Write minimal code to make the test PASS
REFACTOR → Clean up the code while tests stay GREEN
```

## Step 1: Red — Write the Failing Test

### For a bug fix

1. Find or create the test file for the affected module:

```
tests/
├── test_qa_service.py       → src/doc_assistant/services/qa_service.py
├── test_memory_service.py   → src/doc_assistant/memory/service.py
├── test_agent_service.py    → src/doc_assistant/services/agent_service.py
├── ...
```

2. Write a test that:
   - Recreates the exact bug scenario
   - Asserts the CORRECT behavior (which currently fails)
   - Has a descriptive name: `test_<what>_<when>_<then>`

3. Run it and verify it FAILS:

```bash
python -m pytest tests/test_xxx.py::test_yyy -v
```

### For a new feature

```python
# tests/test_new_feature.py
import pytest
from doc_assistant.services.new_service import NewService

def test_new_feature_basic_case():
    """New feature should produce expected output for normal input."""
    service = NewService()
    result = service.do_thing("normal input")
    assert result.status == "success"
    assert len(result.items) > 0

def test_new_feature_empty_input():
    """New feature should handle empty input gracefully."""
    ...

def test_new_feature_invalid_input():
    """New feature should raise clear error on invalid input."""
    ...
```

## Step 2: Green — Minimal Implementation

Write ONLY enough code to make the test pass. No refactoring, no optimization, no "while I'm here" changes.

```bash
python -m pytest tests/test_xxx.py::test_yyy -v  # Must PASS
```

## Step 3: Refactor

Clean up, but keep tests green:

```bash
python -m pytest tests/test_xxx.py -v  # All still passing
```

## Project-Specific Patterns

### Testing async services (pytest-asyncio)

```python
import pytest

@pytest.mark.asyncio
async def test_async_qa_flow():
    from doc_assistant.services.qa_service import DocumentQAService
    service = DocumentQAService(...)
    result = await service.aask("What are the key terms?")
    assert result.answer
    assert len(result.citations) > 0
```

### Mocking the LLM (avoid real API calls in tests)

```python
from unittest.mock import patch, MagicMock

@patch('doc_assistant.models.language_model.build_chat_model')
def test_with_mock_llm(mock_build):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "mocked response"
    mock_build.return_value = mock_llm
    ...
```

### Mocking vector store

```python
@patch('doc_assistant.retrieval.vector_store.DocumentVectorStore')
def test_qa_with_mock_store(mock_store_cls):
    mock_store = MagicMock()
    mock_store.search.return_value = [
        {"content": "...", "metadata": {"source_id": "S1"}}
    ]
    mock_store_cls.return_value = mock_store
    ...
```

## Anti-Patterns

- ❌ Writing the implementation first, then the test
- ❌ Writing a test you already know will pass
- ❌ Testing implementation details instead of behavior
- ❌ Mocking everything (test the real code path when possible)
- ❌ Skipping the test for "trivial one-line fixes" (those are the ones that regress)
