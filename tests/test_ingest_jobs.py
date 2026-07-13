from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from threading import enumerate as enumerate_threads
from types import SimpleNamespace

from api.jobs import IngestJobStatus, IngestJobStore
from api.task_queue import shutdown_background_tasks, submit_background_task
from doc_assistant.schemas.citation import IngestResult


def test_ingest_job_store_hides_jobs_from_other_tenants(tmp_path) -> None:
    store = IngestJobStore()
    job = store.create("tenant-a", "contract.txt", tmp_path / "contract.txt")

    assert store.get(job.job_id, "tenant-a") is not None
    assert store.get(job.job_id, "tenant-b") is None


def test_ingest_job_store_tracks_success_result(tmp_path) -> None:
    store = IngestJobStore()
    job = store.create("tenant-a", "contract.txt", tmp_path / "contract.txt")
    result = IngestResult(
        file_id="abc",
        file_name="contract.txt",
        document_count=1,
        chunk_count=2,
    )

    store.mark_running(job.job_id)
    store.update_progress(job.job_id, "embedding", 70, "sample warning")
    store.mark_succeeded(job.job_id, result)

    finished = store.get(job.job_id, "tenant-a")
    assert finished is not None
    assert finished.status == IngestJobStatus.SUCCEEDED
    assert finished.stage == "completed"
    assert finished.progress == 100
    assert finished.result == result
    assert finished.warnings == ["sample warning"]
    assert finished.started_at is not None
    assert finished.completed_at is not None


def test_ingest_job_store_persists_jobs_to_sqlite(tmp_path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    first_store = IngestJobStore(db_path)
    job = first_store.create("tenant-a", "contract.txt", tmp_path / "contract.txt")
    result = IngestResult(
        file_id="abc",
        file_name="contract.txt",
        document_count=1,
        chunk_count=2,
        document_key="doc-key",
        document_version=3,
        file_extension=".txt",
        warnings=["empty page"],
    )

    first_store.mark_running(job.job_id)
    first_store.mark_succeeded(job.job_id, result)

    second_store = IngestJobStore(db_path)
    loaded = second_store.get(job.job_id, "tenant-a")

    assert loaded is not None
    assert loaded.status == IngestJobStatus.SUCCEEDED
    assert loaded.result == result
    assert loaded.warnings == ["empty page"]


def test_ingest_job_store_claims_sqlite_job_once_and_does_not_restart_running(tmp_path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    first_store = IngestJobStore(db_path)
    second_store = IngestJobStore(db_path)
    job = first_store.create("tenant-a", "contract.txt", tmp_path / "contract.txt")
    barrier = Barrier(2)

    def claim(store: IngestJobStore) -> bool:
        barrier.wait()
        return store.claim(job.job_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(claim, (first_store, second_store)))

    assert sum(claimed) == 1
    assert first_store.list_restartable() == []
    loaded = second_store.get(job.job_id, "tenant-a")
    assert loaded is not None
    assert loaded.status == IngestJobStatus.RUNNING


def test_background_executor_rebuilds_across_lifespans(monkeypatch) -> None:
    from api import main as api_main
    from doc_assistant.models import language_model

    completed: list[str] = []
    monkeypatch.setattr(api_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            default_tenant_id="default",
            api_keys=("test",),
            ensure_directories=lambda: None,
        ),
    )
    monkeypatch.setattr(api_main, "_vector_store", lambda _tenant_id: None)
    monkeypatch.setattr(api_main, "_memory_service", lambda _tenant_id: None)
    monkeypatch.setattr(api_main, "_qa_service", lambda _tenant_id: None)
    monkeypatch.setattr(api_main, "_agent_service", lambda _tenant_id: None)
    monkeypatch.setattr(api_main, "_recover_background_work", lambda: None)
    monkeypatch.setattr(language_model, "build_chat_model", lambda: object())

    async def run_once(key: str, value: str) -> None:
        async with api_main.lifespan(api_main.app):
            assert submit_background_task(key, completed.append, value)

    try:
        asyncio.run(run_once("lifespan-first", "first"))
        asyncio.run(run_once("lifespan-second", "second"))
    finally:
        shutdown_background_tasks()

    assert completed == ["first", "second"]
    assert not any(
        thread.is_alive() and thread.name.startswith("legal-doc-background")
        for thread in enumerate_threads()
    )
