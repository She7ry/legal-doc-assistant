from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from threading import enumerate as enumerate_threads
from types import SimpleNamespace

from ai.rag.schemas import IngestResult
from backend.jobs import IngestJobStatus, IngestJobStore
from backend.task_queue import shutdown_background_tasks, submit_background_task


def test_ingest_job_store_hides_jobs_from_other_users(tmp_path) -> None:
    store = IngestJobStore(tmp_path / "jobs.sqlite3")
    job = store.create("user-a", "contract.txt", tmp_path / "contract.txt")

    assert store.get(job.job_id, "user-a") is not None
    assert store.get(job.job_id, "user-b") is None


def test_ingest_job_store_tracks_success_result(tmp_path) -> None:
    store = IngestJobStore(tmp_path / "jobs.sqlite3")
    job = store.create("user-a", "contract.txt", tmp_path / "contract.txt")
    result = IngestResult(
        file_id="abc",
        file_name="contract.txt",
        document_count=1,
        chunk_count=2,
    )

    assert store.claim(job.job_id)
    store.update_progress(job.job_id, "embedding", 70, "sample warning")
    store.mark_succeeded(job.job_id, result)

    finished = store.get(job.job_id, "user-a")
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
    job = first_store.create("user-a", "contract.txt", tmp_path / "contract.txt")
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

    assert first_store.claim(job.job_id)
    first_store.mark_succeeded(job.job_id, result)

    second_store = IngestJobStore(db_path)
    loaded = second_store.get(job.job_id, "user-a")

    assert loaded is not None
    assert loaded.status == IngestJobStatus.SUCCEEDED
    assert loaded.result == result
    assert loaded.warnings == ["empty page"]


def test_ingest_job_store_claims_sqlite_job_once_and_does_not_restart_running(tmp_path) -> None:
    db_path = tmp_path / "jobs.sqlite3"
    first_store = IngestJobStore(db_path)
    second_store = IngestJobStore(db_path)
    job = first_store.create("user-a", "contract.txt", tmp_path / "contract.txt")
    barrier = Barrier(2)

    def claim(store: IngestJobStore) -> bool:
        barrier.wait()
        return store.claim(job.job_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(claim, (first_store, second_store)))

    assert sum(claimed) == 1
    assert first_store.list_restartable() == []
    loaded = second_store.get(job.job_id, "user-a")
    assert loaded is not None
    assert loaded.status == IngestJobStatus.RUNNING


def test_ingest_job_store_requeues_interrupted_work(tmp_path) -> None:
    store = IngestJobStore(tmp_path / "jobs.sqlite3")
    job = store.create("user-a", "contract.txt", tmp_path / "contract.txt")
    assert store.claim(job.job_id)

    assert store.requeue_interrupted() == 1

    recovered = store.get(job.job_id, "user-a")
    assert recovered is not None
    assert recovered.status == IngestJobStatus.QUEUED
    assert recovered.started_at is None
    assert [record.job_id for record in store.list_restartable()] == [job.job_id]


def test_background_executor_rebuilds_across_lifespans(monkeypatch) -> None:
    from ai import llm as language_model
    from backend import main as api_main

    completed: list[str] = []
    monkeypatch.setattr(api_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        api_main,
        "settings",
        SimpleNamespace(
            ensure_directories=lambda: None,
        ),
    )
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
