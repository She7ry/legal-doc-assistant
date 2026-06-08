from __future__ import annotations

from doc_assistant.schemas.citation import IngestResult

from api.jobs import IngestJobStatus, IngestJobStore


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
