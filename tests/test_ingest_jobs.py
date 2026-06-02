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
    store.mark_succeeded(job.job_id, result)

    finished = store.get(job.job_id, "tenant-a")
    assert finished is not None
    assert finished.status == IngestJobStatus.SUCCEEDED
    assert finished.result == result
    assert finished.started_at is not None
    assert finished.completed_at is not None
