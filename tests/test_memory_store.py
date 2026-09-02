"""In-memory parity for the durable job store, registry, audit log, and assets.

These mirror tests/test_postgres_integration.py so the memory implementations stay
faithful stand-ins for the Postgres ones. Nothing is mocked.
"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    AuditEventType,
    IdempotencyConflict,
    IdentitySubjectTaken,
    JobFailureCode,
    JobStatus,
    JobSubmission,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerAuthError,
    WorkerCapability,
    WorkerIdentity,
    WorkerRegistration,
    WorkerStatus,
)

ECHO = DETERMINISTIC_ECHO_CAPABILITY
OTHER = "pool.other.capability.v1"


def _submission(key: str, **overrides: object) -> JobSubmission:
    base = JobSubmission(
        job_id=str(uuid4()),
        idempotency_key=f"pool:{key}",
        capability_id=ECHO,
        input_keys=(f"inputs/pool/{key}.bin",),
        output_key=f"outputs/pool/{key}.txt",
        payload={"seed": 7, "label": key},
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


async def _submit(store: MemoryJobStore, key: str, **overrides: object) -> str:
    result = await store.submit(_submission(key, **overrides))
    assert result.created is True
    return result.record.job_id


def _expire_lease(store: MemoryJobStore, job_id: str) -> None:
    record = store.records[job_id]
    store.records[job_id] = replace(
        record, lease_until=datetime.now(UTC) - timedelta(seconds=1)
    )


async def test_concurrent_workers_never_share_a_lease() -> None:
    store = MemoryJobStore()
    job_id = await _submit(store, "solo")

    leases = await asyncio.gather(
        *(
            store.lease(
                worker_id=f"worker-{index}", capability_ids=(ECHO,), lease_seconds=600
            )
            for index in range(4)
        )
    )

    granted = [lease for lease in leases if lease is not None]
    assert len(granted) == 1
    assert granted[0].job_id == job_id
    assert granted[0].attempts == 1
    assert granted[0].claim_token
    assert granted[0].capability_id == ECHO
    assert granted[0].input_keys == ("inputs/pool/solo.bin",)
    assert granted[0].request_digest
    record = await store.get(job_id)
    assert record is not None
    assert record.status is JobStatus.PROCESSING
    assert record.leased_by == granted[0].leased_by
    assert record.lease_until is not None


async def test_lease_order_is_priority_then_oldest() -> None:
    store = MemoryJobStore()
    oldest_normal = await _submit(store, "old")
    middle_normal = await _submit(store, "middle")
    newest_urgent = await _submit(store, "urgent", priority=10)

    order = []
    for _ in range(3):
        lease = await store.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        assert lease is not None
        order.append(lease.job_id)

    assert order == [newest_urgent, oldest_normal, middle_normal]
    assert (
        await store.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )


async def test_capability_mismatch_is_never_leased() -> None:
    store = MemoryJobStore()
    await _submit(store, "other", capability_id=OTHER)

    assert (
        await store.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )
    matching = await store.lease(
        worker_id="worker-a", capability_ids=(ECHO, OTHER), lease_seconds=600
    )
    assert matching is not None
    assert matching.capability_id == OTHER


async def test_expired_lease_recovers_and_stale_token_is_fenced() -> None:
    store = MemoryJobStore()
    job_id = await _submit(store, "expiring")
    stale = await store.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert stale is not None and stale.claim_token is not None
    assert (
        await store.heartbeat(
            job_id, stale.claim_token, lease_seconds=600, progress_percent=42
        )
        is True
    )

    _expire_lease(store, job_id)
    fresh = await store.lease(
        worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
    )

    assert fresh is not None
    assert fresh.job_id == job_id
    assert fresh.attempts == 2
    assert fresh.claim_token is not None
    assert fresh.claim_token != stale.claim_token
    record = await store.get(job_id)
    assert record is not None
    assert record.leased_by == "worker-b"
    assert record.progress_percent == 42

    stale_token = stale.claim_token
    assert await store.heartbeat(job_id, stale_token, lease_seconds=600) is False
    assert (
        await store.complete(
            job_id,
            stale_token,
            content_type="text/plain",
            sha256="0" * 64,
            byte_length=4,
        )
        is False
    )
    assert await store.release(job_id, stale_token, "stale") is False
    assert await store.fail(job_id, stale_token, "stale") is False
    record = await store.get(job_id)
    assert record is not None
    assert record.status is JobStatus.PROCESSING
    assert record.leased_by == "worker-b"

    assert (
        await store.complete(
            job_id,
            fresh.claim_token,
            content_type="text/plain",
            sha256="a" * 64,
            byte_length=4,
        )
        is True
    )
    settled = await store.get(job_id)
    assert settled is not None
    assert settled.status is JobStatus.COMPLETED
    assert settled.leased_by is None
    assert settled.lease_until is None
    assert settled.claim_token is None
    assert settled.output_content_type == "text/plain"
    assert settled.output_sha256 == "a" * 64
    assert settled.output_byte_length == 4


async def test_release_requeues_the_job_for_another_worker() -> None:
    store = MemoryJobStore()
    job_id = await _submit(store, "released")
    lease = await store.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert lease is not None and lease.claim_token is not None

    assert await store.release(job_id, lease.claim_token, "draining") is True

    record = await store.get(job_id)
    assert record is not None
    assert record.status is JobStatus.QUEUED
    assert record.error == "draining"
    assert record.leased_by is None
    assert record.lease_until is None
    again = await store.lease(
        worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
    )
    assert again is not None
    assert again.attempts == 2


async def test_fail_records_the_terminal_disposition() -> None:
    store = MemoryJobStore()
    job_id = await _submit(store, "failing")
    lease = await store.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert lease is not None and lease.claim_token is not None

    assert (
        await store.fail(
            job_id,
            lease.claim_token,
            "input rejected",
            failure_code=JobFailureCode.INVALID_INPUT,
            failure_message="The request was rejected.",
            retryable=False,
        )
        is True
    )

    record = await store.get(job_id)
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.error == "input rejected"
    assert record.failure_code is JobFailureCode.INVALID_INPUT
    assert record.failure_message == "The request was rejected."
    assert record.retryable is False
    assert record.leased_by is None


async def test_attempt_budget_exhaustion_fails_the_job() -> None:
    store = MemoryJobStore()
    job_id = await _submit(store, "budget", attempt_budget=2)

    for _ in range(2):
        lease = await store.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        assert lease is not None
        _expire_lease(store, job_id)

    assert (
        await store.lease(
            worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )
    assert await store.expire_exhausted() == 1
    assert await store.expire_exhausted() == 0
    record = await store.get(job_id)
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.attempts == 2
    assert record.failure_code is JobFailureCode.TEMPORARY_FAILURE
    assert record.retryable is True
    assert record.leased_by is None


async def test_cancel_clears_the_lease() -> None:
    store = MemoryJobStore()
    job_id = await _submit(store, "cancelled")
    lease = await store.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert lease is not None and lease.claim_token is not None

    assert await store.cancel(job_id) is True

    record = await store.get(job_id)
    assert record is not None
    assert record.status is JobStatus.CANCELLED
    assert record.cancelled_at is not None
    assert record.claim_token is None
    assert record.leased_by is None
    assert record.lease_until is None
    assert await store.heartbeat(job_id, lease.claim_token, lease_seconds=600) is False
    assert (
        await store.lease(
            worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )
    assert await store.cancel(job_id) is False
    assert await store.cancel(str(uuid4())) is False


async def test_submit_replays_the_same_record() -> None:
    store = MemoryJobStore()
    submission = _submission("replay")

    first = await store.submit(submission)
    replay = await store.submit(submission)

    assert first.created is True
    assert replay.created is False
    assert replay.record.job_id == first.record.job_id
    assert replay.record.request_digest == first.record.request_digest
    assert len(first.record.request_digest) == 64


async def test_reused_idempotency_key_with_a_different_request_conflicts() -> None:
    store = MemoryJobStore()
    submission = _submission("conflict")
    await store.submit(submission)

    with pytest.raises(IdempotencyConflict):
        await store.submit(
            replace(
                submission, job_id=str(uuid4()), output_key="outputs/pool/other.txt"
            )
        )


async def test_list_for_tenant_returns_only_that_tenant() -> None:
    store = MemoryJobStore()
    mine = await _submit(store, "mine", tenant_id="tenant-a")
    also_mine = await _submit(store, "also-mine", tenant_id="tenant-a")
    await _submit(store, "theirs", tenant_id="tenant-b")
    await _submit(store, "untenanted")

    records = await store.list_for_tenant("tenant-a")

    assert [record.job_id for record in records] == [mine, also_mine]
    assert await store.list_for_tenant("tenant-c") == ()


async def test_get_returns_none_for_an_unknown_job() -> None:
    assert await MemoryJobStore().get(str(uuid4())) is None


async def test_read_urls_are_limited_to_the_allowed_read_prefixes() -> None:
    assets = MemoryAssetStore()
    await assets.write_once("inputs/pool/echo.bin", b"echo", "application/octet-stream")

    assert (
        await assets.create_read_url("inputs/pool/echo.bin")
        == "memory://read/inputs/pool/echo.bin"
    )
    with pytest.raises(ValueError, match="inputs/"):
        await assets.create_read_url("secrets/pool/echo.bin")
    with pytest.raises(FileNotFoundError):
        await assets.create_read_url("inputs/pool/missing.bin")


async def test_output_upload_grants_are_limited_to_the_output_namespace() -> None:
    assets = MemoryAssetStore()
    await assets.write_once("outputs/pool/echo.txt", b"echo", "text/plain")

    assert (
        await assets.create_output_upload_url("outputs/pool/echo.txt", "text/plain")
        == "memory://upload/outputs/pool/echo.txt"
    )
    descriptor = await assets.describe("outputs/pool/echo.txt")
    assert descriptor is not None
    assert descriptor.size == 4
    assert descriptor.content_type == "text/plain"
    assert await assets.describe("outputs/pool/missing.txt") is None
    with pytest.raises(ValueError, match="outputs/"):
        await assets.create_output_upload_url("inputs/pool/echo.bin", "text/plain")


async def test_write_once_never_overwrites_and_reads_stay_bounded() -> None:
    assets = MemoryAssetStore()

    assert await assets.write_once("outputs/pool/a.txt", b"first", "text/plain") is True
    assert (
        await assets.write_once("outputs/pool/a.txt", b"second", "text/plain") is False
    )

    assert await assets.read_limited("outputs/pool/a.txt", 8) == b"first"
    with pytest.raises(ValueError, match="bytes"):
        await assets.read_limited("outputs/pool/a.txt", 2)
    with pytest.raises(FileNotFoundError):
        await assets.read_limited("outputs/pool/missing.txt", 8)


async def test_custom_prefixes_replace_the_defaults() -> None:
    assets = MemoryAssetStore(
        allowed_read_prefixes=("shared/",), allowed_output_prefixes=("shared/out/",)
    )
    await assets.write_once("shared/a.bin", b"a", "application/octet-stream")

    assert await assets.create_read_url("shared/a.bin") == "memory://read/shared/a.bin"
    assert (
        await assets.create_output_upload_url("shared/out/b.bin", "text/plain")
        == "memory://upload/shared/out/b.bin"
    )
    with pytest.raises(ValueError, match="shared/"):
        await assets.create_read_url("inputs/a.bin")
    with pytest.raises(ValueError, match="shared/out/"):
        await assets.create_output_upload_url("outputs/b.bin", "text/plain")


async def test_revoked_worker_stays_revoked_across_heartbeats() -> None:
    registry = MemoryWorkerRegistry()
    registration = WorkerRegistration(
        worker_id="worker-a",
        capabilities=(
            WorkerCapability(
                capability_id=ECHO, plugin_id="deterministic-echo", plugin_version="1"
            ),
        ),
        gpu_model="RTX 4090",
        vram_mb=24576,
        runtime_versions={"driver": "560.35"},
    )

    created = await registry.upsert(registration, identity_subject="static:worker-a")
    assert created.status is WorkerStatus.ACTIVE
    assert created.capability_ids == (ECHO,)
    assert created.last_heartbeat_at is not None

    draining = await registry.upsert(
        replace(registration, draining=True), identity_subject="static:worker-a"
    )
    assert draining.status is WorkerStatus.DRAINING

    assert await registry.set_status("worker-a", WorkerStatus.REVOKED) is True
    assert await registry.set_status("missing", WorkerStatus.REVOKED) is False
    revoked = await registry.get("worker-a")
    assert revoked is not None
    assert revoked.status is WorkerStatus.REVOKED
    assert revoked.revoked_at is not None

    still_revoked = await registry.upsert(
        registration, identity_subject="static:worker-a"
    )
    assert still_revoked.status is WorkerStatus.REVOKED
    assert await registry.get("worker-b") is None
    assert [worker.worker_id for worker in await registry.list()] == ["worker-a"]


async def test_one_identity_subject_enrolls_exactly_one_worker() -> None:
    registry = MemoryWorkerRegistry()
    registration = WorkerRegistration(
        worker_id="worker-a",
        capabilities=(
            WorkerCapability(
                capability_id=ECHO, plugin_id="deterministic-echo", plugin_version="1"
            ),
        ),
    )
    await registry.upsert(registration, identity_subject="worker@pool.invalid")

    found = await registry.find_by_identity_subject("worker@pool.invalid")

    assert found is not None
    assert found.worker_id == "worker-a"
    assert await registry.find_by_identity_subject("stranger@pool.invalid") is None
    with pytest.raises(IdentitySubjectTaken):
        await registry.upsert(
            replace(registration, worker_id="worker-b"),
            identity_subject="worker@pool.invalid",
        )


async def test_audit_log_lists_one_job_in_order() -> None:
    audit = MemoryAuditLog()

    await audit.record(
        AuditEventType.LEASE_GRANTED,
        worker_id="worker-a",
        job_id="job-1",
        detail={"attempt": 1},
    )
    await audit.record(
        AuditEventType.JOB_COMPLETED, worker_id="worker-a", job_id="job-1"
    )
    await audit.record(
        AuditEventType.LEASE_GRANTED, worker_id="worker-b", job_id="job-2"
    )
    await audit.record(AuditEventType.AUTH_REJECTED, detail={"reason": "x"})

    events = await audit.list_for_job("job-1")

    assert [event.event_type for event in events] == [
        AuditEventType.LEASE_GRANTED,
        AuditEventType.JOB_COMPLETED,
    ]
    assert events[0].detail == {"attempt": 1}
    assert events[1].detail == {}
    assert events[0].created_at <= events[1].created_at


async def test_authenticator_resolves_only_known_bearer_tokens() -> None:
    identity = WorkerIdentity(
        worker_id="worker-a", subject="static:worker-a", method="static"
    )
    authenticator = MemoryWorkerAuthenticator({"token-a": identity})

    assert await authenticator.authenticate("Bearer token-a") == identity
    for authorization in (None, "token-a", "Bearer token-b"):
        with pytest.raises(WorkerAuthError):
            await authenticator.authenticate(authorization)
