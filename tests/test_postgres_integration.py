"""Postgres-backed contract tests for the durable job store, registry, and audit log.

Every test runs against the real test database in an isolated schema, and nothing
is mocked: the lease, the fencing predicate, and the idempotent replay are SQL
behaviors, so a double would prove nothing.

Point `OUTBOUND_GPU_WORKER_POOL_TEST_DATABASE_URL` at any database; each test
creates and drops its own schema.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from uuid import uuid4

import asyncpg
import pytest

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    AuditEventType,
    IdempotencyConflict,
    IdentitySubjectTaken,
    JobFailureCode,
    JobStatus,
    JobSubmission,
    WorkerCapability,
    WorkerRegistration,
    WorkerStatus,
)
from outbound_gpu_worker_pool.postgres import (
    POOL_JOBS_TABLE,
    PostgresAuditLog,
    PostgresJobStore,
    PostgresWorkerRegistry,
)

DATABASE_URL = os.environ.get(
    "OUTBOUND_GPU_WORKER_POOL_TEST_DATABASE_URL",
    "postgresql://postgres:test@127.0.0.1:15432/worker_pool",
)

ECHO = DETERMINISTIC_ECHO_CAPABILITY
OTHER = "pool.other.capability.v1"


@dataclass
class PoolFixture:
    jobs: PostgresJobStore
    registry: PostgresWorkerRegistry
    audit: PostgresAuditLog
    pool: asyncpg.Pool


@pytest.fixture
async def pool_fixture() -> AsyncIterator[PoolFixture]:
    schema = f"test_{uuid4().hex}"
    admin = await asyncpg.connect(DATABASE_URL)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        server_settings={"search_path": schema},
    )
    jobs = PostgresJobStore(DATABASE_URL, pool=pool)
    await jobs.start()
    # Migrations must be idempotent: a second start on the same schema is a no-op.
    await jobs.start()
    registry = PostgresWorkerRegistry(DATABASE_URL, pool=pool)
    await registry.start()
    audit = PostgresAuditLog(DATABASE_URL, pool=pool)
    await audit.start()
    yield PoolFixture(jobs=jobs, registry=registry, audit=audit, pool=pool)
    await audit.stop()
    await registry.stop()
    await jobs.stop()
    await pool.close()
    await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
    await admin.close()


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


async def _submit(fixture: PoolFixture, key: str, **overrides: object) -> str:
    result = await fixture.jobs.submit(_submission(key, **overrides))
    assert result.created is True
    return result.record.job_id


async def _expire_lease(fixture: PoolFixture, job_id: str) -> None:
    async with fixture.pool.acquire() as connection:
        await connection.execute(
            f"UPDATE {POOL_JOBS_TABLE} SET lease_until = now() - interval '1 second'"
            " WHERE job_id = $1::uuid",
            job_id,
        )


async def _set_created_at(
    fixture: PoolFixture, job_id: str, offset_seconds: int
) -> None:
    async with fixture.pool.acquire() as connection:
        await connection.execute(
            f"UPDATE {POOL_JOBS_TABLE}"
            " SET created_at = now() + ($2::int * interval '1 second')"
            " WHERE job_id = $1::uuid",
            job_id,
            offset_seconds,
        )


async def test_concurrent_workers_never_share_a_lease(
    pool_fixture: PoolFixture,
) -> None:
    job_id = await _submit(pool_fixture, "solo")

    leases = await asyncio.gather(
        *(
            pool_fixture.jobs.lease(
                worker_id=f"worker-{index}",
                capability_ids=(ECHO,),
                lease_seconds=600,
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
    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.PROCESSING
    assert record.leased_by == granted[0].leased_by
    assert record.leased_by is not None and record.leased_by.startswith("worker-")
    assert record.lease_until is not None
    assert record.request_digest == granted[0].request_digest


async def test_second_queued_job_goes_to_the_other_worker(
    pool_fixture: PoolFixture,
) -> None:
    first = await _submit(pool_fixture, "first")
    second = await _submit(pool_fixture, "second")
    await _set_created_at(pool_fixture, first, -20)
    await _set_created_at(pool_fixture, second, -10)

    lease_a = await pool_fixture.jobs.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    lease_b = await pool_fixture.jobs.lease(
        worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
    )
    lease_c = await pool_fixture.jobs.lease(
        worker_id="worker-c", capability_ids=(ECHO,), lease_seconds=600
    )

    assert lease_a is not None and lease_a.job_id == first
    assert lease_b is not None and lease_b.job_id == second
    assert lease_c is None


async def test_expired_lease_recovers_and_stale_token_is_fenced(
    pool_fixture: PoolFixture,
) -> None:
    job_id = await _submit(pool_fixture, "expiring")
    stale = await pool_fixture.jobs.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert stale is not None
    assert stale.claim_token is not None
    assert (
        await pool_fixture.jobs.heartbeat(
            job_id, stale.claim_token, lease_seconds=600, progress_percent=42
        )
        is True
    )

    await _expire_lease(pool_fixture, job_id)
    fresh = await pool_fixture.jobs.lease(
        worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
    )

    assert fresh is not None
    assert fresh.job_id == job_id
    assert fresh.attempts == 2
    assert fresh.claim_token is not None
    assert fresh.claim_token != stale.claim_token
    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.leased_by == "worker-b"
    assert record.progress_percent == 42

    stale_token = stale.claim_token
    assert (
        await pool_fixture.jobs.heartbeat(job_id, stale_token, lease_seconds=600)
        is False
    )
    assert (
        await pool_fixture.jobs.complete(
            job_id,
            stale_token,
            content_type="text/plain",
            sha256="0" * 64,
            byte_length=4,
        )
        is False
    )
    assert await pool_fixture.jobs.release(job_id, stale_token, "stale") is False
    assert await pool_fixture.jobs.fail(job_id, stale_token, "stale") is False
    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.PROCESSING
    assert record.leased_by == "worker-b"

    assert (
        await pool_fixture.jobs.complete(
            job_id,
            fresh.claim_token,
            content_type="text/plain",
            sha256="a" * 64,
            byte_length=4,
        )
        is True
    )
    settled = await pool_fixture.jobs.get(job_id)
    assert settled is not None
    assert settled.status is JobStatus.COMPLETED
    assert settled.leased_by is None
    assert settled.lease_until is None
    assert settled.claim_token is None
    assert settled.output_content_type == "text/plain"
    assert settled.output_sha256 == "a" * 64
    assert settled.output_byte_length == 4


async def test_release_requeues_the_job_for_another_worker(
    pool_fixture: PoolFixture,
) -> None:
    job_id = await _submit(pool_fixture, "released")
    lease = await pool_fixture.jobs.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert lease is not None and lease.claim_token is not None

    assert await pool_fixture.jobs.release(job_id, lease.claim_token, "draining") is True

    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.QUEUED
    assert record.error == "draining"
    assert record.leased_by is None
    assert record.lease_until is None
    again = await pool_fixture.jobs.lease(
        worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
    )
    assert again is not None
    assert again.attempts == 2


async def test_fail_records_the_terminal_disposition(
    pool_fixture: PoolFixture,
) -> None:
    job_id = await _submit(pool_fixture, "failing")
    lease = await pool_fixture.jobs.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert lease is not None and lease.claim_token is not None

    assert (
        await pool_fixture.jobs.fail(
            job_id,
            lease.claim_token,
            "input rejected",
            failure_code=JobFailureCode.INVALID_INPUT,
            failure_message="The request was rejected.",
            retryable=False,
        )
        is True
    )

    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.error == "input rejected"
    assert record.failure_code is JobFailureCode.INVALID_INPUT
    assert record.failure_message == "The request was rejected."
    assert record.retryable is False
    assert record.leased_by is None


async def test_lease_order_is_priority_then_oldest(pool_fixture: PoolFixture) -> None:
    newest_urgent = await _submit(pool_fixture, "urgent", priority=10)
    oldest_normal = await _submit(pool_fixture, "old", priority=100)
    middle_normal = await _submit(pool_fixture, "middle", priority=100)
    await _set_created_at(pool_fixture, oldest_normal, -30)
    await _set_created_at(pool_fixture, middle_normal, -20)
    await _set_created_at(pool_fixture, newest_urgent, -10)

    order = []
    for _ in range(3):
        lease = await pool_fixture.jobs.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        assert lease is not None
        order.append(lease.job_id)

    assert order == [newest_urgent, oldest_normal, middle_normal]


async def test_capability_mismatch_is_never_leased(pool_fixture: PoolFixture) -> None:
    await _submit(pool_fixture, "other", capability_id=OTHER)

    assert (
        await pool_fixture.jobs.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )
    matching = await pool_fixture.jobs.lease(
        worker_id="worker-a", capability_ids=(ECHO, OTHER), lease_seconds=600
    )
    assert matching is not None
    assert matching.capability_id == OTHER


async def test_attempt_budget_exhaustion_fails_the_job(
    pool_fixture: PoolFixture,
) -> None:
    job_id = await _submit(pool_fixture, "budget", attempt_budget=2)

    for _ in range(2):
        lease = await pool_fixture.jobs.lease(
            worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
        )
        assert lease is not None
        await _expire_lease(pool_fixture, job_id)

    assert (
        await pool_fixture.jobs.lease(
            worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )
    assert await pool_fixture.jobs.expire_exhausted() == 1
    assert await pool_fixture.jobs.expire_exhausted() == 0
    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.FAILED
    assert record.attempts == 2
    assert record.failure_code is JobFailureCode.TEMPORARY_FAILURE
    assert record.retryable is True
    assert record.leased_by is None


async def test_cancel_clears_the_lease(pool_fixture: PoolFixture) -> None:
    job_id = await _submit(pool_fixture, "cancelled")
    lease = await pool_fixture.jobs.lease(
        worker_id="worker-a", capability_ids=(ECHO,), lease_seconds=600
    )
    assert lease is not None and lease.claim_token is not None

    assert await pool_fixture.jobs.cancel(job_id) is True

    record = await pool_fixture.jobs.get(job_id)
    assert record is not None
    assert record.status is JobStatus.CANCELLED
    assert record.cancelled_at is not None
    assert record.claim_token is None
    assert record.leased_by is None
    assert record.lease_until is None
    assert (
        await pool_fixture.jobs.heartbeat(job_id, lease.claim_token, lease_seconds=600)
        is False
    )
    assert (
        await pool_fixture.jobs.lease(
            worker_id="worker-b", capability_ids=(ECHO,), lease_seconds=600
        )
        is None
    )
    assert await pool_fixture.jobs.cancel(job_id) is False


async def test_submit_replays_the_same_record(pool_fixture: PoolFixture) -> None:
    submission = _submission("replay")

    first = await pool_fixture.jobs.submit(submission)
    replay = await pool_fixture.jobs.submit(submission)

    assert first.created is True
    assert replay.created is False
    assert replay.record.job_id == first.record.job_id
    assert replay.record.request_digest == first.record.request_digest
    assert len(first.record.request_digest) == 64
    other = await pool_fixture.jobs.submit(_submission("replay-2"))
    assert other.record.request_digest != first.record.request_digest


async def test_reused_idempotency_key_with_a_different_request_conflicts(
    pool_fixture: PoolFixture,
) -> None:
    submission = _submission("conflict")
    await pool_fixture.jobs.submit(submission)

    with pytest.raises(IdempotencyConflict):
        await pool_fixture.jobs.submit(
            replace(
                submission,
                job_id=str(uuid4()),
                output_key="outputs/pool/other.txt",
            )
        )


async def test_list_for_tenant_returns_only_that_tenant(
    pool_fixture: PoolFixture,
) -> None:
    mine = await _submit(pool_fixture, "mine", tenant_id="tenant-a")
    also_mine = await _submit(pool_fixture, "also-mine", tenant_id="tenant-a")
    await _submit(pool_fixture, "theirs", tenant_id="tenant-b")
    await _submit(pool_fixture, "untenanted")
    await _set_created_at(pool_fixture, mine, -20)
    await _set_created_at(pool_fixture, also_mine, -10)

    records = await pool_fixture.jobs.list_for_tenant("tenant-a")

    assert [record.job_id for record in records] == [mine, also_mine]
    assert await pool_fixture.jobs.list_for_tenant("tenant-c") == ()


async def test_worker_registry_and_audit_round_trip(
    pool_fixture: PoolFixture,
) -> None:
    registration = WorkerRegistration(
        worker_id="worker-a",
        capabilities=(
            WorkerCapability(
                capability_id=ECHO,
                plugin_id="deterministic-echo",
                plugin_version="1",
                concurrency=2,
            ),
        ),
        gpu_model="RTX 4090",
        vram_mb=24576,
        runtime_versions={"driver": "560.35", "cuda": "12.6"},
    )

    created = await pool_fixture.registry.upsert(
        registration, identity_subject="static:worker-a"
    )
    assert created.worker_id == "worker-a"
    assert created.status is WorkerStatus.ACTIVE
    assert created.identity_subject == "static:worker-a"
    assert created.capabilities == registration.capabilities
    assert created.capability_ids == (ECHO,)
    assert created.last_heartbeat_at is not None

    updated = await pool_fixture.registry.upsert(
        replace(registration, draining=True), identity_subject="static:worker-a"
    )
    assert updated.status is WorkerStatus.DRAINING

    assert (
        await pool_fixture.registry.set_status("worker-a", WorkerStatus.REVOKED) is True
    )
    assert (
        await pool_fixture.registry.set_status("missing", WorkerStatus.REVOKED) is False
    )
    revoked = await pool_fixture.registry.get("worker-a")
    assert revoked is not None
    assert revoked.status is WorkerStatus.REVOKED
    assert revoked.revoked_at is not None
    # A revoked worker stays revoked even when its agent keeps heartbeating.
    still_revoked = await pool_fixture.registry.upsert(
        registration, identity_subject="static:worker-a"
    )
    assert still_revoked.status is WorkerStatus.REVOKED
    assert await pool_fixture.registry.get("worker-b") is None
    assert [worker.worker_id for worker in await pool_fixture.registry.list()] == [
        "worker-a"
    ]

    job_id = await _submit(pool_fixture, "audited")
    await pool_fixture.audit.record(
        AuditEventType.LEASE_GRANTED,
        worker_id="worker-a",
        job_id=job_id,
        detail={"attempt": 1},
    )
    await pool_fixture.audit.record(
        AuditEventType.JOB_COMPLETED, worker_id="worker-a", job_id=job_id
    )
    await pool_fixture.audit.record(
        AuditEventType.AUTH_REJECTED, detail={"reason": "x"}
    )

    events = await pool_fixture.audit.list_for_job(job_id)
    assert [event.event_type for event in events] == [
        AuditEventType.LEASE_GRANTED,
        AuditEventType.JOB_COMPLETED,
    ]
    assert events[0].worker_id == "worker-a"
    assert events[0].job_id == job_id
    assert events[0].detail == {"attempt": 1}
    assert events[1].detail == {}
    assert events[0].created_at <= events[1].created_at


async def test_one_identity_subject_enrolls_exactly_one_worker(
    pool_fixture: PoolFixture,
) -> None:
    registration = WorkerRegistration(
        worker_id="worker-a",
        capabilities=(
            WorkerCapability(
                capability_id=ECHO, plugin_id="deterministic-echo", plugin_version="1"
            ),
        ),
    )
    await pool_fixture.registry.upsert(
        registration, identity_subject="worker@pool.invalid"
    )

    found = await pool_fixture.registry.find_by_identity_subject("worker@pool.invalid")

    assert found is not None
    assert found.worker_id == "worker-a"
    assert (
        await pool_fixture.registry.find_by_identity_subject("stranger@pool.invalid")
        is None
    )
    with pytest.raises(IdentitySubjectTaken):
        await pool_fixture.registry.upsert(
            replace(registration, worker_id="worker-b"),
            identity_subject="worker@pool.invalid",
        )
