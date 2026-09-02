"""Framework-free coordinator logic for the outbound GPU worker pool.

Every rule `WorkerPoolService` owns is exercised here without HTTP: submission,
the host-facing reads, worker administration, lease grants, the completion
verification order, and the attempt-budget behaviour of a release.
"""

import hashlib
from dataclasses import dataclass, replace
from uuid import uuid4

import pytest

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
    AuditEventType,
    IdempotencyConflict,
    JobFailureCode,
    JobRecord,
    JobStatus,
    JobSubmission,
    LeaseGrant,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    OutputManifest,
    WorkerCapability,
    WorkerIdentity,
    WorkerRegistration,
    WorkerStatus,
    job_request_digest,
)
from outbound_gpu_worker_pool.plugins import (
    DeterministicEchoPlugin,
    capability_schemas_from_plugins,
)
from outbound_gpu_worker_pool.service import (
    OUTPUT_UPLOAD_CONTENT_TYPE,
    CompletionRejected,
    WorkerPoolService,
)

ECHO = DETERMINISTIC_ECHO_CAPABILITY
ECHO_SCHEMAS = capability_schemas_from_plugins((DeterministicEchoPlugin(),))
OUTPUT_BYTES = b"deterministic-echo/v1\nlabel=job-1\nseed=7\n"
WORKER_A = WorkerIdentity("worker-a", "static:worker-a", "static")

# reason, output published, verification budget below the real size, manifest fault
VERIFICATION_ORDER: tuple[tuple[str, bool, bool, dict[str, object]], ...] = (
    ("output_key_mismatch", False, True, {"output_key": "outputs/pool/other.txt"}),
    ("idempotency_key_mismatch", False, True, {"idempotency_key": "pool:other"}),
    ("request_digest_mismatch", False, True, {"request_digest": "0" * 64}),
    ("publication_mode_mismatch", False, True, {"publication_mode": "mutable_overwrite"}),
    ("output_missing", False, True, {}),
    ("byte_length_mismatch", True, True, {"byte_length": 3}),
    ("output_too_large", True, True, {}),
    ("sha256_mismatch", True, False, {"sha256": "1" * 64}),
)


@dataclass
class _Harness:
    jobs: MemoryJobStore
    assets: MemoryAssetStore
    registry: MemoryWorkerRegistry
    audit: MemoryAuditLog
    service: WorkerPoolService


def _harness(**options: object) -> _Harness:
    jobs = MemoryJobStore()
    assets = MemoryAssetStore()
    registry = MemoryWorkerRegistry()
    audit = MemoryAuditLog()
    service = WorkerPoolService(
        jobs,
        assets,
        registry,
        audit,
        MemoryWorkerAuthenticator({"token-a": WORKER_A}),
        ECHO_SCHEMAS,
        **options,  # type: ignore[arg-type]
    )
    return _Harness(
        jobs=jobs, assets=assets, registry=registry, audit=audit, service=service
    )


def _submission(key: str = "job-1", **overrides: object) -> JobSubmission:
    base = JobSubmission(
        job_id=str(uuid4()),
        idempotency_key=f"pool:{key}",
        capability_id=ECHO,
        input_keys=(f"inputs/pool/{key}.bin",),
        output_key=f"outputs/pool/{key}.txt",
        payload={"seed": 7, "label": key},
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


async def _submit(harness: _Harness, submission: JobSubmission) -> JobRecord:
    for index, key in enumerate(submission.input_keys):
        await harness.assets.write_once(
            key, f"pool-input-{index}".encode(), "application/octet-stream"
        )
    return await harness.service.submit(submission)


async def _registered(harness: _Harness) -> WorkerIdentity:
    await harness.service.register_heartbeat(
        WORKER_A,
        WorkerRegistration(
            worker_id=WORKER_A.worker_id,
            capabilities=(
                WorkerCapability(
                    capability_id=ECHO,
                    plugin_id="deterministic-echo",
                    plugin_version="1",
                ),
            ),
        ),
    )
    return WORKER_A


async def _leased(
    harness: _Harness, submission: JobSubmission | None = None
) -> LeaseGrant:
    await _submit(harness, submission if submission is not None else _submission())
    identity = await _registered(harness)
    grant = await harness.service.lease(identity, (ECHO,))
    assert grant is not None
    return grant


def _manifest(grant: LeaseGrant, content: bytes, **overrides: object) -> OutputManifest:
    base = OutputManifest(
        output_key=grant.output_key,
        content_type="text/plain",
        byte_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        idempotency_key=grant.idempotency_key,
        request_digest=grant.request_digest,
        plugin_id="deterministic-echo",
        plugin_version="1",
        model_id="deterministic-echo",
        model_version="1",
        publication_mode=PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
        seed=7,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


async def test_submit_validates_the_submission() -> None:
    harness = _harness()

    with pytest.raises(ValueError):
        await harness.service.submit(_submission(input_keys=("",)))


async def test_submit_stores_the_canonical_request_digest() -> None:
    harness = _harness()
    submission = _submission()

    record = await _submit(harness, submission)

    assert record.request_digest == job_request_digest(submission)
    assert record.status is JobStatus.QUEUED


async def test_an_equal_resubmission_replays_the_same_record() -> None:
    harness = _harness()
    submission = _submission()

    first = await _submit(harness, submission)
    second = await harness.service.submit(replace(submission, job_id=str(uuid4())))

    assert second.job_id == first.job_id
    assert len(harness.jobs.records) == 1


async def test_a_reused_idempotency_key_for_a_different_request_conflicts() -> None:
    harness = _harness()
    submission = _submission()
    await _submit(harness, submission)

    with pytest.raises(IdempotencyConflict):
        await harness.service.submit(
            replace(submission, job_id=str(uuid4()), payload={"seed": 8})
        )


async def test_a_leased_job_read_by_a_host_carries_no_claim_token() -> None:
    harness = _harness()
    grant = await _leased(harness)

    record = await harness.service.get(grant.job_id)

    assert harness.jobs.records[grant.job_id].claim_token == grant.claim_token
    assert record is not None
    assert record.claim_token is None
    assert record.status is JobStatus.PROCESSING


async def test_submit_and_list_for_tenant_hide_the_claim_token() -> None:
    harness = _harness()
    submitted = await _submit(harness, _submission(tenant_id="tenant-a"))
    await _submit(harness, _submission("job-2", tenant_id="tenant-b"))

    listed = await harness.service.list_for_tenant("tenant-a")

    assert submitted.claim_token is None
    assert [record.job_id for record in listed] == [submitted.job_id]
    assert all(record.claim_token is None for record in listed)


async def test_get_returns_none_for_an_unknown_job() -> None:
    harness = _harness()

    assert await harness.service.get(str(uuid4())) is None


async def test_cancel_settles_a_queued_job_and_is_idempotent() -> None:
    harness = _harness()
    record = await _submit(harness, _submission())

    cancelled = await harness.service.cancel(record.job_id)
    again = await harness.service.cancel(record.job_id)
    unknown = await harness.service.cancel(str(uuid4()))

    assert (cancelled, again, unknown) == (True, False, False)
    stored = await harness.service.get(record.job_id)
    assert stored is not None
    assert stored.status is JobStatus.CANCELLED


async def test_setting_a_worker_status_is_audited() -> None:
    harness = _harness()
    await _registered(harness)

    changed = await harness.service.set_worker_status(
        WORKER_A.worker_id, WorkerStatus.DRAINING
    )

    assert changed is True
    assert [event.event_type for event in harness.audit.events] == [
        AuditEventType.WORKER_HEARTBEAT,
        AuditEventType.WORKER_STATUS_CHANGED,
    ]
    assert harness.audit.events[-1].detail == {"status": "draining"}
    assert harness.audit.events[-1].worker_id == WORKER_A.worker_id


async def test_revoking_a_worker_records_a_revocation_event() -> None:
    harness = _harness()
    await _registered(harness)

    changed = await harness.service.set_worker_status(
        WORKER_A.worker_id, WorkerStatus.REVOKED
    )

    assert changed is True
    assert [event.event_type for event in harness.audit.events][-2:] == [
        AuditEventType.WORKER_STATUS_CHANGED,
        AuditEventType.WORKER_REVOKED,
    ]
    workers = await harness.service.list_workers()
    assert [worker.status for worker in workers] == [WorkerStatus.REVOKED]


async def test_setting_the_status_of_an_unknown_worker_records_nothing() -> None:
    harness = _harness()

    changed = await harness.service.set_worker_status("worker-z", WorkerStatus.REVOKED)

    assert changed is False
    assert harness.audit.events == []


async def test_audit_for_job_returns_only_that_job() -> None:
    harness = _harness()
    grant = await _leased(harness)

    events = await harness.service.audit_for_job(grant.job_id)

    assert [event.event_type for event in events] == [AuditEventType.LEASE_GRANTED]
    assert await harness.service.audit_for_job(str(uuid4())) == ()


async def test_a_lease_grants_one_read_url_per_input_key_in_order() -> None:
    harness = _harness()
    submission = _submission(
        input_keys=("inputs/pool/a.bin", "inputs/pool/b.bin", "inputs/pool/c.bin")
    )

    grant = await _leased(harness, submission)

    assert [item.key for item in grant.input_grants] == list(submission.input_keys)
    assert {item.method for item in grant.input_grants} == {"GET"}
    assert [item.url for item in grant.input_grants] == [
        f"memory://read/{key}" for key in submission.input_keys
    ]
    assert grant.output_grant.key == submission.output_key
    assert grant.output_grant.method == "PUT"
    assert grant.output_grant.content_type == OUTPUT_UPLOAD_CONTENT_TYPE
    assert grant.claim_token
    assert grant.request_digest == job_request_digest(submission)


async def test_a_lease_is_audited_and_so_is_an_empty_one() -> None:
    harness = _harness()
    grant = await _leased(harness)

    empty = await harness.service.lease(WORKER_A, (ECHO,))

    assert empty is None
    assert [event.event_type for event in harness.audit.events][-2:] == [
        AuditEventType.LEASE_GRANTED,
        AuditEventType.LEASE_EMPTY,
    ]
    assert harness.audit.events[-2].job_id == grant.job_id


async def test_a_verified_completion_stores_the_attested_output_facts() -> None:
    harness = _harness()
    grant = await _leased(harness)
    await harness.assets.write_once(grant.output_key, OUTPUT_BYTES, "text/plain")

    await harness.service.complete(
        WORKER_A, grant.job_id, grant.claim_token, _manifest(grant, OUTPUT_BYTES)
    )

    record = harness.jobs.records[grant.job_id]
    assert record.status is JobStatus.COMPLETED
    assert record.output_content_type == "text/plain"
    assert record.output_sha256 == hashlib.sha256(OUTPUT_BYTES).hexdigest()
    assert record.output_byte_length == len(OUTPUT_BYTES)


@pytest.mark.parametrize("index", range(len(VERIFICATION_ORDER)))
async def test_completion_verification_runs_in_a_fixed_order(index: int) -> None:
    """Each case carries its own fault and every later one, so only order passes."""
    reason, published, tight_budget, _fault = VERIFICATION_ORDER[index]
    overrides: dict[str, object] = {}
    for _reason, _published, _tight, fault in VERIFICATION_ORDER[index:]:
        overrides.update(fault)
    harness = _harness(
        max_verify_bytes=(
            len(OUTPUT_BYTES) - 1 if tight_budget else len(OUTPUT_BYTES)
        )
    )
    grant = await _leased(harness)
    if published:
        await harness.assets.write_once(grant.output_key, OUTPUT_BYTES, "text/plain")

    with pytest.raises(CompletionRejected) as caught:
        await harness.service.complete(
            WORKER_A,
            grant.job_id,
            grant.claim_token,
            _manifest(grant, OUTPUT_BYTES, **overrides),
        )

    assert caught.value.reason == reason
    record = harness.jobs.records[grant.job_id]
    assert record.status is JobStatus.FAILED
    assert record.retryable is False
    assert record.failure_code is JobFailureCode.INVALID_INPUT


async def test_a_rejected_completion_is_audited_with_its_reason() -> None:
    harness = _harness()
    grant = await _leased(harness)

    with pytest.raises(CompletionRejected):
        await harness.service.complete(
            WORKER_A, grant.job_id, grant.claim_token, _manifest(grant, OUTPUT_BYTES)
        )

    rejected = harness.audit.events[-1]
    assert rejected.event_type is AuditEventType.COMPLETION_REJECTED
    assert rejected.detail == {"reason": "output_missing"}


async def test_a_release_within_the_attempt_budget_requeues_the_job() -> None:
    harness = _harness()
    grant = await _leased(harness, _submission(attempt_budget=2))

    await harness.service.release(
        WORKER_A, grant.job_id, grant.claim_token, "TransferError"
    )

    record = harness.jobs.records[grant.job_id]
    assert record.status is JobStatus.QUEUED
    assert harness.audit.events[-1].event_type is AuditEventType.JOB_RELEASED
    assert harness.audit.events[-1].detail == {"reason": "TransferError"}


async def test_a_release_on_the_last_attempt_fails_the_job_retryably() -> None:
    harness = _harness()
    grant = await _leased(harness, _submission(attempt_budget=1))

    await harness.service.release(
        WORKER_A, grant.job_id, grant.claim_token, "RuntimeError"
    )

    record = harness.jobs.records[grant.job_id]
    assert record.status is JobStatus.FAILED
    assert record.retryable is True
    assert record.failure_code is JobFailureCode.TEMPORARY_FAILURE
    assert harness.audit.events[-1].detail == {
        "reason": "RuntimeError",
        "retryable": True,
        "failure_code": "temporary_failure",
        "budget_exhausted": True,
    }


async def test_a_worker_reported_failure_is_audited() -> None:
    harness = _harness()
    grant = await _leased(harness)

    await harness.service.fail(
        WORKER_A,
        grant.job_id,
        grant.claim_token,
        "the plugin rejected the contract version",
        retryable=False,
        failure_code=JobFailureCode.UNSUPPORTED_OPERATION,
    )

    record = harness.jobs.records[grant.job_id]
    assert record.status is JobStatus.FAILED
    assert record.retryable is False
    assert record.failure_code is JobFailureCode.UNSUPPORTED_OPERATION
    assert harness.audit.events[-1].detail == {
        "reason": "the plugin rejected the contract version",
        "retryable": False,
        "failure_code": "unsupported_operation",
    }


async def test_a_job_heartbeat_extends_the_lease_and_records_progress() -> None:
    harness = _harness()
    grant = await _leased(harness)

    lease_until = await harness.service.job_heartbeat(
        WORKER_A, grant.job_id, grant.claim_token, 42
    )

    assert lease_until >= grant.lease_until
    assert harness.jobs.records[grant.job_id].progress_percent == 42
    assert harness.audit.events[-1].event_type is AuditEventType.JOB_HEARTBEAT


async def test_the_published_capability_schema_describes_the_echo_contract() -> None:
    harness = _harness()

    schema = harness.service.capabilities_schema()["capabilities"][ECHO]

    assert schema["contract_version"] == 1
    assert set(schema["input_schema"]["properties"]) == {"seed", "label"}
