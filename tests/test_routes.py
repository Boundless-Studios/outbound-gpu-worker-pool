"""Coordinator API tests for the outbound GPU worker pool.

Everything runs against the in-memory stores through a real FastAPI TestClient so
the tests exercise the router, the DTO allowlist, and `WorkerPoolService`
together. Nothing is mocked: the registry, audit log, job store, and asset store
are the same in-memory implementations the Postgres stores are checked against.
"""

import asyncio
import hashlib
from dataclasses import dataclass, replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
    AuditEventType,
    JobFailureCode,
    JobStatus,
    JobSubmission,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerIdentity,
    WorkerStatus,
)
from outbound_gpu_worker_pool import routes as routes_module
from outbound_gpu_worker_pool.coordinator import create_coordinator_app
from outbound_gpu_worker_pool.plugins import (
    DeterministicEchoPlugin,
    capability_schemas_from_plugins,
)
from outbound_gpu_worker_pool.service import WorkerPoolService

ECHO = DETERMINISTIC_ECHO_CAPABILITY
ECHO_SCHEMAS = capability_schemas_from_plugins((DeterministicEchoPlugin(),))
OTHER = "pool.other.capability.v1"
OUTPUT_BYTES = b"deterministic-echo/v1\nlabel=job-1\nseed=7\n"


@dataclass
class _Harness:
    client: TestClient
    jobs: MemoryJobStore
    assets: MemoryAssetStore
    registry: MemoryWorkerRegistry
    audit: MemoryAuditLog
    service: WorkerPoolService


def _harness(
    *,
    identities: dict[str, WorkerIdentity] | None = None,
    **service_options: object,
) -> _Harness:
    jobs = MemoryJobStore()
    assets = MemoryAssetStore()
    registry = MemoryWorkerRegistry()
    audit = MemoryAuditLog()
    authenticator = MemoryWorkerAuthenticator(
        identities
        if identities is not None
        else {
            "token-a": WorkerIdentity("worker-a", "static:worker-a", "static"),
            "token-b": WorkerIdentity("worker-b", "static:worker-b", "static"),
        }
    )
    service = WorkerPoolService(
        jobs,
        assets,
        registry,
        audit,
        authenticator,
        ECHO_SCHEMAS,
        **service_options,  # type: ignore[arg-type]
    )
    return _Harness(
        client=TestClient(create_coordinator_app(service)),
        jobs=jobs,
        assets=assets,
        registry=registry,
        audit=audit,
        service=service,
    )


def _authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def heartbeat(
    client: TestClient,
    token: str,
    worker_id: str,
    capability_ids: tuple[str, ...] = (ECHO,),
    *,
    draining: bool = False,
):
    return client.post(
        "/worker/v1/heartbeat",
        headers=_authorization(token),
        json={
            "worker_id": worker_id,
            "capabilities": [
                {
                    "capability_id": capability_id,
                    "plugin_id": "deterministic-echo",
                    "plugin_version": "1",
                    "concurrency": 1,
                }
                for capability_id in capability_ids
            ],
            "runtime_versions": {"python": "3.13"},
            "draining": draining,
        },
    )


def submit_pool_job(harness: _Harness, key: str = "job-1", **overrides: object) -> str:
    base = JobSubmission(
        job_id=str(uuid4()),
        idempotency_key=f"pool:{key}",
        capability_id=ECHO,
        input_keys=(f"inputs/pool/{key}.bin",),
        output_key=f"outputs/pool/{key}.txt",
        payload={"seed": 7, "label": key},
        tenant_id="tenant-a",
    )
    submission = replace(base, **overrides)  # type: ignore[arg-type]
    for index, input_key in enumerate(submission.input_keys):
        asyncio.run(
            harness.assets.write_once(
                input_key, f"pool-input-{index}".encode(), "application/octet-stream"
            )
        )
    return asyncio.run(harness.service.submit(submission)).job_id


def _lease(client: TestClient, token: str, capability_ids: tuple[str, ...] = (ECHO,)):
    return client.post(
        "/worker/v1/lease",
        headers=_authorization(token),
        json={"capability_ids": list(capability_ids)},
    )


def _manifest(lease: dict[str, object], content: bytes, **overrides: object) -> dict:
    manifest = {
        "output_key": lease["output_key"],
        "content_type": "text/plain",
        "byte_length": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "idempotency_key": lease["idempotency_key"],
        "request_digest": lease["request_digest"],
        "plugin_id": "deterministic-echo",
        "plugin_version": "1",
        "model_id": "deterministic-echo",
        "model_version": "1",
        "publication_mode": PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
        "seed": 7,
        "diagnostics": {},
    }
    manifest.update(overrides)
    return manifest


def _event_types(audit: MemoryAuditLog, job_id: str) -> list[AuditEventType]:
    return [event.event_type for event in audit.events if event.job_id == job_id]


def _leased(harness: _Harness) -> tuple[str, dict]:
    job_id = submit_pool_job(harness)
    heartbeat(harness.client, "token-a", "worker-a")
    granted = _lease(harness.client, "token-a")
    assert granted.status_code == 200, granted.text
    return job_id, granted.json()


def test_a_request_without_a_credential_is_rejected() -> None:
    harness = _harness()

    response = harness.client.post("/worker/v1/lease", json={"capability_ids": [ECHO]})

    assert response.status_code == 401


def test_an_unknown_credential_is_rejected_and_audited() -> None:
    harness = _harness()

    response = _lease(harness.client, "token-unknown")

    assert response.status_code == 401
    assert [event.event_type for event in harness.audit.events] == [
        AuditEventType.AUTH_REJECTED
    ]
    assert harness.audit.events[0].detail == {"reason": "invalid_credential"}
    assert "token-unknown" not in repr(harness.audit.events)


def test_one_worker_is_revoked_without_touching_the_others() -> None:
    harness = _harness()
    submit_pool_job(harness)
    assert heartbeat(harness.client, "token-a", "worker-a").status_code == 200
    assert heartbeat(harness.client, "token-b", "worker-b").status_code == 200
    asyncio.run(harness.registry.set_status("worker-b", WorkerStatus.REVOKED))

    revoked = _lease(harness.client, "token-b")
    revoked_heartbeat = heartbeat(harness.client, "token-b", "worker-b")
    survivor = _lease(harness.client, "token-a")

    assert revoked.status_code == 403
    assert revoked_heartbeat.status_code == 403
    assert survivor.status_code == 200


def test_leasing_before_registering_is_a_conflict() -> None:
    harness = _harness()
    submit_pool_job(harness)

    response = _lease(harness.client, "token-a")

    assert response.status_code == 409


def test_a_lease_carries_scoped_grants_and_is_handed_out_once() -> None:
    harness = _harness()
    job_id = submit_pool_job(
        harness, input_keys=("inputs/pool/a.bin", "inputs/pool/b.bin")
    )
    heartbeat(harness.client, "token-a", "worker-a")
    heartbeat(harness.client, "token-b", "worker-b")

    granted = _lease(harness.client, "token-a")
    exhausted = _lease(harness.client, "token-b")

    assert granted.status_code == 200
    body = granted.json()
    record = harness.jobs.records[job_id]
    assert body["job_id"] == job_id
    assert body["claim_token"]
    assert body["request_digest"] == record.request_digest
    assert body["capability_id"] == ECHO
    assert body["contract_version"] == record.contract_version
    assert body["execution_deadline_seconds"] == record.execution_deadline_seconds
    assert body["input_keys"] == list(record.input_keys)
    assert [grant["key"] for grant in body["input_grants"]] == list(record.input_keys)
    assert {grant["method"] for grant in body["input_grants"]} == {"GET"}
    assert body["output_grant"]["key"] == record.output_key
    assert body["output_grant"]["method"] == "PUT"
    assert body["output_grant"]["content_type"] == "application/octet-stream"
    assert exhausted.status_code == 204
    assert AuditEventType.LEASE_GRANTED in _event_types(harness.audit, job_id)
    assert AuditEventType.LEASE_EMPTY in [
        event.event_type for event in harness.audit.events
    ]


def test_a_lease_never_exposes_the_tenant() -> None:
    harness = _harness()
    _job_id, grant = _leased(harness)

    assert "tenant_id" not in grant


def test_a_draining_worker_is_offered_no_work() -> None:
    harness = _harness()
    submit_pool_job(harness)
    heartbeat(harness.client, "token-a", "worker-a", draining=True)

    response = _lease(harness.client, "token-a")

    assert response.status_code == 204


def test_a_capability_the_worker_never_registered_is_not_leased() -> None:
    harness = _harness()
    submit_pool_job(harness)
    heartbeat(harness.client, "token-a", "worker-a", (OTHER,))

    response = _lease(harness.client, "token-a", (ECHO, OTHER))

    assert response.status_code == 204


@pytest.mark.parametrize("suffix", ["heartbeat", "complete", "fail", "release"])
def test_a_stale_claim_token_cannot_settle_a_job(suffix: str) -> None:
    harness = _harness()
    job_id, grant = _leased(harness)
    asyncio.run(
        harness.assets.write_once(str(grant["output_key"]), OUTPUT_BYTES, "text/plain")
    )
    bodies = {
        "heartbeat": {"claim_token": "stale-token", "progress_percent": 10},
        "complete": {
            "claim_token": "stale-token",
            "manifest": _manifest(grant, OUTPUT_BYTES),
        },
        "fail": {
            "claim_token": "stale-token",
            "reason": "plugin crashed",
            "retryable": True,
            "failure_code": "temporary_failure",
        },
        "release": {"claim_token": "stale-token", "reason": "draining"},
    }

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/{suffix}",
        headers=_authorization("token-a"),
        json=bodies[suffix],
    )

    assert response.status_code == 409
    assert harness.jobs.records[job_id].status is JobStatus.PROCESSING


@pytest.mark.parametrize("suffix", ["heartbeat", "complete", "fail", "release"])
def test_another_worker_cannot_settle_a_live_lease(suffix: str) -> None:
    harness = _harness()
    job_id, grant = _leased(harness)
    heartbeat(harness.client, "token-b", "worker-b")
    bodies = {
        "heartbeat": {"claim_token": grant["claim_token"]},
        "complete": {
            "claim_token": grant["claim_token"],
            "manifest": _manifest(grant, OUTPUT_BYTES),
        },
        "fail": {
            "claim_token": grant["claim_token"],
            "reason": "plugin crashed",
            "retryable": True,
            "failure_code": "temporary_failure",
        },
        "release": {"claim_token": grant["claim_token"], "reason": "draining"},
    }

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/{suffix}",
        headers=_authorization("token-b"),
        json=bodies[suffix],
    )

    assert response.status_code == 409
    assert harness.jobs.records[job_id].status is JobStatus.PROCESSING


def test_an_unknown_job_is_not_found() -> None:
    harness = _harness()
    heartbeat(harness.client, "token-a", "worker-a")

    response = harness.client.post(
        f"/worker/v1/jobs/{uuid4()}/heartbeat",
        headers=_authorization("token-a"),
        json={"claim_token": "any-token"},
    )

    assert response.status_code == 404


def test_a_live_lease_is_extended_by_a_job_heartbeat() -> None:
    harness = _harness()
    job_id, grant = _leased(harness)

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/heartbeat",
        headers=_authorization("token-a"),
        json={"claim_token": grant["claim_token"], "progress_percent": 42},
    )

    assert response.status_code == 200
    assert response.json()["lease_until"]
    assert harness.jobs.records[job_id].progress_percent == 42
    assert AuditEventType.JOB_HEARTBEAT in _event_types(harness.audit, job_id)


@pytest.mark.parametrize(
    ("reason", "published", "overrides"),
    [
        ("output_key_mismatch", True, {"output_key": "outputs/pool/other.txt"}),
        ("idempotency_key_mismatch", True, {"idempotency_key": "pool:other"}),
        ("request_digest_mismatch", True, {"request_digest": "0" * 64}),
        ("publication_mode_mismatch", True, {"publication_mode": "mutable_overwrite"}),
        ("output_missing", False, {}),
        ("byte_length_mismatch", True, {"byte_length": 3}),
        ("sha256_mismatch", True, {"sha256": "1" * 64}),
    ],
)
def test_completion_verification_fails_the_job_terminally(
    reason: str, published: bool, overrides: dict
) -> None:
    harness = _harness()
    job_id, grant = _leased(harness)
    if published:
        asyncio.run(
            harness.assets.write_once(
                str(grant["output_key"]), OUTPUT_BYTES, "text/plain"
            )
        )

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/complete",
        headers=_authorization("token-a"),
        json={
            "claim_token": grant["claim_token"],
            "manifest": _manifest(grant, OUTPUT_BYTES, **overrides),
        },
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"]["reason"] == reason
    record = harness.jobs.records[job_id]
    assert record.status is JobStatus.FAILED
    assert record.retryable is False
    assert record.failure_code is JobFailureCode.INVALID_INPUT
    assert AuditEventType.COMPLETION_REJECTED in _event_types(harness.audit, job_id)


def test_completion_rejects_an_output_beyond_the_verification_budget() -> None:
    harness = _harness(max_verify_bytes=8)
    job_id, grant = _leased(harness)
    asyncio.run(
        harness.assets.write_once(str(grant["output_key"]), OUTPUT_BYTES, "text/plain")
    )

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/complete",
        headers=_authorization("token-a"),
        json={
            "claim_token": grant["claim_token"],
            "manifest": _manifest(grant, OUTPUT_BYTES),
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "output_too_large"
    assert harness.jobs.records[job_id].status is JobStatus.FAILED


def test_a_verified_completion_settles_the_job() -> None:
    harness = _harness()
    job_id, grant = _leased(harness)
    asyncio.run(
        harness.assets.write_once(str(grant["output_key"]), OUTPUT_BYTES, "text/plain")
    )

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/complete",
        headers=_authorization("token-a"),
        json={
            "claim_token": grant["claim_token"],
            "manifest": _manifest(grant, OUTPUT_BYTES),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "status": "completed"}
    record = harness.jobs.records[job_id]
    assert record.status is JobStatus.COMPLETED
    assert record.output_key == grant["output_key"]
    assert record.output_sha256 == hashlib.sha256(OUTPUT_BYTES).hexdigest()
    assert _event_types(harness.audit, job_id) == [
        AuditEventType.LEASE_GRANTED,
        AuditEventType.JOB_COMPLETED,
    ]
    completed = harness.audit.events[-1]
    assert completed.detail == {
        "byte_length": len(OUTPUT_BYTES),
        "sha256": hashlib.sha256(OUTPUT_BYTES).hexdigest(),
        "plugin_id": "deterministic-echo",
        "plugin_version": "1",
    }


def test_a_worker_reported_failure_is_terminal_when_not_retryable() -> None:
    harness = _harness()
    job_id, grant = _leased(harness)

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/fail",
        headers=_authorization("token-a"),
        json={
            "claim_token": grant["claim_token"],
            "reason": "the plugin rejected the contract version",
            "retryable": False,
            "failure_code": "unsupported_operation",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "status": "failed"}
    record = harness.jobs.records[job_id]
    assert record.status is JobStatus.FAILED
    assert record.retryable is False
    assert record.failure_code is JobFailureCode.UNSUPPORTED_OPERATION
    assert AuditEventType.JOB_FAILED in _event_types(harness.audit, job_id)


def test_a_released_lease_returns_the_job_to_the_queue() -> None:
    harness = _harness()
    job_id, grant = _leased(harness)

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/release",
        headers=_authorization("token-a"),
        json={"claim_token": grant["claim_token"], "reason": "draining"},
    )

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "status": "released"}
    assert harness.jobs.records[job_id].status is JobStatus.QUEUED
    assert AuditEventType.JOB_RELEASED in _event_types(harness.audit, job_id)


def _leased_with_budget(harness: _Harness, attempt_budget: int) -> tuple[str, dict]:
    job_id = submit_pool_job(harness, attempt_budget=attempt_budget)
    heartbeat(harness.client, "token-a", "worker-a")
    granted = _lease(harness.client, "token-a")
    assert granted.status_code == 200, granted.text
    return job_id, granted.json()


def test_a_release_on_the_last_attempt_settles_the_job_terminally() -> None:
    harness = _harness()
    job_id, grant = _leased_with_budget(harness, 1)

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/release",
        headers=_authorization("token-a"),
        json={"claim_token": grant["claim_token"], "reason": "RuntimeError"},
    )

    assert response.status_code == 200
    assert response.json() == {"job_id": job_id, "status": "released"}
    record = harness.jobs.records[job_id]
    assert record.status is JobStatus.FAILED
    assert record.retryable is True
    assert record.failure_code is JobFailureCode.TEMPORARY_FAILURE
    failed = [
        event
        for event in harness.audit.events
        if event.job_id == job_id and event.event_type is AuditEventType.JOB_FAILED
    ]
    assert failed[-1].detail == {
        "reason": "RuntimeError",
        "retryable": True,
        "failure_code": "temporary_failure",
        "budget_exhausted": True,
    }


def test_a_release_within_the_attempt_budget_is_leasable_again() -> None:
    harness = _harness()
    job_id, grant = _leased_with_budget(harness, 2)

    response = harness.client.post(
        f"/worker/v1/jobs/{job_id}/release",
        headers=_authorization("token-a"),
        json={"claim_token": grant["claim_token"], "reason": "TransferError"},
    )

    assert response.status_code == 200
    assert harness.jobs.records[job_id].status is JobStatus.QUEUED
    released = _lease(harness.client, "token-a")
    assert released.status_code == 200, released.text
    assert released.json()["job_id"] == job_id
    assert harness.jobs.records[job_id].attempts == 2


def test_a_cancelled_job_can_no_longer_be_completed() -> None:
    harness = _harness()
    job_id, grant = _leased(harness)
    asyncio.run(
        harness.assets.write_once(str(grant["output_key"]), OUTPUT_BYTES, "text/plain")
    )

    cancelled = asyncio.run(harness.service.cancel(job_id))
    completed = harness.client.post(
        f"/worker/v1/jobs/{job_id}/complete",
        headers=_authorization("token-a"),
        json={
            "claim_token": grant["claim_token"],
            "manifest": _manifest(grant, OUTPUT_BYTES),
        },
    )

    assert cancelled is True
    assert completed.status_code == 409


def test_a_worker_over_its_rate_limit_is_throttled() -> None:
    harness = _harness(per_worker_limit_per_minute=3)

    statuses = [
        heartbeat(harness.client, "token-a", "worker-a").status_code for _ in range(4)
    ]

    assert statuses == [200, 200, 200, 429]
    assert harness.audit.events[-1].event_type is AuditEventType.RATE_LIMITED
    assert harness.audit.events[-1].worker_id == "worker-a"


def test_two_workers_cannot_share_one_identity_subject() -> None:
    harness = _harness(
        identities={
            "token-a": WorkerIdentity("worker-a", "static:worker-a", "static"),
            "token-c": WorkerIdentity("worker-c", "static:worker-a", "static"),
        }
    )
    assert heartbeat(harness.client, "token-a", "worker-a").status_code == 200

    response = heartbeat(harness.client, "token-c", "worker-c")

    assert response.status_code == 409


def test_a_registration_for_another_worker_id_is_a_conflict() -> None:
    harness = _harness()

    response = heartbeat(harness.client, "token-a", "worker-b")

    assert response.status_code == 409


def test_the_capability_schema_describes_the_injected_capability() -> None:
    harness = _harness()

    response = harness.client.get(
        "/worker/v1/capabilities/schema", headers=_authorization("token-a")
    )

    assert response.status_code == 200
    schema = response.json()["capabilities"][ECHO]
    assert schema["contract_version"] == 1
    assert set(schema["input_schema"]["properties"]) == {"seed", "label"}


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/worker/v1/heartbeat",
            {
                "worker_id": "worker-a",
                "capabilities": [
                    {
                        "capability_id": ECHO,
                        "plugin_id": "deterministic-echo",
                        "plugin_version": "1",
                    }
                ],
                "unexpected": 1,
            },
        ),
        ("/worker/v1/lease", {"capability_ids": [ECHO], "unexpected": 1}),
        ("/worker/v1/jobs/{job_id}/heartbeat", {"claim_token": "token", "unexpected": 1}),
        (
            "/worker/v1/jobs/{job_id}/complete",
            {
                "claim_token": "token",
                "manifest": {
                    "output_key": "outputs/pool/job-1.txt",
                    "content_type": "text/plain",
                    "byte_length": 1,
                    "sha256": "0" * 64,
                    "idempotency_key": "pool:job-1",
                    "request_digest": "0" * 64,
                    "plugin_id": "deterministic-echo",
                    "plugin_version": "1",
                    "model_id": "deterministic-echo",
                    "model_version": "1",
                    "publication_mode": PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
                    "unexpected": 1,
                },
            },
        ),
        (
            "/worker/v1/jobs/{job_id}/fail",
            {
                "claim_token": "token",
                "reason": "boom",
                "retryable": True,
                "failure_code": "temporary_failure",
                "unexpected": 1,
            },
        ),
        (
            "/worker/v1/jobs/{job_id}/release",
            {"claim_token": "token", "reason": "draining", "unexpected": 1},
        ),
    ],
)
def test_every_request_dto_rejects_an_unknown_field(path: str, body: dict) -> None:
    harness = _harness()

    response = harness.client.post(
        path.format(job_id=uuid4()), headers=_authorization("token-a"), json=body
    )

    assert response.status_code == 422


def test_every_request_dto_is_closed_and_string_bounded() -> None:
    models = [
        value
        for value in vars(routes_module).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == routes_module.__name__
        and (value.__name__.endswith("Request") or value.__name__.endswith("Dto"))
    ]

    assert {model.__name__ for model in models} == {
        "WorkerCapabilityDto",
        "WorkerHeartbeatRequest",
        "LeaseRequest",
        "JobHeartbeatRequest",
        "OutputManifestDto",
        "CompleteRequest",
        "FailRequest",
        "ReleaseRequest",
    }
    for model in models:
        assert model.model_config["extra"] == "forbid", model.__name__
        for name, field in model.model_fields.items():
            if field.annotation not in (str, str | None):
                continue
            assert any(
                getattr(constraint, "max_length", None) is not None
                for constraint in field.metadata
            ), f"{model.__name__}.{name} is an unbounded string"
