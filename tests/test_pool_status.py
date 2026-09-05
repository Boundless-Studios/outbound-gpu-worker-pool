"""Host-facing pool status tests: `GET /pool/workers` and `GET /pool/queue`.

`create_pool_status_router` is what a host mounts next to its own authenticated
`/pool/jobs` routes (the library ships none), so these tests build a small
FastAPI app the same way a host would: the coordinator's worker router plus the
pool status router, sharing one `WorkerPoolService`.
"""

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    JobStatus,
    JobSubmission,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerCapability,
    WorkerIdentity,
    WorkerRegistration,
)
from outbound_gpu_worker_pool.plugins import (
    DeterministicEchoPlugin,
    capability_schemas_from_plugins,
)
from outbound_gpu_worker_pool.routes import (
    create_pool_status_router,
    create_worker_router,
)
from outbound_gpu_worker_pool.service import WorkerPoolService

ECHO = DETERMINISTIC_ECHO_CAPABILITY
OTHER = "pool.other.capability.v1"
ECHO_SCHEMAS = capability_schemas_from_plugins((DeterministicEchoPlugin(),))
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class _Harness:
    client: TestClient
    jobs: MemoryJobStore
    assets: MemoryAssetStore
    registry: MemoryWorkerRegistry
    audit: MemoryAuditLog
    service: WorkerPoolService
    now: list[datetime]


def _harness(**service_options: object) -> _Harness:
    jobs = MemoryJobStore()
    assets = MemoryAssetStore()
    registry = MemoryWorkerRegistry()
    audit = MemoryAuditLog()
    now = [EPOCH]
    authenticator = MemoryWorkerAuthenticator(
        {
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
        clock=lambda: now[0],
        **service_options,  # type: ignore[arg-type]
    )
    app = FastAPI()
    app.include_router(create_worker_router(service))
    app.include_router(create_pool_status_router(service))
    return _Harness(
        client=TestClient(app),
        jobs=jobs,
        assets=assets,
        registry=registry,
        audit=audit,
        service=service,
        now=now,
    )


def _authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _heartbeat(
    harness: _Harness,
    token: str,
    worker_id: str,
    *,
    capability_ids: tuple[str, ...] = (ECHO,),
    draining: bool = False,
    gpus: list[dict] | None = None,
    busy_job_id: str | None = None,
):
    response = harness.client.post(
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
            "draining": draining,
            "gpus": gpus or [],
            "busy_job_id": busy_job_id,
        },
    )
    assert response.status_code == 200
    return response


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


GPU_A = {
    "index": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "utilization_pct": 12,
    "memory_used_mb": 4096,
    "memory_total_mb": 24576,
}


def test_a_freshly_heartbeated_worker_is_online_and_carries_its_gpus() -> None:
    harness = _harness()

    _heartbeat(harness, "token-a", "worker-a", gpus=[GPU_A])

    response = harness.client.get("/pool/workers")
    assert response.status_code == 200
    body = response.json()
    assert len(body["workers"]) == 1
    worker = body["workers"][0]
    assert worker["worker_id"] == "worker-a"
    assert worker["status"] == "online"
    assert worker["draining"] is False
    assert worker["busy_job_id"] is None
    assert worker["capabilities"] == [ECHO]
    assert worker["gpus"] == [GPU_A]


def test_a_worker_with_a_busy_job_id_is_busy_even_if_also_draining() -> None:
    harness = _harness()

    _heartbeat(harness, "token-a", "worker-a", draining=True, busy_job_id="job-1")

    worker = harness.client.get("/pool/workers").json()["workers"][0]
    assert worker["status"] == "busy"
    assert worker["draining"] is True
    assert worker["busy_job_id"] == "job-1"


def test_a_draining_worker_with_no_busy_job_is_draining() -> None:
    harness = _harness()

    _heartbeat(harness, "token-a", "worker-a", draining=True)

    worker = harness.client.get("/pool/workers").json()["workers"][0]
    assert worker["status"] == "draining"


def _set_last_heartbeat_at(harness: _Harness, worker_id: str, at: datetime) -> None:
    # MemoryWorkerRegistry stamps `last_heartbeat_at` from the wall clock, not
    # the service's injected clock, so a frozen-time test sets it directly —
    # the same seam test_pool_status's queue test uses for job records.
    record = harness.registry.workers[worker_id]
    harness.registry.workers[worker_id] = replace(record, last_heartbeat_at=at)


def test_a_worker_heartbeated_over_90_seconds_ago_is_offline() -> None:
    harness = _harness()
    _heartbeat(harness, "token-a", "worker-a")
    _set_last_heartbeat_at(harness, "worker-a", EPOCH)

    harness.now[0] = EPOCH + timedelta(seconds=91)

    worker = harness.client.get("/pool/workers").json()["workers"][0]
    assert worker["status"] == "offline"


def test_a_worker_at_exactly_90_seconds_is_still_online() -> None:
    harness = _harness()
    _heartbeat(harness, "token-a", "worker-a")
    _set_last_heartbeat_at(harness, "worker-a", EPOCH)

    harness.now[0] = EPOCH + timedelta(seconds=90)

    worker = harness.client.get("/pool/workers").json()["workers"][0]
    assert worker["status"] == "online"


def test_a_worker_never_heard_from_in_24_hours_is_omitted() -> None:
    harness = _harness()
    _heartbeat(harness, "token-a", "worker-a")
    _heartbeat(harness, "token-b", "worker-b")
    _set_last_heartbeat_at(harness, "worker-a", EPOCH)
    _set_last_heartbeat_at(harness, "worker-b", EPOCH + timedelta(hours=23, minutes=59))

    harness.now[0] = EPOCH + timedelta(hours=24, seconds=1)

    worker_ids = {
        w["worker_id"] for w in harness.client.get("/pool/workers").json()["workers"]
    }
    assert worker_ids == {"worker-b"}


async def _seed_input(harness: _Harness, key: str) -> None:
    await harness.assets.write_once(
        f"inputs/pool/{key}.bin", b"pool-input", "application/octet-stream"
    )


def test_the_queue_route_counts_by_capability_and_status() -> None:
    harness = _harness()

    async def _seed() -> None:
        for i in range(3):
            key = f"echo-queued-{i}"
            await _seed_input(harness, key)
            await harness.service.submit(_submission(key))
        for i in range(2):
            key = f"other-queued-{i}"
            await _seed_input(harness, key)
            await harness.service.submit(_submission(key, capability_id=OTHER))
        # A completed job must not be counted at all.
        await _seed_input(harness, "echo-done")
        completed = await harness.service.submit(_submission("echo-done"))
        record = await harness.jobs.get(completed.job_id)
        assert record is not None
        harness.jobs.records[completed.job_id] = replace(
            record, status=JobStatus.COMPLETED
        )
        # A leased job counts as processing, under its own capability.
        await _seed_input(harness, "echo-processing")
        await harness.service.submit(_submission("echo-processing"))
        identity = WorkerIdentity("worker-a", "static:worker-a", "static")
        await harness.service.register_heartbeat(
            identity,
            WorkerRegistration(
                worker_id="worker-a",
                capabilities=(WorkerCapability(ECHO, "deterministic-echo", "1"),),
            ),
        )
        grant = await harness.service.lease(identity, (ECHO,))
        assert grant is not None

    asyncio.run(_seed())

    response = harness.client.get("/pool/queue")
    assert response.status_code == 200
    body = response.json()
    assert body["queued"] == 5
    assert body["processing"] == 1
    assert body["by_capability"][ECHO] == {"queued": 3, "processing": 1}
    assert body["by_capability"][OTHER] == {"queued": 2, "processing": 0}
