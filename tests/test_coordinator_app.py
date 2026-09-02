"""The standalone coordinator application and its environment runner.

The app a host mounts is exactly `/health` plus the worker router: no job
submission surface ships with the library. `build_from_env` is the thin wiring a
standalone deployment uses, and every misconfiguration it can see is a ValueError
rather than a half-built app.
"""

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerIdentity,
)
from outbound_gpu_worker_pool import coordinator_main
from outbound_gpu_worker_pool.coordinator import create_coordinator_app
from outbound_gpu_worker_pool.plugins import (
    DeterministicEchoPlugin,
    capability_schemas_from_plugins,
)
from outbound_gpu_worker_pool.service import WorkerPoolService

ECHO = DETERMINISTIC_ECHO_CAPABILITY
ECHO_SCHEMAS = capability_schemas_from_plugins((DeterministicEchoPlugin(),))
TOKEN_DIGEST = hashlib.sha256(b"token-a").hexdigest()
BASE_ENVIRONMENT = {
    "OGWP_DATABASE_URL": "postgresql://coordinator.invalid/pool",
    "OGWP_ASSET_BACKEND": "memory",
    "OGWP_WORKER_AUTH": "static",
    "OGWP_WORKER_TOKENS": f"worker-a:{TOKEN_DIGEST}",
}


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")


def _service(**options: object) -> WorkerPoolService:
    return WorkerPoolService(
        MemoryJobStore(),
        MemoryAssetStore(),
        MemoryWorkerRegistry(),
        MemoryAuditLog(),
        MemoryWorkerAuthenticator(
            {"token-a": WorkerIdentity("worker-a", "static:worker-a", "static")}
        ),
        ECHO_SCHEMAS,
        **options,  # type: ignore[arg-type]
    )


def test_health_reports_the_auth_method_and_published_capabilities() -> None:
    client = TestClient(create_coordinator_app(_service()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "worker_pool": {"auth": "static", "capabilities": [ECHO]},
    }


def test_the_coordinator_serves_the_worker_router_and_nothing_else() -> None:
    application = create_coordinator_app(_service(auth_method="google_oidc"))
    client = TestClient(application)

    paths = set(application.openapi()["paths"])

    assert client.get("/jobs").status_code == 404
    assert client.post("/jobs", json={}).status_code == 404
    assert client.get("/health").json()["worker_pool"]["auth"] == "google_oidc"
    assert paths == {
        "/health",
        "/worker/v1/heartbeat",
        "/worker/v1/lease",
        "/worker/v1/jobs/{job_id}/heartbeat",
        "/worker/v1/jobs/{job_id}/complete",
        "/worker/v1/jobs/{job_id}/fail",
        "/worker/v1/jobs/{job_id}/release",
        "/worker/v1/capabilities/schema",
    }


def test_the_lifespan_starts_and_stops_every_given_store() -> None:
    first = _Recorder()
    second = _Recorder()

    with TestClient(
        create_coordinator_app(_service(), lifecycle=(first, second))
    ) as client:
        assert client.get("/health").status_code == 200
        assert (first.events, second.events) == (["start"], ["start"])

    assert (first.events, second.events) == (["start", "stop"], ["start", "stop"])


def test_build_from_env_wires_a_static_coordinator() -> None:
    application = coordinator_main.build_from_env(BASE_ENVIRONMENT)

    assert isinstance(application, FastAPI)
    assert "/worker/v1/lease" in set(application.openapi()["paths"])
    assert TestClient(application).get("/health").json()["worker_pool"] == {
        "auth": "static",
        "capabilities": [ECHO],
    }


def test_build_from_env_wires_the_memory_backends() -> None:
    application = coordinator_main.build_from_env(
        {**BASE_ENVIRONMENT, "OGWP_JOB_BACKEND": "memory", "OGWP_DATABASE_URL": ""}
    )

    assert TestClient(application).get("/health").status_code == 200


def test_build_from_env_wires_google_identity_tokens() -> None:
    application = coordinator_main.build_from_env(
        {
            **BASE_ENVIRONMENT,
            "OGWP_WORKER_AUTH": "google_oidc",
            "OGWP_WORKER_TOKENS": "",
            "OGWP_WORKER_AUDIENCE": "https://coordinator.invalid",
        }
    )

    assert TestClient(application).get("/health").json()["worker_pool"]["auth"] == (
        "google_oidc"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"OGWP_WORKER_AUTH": "bogus"}, "worker auth"),
        ({"OGWP_WORKER_AUTH": "none"}, "worker auth"),
        ({"OGWP_WORKER_TOKENS": ""}, "OGWP_WORKER_TOKENS"),
        (
            {"OGWP_WORKER_AUTH": "google_oidc", "OGWP_WORKER_TOKENS": ""},
            "OGWP_WORKER_AUDIENCE",
        ),
        ({"OGWP_DATABASE_URL": ""}, "OGWP_DATABASE_URL"),
        ({"OGWP_JOB_BACKEND": "sqlite"}, "job backend"),
        ({"OGWP_ASSET_BACKEND": "s3"}, "asset backend"),
        ({"OGWP_ASSET_BACKEND": "gcs"}, "OGWP_ASSET_BUCKET"),
        ({"OGWP_CAPABILITY_PLUGINS": "unknown-plugin"}, "capability plugin"),
    ],
)
def test_build_from_env_rejects_an_incomplete_environment(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        coordinator_main.build_from_env({**BASE_ENVIRONMENT, **overrides})
