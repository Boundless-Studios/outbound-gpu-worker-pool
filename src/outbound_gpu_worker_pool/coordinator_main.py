"""Run a standalone coordinator from the environment (`OGWP_*`).

This module is thin on purpose: it chooses backends, builds one
`WorkerPoolService`, and hands it to `create_coordinator_app`. A host that
already has an application wires the service itself and mounts the router.

There is no `none` worker-auth mode. This process exists only to serve workers,
so an unauthenticated coordinator would be a coordinator with no purpose.
"""

import os
from collections.abc import Mapping

from fastapi import FastAPI

from outbound_gpu_worker_pool.auth import (
    GOOGLE_OIDC_AUTH_METHOD,
    STATIC_AUTH_METHOD,
    GoogleIdTokenWorkerAuthenticator,
    StaticTokenWorkerAuthenticator,
)
from outbound_gpu_worker_pool.contracts import (
    DEFAULT_OUTPUT_PREFIXES,
    DEFAULT_READ_PREFIXES,
    AssetStore,
    AuditLog,
    CapabilitySchema,
    CapabilitySchemas,
    JobStore,
    WorkerAuthenticator,
    WorkerRegistry,
)
from outbound_gpu_worker_pool.coordinator import Startable, create_coordinator_app
from outbound_gpu_worker_pool.memory import (
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerRegistry,
)
from outbound_gpu_worker_pool.postgres import (
    PostgresAuditLog,
    PostgresJobStore,
    PostgresWorkerRegistry,
)
from outbound_gpu_worker_pool.service import (
    DEFAULT_CAPABILITY_SCHEMAS,
    WorkerPoolService,
)

POSTGRES_BACKEND = "postgres"
MEMORY_BACKEND = "memory"
GCS_BACKEND = "gcs"
DEFAULT_PORT = 8080
DETERMINISTIC_ECHO_PLUGIN_ID = "deterministic-echo"

# Task 3 registers its plugins here; the schemas a coordinator publishes are
# exactly the ones the enabled plugins declare.
KNOWN_CAPABILITY_SCHEMAS: dict[str, CapabilitySchemas] = {
    DETERMINISTIC_ECHO_PLUGIN_ID: DEFAULT_CAPABILITY_SCHEMAS,
}


def build_from_env(environment: Mapping[str, str]) -> FastAPI:
    job_backend = environment.get("OGWP_JOB_BACKEND", POSTGRES_BACKEND)
    asset_backend = environment.get("OGWP_ASSET_BACKEND", GCS_BACKEND)
    worker_auth = environment.get("OGWP_WORKER_AUTH", STATIC_AUTH_METHOD)
    if job_backend not in {POSTGRES_BACKEND, MEMORY_BACKEND}:
        raise ValueError(f"unsupported job backend: {job_backend}")
    if asset_backend not in {GCS_BACKEND, MEMORY_BACKEND}:
        raise ValueError(f"unsupported asset backend: {asset_backend}")
    if worker_auth not in {STATIC_AUTH_METHOD, GOOGLE_OIDC_AUTH_METHOD}:
        raise ValueError(f"unsupported worker auth mode: {worker_auth}")
    read_prefixes = _prefixes(
        environment.get("OGWP_ALLOWED_READ_PREFIXES"), DEFAULT_READ_PREFIXES
    )
    output_prefixes = _prefixes(
        environment.get("OGWP_ALLOWED_OUTPUT_PREFIXES"), DEFAULT_OUTPUT_PREFIXES
    )
    lifecycle: list[Startable] = []
    jobs: JobStore
    registry: WorkerRegistry
    audit: AuditLog
    if job_backend == POSTGRES_BACKEND:
        database_url = _required(environment, "OGWP_DATABASE_URL")
        postgres_jobs = PostgresJobStore(database_url)
        postgres_registry = PostgresWorkerRegistry(database_url)
        postgres_audit = PostgresAuditLog(database_url)
        lifecycle.extend((postgres_jobs, postgres_registry, postgres_audit))
        jobs, registry, audit = postgres_jobs, postgres_registry, postgres_audit
    else:
        jobs, registry, audit = (
            MemoryJobStore(),
            MemoryWorkerRegistry(),
            MemoryAuditLog(),
        )
    assets: AssetStore
    if asset_backend == GCS_BACKEND:
        from outbound_gpu_worker_pool.assets.gcs import GcsAssetStore

        gcs_assets = GcsAssetStore(
            bucket=_required(environment, "OGWP_ASSET_BUCKET"),
            signing_service_account_email=environment.get(
                "OGWP_SIGNING_SERVICE_ACCOUNT"
            ),
            allowed_read_prefixes=read_prefixes,
            allowed_output_prefixes=output_prefixes,
        )
        lifecycle.append(gcs_assets)
        assets = gcs_assets
    else:
        assets = MemoryAssetStore(
            allowed_read_prefixes=read_prefixes,
            allowed_output_prefixes=output_prefixes,
        )
    authenticator: WorkerAuthenticator
    if worker_auth == STATIC_AUTH_METHOD:
        authenticator = StaticTokenWorkerAuthenticator.from_env_value(
            _required(environment, "OGWP_WORKER_TOKENS")
        )
    else:
        authenticator = GoogleIdTokenWorkerAuthenticator(
            audience=_required(environment, "OGWP_WORKER_AUDIENCE"), registry=registry
        )
    service = WorkerPoolService(
        jobs,
        assets,
        registry,
        audit,
        authenticator,
        _capability_schemas(
            environment.get("OGWP_CAPABILITY_PLUGINS", DETERMINISTIC_ECHO_PLUGIN_ID)
        ),
        auth_method=worker_auth,
    )
    return create_coordinator_app(service, lifecycle=lifecycle)


def main(environment: Mapping[str, str] = os.environ) -> int:
    import uvicorn

    uvicorn.run(
        build_from_env(environment),
        host="0.0.0.0",
        port=int(environment.get("OGWP_PORT", str(DEFAULT_PORT))),
    )
    return 0


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _prefixes(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    entries = tuple(entry.strip() for entry in (value or "").split(",") if entry.strip())
    return entries if entries else default


def _capability_schemas(names: str) -> CapabilitySchemas:
    schemas: dict[str, CapabilitySchema] = {}
    for name in names.split(","):
        plugin_id = name.strip()
        if not plugin_id:
            continue
        known = KNOWN_CAPABILITY_SCHEMAS.get(plugin_id)
        if known is None:
            raise ValueError(f"unknown capability plugin: {plugin_id}")
        schemas.update(known)
    if not schemas:
        raise ValueError("OGWP_CAPABILITY_PLUGINS must name at least one plugin")
    return schemas


if __name__ == "__main__":
    raise SystemExit(main())
