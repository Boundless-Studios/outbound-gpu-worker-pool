---
artifact: plan
status: active
source_issue: none
branch: extract-library
---

# Extract the worker pool into a provider-neutral library

**Goal:** turn the first product-embedded implementation of this design (a `worker_pool`
package inside a private video service, with its migration and tests) into this public,
provider-neutral Python package, then have that product consume the library.

**Source of truth for behavior:** the product implementation's tests, which encode every
acceptance criterion of the design's Phase 1. Port the behavior and the tests; do not
redesign. `docs/design.md` is the architecture.

**Package:** `outbound_gpu_worker_pool` (distribution `outbound-gpu-worker-pool`),
`src/` layout, hatchling, Python 3.13, MIT. Core depends only on pydantic; `coordinator`
(asyncpg, fastapi), `agent` (httpx), `gcs`, and `google-auth` are extras so a worker
machine installs no server code and the coordinator installs no HTTP client.

**Run tests:** `pip install -e ".[dev]" && pytest -q`. Postgres tests read
`OUTBOUND_GPU_WORKER_POOL_TEST_DATABASE_URL` (default
`postgresql://postgres:test@127.0.0.1:15432/worker_pool`; any database works because every
test creates its own schema).

## Generalization rules (what changes versus the private code)

1. **Own job model.** No `VideoJob`, `JobOperation`, `requested_outputs`, `project_id`,
   `manifest_operation_id`, or `result` derivatives. The pool job is:
   ```python
   @dataclass(frozen=True)
   class JobSubmission:
       job_id: str; idempotency_key: str; capability_id: str; contract_version: int = 1
       input_keys: tuple[str, ...]; output_key: str; payload: JobPayload = {}
       tenant_id: str | None = None; priority: int = 100; attempt_budget: int = 5
       execution_deadline_seconds: int = 1200
   ```
   `JobRecord` adds `status`, `attempts`, `request_digest`, `leased_by`, `lease_until`,
   `claim_token` (never serialized outward), `output_content_type`, `output_sha256`,
   `output_byte_length`, `error`, `failure_code`, `failure_message`, `retryable`,
   `progress_percent`, `created_at`, `updated_at`, `cancelled_at`. Status enum:
   `queued, processing, completed, failed, cancelled`. Failure codes:
   `invalid_input, unsupported_operation, provider_unavailable, temporary_failure`.
   `job_request_digest` hashes `{capability_id, contract_version, input_keys, output_key, payload}`.
   Payload validation (JSON tree, depth, bytes, finite numbers) is lifted verbatim.
2. **Own tables.** Migration `001_worker_pool.sql` creates `pool_jobs`, `pool_workers`,
   `pool_audit_events` with the same columns and indexes as the private `006` plus the job
   columns above. `PostgresJobStore.start()` applies migrations under an advisory lock;
   registry and audit stores never apply DDL. Every table name is a module constant so a
   host can read them.
3. **Asset seam is a small Protocol** named `AssetStore`:
   `create_read_url(key)`, `create_output_upload_url(key, content_type)`, `describe(key)`,
   `read_limited(key, max_bytes)`, `write_once(key, content, content_type)` (the last only
   for tests and the memory implementation). Namespace policy (which prefixes may be read or
   written) is a constructor argument `allowed_prefixes`, not hardcoded.
4. **Multiple inputs.** `LeaseGrant.input_grants` has one grant per `input_keys` entry, in
   order; `ExecutionContext.input_paths` is keyed by asset key. The deterministic echo plugin
   hashes all inputs in key order.
5. **No product routes.** The library ships `create_worker_router(service)` and a
   `create_coordinator_app(service, *, health=...)` factory that mounts `/health` and the
   router. Job submission is `WorkerPoolService.submit(JobSubmission) -> JobRecord`,
   `get(job_id)`, `cancel(job_id)`, `list_for_tenant(tenant_id)` — host apps put their own
   authenticated routes in front. Admin operations are service methods too:
   `set_worker_status(worker_id, status)`, `list_workers()`, `audit_for_job(job_id)`.
6. **Environment wiring is optional.** `outbound_gpu_worker_pool.coordinator.__main__`
   runs a standalone coordinator from `OGWP_*` variables (database URL, worker auth mode,
   tokens or audience, asset backend `gcs|memory`, bucket, signing account,
   allowed prefixes); `outbound_gpu_worker_pool.agent.__main__` runs an agent from
   `OGWP_WORKER_*` variables. Both are thin.
7. **Everything else is a straight port**: worker registry, audit log, authenticators,
   rate limiter, completion verification order, release-with-budget semantics,
   `PluginRequestRejected` terminal gate, agent backoff/drain/heartbeat/workspace cleanup,
   log redaction, `HttpAssetTransfer` create-once PUT with 412 as already-published.

## Module layout

```
src/outbound_gpu_worker_pool/
  __init__.py            re-exports the public contracts
  contracts.py           job + worker + grant + manifest models, Protocols, exceptions
  validation.py          payload/json-tree validation, capability id, digest
  memory.py              MemoryJobStore, MemoryAssetStore, MemoryWorkerRegistry,
                         MemoryAuditLog, MemoryWorkerAuthenticator, MemoryAssetTransfer
  postgres.py            PostgresJobStore, PostgresWorkerRegistry, PostgresAuditLog
  migrations/001_worker_pool.sql
  assets/gcs.py          GcsAssetStore (extra: gcs)
  auth.py                StaticTokenWorkerAuthenticator, GoogleIdTokenWorkerAuthenticator
  service.py             WorkerPoolService, RateLimiter, exceptions
  routes.py              /worker/v1 router (extra: coordinator)
  coordinator/app.py     create_coordinator_app; coordinator/__main__.py env runner
  plugins.py             plugin contract + DeterministicEchoPlugin + schemas helper
  agent.py               AssetTransfer, HttpAssetTransfer, WorkerAgent (extra: agent)
  agent/__main__.py      env runner (module path: outbound_gpu_worker_pool.agent_main)
tests/
  test_validation.py, test_memory_store.py, test_postgres_integration.py, test_gcs.py,
  test_auth.py, test_service.py, test_routes.py, test_agent.py, test_agent_main.py,
  test_coordinator_app.py
```

Note: `agent.py` (module) and `agent/` (package) cannot coexist; use `agent.py` plus
`agent_main.py` for the runner, and `coordinator.py` plus `coordinator_main.py` likewise.

## Tasks

### Task 1 — contracts, validation, stores, migrations (RED tests first)
- Port `tests/test_worker_pool_integration.py` and `tests/test_memory_worker_pool.py` to the
  new job model as `tests/test_postgres_integration.py` and `tests/test_memory_store.py`;
  add `tests/test_validation.py`. Run: RED.
- Implement `contracts.py`, `validation.py`, `memory.py`, `postgres.py`,
  `migrations/001_worker_pool.sql`, `assets/gcs.py` (+ `tests/test_gcs.py` ported).
- GREEN. Commit `feat: add durable capability leases, registry, and audit log`.

### Task 2 — auth, service, routes, coordinator app
- Port `tests/test_worker_pool_api.py` to `tests/test_routes.py` (+ `tests/test_service.py`
  for submit/get/cancel/list/admin methods and `tests/test_coordinator_app.py` for the
  factory and env runner, `tests/test_auth.py`). RED.
- Implement `auth.py`, `service.py`, `routes.py`, `coordinator.py`, `coordinator_main.py`.
- GREEN. Commit `feat: add authenticated coordinator API`.

### Task 3 — plugins, agent, runner, README
- Port `tests/test_worker_agent.py` and `tests/test_worker_pool_main.py` to
  `tests/test_agent.py` and `tests/test_agent_main.py` (multi-input echo). RED.
- Implement `plugins.py`, `agent.py`, `agent_main.py`, `MemoryAssetTransfer`.
- Write `README.md`: what it is, the trust model in five bullets, install extras,
  quickstart (memory coordinator + agent in one process), host integration
  (submit/get/cancel), enrolling and revoking workers, running the agent, config tables for
  both runners, link to `docs/design.md`, status "slice 1 / alpha".
- GREEN. Commit `feat: add plugin contract and outbound worker agent`.

### Task 4 — the originating product consumes the library (separate repo, separate PR)
- Add the dependency, delete the product's embedded `worker_pool/` package, keep pool
  columns on the product's own job table only if its UI needs them; otherwise submit pool
  jobs through the library's store and map completion back into the product's job model.
  Out of scope for this repository's first PR.
