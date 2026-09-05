"""Postgres-backed job store, worker registry, and audit log.

The lease is a single compare-and-set statement over `FOR UPDATE SKIP LOCKED`, and
every settlement predicate is fenced on `(job_id, claim_token)`, so a replaced
claim token can neither heartbeat nor settle a job. Migrations belong to
`PostgresJobStore.start()`; the registry and the audit log never apply DDL.
"""

import json
from dataclasses import asdict
from importlib.resources import files
from typing import Any
from uuid import uuid4

import asyncpg

from outbound_gpu_worker_pool.contracts import (
    ATTEMPT_BUDGET_EXHAUSTED,
    AuditEvent,
    AuditEventType,
    CapabilityQueueDepth,
    CostClass,
    GpuTelemetry,
    IdempotencyConflict,
    IdentitySubjectTaken,
    JobFailureCode,
    JobPayloadValue,
    JobRecord,
    JobStatus,
    JobSubmission,
    QueueDepth,
    SubmissionResult,
    WorkerCapability,
    WorkerRecord,
    WorkerRegistration,
    WorkerStatus,
)
from outbound_gpu_worker_pool.validation import (
    job_request_digest,
    validate_job_submission,
)

POOL_JOBS_TABLE = "pool_jobs"
POOL_WORKERS_TABLE = "pool_workers"
POOL_AUDIT_EVENTS_TABLE = "pool_audit_events"

MIGRATIONS_PACKAGE = "outbound_gpu_worker_pool.migrations"
MIGRATION_FILES = ("001_worker_pool.sql", "002_worker_telemetry.sql")
MIGRATION_ADVISORY_LOCK_KEY = "outbound_gpu_worker_pool.migrations"


class _PoolOwner:
    def __init__(self, database_url: str, *, pool: Any | None = None) -> None:
        self._database_url = database_url
        self._pool = pool
        self._owns_pool = False

    async def start(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._database_url, min_size=1, max_size=4
            )
            self._owns_pool = True

    async def stop(self) -> None:
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None
            self._owns_pool = False

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgreSQL worker pool store has not started")
        return self._pool

    async def _execute(self, query: str, *args: Any) -> bool:
        async with self._require_pool().acquire() as connection:
            result = await connection.execute(query, *args)
        return result == "UPDATE 1"


class PostgresJobStore(_PoolOwner):
    async def start(self) -> None:
        await super().start()
        migrations = files(MIGRATIONS_PACKAGE)
        async with self._require_pool().acquire() as connection:
            await connection.execute(
                f"SELECT pg_advisory_lock(hashtext('{MIGRATION_ADVISORY_LOCK_KEY}'))"
            )
            try:
                for name in MIGRATION_FILES:
                    await connection.execute(migrations.joinpath(name).read_text())
            finally:
                await connection.execute(
                    f"SELECT pg_advisory_unlock(hashtext('{MIGRATION_ADVISORY_LOCK_KEY}'))"
                )

    async def submit(self, submission: JobSubmission) -> SubmissionResult:
        validate_job_submission(submission)
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                f"""
                INSERT INTO {POOL_JOBS_TABLE} (
                    job_id, idempotency_key, capability_id, contract_version,
                    input_keys, output_key, payload, tenant_id, priority,
                    attempt_budget, execution_deadline_seconds, request_digest
                )
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8, $9, $10,
                        $11, $12)
                ON CONFLICT (idempotency_key) DO UPDATE
                    SET idempotency_key = EXCLUDED.idempotency_key
                    WHERE {POOL_JOBS_TABLE}.capability_id = EXCLUDED.capability_id
                      AND {POOL_JOBS_TABLE}.contract_version = EXCLUDED.contract_version
                      AND {POOL_JOBS_TABLE}.input_keys = EXCLUDED.input_keys
                      AND {POOL_JOBS_TABLE}.output_key = EXCLUDED.output_key
                      AND {POOL_JOBS_TABLE}.payload = EXCLUDED.payload
                      AND {POOL_JOBS_TABLE}.tenant_id IS NOT DISTINCT FROM EXCLUDED.tenant_id
                      AND {POOL_JOBS_TABLE}.priority = EXCLUDED.priority
                      AND {POOL_JOBS_TABLE}.attempt_budget = EXCLUDED.attempt_budget
                      AND {POOL_JOBS_TABLE}.execution_deadline_seconds
                          = EXCLUDED.execution_deadline_seconds
                RETURNING *, (xmax = 0) AS created
                """,
                submission.job_id,
                submission.idempotency_key,
                submission.capability_id,
                submission.contract_version,
                json.dumps(list(submission.input_keys)),
                submission.output_key,
                json.dumps(submission.payload),
                submission.tenant_id,
                submission.priority,
                submission.attempt_budget,
                submission.execution_deadline_seconds,
                job_request_digest(submission),
            )
        if row is None:
            raise IdempotencyConflict(submission.idempotency_key)
        return SubmissionResult(record=_record(row), created=row["created"])

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT * FROM {POOL_JOBS_TABLE} WHERE job_id = $1::uuid", job_id
            )
        return _record(row) if row is not None else None

    async def list_for_tenant(self, tenant_id: str) -> tuple[JobRecord, ...]:
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(
                f"""SELECT * FROM {POOL_JOBS_TABLE} WHERE tenant_id = $1
                    ORDER BY created_at, job_id""",
                tenant_id,
            )
        return tuple(_record(row) for row in rows)

    async def cancel(self, job_id: str) -> bool:
        return await self._execute(
            f"""UPDATE {POOL_JOBS_TABLE}
                   SET status = 'cancelled', cancelled_at = now(), claim_token = NULL,
                       leased_by = NULL, lease_until = NULL, updated_at = now()
               WHERE job_id = $1::uuid AND status IN ('queued', 'processing')""",
            job_id,
        )

    async def lease(
        self, *, worker_id: str, capability_ids: tuple[str, ...], lease_seconds: int
    ) -> JobRecord | None:
        token = uuid4().hex
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                f"""
                WITH candidate AS (
                    SELECT job_id FROM {POOL_JOBS_TABLE}
                    WHERE capability_id = ANY($1::text[])
                      AND attempts < attempt_budget
                      AND (status = 'queued'
                           OR (status = 'processing' AND lease_until < now()))
                    ORDER BY priority ASC, created_at ASC, job_id ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE {POOL_JOBS_TABLE} AS j
                   SET status = 'processing', attempts = j.attempts + 1,
                       claim_token = $2, leased_by = $3,
                       lease_until = now() + ($4::int * interval '1 second'),
                       updated_at = now()
                  FROM candidate
                 WHERE j.job_id = candidate.job_id
                RETURNING j.*
                """,
                list(capability_ids),
                token,
                worker_id,
                lease_seconds,
            )
        return _record(row) if row is not None else None

    async def heartbeat(
        self,
        job_id: str,
        claim_token: str,
        *,
        lease_seconds: int,
        progress_percent: int | None = None,
    ) -> bool:
        return await self._execute(
            f"""UPDATE {POOL_JOBS_TABLE}
                    SET lease_until = now() + ($3::int * interval '1 second'),
                        progress_percent = COALESCE($4, progress_percent),
                        updated_at = now()
                WHERE job_id = $1::uuid AND status = 'processing'
                  AND claim_token = $2""",
            job_id,
            claim_token,
            lease_seconds,
            progress_percent,
        )

    async def complete(
        self,
        job_id: str,
        claim_token: str,
        *,
        content_type: str,
        sha256: str,
        byte_length: int,
    ) -> bool:
        return await self._execute(
            f"""UPDATE {POOL_JOBS_TABLE}
                    SET status = 'completed', output_content_type = $3,
                        output_sha256 = $4, output_byte_length = $5, error = NULL,
                        claim_token = NULL, leased_by = NULL, lease_until = NULL,
                        updated_at = now()
                WHERE job_id = $1::uuid AND status = 'processing'
                  AND claim_token = $2""",
            job_id,
            claim_token,
            content_type,
            sha256,
            byte_length,
        )

    async def fail(
        self,
        job_id: str,
        claim_token: str,
        reason: str,
        *,
        failure_code: JobFailureCode = JobFailureCode.TEMPORARY_FAILURE,
        failure_message: str | None = None,
        retryable: bool = True,
    ) -> bool:
        return await self._execute(
            f"""UPDATE {POOL_JOBS_TABLE}
                    SET status = 'failed', error = $3, failure_code = $4,
                        failure_message = $5, retryable = $6, claim_token = NULL,
                        leased_by = NULL, lease_until = NULL, updated_at = now()
                WHERE job_id = $1::uuid AND status = 'processing'
                  AND claim_token = $2""",
            job_id,
            claim_token,
            reason,
            failure_code,
            failure_message,
            retryable,
        )

    async def release(self, job_id: str, claim_token: str, reason: str) -> bool:
        return await self._execute(
            f"""UPDATE {POOL_JOBS_TABLE}
                    SET status = 'queued', error = $3, claim_token = NULL,
                        leased_by = NULL, lease_until = NULL, updated_at = now()
                WHERE job_id = $1::uuid AND status = 'processing'
                  AND claim_token = $2""",
            job_id,
            claim_token,
            reason,
        )

    async def expire_exhausted(self) -> int:
        async with self._require_pool().acquire() as connection:
            result = await connection.execute(
                f"""UPDATE {POOL_JOBS_TABLE}
                        SET status = 'failed', error = $1,
                            failure_code = 'temporary_failure', retryable = true,
                            claim_token = NULL, leased_by = NULL, lease_until = NULL,
                            updated_at = now()
                    WHERE status = 'processing' AND lease_until < now()
                      AND attempts >= attempt_budget""",
                ATTEMPT_BUDGET_EXHAUSTED,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def queue_depth(self) -> QueueDepth:
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(
                f"""SELECT capability_id, status, count(*) AS n
                    FROM {POOL_JOBS_TABLE}
                    WHERE status IN ('queued', 'processing')
                    GROUP BY capability_id, status"""
            )
        totals = {"queued": 0, "processing": 0}
        by_capability: dict[str, dict[str, int]] = {}
        for row in rows:
            status = str(row["status"])
            count = int(row["n"])
            totals[status] += count
            counts = by_capability.setdefault(
                row["capability_id"], {"queued": 0, "processing": 0}
            )
            counts[status] = count
        return QueueDepth(
            queued=totals["queued"],
            processing=totals["processing"],
            by_capability={
                capability_id: CapabilityQueueDepth(**counts)
                for capability_id, counts in by_capability.items()
            },
        )


class PostgresWorkerRegistry(_PoolOwner):
    async def upsert(
        self, registration: WorkerRegistration, *, identity_subject: str
    ) -> WorkerRecord:
        status = WorkerStatus.DRAINING if registration.draining else WorkerStatus.ACTIVE
        async with self._require_pool().acquire() as connection:
            try:
                row = await connection.fetchrow(
                    f"""
                    INSERT INTO {POOL_WORKERS_TABLE} (
                        worker_id, identity_subject, status, capabilities, gpu_model,
                        vram_mb, runtime_versions, cost_class, labels, active_leases,
                        last_heartbeat_at, updated_at, gpus, busy_job_id
                    )
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7::jsonb, $8, $9::jsonb,
                            $10, now(), now(), $11::jsonb, $12::uuid)
                    ON CONFLICT (worker_id) DO UPDATE SET
                        identity_subject = EXCLUDED.identity_subject,
                        status = CASE
                            WHEN {POOL_WORKERS_TABLE}.status = 'revoked' THEN 'revoked'
                            ELSE EXCLUDED.status
                        END,
                        capabilities = EXCLUDED.capabilities,
                        gpu_model = EXCLUDED.gpu_model,
                        vram_mb = EXCLUDED.vram_mb,
                        runtime_versions = EXCLUDED.runtime_versions,
                        cost_class = EXCLUDED.cost_class,
                        labels = EXCLUDED.labels,
                        active_leases = EXCLUDED.active_leases,
                        last_heartbeat_at = now(),
                        updated_at = now(),
                        gpus = EXCLUDED.gpus,
                        busy_job_id = EXCLUDED.busy_job_id
                    RETURNING *
                    """,
                    registration.worker_id,
                    identity_subject,
                    status,
                    json.dumps([asdict(item) for item in registration.capabilities]),
                    registration.gpu_model,
                    registration.vram_mb,
                    json.dumps(registration.runtime_versions),
                    registration.cost_class,
                    json.dumps(list(registration.labels)),
                    registration.active_leases,
                    json.dumps([asdict(item) for item in registration.gpus]),
                    registration.busy_job_id,
                )
            except asyncpg.UniqueViolationError as exc:
                raise IdentitySubjectTaken(identity_subject) from exc
        return _worker_record(row)

    async def get(self, worker_id: str) -> WorkerRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT * FROM {POOL_WORKERS_TABLE} WHERE worker_id = $1", worker_id
            )
        return _worker_record(row) if row is not None else None

    async def find_by_identity_subject(self, subject: str) -> WorkerRecord | None:
        async with self._require_pool().acquire() as connection:
            row = await connection.fetchrow(
                f"SELECT * FROM {POOL_WORKERS_TABLE} WHERE identity_subject = $1",
                subject,
            )
        return _worker_record(row) if row is not None else None

    async def list(self) -> tuple[WorkerRecord, ...]:
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(
                f"SELECT * FROM {POOL_WORKERS_TABLE} ORDER BY worker_id"
            )
        return tuple(_worker_record(row) for row in rows)

    async def set_status(self, worker_id: str, status: WorkerStatus) -> bool:
        return await self._execute(
            f"""UPDATE {POOL_WORKERS_TABLE}
                    SET status = $2,
                        revoked_at = CASE
                            WHEN $2 = 'revoked' THEN now() ELSE revoked_at
                        END,
                        updated_at = now()
                WHERE worker_id = $1""",
            worker_id,
            status,
        )


class PostgresAuditLog(_PoolOwner):
    async def record(
        self,
        event_type: AuditEventType,
        *,
        worker_id: str | None = None,
        job_id: str | None = None,
        detail: dict[str, JobPayloadValue] | None = None,
    ) -> None:
        async with self._require_pool().acquire() as connection:
            await connection.execute(
                f"""INSERT INTO {POOL_AUDIT_EVENTS_TABLE}
                        (event_type, worker_id, job_id, detail)
                    VALUES ($1, $2, $3::uuid, $4::jsonb)""",
                event_type,
                worker_id,
                job_id,
                json.dumps(detail or {}),
            )

    async def list_for_job(self, job_id: str) -> tuple[AuditEvent, ...]:
        async with self._require_pool().acquire() as connection:
            rows = await connection.fetch(
                f"""SELECT * FROM {POOL_AUDIT_EVENTS_TABLE}
                    WHERE job_id = $1::uuid ORDER BY event_id""",
                job_id,
            )
        return tuple(_audit_event(row) for row in rows)


def _record(row: Any) -> JobRecord:
    return JobRecord(
        job_id=str(row["job_id"]),
        idempotency_key=row["idempotency_key"],
        capability_id=row["capability_id"],
        status=JobStatus(row["status"]),
        input_keys=tuple(_json_value(row["input_keys"])),
        output_key=row["output_key"],
        attempts=row["attempts"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        contract_version=row["contract_version"],
        payload=_json_value(row["payload"]),
        tenant_id=row["tenant_id"],
        priority=row["priority"],
        attempt_budget=row["attempt_budget"],
        execution_deadline_seconds=row["execution_deadline_seconds"],
        request_digest=row["request_digest"],
        claim_token=row["claim_token"],
        leased_by=row["leased_by"],
        lease_until=row["lease_until"],
        output_content_type=row["output_content_type"],
        output_sha256=row["output_sha256"],
        output_byte_length=row["output_byte_length"],
        error=row["error"],
        failure_code=(
            JobFailureCode(row["failure_code"]) if row["failure_code"] else None
        ),
        failure_message=row["failure_message"],
        retryable=row["retryable"],
        progress_percent=row["progress_percent"],
        cancelled_at=row["cancelled_at"],
    )


def _worker_record(row: Any) -> WorkerRecord:
    return WorkerRecord(
        worker_id=row["worker_id"],
        identity_subject=row["identity_subject"],
        status=WorkerStatus(row["status"]),
        capabilities=tuple(
            WorkerCapability(**capability)
            for capability in _json_value(row["capabilities"])
        ),
        gpu_model=row["gpu_model"],
        vram_mb=row["vram_mb"],
        runtime_versions=_json_value(row["runtime_versions"]),
        cost_class=CostClass(row["cost_class"]),
        labels=tuple(_json_value(row["labels"])),
        active_leases=row["active_leases"],
        last_heartbeat_at=row["last_heartbeat_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        revoked_at=row["revoked_at"],
        gpus=tuple(GpuTelemetry(**gpu) for gpu in _json_value(row["gpus"])),
        busy_job_id=(
            str(row["busy_job_id"]) if row["busy_job_id"] is not None else None
        ),
    )


def _audit_event(row: Any) -> AuditEvent:
    job_id = row["job_id"]
    return AuditEvent(
        event_type=AuditEventType(row["event_type"]),
        worker_id=row["worker_id"],
        job_id=str(job_id) if job_id is not None else None,
        detail=_json_value(row["detail"]),
        created_at=row["created_at"],
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value
