"""Framework-free coordinator logic for the outbound GPU worker pool.

`WorkerPoolService` owns every rule a worker request must satisfy: authentication,
rate limiting, revocation, registration, lease selection, worker binding, and
completion verification. It raises typed exceptions; the FastAPI router in
`routes.py` is the only place those become status codes. The host-facing methods
(`submit`, `get`, `cancel`, `list_for_tenant`, and the worker administration
calls) are the seam a host puts its own authenticated routes in front of.

Audit rows carry ids, counts, digests, statuses, and truncated reasons. They never
carry credentials, signed URLs, prompts, or asset bytes. Neither does any record
handed to a host: `claim_token` is the lease secret and leaves only in a grant.
"""

import hashlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from outbound_gpu_worker_pool.contracts import (
    DETERMINISTIC_ECHO_CAPABILITY,
    MAX_AUDIT_REASON_LENGTH,
    PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
    AssetGrant,
    AssetStore,
    AuditEvent,
    AuditEventType,
    AuditLog,
    CapabilitySchema,
    CapabilitySchemas,
    IdentitySubjectTaken,
    JobFailureCode,
    JobPayloadValue,
    JobRecord,
    JobStatus,
    JobStore,
    JobSubmission,
    LeaseGrant,
    OutputManifest,
    WorkerAuthenticator,
    WorkerAuthError,
    WorkerIdentity,
    WorkerRecord,
    WorkerRegistration,
    WorkerRegistry,
    WorkerStatus,
)
from outbound_gpu_worker_pool.validation import validate_capability_id

OUTPUT_UPLOAD_CONTENT_TYPE = "application/octet-stream"
GLOBAL_RATE_LIMIT_KEY = "*"

DEFAULT_CAPABILITY_SCHEMAS: CapabilitySchemas = {
    DETERMINISTIC_ECHO_CAPABILITY: CapabilitySchema(
        capability_id=DETERMINISTIC_ECHO_CAPABILITY,
        contract_version=1,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "seed": {"type": "integer", "minimum": 0},
                "label": {"type": "string", "maxLength": 64},
            },
        },
    )
}


class RateLimited(RuntimeError):
    """The caller exceeded its request budget."""


class WorkerRevoked(PermissionError):
    """The credential is valid but the worker is revoked."""


class WorkerMismatch(ValueError):
    """The registration does not belong to the authenticated worker."""


class WorkerNotRegistered(LookupError):
    """The worker must heartbeat before it can lease work."""


class JobNotFound(LookupError):
    """No job exists under that identifier."""


class StaleLease(RuntimeError):
    """The claim token or worker binding no longer owns the job."""


class CompletionRejected(ValueError):
    """The attested output did not match the leased request."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class RateLimiter:
    """A token bucket per key, refilled continuously from the injected clock."""

    def __init__(self, limit_per_minute: int, clock: Callable[[], datetime]) -> None:
        self._limit = float(limit_per_minute)
        self._clock = clock
        self._buckets: dict[str, tuple[float, datetime]] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        tokens, updated = self._buckets.get(key, (self._limit, now))
        tokens = min(
            self._limit,
            tokens + (now - updated).total_seconds() * self._limit / 60.0,
        )
        allowed = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
        return allowed


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _without_claim_token(record: JobRecord) -> JobRecord:
    return replace(record, claim_token=None)


class WorkerPoolService:
    def __init__(
        self,
        jobs: JobStore,
        assets: AssetStore,
        registry: WorkerRegistry,
        audit: AuditLog,
        authenticator: WorkerAuthenticator,
        capability_schemas: CapabilitySchemas,
        *,
        clock: Callable[[], datetime] = _utc_now,
        auth_method: str = "static",
        per_worker_limit_per_minute: int = 120,
        global_limit_per_minute: int = 1200,
        default_lease_seconds: int = 1200,
        max_verify_bytes: int = 256 * 1024 * 1024,
        grant_ttl_seconds: int = 900,
    ) -> None:
        self._jobs = jobs
        self._assets = assets
        self._registry = registry
        self._audit = audit
        self._authenticator = authenticator
        self._capability_schemas = capability_schemas
        self._clock = clock
        self.auth_method = auth_method
        self._default_lease_seconds = default_lease_seconds
        self._max_verify_bytes = max_verify_bytes
        self._grant_ttl_seconds = grant_ttl_seconds
        self._worker_limiter = RateLimiter(per_worker_limit_per_minute, clock)
        self._global_limiter = RateLimiter(global_limit_per_minute, clock)

    async def submit(self, submission: JobSubmission) -> JobRecord:
        """Insert the job, or replay the record an equal submission created."""
        result = await self._jobs.submit(submission)
        return _without_claim_token(result.record)

    async def get(self, job_id: str) -> JobRecord | None:
        record = await self._jobs.get(job_id)
        return None if record is None else _without_claim_token(record)

    async def cancel(self, job_id: str) -> bool:
        return await self._jobs.cancel(job_id)

    async def list_for_tenant(self, tenant_id: str) -> tuple[JobRecord, ...]:
        return tuple(
            _without_claim_token(record)
            for record in await self._jobs.list_for_tenant(tenant_id)
        )

    async def set_worker_status(self, worker_id: str, status: WorkerStatus) -> bool:
        changed = await self._registry.set_status(worker_id, status)
        if not changed:
            return False
        await self._audit.record(
            AuditEventType.WORKER_STATUS_CHANGED,
            worker_id=worker_id,
            detail={"status": str(status)},
        )
        if status is WorkerStatus.REVOKED:
            await self._audit.record(
                AuditEventType.WORKER_REVOKED, worker_id=worker_id
            )
        return True

    async def list_workers(self) -> tuple[WorkerRecord, ...]:
        return await self._registry.list()

    async def audit_for_job(self, job_id: str) -> tuple[AuditEvent, ...]:
        return await self._audit.list_for_job(job_id)

    async def authenticate(self, authorization: str | None) -> WorkerIdentity:
        try:
            identity = await self._authenticator.authenticate(authorization)
        except WorkerAuthError:
            await self._audit.record(
                AuditEventType.AUTH_REJECTED,
                detail={"reason": "invalid_credential"},
            )
            raise
        if not self._worker_limiter.allow(
            identity.worker_id
        ) or not self._global_limiter.allow(GLOBAL_RATE_LIMIT_KEY):
            await self._audit.record(
                AuditEventType.RATE_LIMITED, worker_id=identity.worker_id
            )
            raise RateLimited(identity.worker_id)
        record = await self._registry.get(identity.worker_id)
        if record is not None and record.status is WorkerStatus.REVOKED:
            raise WorkerRevoked(identity.worker_id)
        return identity

    async def register_heartbeat(
        self, identity: WorkerIdentity, registration: WorkerRegistration
    ) -> WorkerRecord:
        if registration.worker_id != identity.worker_id:
            raise WorkerMismatch(registration.worker_id)
        for capability in registration.capabilities:
            validate_capability_id(capability.capability_id)
        try:
            record = await self._registry.upsert(
                registration, identity_subject=identity.subject
            )
        except IdentitySubjectTaken as exc:
            raise WorkerMismatch(identity.subject) from exc
        await self._audit.record(
            AuditEventType.WORKER_HEARTBEAT,
            worker_id=record.worker_id,
            detail={
                "capabilities": len(registration.capabilities),
                "draining": registration.draining,
            },
        )
        return record

    async def lease(
        self,
        identity: WorkerIdentity,
        capability_ids: tuple[str, ...],
        lease_seconds: int | None = None,
    ) -> LeaseGrant | None:
        worker = await self._registry.get(identity.worker_id)
        if worker is None:
            raise WorkerNotRegistered(identity.worker_id)
        if worker.status is WorkerStatus.DRAINING:
            return None
        registered = set(worker.capability_ids)
        effective = tuple(
            capability_id
            for capability_id in capability_ids
            if capability_id in registered
        )
        if not effective:
            return None
        seconds = (
            lease_seconds if lease_seconds is not None else self._default_lease_seconds
        )
        await self._jobs.expire_exhausted()
        job = await self._jobs.lease(
            worker_id=identity.worker_id,
            capability_ids=effective,
            lease_seconds=seconds,
        )
        if job is None:
            await self._audit.record(
                AuditEventType.LEASE_EMPTY, worker_id=identity.worker_id
            )
            return None
        if job.claim_token is None:
            raise StaleLease(job.job_id)
        now = self._clock()
        expires_at = now + timedelta(seconds=self._grant_ttl_seconds)
        input_grants: list[AssetGrant] = []
        for input_key in job.input_keys:
            input_grants.append(
                AssetGrant(
                    key=input_key,
                    url=await self._assets.create_read_url(input_key),
                    method="GET",
                    expires_at=expires_at,
                )
            )
        output_grant = AssetGrant(
            key=job.output_key,
            url=await self._assets.create_output_upload_url(
                job.output_key, OUTPUT_UPLOAD_CONTENT_TYPE
            ),
            method="PUT",
            content_type=OUTPUT_UPLOAD_CONTENT_TYPE,
            expires_at=expires_at,
        )
        await self._audit.record(
            AuditEventType.LEASE_GRANTED,
            worker_id=identity.worker_id,
            job_id=job.job_id,
            detail={"attempt": job.attempts, "capability_id": job.capability_id},
        )
        return LeaseGrant(
            job_id=job.job_id,
            claim_token=job.claim_token,
            lease_until=now + timedelta(seconds=seconds),
            execution_deadline_seconds=job.execution_deadline_seconds,
            capability_id=job.capability_id,
            contract_version=job.contract_version,
            request_digest=job.request_digest,
            idempotency_key=job.idempotency_key,
            input_keys=job.input_keys,
            output_key=job.output_key,
            payload=job.payload,
            input_grants=tuple(input_grants),
            output_grant=output_grant,
            tenant_id=job.tenant_id,
        )

    async def job_heartbeat(
        self,
        identity: WorkerIdentity,
        job_id: str,
        claim_token: str,
        progress_percent: int | None = None,
    ) -> datetime:
        record = await self._owned_job(identity, job_id)
        accepted = await self._jobs.heartbeat(
            record.job_id,
            claim_token,
            lease_seconds=self._default_lease_seconds,
            progress_percent=progress_percent,
        )
        if not accepted:
            raise StaleLease(job_id)
        await self._audit.record(
            AuditEventType.JOB_HEARTBEAT,
            worker_id=identity.worker_id,
            job_id=record.job_id,
            detail={"progress_percent": progress_percent},
        )
        return self._clock() + timedelta(seconds=self._default_lease_seconds)

    async def complete(
        self,
        identity: WorkerIdentity,
        job_id: str,
        claim_token: str,
        manifest: OutputManifest,
    ) -> None:
        record = await self._owned_job(identity, job_id)
        try:
            await self._verify_output(record, manifest)
        except CompletionRejected as exc:
            await self._jobs.fail(
                record.job_id,
                claim_token,
                exc.reason,
                failure_code=JobFailureCode.INVALID_INPUT,
                retryable=False,
            )
            await self._audit.record(
                AuditEventType.COMPLETION_REJECTED,
                worker_id=identity.worker_id,
                job_id=record.job_id,
                detail={"reason": exc.reason},
            )
            raise
        accepted = await self._jobs.complete(
            record.job_id,
            claim_token,
            content_type=manifest.content_type,
            sha256=manifest.sha256,
            byte_length=manifest.byte_length,
        )
        if not accepted:
            raise StaleLease(job_id)
        await self._audit.record(
            AuditEventType.JOB_COMPLETED,
            worker_id=identity.worker_id,
            job_id=record.job_id,
            detail={
                "byte_length": manifest.byte_length,
                "sha256": manifest.sha256,
                "plugin_id": manifest.plugin_id,
                "plugin_version": manifest.plugin_version,
            },
        )

    async def fail(
        self,
        identity: WorkerIdentity,
        job_id: str,
        claim_token: str,
        reason: str,
        *,
        retryable: bool,
        failure_code: JobFailureCode,
    ) -> None:
        record = await self._owned_job(identity, job_id)
        accepted = await self._jobs.fail(
            record.job_id,
            claim_token,
            reason,
            failure_code=failure_code,
            retryable=retryable,
        )
        if not accepted:
            raise StaleLease(job_id)
        await self._audit.record(
            AuditEventType.JOB_FAILED,
            worker_id=identity.worker_id,
            job_id=record.job_id,
            detail={
                "reason": reason[:MAX_AUDIT_REASON_LENGTH],
                "retryable": retryable,
                "failure_code": str(failure_code),
            },
        )

    async def release(
        self, identity: WorkerIdentity, job_id: str, claim_token: str, reason: str
    ) -> None:
        """Requeue the job, or settle it terminally once its attempts are spent."""
        record = await self._owned_job(identity, job_id)
        if record.attempts >= record.attempt_budget:
            accepted = await self._jobs.fail(
                record.job_id,
                claim_token,
                reason,
                failure_code=JobFailureCode.TEMPORARY_FAILURE,
                retryable=True,
            )
            if not accepted:
                raise StaleLease(job_id)
            await self._audit.record(
                AuditEventType.JOB_FAILED,
                worker_id=identity.worker_id,
                job_id=record.job_id,
                detail={
                    "reason": reason[:MAX_AUDIT_REASON_LENGTH],
                    "retryable": True,
                    "failure_code": str(JobFailureCode.TEMPORARY_FAILURE),
                    "budget_exhausted": True,
                },
            )
            return
        accepted = await self._jobs.release(record.job_id, claim_token, reason)
        if not accepted:
            raise StaleLease(job_id)
        await self._audit.record(
            AuditEventType.JOB_RELEASED,
            worker_id=identity.worker_id,
            job_id=record.job_id,
            detail={"reason": reason[:MAX_AUDIT_REASON_LENGTH]},
        )

    def capabilities_schema(self) -> dict[str, dict[str, JobPayloadValue]]:
        return {
            "capabilities": {
                capability_id: {
                    "contract_version": schema.contract_version,
                    "input_schema": schema.input_schema,
                }
                for capability_id, schema in self._capability_schemas.items()
            }
        }

    async def _owned_job(self, identity: WorkerIdentity, job_id: str) -> JobRecord:
        record = await self._jobs.get(job_id)
        if record is None:
            raise JobNotFound(job_id)
        if (
            record.leased_by != identity.worker_id
            or record.status is not JobStatus.PROCESSING
        ):
            raise StaleLease(job_id)
        return record

    async def _verify_output(self, record: JobRecord, manifest: OutputManifest) -> None:
        if manifest.output_key != record.output_key:
            raise CompletionRejected("output_key_mismatch")
        if manifest.idempotency_key != record.idempotency_key:
            raise CompletionRejected("idempotency_key_mismatch")
        if manifest.request_digest != record.request_digest:
            raise CompletionRejected("request_digest_mismatch")
        if manifest.publication_mode != PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE:
            raise CompletionRejected("publication_mode_mismatch")
        descriptor = await self._assets.describe(record.output_key)
        if descriptor is None:
            raise CompletionRejected("output_missing")
        if descriptor.size != manifest.byte_length:
            raise CompletionRejected("byte_length_mismatch")
        if manifest.byte_length > self._max_verify_bytes:
            raise CompletionRejected("output_too_large")
        content = await self._assets.read_limited(
            record.output_key, self._max_verify_bytes
        )
        if hashlib.sha256(content).hexdigest() != manifest.sha256:
            raise CompletionRejected("sha256_mismatch")
