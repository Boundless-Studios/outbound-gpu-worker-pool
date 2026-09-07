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
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from outbound_gpu_worker_pool.contracts import (
    MAX_AUDIT_REASON_LENGTH,
    POOL_WORKER_ONLINE_WINDOW_SECONDS,
    POOL_WORKER_VISIBILITY_WINDOW_SECONDS,
    PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
    AssetGrant,
    AssetStore,
    AuditEvent,
    AuditEventType,
    AuditLog,
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
    PoolWorkerStatus,
    PoolWorkerView,
    QueueDepth,
    WorkerAuthenticator,
    WorkerAuthError,
    WorkerAuthBusy,
    WorkerIdentityMismatch,
    WorkerTenantMismatch,
    WorkerIdentity,
    WorkerRecord,
    WorkerRegistration,
    WorkerRegistry,
    WorkerStatus,
)
from outbound_gpu_worker_pool.enrollment import WorkerEnrollment, validate_enrollments
from outbound_gpu_worker_pool.validation import validate_capability_id

OUTPUT_UPLOAD_CONTENT_TYPE = "application/octet-stream"
GLOBAL_RATE_LIMIT_KEY = "*"


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
    """Bounded per-key token buckets; idle keys expire after a full refill.

    Active buckets are never evicted to admit a new key: doing so would let key
    churn reset a caller's budget. At capacity, new keys fail closed. Source
    checks precede the global budget, so one saturated peer cannot consume it.
    """

    def __init__(
        self, limit_per_minute: int, clock: Callable[[], datetime], *, max_keys: int = 4096
    ) -> None:
        if limit_per_minute < 1 or max_keys < 1:
            raise ValueError("rate limits and bucket capacity must be positive")
        self._limit = float(limit_per_minute)
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, datetime]] = OrderedDict()

    def allow(self, key: str) -> bool:
        now = self._clock()
        while self._buckets:
            _, (_, oldest) = next(iter(self._buckets.items()))
            if (now - oldest).total_seconds() < 60:
                break
            self._buckets.popitem(last=False)
        if key not in self._buckets and len(self._buckets) >= self._max_keys:
            return False
        tokens, updated = self._buckets.get(key, (self._limit, now))
        tokens = min(
            self._limit,
            tokens + max(0.0, (now - updated).total_seconds()) * self._limit / 60.0,
        )
        allowed = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
        self._buckets.move_to_end(key)
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
        enrollments: Mapping[str, WorkerEnrollment] | None = None,
        pre_auth_limit_per_minute: int = 1200,
        per_source_limit_per_minute: int = 120,
        auth_audit_limit_per_minute: int = 30,
        max_auth_concurrency: int = 16,
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
        self._enrollments = validate_enrollments(enrollments)
        self._pre_auth_limiter = RateLimiter(pre_auth_limit_per_minute, clock, max_keys=1)
        self._source_limiter = RateLimiter(per_source_limit_per_minute, clock)
        self._auth_audit_limiter = RateLimiter(auth_audit_limit_per_minute, clock, max_keys=1)
        if max_auth_concurrency < 1:
            raise ValueError("authentication concurrency must be positive")
        self._max_auth_concurrency = max_auth_concurrency
        self._active_auth = 0

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

    async def worker_views(
        self,
        *,
        tenant_id: str | None = None,
        online_window_seconds: int = POOL_WORKER_ONLINE_WINDOW_SECONDS,
        visibility_window_seconds: int = POOL_WORKER_VISIBILITY_WINDOW_SECONDS,
    ) -> tuple[PoolWorkerView, ...]:
        """The rows for a host's `GET /pool/workers`.

        `tenant_id` narrows to one tenant's machines exactly, so a host can show a
        user their own hardware; omit it for every worker the pool knows. A worker
        never heard from, or not heard from in `visibility_window_seconds`
        (default 24h), is omitted entirely rather than reported offline.
        """
        now = self._clock()
        views: list[PoolWorkerView] = []
        records = (
            await self._registry.list()
            if tenant_id is None
            else await self._registry.list_by_tenant(tenant_id)
        )
        for record in records:
            if record.last_heartbeat_at is None:
                continue
            age_seconds = (now - record.last_heartbeat_at).total_seconds()
            if age_seconds > visibility_window_seconds:
                continue
            draining = record.status is WorkerStatus.DRAINING
            if record.busy_job_id is not None:
                status = PoolWorkerStatus.BUSY
            elif draining:
                status = PoolWorkerStatus.DRAINING
            elif age_seconds <= online_window_seconds:
                status = PoolWorkerStatus.ONLINE
            else:
                status = PoolWorkerStatus.OFFLINE
            views.append(
                PoolWorkerView(
                    worker_id=record.worker_id,
                    status=status,
                    last_heartbeat_at=record.last_heartbeat_at,
                    capability_ids=record.capability_ids,
                    gpus=record.gpus,
                    busy_job_id=record.busy_job_id,
                    draining=draining,
                    tenant_id=record.tenant_id,
                )
            )
        return tuple(views)

    async def queue_depth(self) -> QueueDepth:
        """Bounded queue depth plus online worker counts for `GET /pool/queue`.

        The worker counts come from the registry's own bounded rows, so telling
        "nothing can run this capability" from "waiting its turn" costs no job scan.
        """
        depth = await self._jobs.queue_depth()
        now = self._clock()
        online: dict[str, int] = {}
        for record in await self._registry.list():
            if record.last_heartbeat_at is None:
                continue
            if record.status is WorkerStatus.REVOKED:
                continue
            age_seconds = (now - record.last_heartbeat_at).total_seconds()
            if age_seconds > POOL_WORKER_ONLINE_WINDOW_SECONDS:
                continue
            for capability_id in record.capability_ids:
                online[capability_id] = online.get(capability_id, 0) + 1
        return replace(depth, online_workers_by_capability=online)

    async def audit_for_job(self, job_id: str) -> tuple[AuditEvent, ...]:
        return await self._audit.list_for_job(job_id)

    async def enroll_worker(
        self, worker_id: str, enrollment: WorkerEnrollment
    ) -> WorkerRecord:
        """Host/admin-only admission. Never expose this as a worker endpoint.

        Enrollment is idempotent but not an identity/tenant rotation API. Existing
        bindings cannot be changed with this method or a worker heartbeat.
        """
        validate_enrollments({worker_id: enrollment})
        existing = await self._registry.get(worker_id)
        if existing is not None:
            if existing.identity_subject != enrollment.identity_subject:
                raise WorkerIdentityMismatch(worker_id)
            if existing.tenant_id != enrollment.tenant_id:
                raise WorkerTenantMismatch(worker_id)
            return existing
        return await self._registry.upsert(
            WorkerRegistration(worker_id=worker_id, capabilities=(), tenant_id=enrollment.tenant_id),
            identity_subject=enrollment.identity_subject,
        )

    async def _approved_enrollment(self, identity: WorkerIdentity) -> WorkerEnrollment:
        record = await self._registry.get(identity.worker_id)
        if record is not None:
            if record.identity_subject != identity.subject:
                raise WorkerIdentityMismatch(identity.worker_id)
            if record.status is WorkerStatus.REVOKED:
                raise WorkerRevoked(identity.worker_id)
            return WorkerEnrollment(record.identity_subject, record.tenant_id)
        enrollment = self._enrollments.get(identity.worker_id)
        if enrollment is None or enrollment.identity_subject != identity.subject:
            raise WorkerAuthError("worker has no approved enrollment")
        return enrollment

    async def _audit_auth(self, event_type: AuditEventType, *, worker_id: str | None = None) -> None:
        # A shared, bounded sampling budget also covers rate-limit rejections.
        if self._auth_audit_limiter.allow(GLOBAL_RATE_LIMIT_KEY):
            await self._audit.record(
                event_type, worker_id=worker_id,
                detail={"reason": "invalid_credential" if event_type is AuditEventType.AUTH_REJECTED else "admission_rejected"},
            )

    async def authenticate(
        self, authorization: str | None, *, source: str = "direct-service-call"
    ) -> WorkerIdentity:
        # No external verification or database access before admission. The
        # source must come from ASGI/trusted ingress, never an arbitrary header.
        if (
            not self._source_limiter.allow(source)
            or not self._pre_auth_limiter.allow(GLOBAL_RATE_LIMIT_KEY)
            or self._active_auth >= self._max_auth_concurrency
        ):
            # No per-request audit write on this hot rejection path.
            raise RateLimited("pre-authentication admission limit")
        if authorization is None or len(authorization) > 8192:
            await self._audit_auth(AuditEventType.AUTH_REJECTED)
            raise WorkerAuthError("missing or oversized credential")
        self._active_auth += 1
        try:
            identity = await self._authenticator.authenticate(authorization)
            await self._approved_enrollment(identity)
            if not self._worker_limiter.allow(identity.worker_id) or not self._global_limiter.allow(
                GLOBAL_RATE_LIMIT_KEY
            ):
                await self._audit_auth(AuditEventType.RATE_LIMITED, worker_id=identity.worker_id)
                raise RateLimited(identity.worker_id)
            return identity
        except WorkerAuthBusy as exc:
            raise RateLimited("identity verification is busy") from exc
        except WorkerAuthError:
            await self._audit_auth(AuditEventType.AUTH_REJECTED)
            raise
        finally:
            self._active_auth -= 1

    async def register_heartbeat(
        self, identity: WorkerIdentity, registration: WorkerRegistration
    ) -> WorkerRecord:
        if registration.worker_id != identity.worker_id:
            raise WorkerMismatch(registration.worker_id)
        enrollment = await self._approved_enrollment(identity)
        if registration.tenant_id != enrollment.tenant_id:
            raise WorkerTenantMismatch(registration.worker_id)
        registration = replace(registration, tenant_id=enrollment.tenant_id)
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
        await self._approved_enrollment(identity)
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
            tenant_id=worker.tenant_id,
            vram_mb=worker.vram_mb,
            gpu_model=worker.gpu_model,
            labels=worker.labels,
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
        await self._approved_enrollment(identity)
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
