"""In-memory job store, asset store, worker registry, audit log, and authenticator.

These are faithful stand-ins for the Postgres and cloud implementations: they
enforce the same lease, fencing, idempotency, and namespace rules so a host can
run the whole pool in one process for tests and local development.
"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from outbound_gpu_worker_pool.contracts import (
    ATTEMPT_BUDGET_EXHAUSTED,
    DEFAULT_OUTPUT_PREFIXES,
    DEFAULT_READ_PREFIXES,
    AssetDescriptor,
    AssetNotFound,
    AssetTooLarge,
    AuditEvent,
    AuditEventType,
    IdempotencyConflict,
    IdentitySubjectTaken,
    JobFailureCode,
    JobPayloadValue,
    JobRecord,
    JobStatus,
    JobSubmission,
    SubmissionResult,
    WorkerAuthError,
    WorkerIdentity,
    WorkerRecord,
    WorkerRegistration,
    WorkerStatus,
)
from outbound_gpu_worker_pool.validation import (
    job_request_digest,
    validate_job_submission,
)

DEFAULT_ASSET_CONTENT_TYPE = "application/octet-stream"
MEMORY_READ_PREFIX = "memory://read/"
MEMORY_UPLOAD_PREFIX = "memory://upload/"


@dataclass
class MemoryJobStore:
    records: dict[str, JobRecord] = field(default_factory=dict)

    async def submit(self, submission: JobSubmission) -> SubmissionResult:
        validate_job_submission(submission)
        for record in self.records.values():
            if record.idempotency_key == submission.idempotency_key:
                if not _replays(record, submission):
                    raise IdempotencyConflict(submission.idempotency_key)
                return SubmissionResult(record=record, created=False)
        now = datetime.now(UTC)
        record = JobRecord(
            job_id=submission.job_id,
            idempotency_key=submission.idempotency_key,
            capability_id=submission.capability_id,
            status=JobStatus.QUEUED,
            input_keys=submission.input_keys,
            output_key=submission.output_key,
            attempts=0,
            created_at=now,
            updated_at=now,
            contract_version=submission.contract_version,
            payload=deepcopy(submission.payload),
            tenant_id=submission.tenant_id,
            priority=submission.priority,
            attempt_budget=submission.attempt_budget,
            execution_deadline_seconds=submission.execution_deadline_seconds,
            request_digest=job_request_digest(submission),
        )
        self.records[record.job_id] = record
        return SubmissionResult(record=record, created=True)

    async def get(self, job_id: str) -> JobRecord | None:
        return self.records.get(job_id)

    async def list_for_tenant(self, tenant_id: str) -> tuple[JobRecord, ...]:
        return tuple(
            record
            for record in self.records.values()
            if record.tenant_id == tenant_id
        )

    async def cancel(self, job_id: str) -> bool:
        record = self.records.get(job_id)
        if record is None or record.status not in {
            JobStatus.QUEUED,
            JobStatus.PROCESSING,
        }:
            return False
        now = datetime.now(UTC)
        self.records[job_id] = replace(
            record,
            status=JobStatus.CANCELLED,
            cancelled_at=now,
            updated_at=now,
            claim_token=None,
            leased_by=None,
            lease_until=None,
        )
        return True

    async def lease(
        self, *, worker_id: str, capability_ids: tuple[str, ...], lease_seconds: int
    ) -> JobRecord | None:
        now = datetime.now(UTC)
        candidates = [
            record
            for record in self.records.values()
            if record.capability_id in capability_ids
            and record.attempts < record.attempt_budget
            and _is_leasable(record, now)
        ]
        if not candidates:
            return None
        record = min(
            candidates, key=lambda candidate: (candidate.priority, candidate.created_at)
        )
        leased = replace(
            record,
            status=JobStatus.PROCESSING,
            attempts=record.attempts + 1,
            claim_token=uuid4().hex,
            leased_by=worker_id,
            lease_until=now + timedelta(seconds=lease_seconds),
            updated_at=now,
        )
        self.records[record.job_id] = leased
        return leased

    async def heartbeat(
        self,
        job_id: str,
        claim_token: str,
        *,
        lease_seconds: int,
        progress_percent: int | None = None,
    ) -> bool:
        record = self._leased(job_id, claim_token)
        if record is None:
            return False
        now = datetime.now(UTC)
        self.records[job_id] = replace(
            record,
            lease_until=now + timedelta(seconds=lease_seconds),
            progress_percent=(
                record.progress_percent
                if progress_percent is None
                else progress_percent
            ),
            updated_at=now,
        )
        return True

    async def complete(
        self,
        job_id: str,
        claim_token: str,
        *,
        content_type: str,
        sha256: str,
        byte_length: int,
    ) -> bool:
        record = self._leased(job_id, claim_token)
        if record is None:
            return False
        self.records[job_id] = replace(
            _settled(record),
            status=JobStatus.COMPLETED,
            output_content_type=content_type,
            output_sha256=sha256,
            output_byte_length=byte_length,
            error=None,
        )
        return True

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
        record = self._leased(job_id, claim_token)
        if record is None:
            return False
        self.records[job_id] = replace(
            _settled(record),
            status=JobStatus.FAILED,
            error=reason,
            failure_code=failure_code,
            failure_message=failure_message,
            retryable=retryable,
        )
        return True

    async def release(self, job_id: str, claim_token: str, reason: str) -> bool:
        record = self._leased(job_id, claim_token)
        if record is None:
            return False
        self.records[job_id] = replace(
            _settled(record), status=JobStatus.QUEUED, error=reason
        )
        return True

    async def expire_exhausted(self) -> int:
        now = datetime.now(UTC)
        expired = 0
        for job_id, record in list(self.records.items()):
            if (
                record.status is not JobStatus.PROCESSING
                or record.lease_until is None
                or record.lease_until >= now
                or record.attempts < record.attempt_budget
            ):
                continue
            self.records[job_id] = replace(
                _settled(record),
                status=JobStatus.FAILED,
                error=ATTEMPT_BUDGET_EXHAUSTED,
                failure_code=JobFailureCode.TEMPORARY_FAILURE,
                retryable=True,
            )
            expired += 1
        return expired

    def _leased(self, job_id: str, claim_token: str) -> JobRecord | None:
        record = self.records.get(job_id)
        if (
            record is None
            or record.status is not JobStatus.PROCESSING
            or record.claim_token is None
            or record.claim_token != claim_token
        ):
            return None
        return record


@dataclass
class MemoryAssetStore:
    assets: dict[str, bytes] = field(default_factory=dict)
    content_types: dict[str, str] = field(default_factory=dict)
    publish_count: int = 0
    allowed_read_prefixes: tuple[str, ...] = DEFAULT_READ_PREFIXES
    allowed_output_prefixes: tuple[str, ...] = DEFAULT_OUTPUT_PREFIXES

    async def create_read_url(self, key: str) -> str:
        _require_prefix(key, self.allowed_read_prefixes, "read")
        if key not in self.assets:
            raise AssetNotFound(key)
        return f"{MEMORY_READ_PREFIX}{key}"

    async def create_output_upload_url(self, key: str, content_type: str) -> str:
        _require_prefix(key, self.allowed_output_prefixes, "output upload")
        return f"{MEMORY_UPLOAD_PREFIX}{key}"

    async def describe(self, key: str) -> AssetDescriptor | None:
        content = self.assets.get(key)
        if content is None:
            return None
        return AssetDescriptor(
            key=key,
            size=len(content),
            content_type=self.content_types.get(key, DEFAULT_ASSET_CONTENT_TYPE),
        )

    async def read(self, key: str) -> bytes:
        try:
            return self.assets[key]
        except KeyError as exc:
            raise AssetNotFound(key) from exc

    async def read_limited(self, key: str, max_bytes: int) -> bytes:
        content = await self.read(key)
        if len(content) > max_bytes:
            raise AssetTooLarge(key, max_bytes, len(content))
        return content

    async def write_once(self, key: str, content: bytes, content_type: str) -> bool:
        if key in self.assets:
            return False
        self.assets[key] = content
        self.content_types[key] = content_type
        self.publish_count += 1
        return True


@dataclass
class MemoryAssetTransfer:
    """Moves bytes between a MemoryAssetStore and a worker workspace.

    This is the agent's `AssetTransfer` seam without a network: it resolves the
    grant URLs `MemoryAssetStore` mints and nothing else. It accepts the caller's
    `max_bytes` for protocol conformance but has no stream to stop; the agent
    enforces the bound on what it is handed back.
    """

    store: MemoryAssetStore
    downloads: list[str] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)

    async def download(
        self, url: str, destination: Path, max_bytes: int | None = None
    ) -> int:
        key = _grant_key(url, MEMORY_READ_PREFIX)
        content = await self.store.read(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self.downloads.append(key)
        return len(content)

    async def upload(self, url: str, source: Path, content_type: str) -> None:
        key = _grant_key(url, MEMORY_UPLOAD_PREFIX)
        await self.store.write_once(key, source.read_bytes(), content_type)
        self.uploads.append(key)


@dataclass
class MemoryWorkerRegistry:
    workers: dict[str, WorkerRecord] = field(default_factory=dict)

    async def upsert(
        self, registration: WorkerRegistration, *, identity_subject: str
    ) -> WorkerRecord:
        now = datetime.now(UTC)
        status = WorkerStatus.DRAINING if registration.draining else WorkerStatus.ACTIVE
        existing = self.workers.get(registration.worker_id)
        if existing is not None and existing.status is WorkerStatus.REVOKED:
            status = WorkerStatus.REVOKED
        for other in self.workers.values():
            if (
                other.identity_subject == identity_subject
                and other.worker_id != registration.worker_id
            ):
                raise IdentitySubjectTaken(identity_subject)
        record = WorkerRecord(
            worker_id=registration.worker_id,
            identity_subject=identity_subject,
            status=status,
            capabilities=registration.capabilities,
            gpu_model=registration.gpu_model,
            vram_mb=registration.vram_mb,
            runtime_versions=deepcopy(registration.runtime_versions),
            cost_class=registration.cost_class,
            labels=registration.labels,
            active_leases=registration.active_leases,
            last_heartbeat_at=now,
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            revoked_at=existing.revoked_at if existing is not None else None,
        )
        self.workers[record.worker_id] = record
        return record

    async def get(self, worker_id: str) -> WorkerRecord | None:
        return self.workers.get(worker_id)

    async def find_by_identity_subject(self, subject: str) -> WorkerRecord | None:
        for record in self.workers.values():
            if record.identity_subject == subject:
                return record
        return None

    async def list(self) -> tuple[WorkerRecord, ...]:
        return tuple(self.workers[worker_id] for worker_id in sorted(self.workers))

    async def set_status(self, worker_id: str, status: WorkerStatus) -> bool:
        record = self.workers.get(worker_id)
        if record is None:
            return False
        now = datetime.now(UTC)
        self.workers[worker_id] = replace(
            record,
            status=status,
            updated_at=now,
            revoked_at=now if status is WorkerStatus.REVOKED else record.revoked_at,
        )
        return True


@dataclass
class MemoryAuditLog:
    events: list[AuditEvent] = field(default_factory=list)

    async def record(
        self,
        event_type: AuditEventType,
        *,
        worker_id: str | None = None,
        job_id: str | None = None,
        detail: dict[str, JobPayloadValue] | None = None,
    ) -> None:
        self.events.append(
            AuditEvent(
                event_type=event_type,
                worker_id=worker_id,
                job_id=job_id,
                detail=deepcopy(detail) if detail else {},
                created_at=datetime.now(UTC),
            )
        )

    async def list_for_job(self, job_id: str) -> tuple[AuditEvent, ...]:
        return tuple(event for event in self.events if event.job_id == job_id)


@dataclass
class MemoryWorkerAuthenticator:
    identities: dict[str, WorkerIdentity]

    async def authenticate(self, authorization: str | None) -> WorkerIdentity:
        if authorization is None or not authorization.startswith("Bearer "):
            raise WorkerAuthError("missing bearer credential")
        identity = self.identities.get(authorization.removeprefix("Bearer "))
        if identity is None:
            raise WorkerAuthError("unknown worker credential")
        return identity


def _grant_key(url: str, prefix: str) -> str:
    if not url.startswith(prefix):
        # Imported here so the core package still installs without the `agent`
        # extra: only a worker that moves bytes ever reaches this branch.
        from outbound_gpu_worker_pool.agent import TransferError

        raise TransferError("asset grant is not addressable in memory")
    return url.removeprefix(prefix)


def _require_prefix(key: str, prefixes: tuple[str, ...], grant: str) -> None:
    if not key.startswith(prefixes):
        allowed = ", ".join(prefixes)
        raise ValueError(f"signed {grant} key must be in one of: {allowed}")


def _settled(record: JobRecord) -> JobRecord:
    return replace(
        record,
        claim_token=None,
        leased_by=None,
        lease_until=None,
        updated_at=datetime.now(UTC),
    )


def _is_leasable(record: JobRecord, now: datetime) -> bool:
    if record.status is JobStatus.QUEUED:
        return True
    return (
        record.status is JobStatus.PROCESSING
        and record.lease_until is not None
        and record.lease_until < now
    )


def _replays(record: JobRecord, submission: JobSubmission) -> bool:
    return (
        record.capability_id == submission.capability_id
        and record.contract_version == submission.contract_version
        and record.input_keys == submission.input_keys
        and record.output_key == submission.output_key
        and record.payload == submission.payload
        and record.tenant_id == submission.tenant_id
        and record.priority == submission.priority
        and record.attempt_budget == submission.attempt_budget
        and record.execution_deadline_seconds == submission.execution_deadline_seconds
    )
