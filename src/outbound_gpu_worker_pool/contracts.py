"""Typed contracts for the outbound GPU worker pool.

Everything a worker and the coordinator exchange is a typed value here. Nothing in
this module carries a URL that a worker could choose, a filesystem path, a shell
command, or a workflow graph: the coordinator hands out validated capability DTOs
and short-lived asset grants, and the worker hands back an attested manifest.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

type JobPayloadValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JobPayloadValue"]
    | dict[str, "JobPayloadValue"]
)
type JobPayload = dict[str, JobPayloadValue]

MAX_JOB_ID_LENGTH = 255
MAX_ASSET_KEY_LENGTH = 1024
MAX_JOB_INPUT_KEYS = 16  # upper bound only: a job may carry no input keys at all
MAX_JOB_PAYLOAD_BYTES = 16_384
MAX_JOB_PAYLOAD_DEPTH = 8
MIN_JOB_PRIORITY = 0
MAX_JOB_PRIORITY = 1_000
MIN_JOB_ATTEMPT_BUDGET = 1
MAX_JOB_ATTEMPT_BUDGET = 20
MIN_EXECUTION_DEADLINE_SECONDS = 60
MAX_EXECUTION_DEADLINE_SECONDS = 7_200
MAX_CAPABILITY_ID_LENGTH = 128
MAX_WORKER_ID_LENGTH = 128
MAX_AUDIT_REASON_LENGTH = 300
MAX_WORKER_GPUS = 16
MAX_GPU_NAME_LENGTH = 128
POOL_WORKER_ONLINE_WINDOW_SECONDS = 90
POOL_WORKER_VISIBILITY_WINDOW_SECONDS = 24 * 60 * 60

PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE = "immutable_create_once"
DETERMINISTIC_ECHO_CAPABILITY = "test.deterministic.echo.v1"
ATTEMPT_BUDGET_EXHAUSTED = "attempt budget exhausted"
DEFAULT_READ_PREFIXES = ("inputs/", "outputs/")
DEFAULT_OUTPUT_PREFIXES = ("outputs/",)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobFailureCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TEMPORARY_FAILURE = "temporary_failure"


@dataclass(frozen=True)
class JobSubmission:
    """One unit of pool work as a host submits it."""

    job_id: str
    idempotency_key: str
    capability_id: str
    input_keys: tuple[str, ...]
    output_key: str
    contract_version: int = 1
    payload: JobPayload = field(default_factory=dict)
    tenant_id: str | None = None
    priority: int = 100
    attempt_budget: int = 5
    execution_deadline_seconds: int = 1200


@dataclass(frozen=True)
class JobRecord:
    """The durable row for one pool job.

    `claim_token` is the lease fencing secret. It is readable here so a store can
    hand a freshly leased job straight to the coordinator, and it must never be
    serialized outward to a host API response or an audit row.
    """

    job_id: str
    idempotency_key: str
    capability_id: str
    status: JobStatus
    input_keys: tuple[str, ...]
    output_key: str
    attempts: int
    created_at: datetime
    updated_at: datetime
    contract_version: int = 1
    payload: JobPayload = field(default_factory=dict)
    tenant_id: str | None = None
    priority: int = 100
    attempt_budget: int = 5
    execution_deadline_seconds: int = 1200
    request_digest: str = ""
    claim_token: str | None = None
    leased_by: str | None = None
    lease_until: datetime | None = None
    output_content_type: str | None = None
    output_sha256: str | None = None
    output_byte_length: int | None = None
    error: str | None = None
    failure_code: JobFailureCode | None = None
    failure_message: str | None = None
    retryable: bool = False
    progress_percent: int | None = None
    cancelled_at: datetime | None = None


@dataclass(frozen=True)
class SubmissionResult:
    record: JobRecord
    created: bool


@dataclass(frozen=True)
class AssetDescriptor:
    key: str
    size: int
    content_type: str | None


class AssetStore(Protocol):
    async def create_read_url(self, key: str) -> str:
        """Short-lived GET grant for one exact key under an allowed read prefix."""
        ...

    async def create_output_upload_url(self, key: str, content_type: str) -> str:
        """Short-lived create-once PUT grant for one exact output key."""
        ...

    async def describe(self, key: str) -> AssetDescriptor | None:
        """Size and content type of an existing object, or None when absent."""
        ...

    async def read_limited(self, key: str, max_bytes: int) -> bytes:
        """Read at most max_bytes or raise AssetTooLarge/AssetNotFound."""
        ...

    async def write_once(self, key: str, content: bytes, content_type: str) -> bool:
        """Create an immutable object; False means that exact key already exists."""
        ...


class JobStore(Protocol):
    async def submit(self, submission: JobSubmission) -> SubmissionResult:
        """Insert the job, or replay the record an equal submission already created.

        Raises IdempotencyConflict when the key was reused for a different request.
        """
        ...

    async def get(self, job_id: str) -> JobRecord | None: ...

    async def list_for_tenant(self, tenant_id: str) -> tuple[JobRecord, ...]: ...

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued or processing job and drop its lease."""
        ...

    async def lease(
        self, *, worker_id: str, capability_ids: tuple[str, ...], lease_seconds: int
    ) -> JobRecord | None:
        """Atomically lease the oldest claimable job for one of capability_ids.

        Ordered by priority, then creation time. A job is claimable while its
        attempts are below its budget and it is either queued or processing with
        an expired lease. Safe under concurrent callers.
        """
        ...

    async def heartbeat(
        self,
        job_id: str,
        claim_token: str,
        *,
        lease_seconds: int,
        progress_percent: int | None = None,
    ) -> bool:
        """Extend the lease of a live (job_id, claim_token); False when stale."""
        ...

    async def complete(
        self,
        job_id: str,
        claim_token: str,
        *,
        content_type: str,
        sha256: str,
        byte_length: int,
    ) -> bool: ...

    async def fail(
        self,
        job_id: str,
        claim_token: str,
        reason: str,
        *,
        failure_code: JobFailureCode = JobFailureCode.TEMPORARY_FAILURE,
        failure_message: str | None = None,
        retryable: bool = True,
    ) -> bool: ...

    async def release(self, job_id: str, claim_token: str, reason: str) -> bool:
        """Requeue a leased job without spending a further attempt."""
        ...

    async def expire_exhausted(self) -> int:
        """Fail processing jobs whose lease expired past their attempt budget."""
        ...

    async def queue_depth(self) -> "QueueDepth":
        """Bounded queued/processing counts, overall and per capability.

        Implementations aggregate in the store (a `GROUP BY`, not a full scan
        loaded into memory) so this stays cheap regardless of queue size.
        """
        ...


class WorkerStatus(StrEnum):
    ACTIVE = "active"
    DRAINING = "draining"
    REVOKED = "revoked"


class CostClass(StrEnum):
    LOCAL = "local"
    HOSTED = "hosted"
    PREMIUM = "premium"


class AuditEventType(StrEnum):
    WORKER_HEARTBEAT = "worker_heartbeat"
    WORKER_STATUS_CHANGED = "worker_status_changed"
    WORKER_REVOKED = "worker_revoked"
    LEASE_GRANTED = "lease_granted"
    LEASE_EMPTY = "lease_empty"
    JOB_HEARTBEAT = "job_heartbeat"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_RELEASED = "job_released"
    COMPLETION_REJECTED = "completion_rejected"
    AUTH_REJECTED = "auth_rejected"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class WorkerCapability:
    capability_id: str
    plugin_id: str
    plugin_version: str
    concurrency: int = 1


@dataclass(frozen=True)
class GpuTelemetry:
    """One GPU's live utilization and memory as a worker samples it.

    Any field a sampler could not read (for example `nvidia-smi` reporting
    `[N/A]`) is `None`; the whole tuple is empty when sampling itself failed.
    """

    index: int
    name: str
    utilization_pct: int | None
    memory_used_mb: int | None
    memory_total_mb: int | None


@dataclass(frozen=True)
class WorkerRegistration:
    """What an agent advertises on every heartbeat."""

    worker_id: str
    capabilities: tuple[WorkerCapability, ...]
    gpu_model: str | None = None
    vram_mb: int | None = None
    runtime_versions: dict[str, str] = field(default_factory=dict)
    cost_class: CostClass = CostClass.LOCAL
    labels: tuple[str, ...] = ()
    active_leases: int = 0
    draining: bool = False
    gpus: tuple[GpuTelemetry, ...] = ()
    busy_job_id: str | None = None


@dataclass(frozen=True)
class WorkerRecord:
    """The durable registry row for one worker."""

    worker_id: str
    identity_subject: str
    status: WorkerStatus
    capabilities: tuple[WorkerCapability, ...]
    gpu_model: str | None
    vram_mb: int | None
    runtime_versions: dict[str, str]
    cost_class: CostClass
    labels: tuple[str, ...]
    active_leases: int
    last_heartbeat_at: datetime | None
    created_at: datetime
    updated_at: datetime
    revoked_at: datetime | None = None
    gpus: tuple[GpuTelemetry, ...] = ()
    busy_job_id: str | None = None

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(capability.capability_id for capability in self.capabilities)


class PoolWorkerStatus(StrEnum):
    """The presentation status a host-facing pool status route reports.

    Distinct from `WorkerStatus`, which is the registry's own active / draining
    / revoked lifecycle: this is derived per request from a worker's current
    lease and heartbeat recency, for `GET /pool/workers`.
    """

    BUSY = "busy"
    DRAINING = "draining"
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True)
class PoolWorkerView:
    """One row of `GET /pool/workers`: a worker's derived status plus telemetry."""

    worker_id: str
    status: PoolWorkerStatus
    last_heartbeat_at: datetime | None
    capability_ids: tuple[str, ...]
    gpus: tuple[GpuTelemetry, ...]
    busy_job_id: str | None
    draining: bool


@dataclass(frozen=True)
class CapabilityQueueDepth:
    queued: int
    processing: int


@dataclass(frozen=True)
class QueueDepth:
    """The result of `GET /pool/queue`: bounded counts, never a job listing."""

    queued: int
    processing: int
    by_capability: dict[str, CapabilityQueueDepth]


@dataclass(frozen=True)
class WorkerIdentity:
    """The authenticated principal behind one request."""

    worker_id: str
    subject: str
    method: str


class WorkerAuthError(PermissionError):
    """The request carried no acceptable worker credential."""


class IdentitySubjectTaken(ValueError):
    """Another worker id is already enrolled under this identity subject."""


class IdempotencyConflict(ValueError):
    """An idempotency key was reused for a different canonical request."""


class AssetNotFound(FileNotFoundError):
    """The requested immutable asset is absent from storage."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"asset not found: {key}")


class AssetTooLarge(ValueError):
    """The requested immutable asset exceeds a configured byte budget."""

    def __init__(
        self, key: str, max_bytes: int, actual_bytes: int | None = None
    ) -> None:
        self.key = key
        self.max_bytes = max_bytes
        self.actual_bytes = actual_bytes
        super().__init__(f"asset exceeds {max_bytes} bytes: {key}")


class AssetStorageUnavailable(RuntimeError):
    """The asset provider failed transiently or could not be reached."""


class WorkerAuthenticator(Protocol):
    async def authenticate(self, authorization: str | None) -> WorkerIdentity:
        """Resolve a bearer credential to a worker identity or raise WorkerAuthError."""
        ...


class WorkerRegistry(Protocol):
    async def upsert(
        self, registration: WorkerRegistration, *, identity_subject: str
    ) -> WorkerRecord:
        """Create or refresh the worker row; a revoked worker stays revoked.

        Raises IdentitySubjectTaken when another worker id already owns the subject.
        """
        ...

    async def get(self, worker_id: str) -> WorkerRecord | None: ...

    async def find_by_identity_subject(self, subject: str) -> WorkerRecord | None:
        """The single worker enrolled under this identity subject, if any."""
        ...

    async def list(self) -> tuple[WorkerRecord, ...]: ...

    async def set_status(self, worker_id: str, status: WorkerStatus) -> bool: ...


@dataclass(frozen=True)
class AuditEvent:
    event_type: AuditEventType
    worker_id: str | None
    job_id: str | None
    detail: dict[str, JobPayloadValue]
    created_at: datetime


class AuditLog(Protocol):
    async def record(
        self,
        event_type: AuditEventType,
        *,
        worker_id: str | None = None,
        job_id: str | None = None,
        detail: dict[str, JobPayloadValue] | None = None,
    ) -> None:
        """Append one redacted audit row. Never store URLs, tokens, prompts, or bytes."""
        ...

    async def list_for_job(self, job_id: str) -> tuple[AuditEvent, ...]: ...


@dataclass(frozen=True)
class CapabilitySchema:
    """The typed input contract the coordinator publishes for one capability."""

    capability_id: str
    contract_version: int
    input_schema: dict[str, JobPayloadValue]


type CapabilitySchemas = Mapping[str, CapabilitySchema]


@dataclass(frozen=True)
class AssetGrant:
    """A short-lived, exact-object, single-method signed URL."""

    key: str
    url: str
    method: str
    expires_at: datetime
    content_type: str | None = None


@dataclass(frozen=True)
class LeaseGrant:
    """Everything a worker receives for one leased job."""

    job_id: str
    claim_token: str
    lease_until: datetime
    execution_deadline_seconds: int
    capability_id: str
    contract_version: int
    request_digest: str
    idempotency_key: str
    input_keys: tuple[str, ...]
    output_key: str
    payload: JobPayload
    input_grants: tuple[AssetGrant, ...]
    output_grant: AssetGrant
    tenant_id: str | None = None


@dataclass(frozen=True)
class OutputManifest:
    """What a worker attests when it commits a result."""

    output_key: str
    content_type: str
    byte_length: int
    sha256: str
    idempotency_key: str
    request_digest: str
    plugin_id: str
    plugin_version: str
    model_id: str
    model_version: str
    publication_mode: str
    seed: int | None = None
    diagnostics: dict[str, JobPayloadValue] = field(default_factory=dict)
