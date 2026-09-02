"""FastAPI surface for the outbound GPU worker pool (install the `coordinator` extra).

Every request DTO is closed (`extra="forbid"`) and every string is length bounded,
so nothing a worker sends can widen the coordinator's input surface. No request DTO
accepts a URL, path, command, or workflow graph; the signed grant URLs appear only
in responses, where the coordinator itself minted them. A worker learns nothing
about who a job belongs to: `tenant_id` is not part of any response.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from outbound_gpu_worker_pool.contracts import (
    MAX_ASSET_KEY_LENGTH,
    MAX_AUDIT_REASON_LENGTH,
    MAX_CAPABILITY_ID_LENGTH,
    MAX_JOB_ID_LENGTH,
    MAX_WORKER_ID_LENGTH,
    AssetGrant,
    CostClass,
    JobFailureCode,
    JobPayload,
    JobPayloadValue,
    LeaseGrant,
    OutputManifest,
    WorkerAuthError,
    WorkerCapability,
    WorkerIdentity,
    WorkerRecord,
    WorkerRegistration,
    WorkerStatus,
)
from outbound_gpu_worker_pool.service import (
    CompletionRejected,
    JobNotFound,
    RateLimited,
    StaleLease,
    WorkerMismatch,
    WorkerNotRegistered,
    WorkerPoolService,
    WorkerRevoked,
)
from outbound_gpu_worker_pool.validation import (
    CAPABILITY_ID_PATTERN,
    validate_job_payload,
)

MAX_CLAIM_TOKEN_LENGTH = 64
MAX_PLUGIN_ID_LENGTH = 64
MAX_PLUGIN_VERSION_LENGTH = 32
MAX_MODEL_ID_LENGTH = 128
MAX_MODEL_VERSION_LENGTH = 64
MAX_RUNTIME_VERSIONS = 16
MAX_LABELS = 16
MAX_CAPABILITIES = 32
WORKER_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,127}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

CapabilityIdField = Annotated[
    str,
    Field(max_length=MAX_CAPABILITY_ID_LENGTH, pattern=CAPABILITY_ID_PATTERN.pattern),
]
JobIdPath = Annotated[str, Path(min_length=1, max_length=MAX_JOB_ID_LENGTH)]


@contextmanager
def _mapped_errors() -> Iterator[None]:
    try:
        yield
    except CompletionRejected as exc:
        raise HTTPException(status_code=422, detail={"reason": exc.reason}) from exc
    except WorkerAuthError as exc:
        raise HTTPException(
            status_code=401, detail="worker credential rejected"
        ) from exc
    except WorkerRevoked as exc:
        raise HTTPException(status_code=403, detail="worker is revoked") from exc
    except RateLimited as exc:
        raise HTTPException(status_code=429, detail="worker rate limit") from exc
    except (WorkerMismatch, WorkerNotRegistered) as exc:
        raise HTTPException(
            status_code=409, detail="worker registration conflict"
        ) from exc
    except JobNotFound as exc:
        raise HTTPException(status_code=404, detail="pool job not found") from exc
    except StaleLease as exc:
        raise HTTPException(status_code=409, detail="job lease is stale") from exc


class WorkerCapabilityDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_id: CapabilityIdField
    plugin_id: str = Field(min_length=1, max_length=MAX_PLUGIN_ID_LENGTH)
    plugin_version: str = Field(min_length=1, max_length=MAX_PLUGIN_VERSION_LENGTH)
    concurrency: int = Field(default=1, ge=1, le=64)


class WorkerHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str = Field(
        min_length=1, max_length=MAX_WORKER_ID_LENGTH, pattern=WORKER_ID_PATTERN
    )
    capabilities: list[WorkerCapabilityDto] = Field(
        min_length=1, max_length=MAX_CAPABILITIES
    )
    gpu_model: str | None = Field(default=None, max_length=64)
    vram_mb: int | None = Field(default=None, ge=0, le=1_000_000)
    runtime_versions: dict[
        Annotated[str, Field(min_length=1, max_length=32)],
        Annotated[str, Field(min_length=1, max_length=64)],
    ] = Field(default_factory=dict, max_length=MAX_RUNTIME_VERSIONS)
    cost_class: CostClass = CostClass.LOCAL
    labels: list[Annotated[str, Field(min_length=1, max_length=32)]] = Field(
        default_factory=list, max_length=MAX_LABELS
    )
    active_leases: int = Field(default=0, ge=0, le=64)
    draining: bool = False

    def to_registration(self) -> WorkerRegistration:
        return WorkerRegistration(
            worker_id=self.worker_id,
            capabilities=tuple(
                WorkerCapability(
                    capability_id=capability.capability_id,
                    plugin_id=capability.plugin_id,
                    plugin_version=capability.plugin_version,
                    concurrency=capability.concurrency,
                )
                for capability in self.capabilities
            ),
            gpu_model=self.gpu_model,
            vram_mb=self.vram_mb,
            runtime_versions=dict(self.runtime_versions),
            cost_class=self.cost_class,
            labels=tuple(self.labels),
            active_leases=self.active_leases,
            draining=self.draining,
        )


class WorkerRecordResponse(BaseModel):
    worker_id: str
    status: WorkerStatus
    capability_ids: list[str]
    server_time: datetime

    @classmethod
    def from_record(cls, record: WorkerRecord) -> "WorkerRecordResponse":
        return cls(
            worker_id=record.worker_id,
            status=record.status,
            capability_ids=list(record.capability_ids),
            server_time=record.updated_at,
        )


class LeaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability_ids: list[CapabilityIdField] = Field(
        min_length=1, max_length=MAX_CAPABILITIES
    )
    lease_seconds: int | None = Field(default=None, ge=60, le=7_200)


class AssetGrantResponse(BaseModel):
    key: str
    url: str
    method: str
    content_type: str | None
    expires_at: datetime

    @classmethod
    def from_grant(cls, grant: AssetGrant) -> "AssetGrantResponse":
        return cls(
            key=grant.key,
            url=grant.url,
            method=grant.method,
            content_type=grant.content_type,
            expires_at=grant.expires_at,
        )


class LeaseResponse(BaseModel):
    job_id: str
    claim_token: str
    lease_until: datetime
    execution_deadline_seconds: int
    capability_id: str
    contract_version: int
    request_digest: str
    idempotency_key: str
    input_keys: list[str]
    output_key: str
    payload: JobPayload
    input_grants: list[AssetGrantResponse]
    output_grant: AssetGrantResponse

    @classmethod
    def from_grant(cls, grant: LeaseGrant) -> "LeaseResponse":
        return cls(
            job_id=grant.job_id,
            claim_token=grant.claim_token,
            lease_until=grant.lease_until,
            execution_deadline_seconds=grant.execution_deadline_seconds,
            capability_id=grant.capability_id,
            contract_version=grant.contract_version,
            request_digest=grant.request_digest,
            idempotency_key=grant.idempotency_key,
            input_keys=list(grant.input_keys),
            output_key=grant.output_key,
            payload=grant.payload,
            input_grants=[
                AssetGrantResponse.from_grant(item) for item in grant.input_grants
            ],
            output_grant=AssetGrantResponse.from_grant(grant.output_grant),
        )


class JobHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_token: str = Field(min_length=1, max_length=MAX_CLAIM_TOKEN_LENGTH)
    progress_percent: int | None = Field(default=None, ge=0, le=100)


class JobHeartbeatResponse(BaseModel):
    lease_until: datetime


class OutputManifestDto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_key: str = Field(min_length=1, max_length=MAX_ASSET_KEY_LENGTH)
    content_type: str = Field(min_length=1, max_length=128)
    byte_length: int = Field(ge=0)
    sha256: str = Field(max_length=64, pattern=SHA256_PATTERN)
    idempotency_key: str = Field(min_length=1, max_length=MAX_JOB_ID_LENGTH)
    request_digest: str = Field(max_length=64, pattern=SHA256_PATTERN)
    plugin_id: str = Field(min_length=1, max_length=MAX_PLUGIN_ID_LENGTH)
    plugin_version: str = Field(min_length=1, max_length=MAX_PLUGIN_VERSION_LENGTH)
    model_id: str = Field(min_length=1, max_length=MAX_MODEL_ID_LENGTH)
    model_version: str = Field(min_length=1, max_length=MAX_MODEL_VERSION_LENGTH)
    publication_mode: str = Field(min_length=1, max_length=64)
    seed: int | None = Field(default=None, ge=0)
    diagnostics: JobPayload = Field(default_factory=dict)

    @field_validator("diagnostics")
    @classmethod
    def bounded_diagnostics(cls, value: JobPayload) -> JobPayload:
        validate_job_payload(value)
        return value

    def to_manifest(self) -> OutputManifest:
        return OutputManifest(
            output_key=self.output_key,
            content_type=self.content_type,
            byte_length=self.byte_length,
            sha256=self.sha256,
            idempotency_key=self.idempotency_key,
            request_digest=self.request_digest,
            plugin_id=self.plugin_id,
            plugin_version=self.plugin_version,
            model_id=self.model_id,
            model_version=self.model_version,
            publication_mode=self.publication_mode,
            seed=self.seed,
            diagnostics=self.diagnostics,
        )


class CompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_token: str = Field(min_length=1, max_length=MAX_CLAIM_TOKEN_LENGTH)
    manifest: OutputManifestDto


class FailRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_token: str = Field(min_length=1, max_length=MAX_CLAIM_TOKEN_LENGTH)
    reason: str = Field(min_length=1, max_length=MAX_AUDIT_REASON_LENGTH)
    retryable: bool
    failure_code: JobFailureCode


class ReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_token: str = Field(min_length=1, max_length=MAX_CLAIM_TOKEN_LENGTH)
    reason: str = Field(min_length=1, max_length=MAX_AUDIT_REASON_LENGTH)


class SettleResponse(BaseModel):
    job_id: str
    status: str


def create_worker_router(service: WorkerPoolService) -> APIRouter:
    router = APIRouter(prefix="/worker/v1", tags=["worker-pool"])

    async def worker_identity(
        authorization: Annotated[
            str | None, Header(alias="Authorization", max_length=8_192)
        ] = None,
    ) -> WorkerIdentity:
        with _mapped_errors():
            return await service.authenticate(authorization)

    Identity = Annotated[WorkerIdentity, Depends(worker_identity)]

    @router.post("/heartbeat", response_model=WorkerRecordResponse)
    async def register(
        request: WorkerHeartbeatRequest, identity: Identity
    ) -> WorkerRecordResponse:
        with _mapped_errors():
            record = await service.register_heartbeat(
                identity, request.to_registration()
            )
        return WorkerRecordResponse.from_record(record)

    @router.post(
        "/lease",
        response_model=LeaseResponse,
        responses={204: {"description": "no compatible work is claimable"}},
    )
    async def lease(request: LeaseRequest, identity: Identity):
        with _mapped_errors():
            grant = await service.lease(
                identity, tuple(request.capability_ids), request.lease_seconds
            )
        if grant is None:
            return Response(status_code=204)
        return LeaseResponse.from_grant(grant)

    @router.post("/jobs/{job_id}/heartbeat", response_model=JobHeartbeatResponse)
    async def job_heartbeat(
        job_id: JobIdPath, request: JobHeartbeatRequest, identity: Identity
    ) -> JobHeartbeatResponse:
        with _mapped_errors():
            lease_until = await service.job_heartbeat(
                identity, job_id, request.claim_token, request.progress_percent
            )
        return JobHeartbeatResponse(lease_until=lease_until)

    @router.post("/jobs/{job_id}/complete", response_model=SettleResponse)
    async def complete(
        job_id: JobIdPath, request: CompleteRequest, identity: Identity
    ) -> SettleResponse:
        with _mapped_errors():
            await service.complete(
                identity,
                job_id,
                request.claim_token,
                request.manifest.to_manifest(),
            )
        return SettleResponse(job_id=job_id, status="completed")

    @router.post("/jobs/{job_id}/fail", response_model=SettleResponse)
    async def fail(
        job_id: JobIdPath, request: FailRequest, identity: Identity
    ) -> SettleResponse:
        with _mapped_errors():
            await service.fail(
                identity,
                job_id,
                request.claim_token,
                request.reason,
                retryable=request.retryable,
                failure_code=request.failure_code,
            )
        return SettleResponse(job_id=job_id, status="failed")

    @router.post("/jobs/{job_id}/release", response_model=SettleResponse)
    async def release(
        job_id: JobIdPath, request: ReleaseRequest, identity: Identity
    ) -> SettleResponse:
        with _mapped_errors():
            await service.release(identity, job_id, request.claim_token, request.reason)
        return SettleResponse(job_id=job_id, status="released")

    @router.get("/capabilities/schema")
    async def capabilities_schema(
        _identity: Identity,
    ) -> dict[str, dict[str, JobPayloadValue]]:
        return service.capabilities_schema()

    return router
