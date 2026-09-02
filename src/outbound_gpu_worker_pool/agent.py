"""The outbound worker agent (install the `agent` extra).

The agent only ever makes outbound HTTPS calls: it registers, leases one job,
downloads the granted inputs, runs an approved plugin in a per-job workspace,
uploads the artifact through a create-once grant, and attests the result. It
listens on no port and holds no bucket or database credential.

Nothing here logs a bearer credential or a signed grant URL. Log records carry
ids, byte counts, digests, statuses, and durations; failure reasons are reported
by exception type, because a third-party exception message may quote a URL.
"""

import asyncio
import hashlib
import logging
import random as random_module
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

import httpx

from outbound_gpu_worker_pool.contracts import (
    MAX_AUDIT_REASON_LENGTH,
    PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
    AssetGrant,
    JobFailureCode,
    LeaseGrant,
    WorkerCapability,
)
from outbound_gpu_worker_pool.plugins import (
    ExecutionContext,
    GpuExecutorPlugin,
    PluginOutput,
    PluginRequestRejected,
)

logger = logging.getLogger(__name__)

CREATE_ONCE_HEADER = "x-goog-if-generation-match"
CREATE_ONCE_HEADER_VALUE = "0"
CREATE_ONCE_CONFLICT_STATUS = 412
DIGEST_CHUNK_BYTES = 1024 * 1024
DRAIN_RELEASE_REASON = "draining"
INPUTS_DIRECTORY = "inputs"
# The scratch bound: a machine's disk is the operator's, so one granted input
# may never fill it. Downloading stops at the first chunk that would pass this.
DEFAULT_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024


class TransferError(RuntimeError):
    """An asset transfer did not complete; the URL is never named in the message."""


class AgentOutcome(StrEnum):
    IDLE = "idle"
    COMPLETED = "completed"
    FAILED = "failed"
    RELEASED = "released"
    REJECTED = "rejected"


class AssetTransfer(Protocol):
    async def download(
        self, url: str, destination: Path, max_bytes: int | None = None
    ) -> int:
        """Write the granted object to destination and return the bytes written.

        `max_bytes` is the caller's scratch bound; a transport that can stop
        mid-stream raises `TransferError` rather than writing past it.
        """
        ...

    async def upload(self, url: str, source: Path, content_type: str) -> None:
        """Publish source through a create-once grant; a replay is not an error."""
        ...


class HttpAssetTransfer:
    """Signed-URL transport: streaming GET, create-once PUT."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def download(
        self, url: str, destination: Path, max_bytes: int | None = None
    ) -> int:
        written = 0
        async with self._client.stream("GET", url) as response:
            if response.status_code // 100 != 2:
                raise TransferError(
                    f"asset download failed with status {response.status_code}"
                )
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    if max_bytes is not None and written + len(chunk) > max_bytes:
                        raise TransferError(
                            f"asset download passed the {max_bytes} byte input bound"
                        )
                    handle.write(chunk)
                    written += len(chunk)
        return written

    async def upload(self, url: str, source: Path, content_type: str) -> None:
        response = await self._client.put(
            url,
            content=source.read_bytes(),
            headers={
                "Content-Type": content_type,
                CREATE_ONCE_HEADER: CREATE_ONCE_HEADER_VALUE,
            },
        )
        if response.status_code == CREATE_ONCE_CONFLICT_STATUS:
            return
        if response.status_code // 100 != 2:
            raise TransferError(
                f"asset upload failed with status {response.status_code}"
            )


@dataclass
class _Progress:
    percent: int = 0

    async def report(self, percent: int) -> None:
        self.percent = percent


class WorkerAgent:
    def __init__(
        self,
        *,
        coordinator_url: str,
        worker_id: str,
        credential: Callable[[], str],
        plugins: Sequence[GpuExecutorPlugin],
        transfer: AssetTransfer,
        http: httpx.AsyncClient,
        workspace_root: Path,
        concurrency: int = 1,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        min_poll_seconds: float = 2.0,
        max_poll_seconds: float = 60.0,
        lease_seconds: int = 1200,
        heartbeat_interval_seconds: float = 30.0,
        gpu_model: str | None = None,
        vram_mb: int | None = None,
        runtime_versions: Mapping[str, str] = MappingProxyType({}),
        random: Callable[[], float] = random_module.random,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not plugins:
            raise ValueError("a worker must serve at least one plugin")
        self.worker_id = worker_id
        self.credential = credential
        self._coordinator_url = coordinator_url.rstrip("/")
        self._transfer = transfer
        self._http = http
        self._workspace_root = workspace_root
        self.max_input_bytes = max_input_bytes
        self._min_poll_seconds = min_poll_seconds
        self._max_poll_seconds = max_poll_seconds
        self._lease_seconds = lease_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._gpu_model = gpu_model
        self._vram_mb = vram_mb
        self._runtime_versions = dict(runtime_versions)
        self._random = random
        self._sleep = sleep
        self._stop: asyncio.Event | None = None
        self._plugins: dict[str, GpuExecutorPlugin] = {}
        self._capabilities: dict[str, WorkerCapability] = {}
        for plugin in plugins:
            manifest = plugin.capabilities()
            for capability in manifest.capabilities:
                self._plugins[capability.capability_id] = plugin
                self._capabilities[capability.capability_id] = WorkerCapability(
                    capability_id=capability.capability_id,
                    plugin_id=manifest.plugin_id,
                    plugin_version=manifest.plugin_version,
                    concurrency=concurrency,
                )

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(self._capabilities)

    async def heartbeat(self, draining: bool = False) -> None:
        response = await self._http.post(
            self._route("/worker/v1/heartbeat"),
            headers=self._headers(),
            json={
                "worker_id": self.worker_id,
                "capabilities": [
                    {
                        "capability_id": capability.capability_id,
                        "plugin_id": capability.plugin_id,
                        "plugin_version": capability.plugin_version,
                        "concurrency": capability.concurrency,
                    }
                    for capability in self._capabilities.values()
                ],
                "gpu_model": self._gpu_model,
                "vram_mb": self._vram_mb,
                "runtime_versions": self._runtime_versions,
                "draining": draining,
            },
        )
        response.raise_for_status()

    async def run_once(self) -> AgentOutcome:
        await self.heartbeat()
        response = await self._http.post(
            self._route("/worker/v1/lease"),
            headers=self._headers(),
            json={
                "capability_ids": list(self.capability_ids),
                "lease_seconds": self._lease_seconds,
            },
        )
        if response.status_code == httpx.codes.NO_CONTENT:
            return AgentOutcome.IDLE
        response.raise_for_status()
        return await self._serve(_lease_grant(response.json()))

    async def run_forever(self, stop: asyncio.Event) -> None:
        self._stop = stop
        backoff = self._min_poll_seconds
        try:
            while not stop.is_set():
                try:
                    outcome = await self.run_once()
                except httpx.HTTPError:
                    self._log_unreachable()
                    outcome = AgentOutcome.IDLE
                if outcome is not AgentOutcome.IDLE:
                    backoff = self._min_poll_seconds
                    continue
                await self._sleep(min(self._max_poll_seconds, backoff) * self._random())
                backoff = min(self._max_poll_seconds, backoff * 2)
            try:
                await self.heartbeat(draining=True)
            except httpx.HTTPError:
                self._log_unreachable()
        finally:
            self._stop = None

    async def _serve(self, lease: LeaseGrant) -> AgentOutcome:
        plugin = self._plugins.get(lease.capability_id)
        if plugin is None:
            return await self._reject(lease, "capability is not served by this worker")
        try:
            request = plugin.validate(lease)
        except PluginRequestRejected as exc:
            return await self._reject(lease, str(exc))
        if self._stop is not None and self._stop.is_set():
            return await self._release(lease)
        workspace = self._workspace_root / lease.job_id
        progress = _Progress()
        cancel = asyncio.Event()
        heartbeats: asyncio.Task[None] | None = None
        started = time.monotonic()
        try:
            try:
                input_paths = await self._download_inputs(lease, workspace)
                heartbeats = asyncio.create_task(self._job_heartbeats(lease, progress))
                output = await plugin.execute(
                    ExecutionContext(
                        job_id=lease.job_id,
                        lease_id=lease.claim_token,
                        deadline=datetime.now(UTC)
                        + timedelta(seconds=lease.execution_deadline_seconds),
                        workspace=workspace,
                        input_paths=input_paths,
                        cancel=cancel,
                        progress=progress.report,
                    ),
                    request,
                )
                return await self._publish(lease, output, started)
            except PluginRequestRejected as exc:
                return await self._reject(lease, str(exc))
            except Exception as exc:
                return await self._release(lease, reason=type(exc).__name__)
        finally:
            cancel.set()
            if heartbeats is not None:
                heartbeats.cancel()
                try:
                    await heartbeats
                except asyncio.CancelledError:
                    pass
            shutil.rmtree(workspace, ignore_errors=True)

    async def _download_inputs(
        self, lease: LeaseGrant, workspace: Path
    ) -> dict[str, Path]:
        shutil.rmtree(workspace, ignore_errors=True)
        inputs = workspace / INPUTS_DIRECTORY
        inputs.mkdir(parents=True)
        paths: dict[str, Path] = {}
        for grant in lease.input_grants:
            destination = inputs / PurePosixPath(grant.key).name
            written = await self._transfer.download(
                grant.url, destination, self.max_input_bytes
            )
            # The bound is the agent's, not the transport's: a transfer that
            # cannot stop mid-stream still cannot hand back an oversized input.
            if written > self.max_input_bytes:
                raise TransferError(
                    f"an input passed the {self.max_input_bytes} byte bound"
                )
            paths[grant.key] = destination
            logger.debug(
                "worker downloaded input",
                extra={
                    "job_id": lease.job_id,
                    "worker_id": self.worker_id,
                    "bytes": written,
                },
            )
        return paths

    async def _publish(
        self, lease: LeaseGrant, output: PluginOutput, started: float
    ) -> AgentOutcome:
        digest, byte_length = _digest_file(output.path)
        await self._transfer.upload(
            lease.output_grant.url,
            output.path,
            lease.output_grant.content_type or output.content_type,
        )
        capability = self._capabilities[lease.capability_id]
        response = await self._http.post(
            self._route(f"/worker/v1/jobs/{lease.job_id}/complete"),
            headers=self._headers(),
            json={
                "claim_token": lease.claim_token,
                "manifest": {
                    "output_key": lease.output_key,
                    "content_type": output.content_type,
                    "byte_length": byte_length,
                    "sha256": digest,
                    "idempotency_key": lease.idempotency_key,
                    "request_digest": lease.request_digest,
                    "plugin_id": capability.plugin_id,
                    "plugin_version": capability.plugin_version,
                    "model_id": output.model_id,
                    "model_version": output.model_version,
                    "publication_mode": PUBLICATION_MODE_IMMUTABLE_CREATE_ONCE,
                    "seed": output.seed,
                    "diagnostics": output.diagnostics,
                },
            },
        )
        if response.status_code in {
            httpx.codes.CONFLICT,
            httpx.codes.UNPROCESSABLE_ENTITY,
        }:
            logger.warning(
                "worker completion was not accepted",
                extra={
                    "job_id": lease.job_id,
                    "worker_id": self.worker_id,
                    "outcome": AgentOutcome.FAILED,
                    "status": response.status_code,
                    "reason": _completion_reason(response),
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return AgentOutcome.FAILED
        response.raise_for_status()
        logger.info(
            "worker completed job",
            extra={
                "job_id": lease.job_id,
                "worker_id": self.worker_id,
                "outcome": AgentOutcome.COMPLETED,
                "bytes": byte_length,
                "sha256": digest,
                "duration_ms": _elapsed_ms(started),
            },
        )
        return AgentOutcome.COMPLETED

    async def _job_heartbeats(self, lease: LeaseGrant, progress: _Progress) -> None:
        while True:
            await self._sleep(self._heartbeat_interval_seconds)
            try:
                await self._http.post(
                    self._route(f"/worker/v1/jobs/{lease.job_id}/heartbeat"),
                    headers=self._headers(),
                    json={
                        "claim_token": lease.claim_token,
                        "progress_percent": progress.percent,
                    },
                )
            except httpx.HTTPError:
                logger.debug(
                    "worker job heartbeat did not reach the coordinator",
                    extra={"job_id": lease.job_id, "worker_id": self.worker_id},
                )

    async def _reject(self, lease: LeaseGrant, reason: str) -> AgentOutcome:
        await self._fail(
            lease,
            reason=reason,
            retryable=False,
            failure_code=JobFailureCode.UNSUPPORTED_OPERATION,
        )
        logger.info(
            "worker rejected job",
            extra={
                "job_id": lease.job_id,
                "worker_id": self.worker_id,
                "outcome": AgentOutcome.REJECTED,
            },
        )
        return AgentOutcome.REJECTED

    async def _fail(
        self,
        lease: LeaseGrant,
        *,
        reason: str,
        retryable: bool,
        failure_code: JobFailureCode,
    ) -> None:
        await self._http.post(
            self._route(f"/worker/v1/jobs/{lease.job_id}/fail"),
            headers=self._headers(),
            json={
                "claim_token": lease.claim_token,
                "reason": reason[:MAX_AUDIT_REASON_LENGTH],
                "retryable": retryable,
                "failure_code": str(failure_code),
            },
        )

    async def _release(
        self, lease: LeaseGrant, reason: str = DRAIN_RELEASE_REASON
    ) -> AgentOutcome:
        await self._http.post(
            self._route(f"/worker/v1/jobs/{lease.job_id}/release"),
            headers=self._headers(),
            json={
                "claim_token": lease.claim_token,
                "reason": reason[:MAX_AUDIT_REASON_LENGTH],
            },
        )
        logger.info(
            "worker released job",
            extra={
                "job_id": lease.job_id,
                "worker_id": self.worker_id,
                "outcome": AgentOutcome.RELEASED,
                "reason": reason,
            },
        )
        return AgentOutcome.RELEASED

    def _log_unreachable(self) -> None:
        logger.warning(
            "worker could not reach the coordinator",
            extra={"worker_id": self.worker_id, "outcome": AgentOutcome.IDLE},
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.credential()}"}

    def _route(self, path: str) -> str:
        return f"{self._coordinator_url}{path}"


def _completion_reason(response: httpx.Response) -> str:
    detail = response.json().get("detail")
    if isinstance(detail, dict):
        return str(detail.get("reason", ""))
    return str(detail)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        while chunk := handle.read(DIGEST_CHUNK_BYTES):
            digest.update(chunk)
            byte_length += len(chunk)
    return digest.hexdigest(), byte_length


def _asset_grant(payload: dict) -> AssetGrant:
    return AssetGrant(
        key=payload["key"],
        url=payload["url"],
        method=payload["method"],
        expires_at=datetime.fromisoformat(payload["expires_at"]),
        content_type=payload["content_type"],
    )


def _lease_grant(payload: dict) -> LeaseGrant:
    return LeaseGrant(
        job_id=payload["job_id"],
        claim_token=payload["claim_token"],
        lease_until=datetime.fromisoformat(payload["lease_until"]),
        execution_deadline_seconds=payload["execution_deadline_seconds"],
        capability_id=payload["capability_id"],
        contract_version=payload["contract_version"],
        request_digest=payload["request_digest"],
        idempotency_key=payload["idempotency_key"],
        input_keys=tuple(payload["input_keys"]),
        output_key=payload["output_key"],
        payload=payload["payload"],
        input_grants=tuple(_asset_grant(item) for item in payload["input_grants"]),
        output_grant=_asset_grant(payload["output_grant"]),
    )
