"""Outbound worker agent tests.

The agent runs in-process against the real coordinator app through
`httpx.ASGITransport`, so every case exercises the router, `WorkerPoolService`,
the plugin contract, and the agent loop together. Only the asset transport is
in-memory; it is the same `MemoryAssetStore` the coordinator verifies against.
"""

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    AuditEventType,
    CapabilitySchema,
    JobFailureCode,
    JobStatus,
    JobSubmission,
    LeaseGrant,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerCapability,
    WorkerIdentity,
    WorkerStatus,
)
from outbound_gpu_worker_pool.agent import (
    DEFAULT_MAX_INPUT_BYTES,
    AgentOutcome,
    HttpAssetTransfer,
    TransferError,
    WorkerAgent,
)
from outbound_gpu_worker_pool.coordinator import create_coordinator_app
from outbound_gpu_worker_pool.memory import MemoryAssetTransfer
from outbound_gpu_worker_pool.plugins import (
    CapabilityManifest,
    DeterministicEchoPlugin,
    ExecutionContext,
    GpuExecutorPlugin,
    PluginHealth,
    PluginOutput,
    PluginRequestRejected,
    ValidatedRequest,
    capability_schemas_from_plugins,
)
from outbound_gpu_worker_pool.service import WorkerPoolService

ECHO = DETERMINISTIC_ECHO_CAPABILITY
OTHER = "pool.other.capability.v1"
CREDENTIAL = "token-a"
# A log record may carry ids, counts, digests and durations. It may never carry
# the worker credential or any grant URL, and every grant the memory asset store
# mints is addressed under this scheme.
GRANT_MARKERS = (CREDENTIAL, "memory://")


def _expected_output(sources: Sequence[bytes], label: str, seed: int) -> bytes:
    digest = hashlib.sha256(b"".join(sources)).hexdigest()
    return b"deterministic-echo/v1\n" + (
        f"label={label}\nseed={seed}\nsha256={digest}\n".encode()
    )


class _RejectingPlugin:
    """Serves one capability and refuses every request for it."""

    def capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            plugin_id="rejecting",
            plugin_version="1",
            capabilities=(WorkerCapability(OTHER, "rejecting", "1"),),
            schemas=(
                CapabilitySchema(
                    capability_id=OTHER,
                    contract_version=1,
                    input_schema={"type": "object"},
                ),
            ),
        )

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        raise PluginRequestRejected("capability is not served by this worker")

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        raise AssertionError("execute must not run for a rejected request")

    async def cancel(self, job_id: str) -> bool:
        return False

    async def health(self) -> PluginHealth:
        return PluginHealth(healthy=True)


@dataclass
class _EchoDelegate:
    """Base for stub plugins that keep the echo capability and schema."""

    echo: DeterministicEchoPlugin = field(default_factory=DeterministicEchoPlugin)

    def capabilities(self) -> CapabilityManifest:
        return self.echo.capabilities()

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        return self.echo.validate(lease)

    async def cancel(self, job_id: str) -> bool:
        return False

    async def health(self) -> PluginHealth:
        return PluginHealth(healthy=True)


@dataclass
class _FlakyPlugin(_EchoDelegate):
    """Explodes on its first execution and serves the retry normally."""

    calls: int = 0

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("plugin exploded")
        return await self.echo.execute(context, request)


class _RejectingExecutePlugin(_EchoDelegate):
    """Accepts validation and then refuses the request during execution."""

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        raise PluginRequestRejected("the request is not one this plugin can serve")


@dataclass
class _RecordingPlugin(_EchoDelegate):
    """Reads the working set it was handed, then echoes normally.

    The bytes are captured during execution because the workspace is gone by the
    time the job settles.
    """

    inputs: dict[str, bytes] = field(default_factory=dict)

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        self.inputs = {
            key: path.read_bytes() for key, path in context.input_paths.items()
        }
        return await self.echo.execute(context, request)


class _FailingDownloadTransfer(MemoryAssetTransfer):
    """A transport whose input download never lands."""

    async def download(
        self, url: str, destination: Path, max_bytes: int | None = None
    ) -> int:
        raise TransferError("asset download failed with status 500")


@dataclass
class _ProgressPlugin(_EchoDelegate):
    """Reports progress and waits for the agent's background heartbeat to land."""

    audit: MemoryAuditLog | None = None
    attempts: int = 200

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        assert self.audit is not None
        await context.progress(42)
        for _ in range(self.attempts):
            if any(
                event.event_type is AuditEventType.JOB_HEARTBEAT
                for event in self.audit.events
            ):
                break
            await asyncio.sleep(0)
        return await self.echo.execute(context, request)


@dataclass
class _DrainingPlugin(_EchoDelegate):
    """Sets the drain event in the middle of execution."""

    stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        self.stop.set()
        return await self.echo.execute(context, request)


@dataclass
class _DrainOnValidatePlugin(_EchoDelegate):
    """Sets the drain event after the lease is granted but before any download."""

    stop: asyncio.Event = field(default_factory=asyncio.Event)

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        self.stop.set()
        return self.echo.validate(lease)

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        raise AssertionError("execute must not run after the drain signal")


@dataclass(frozen=True)
class _Submitted:
    job_id: str
    input_keys: tuple[str, ...]
    output_key: str
    sources: tuple[bytes, ...]


@dataclass
class _Harness:
    jobs: MemoryJobStore
    assets: MemoryAssetStore
    registry: MemoryWorkerRegistry
    audit: MemoryAuditLog
    service: WorkerPoolService
    transfer: MemoryAssetTransfer
    http: httpx.AsyncClient
    agent: WorkerAgent
    workspace_root: Path

    async def submit(
        self,
        key: str = "job-1",
        *,
        capability_id: str = ECHO,
        seed: int = 7,
        input_count: int = 1,
    ) -> _Submitted:
        input_keys = tuple(
            f"inputs/pool/{key}-{index}.bin" for index in range(input_count)
        )
        sources = tuple(
            f"pool-input-{key}-{index}".encode() for index in range(input_count)
        )
        for input_key, source in zip(input_keys, sources, strict=True):
            # Seeded straight into the store: `publish_count` then counts only
            # what the worker itself published.
            self.assets.assets[input_key] = source
            self.assets.content_types[input_key] = "application/octet-stream"
        record = await self.service.submit(
            JobSubmission(
                job_id=str(uuid4()),
                idempotency_key=f"pool:{key}",
                capability_id=capability_id,
                input_keys=input_keys,
                output_key=f"outputs/pool/{key}.txt",
                payload={"seed": seed, "label": key},
                tenant_id="tenant-a",
            )
        )
        return _Submitted(
            job_id=record.job_id,
            input_keys=input_keys,
            output_key=record.output_key,
            sources=sources,
        )

    def event_types(self, job_id: str) -> list[AuditEventType]:
        return [
            event.event_type for event in self.audit.events if event.job_id == job_id
        ]


@asynccontextmanager
async def _harness(
    workspace_root: Path,
    *,
    plugins: Sequence[GpuExecutorPlugin] | None = None,
    transfer_factory: Callable[[MemoryAssetStore], MemoryAssetTransfer] = (
        MemoryAssetTransfer
    ),
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> AsyncIterator[_Harness]:
    jobs = MemoryJobStore()
    assets = MemoryAssetStore()
    registry = MemoryWorkerRegistry()
    audit = MemoryAuditLog()
    active: Sequence[GpuExecutorPlugin] = (
        plugins if plugins is not None else (DeterministicEchoPlugin(),)
    )
    service = WorkerPoolService(
        jobs,
        assets,
        registry,
        audit,
        MemoryWorkerAuthenticator(
            {CREDENTIAL: WorkerIdentity("worker-a", "static:worker-a", "static")}
        ),
        capability_schemas_from_plugins(active),
        per_worker_limit_per_minute=100_000,
        global_limit_per_minute=100_000,
    )
    transfer = transfer_factory(assets)

    async def sleep(delay: float) -> None:
        await asyncio.sleep(0)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_coordinator_app(service)),
        base_url="http://coordinator",
    ) as http:
        yield _Harness(
            jobs=jobs,
            assets=assets,
            registry=registry,
            audit=audit,
            service=service,
            transfer=transfer,
            http=http,
            agent=WorkerAgent(
                coordinator_url="http://coordinator",
                worker_id="worker-a",
                credential=lambda: CREDENTIAL,
                plugins=active,
                transfer=transfer,
                http=http,
                workspace_root=workspace_root,
                max_input_bytes=max_input_bytes,
                min_poll_seconds=0.01,
                max_poll_seconds=0.04,
                random=lambda: 1.0,
                sleep=sleep,
            ),
            workspace_root=workspace_root,
        )


async def test_a_round_trip_publishes_the_deterministic_output(tmp_path: Path) -> None:
    root = tmp_path / "workspaces"
    async with _harness(root) as harness:
        job = await harness.submit()

        outcome = await harness.agent.run_once()

        assert outcome is AgentOutcome.COMPLETED
        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.COMPLETED
        assert harness.assets.assets[job.output_key] == (
            _expected_output(job.sources, "job-1", 7)
        )
        assert harness.transfer.downloads == list(job.input_keys)
        assert harness.transfer.uploads == [job.output_key]
        types = harness.event_types(job.job_id)
        assert AuditEventType.LEASE_GRANTED in types
        assert AuditEventType.JOB_COMPLETED in types
        assert types.index(AuditEventType.LEASE_GRANTED) < types.index(
            AuditEventType.JOB_COMPLETED
        )
        # (f) the per-job workspace is gone once the job settles.
        assert not (root / job.job_id).exists()


async def test_a_several_inputs_are_hashed_in_key_order(tmp_path: Path) -> None:
    plugin = _RecordingPlugin()
    async with _harness(tmp_path / "workspaces", plugins=(plugin,)) as harness:
        job = await harness.submit(input_count=2)

        assert await harness.agent.run_once() is AgentOutcome.COMPLETED

        ordered = [
            source for _, source in sorted(zip(job.input_keys, job.sources, strict=True))
        ]
        assert harness.assets.assets[job.output_key] == (
            _expected_output(ordered, "job-1", 7)
        )
        assert harness.transfer.downloads == list(job.input_keys)
        # The working set is addressed by asset key, not by local file name.
        assert plugin.inputs == dict(zip(job.input_keys, job.sources, strict=True))


async def test_a_job_without_inputs_hashes_the_empty_digest(tmp_path: Path) -> None:
    async with _harness(tmp_path / "workspaces") as harness:
        job = await harness.submit(input_count=0)

        assert await harness.agent.run_once() is AgentOutcome.COMPLETED

        assert job.input_keys == ()
        assert harness.transfer.downloads == []
        assert harness.assets.assets[job.output_key] == (
            _expected_output((), "job-1", 7)
        )
        assert (
            hashlib.sha256(b"").hexdigest().encode()
            in harness.assets.assets[job.output_key]
        )


async def test_b_a_replayed_submission_is_one_job_published_once(
    tmp_path: Path,
) -> None:
    async with _harness(tmp_path / "workspaces") as harness:
        job = await harness.submit()
        assert await harness.agent.run_once() is AgentOutcome.COMPLETED
        published = harness.assets.publish_count

        replayed = await harness.submit()

        assert replayed.job_id == job.job_id
        assert await harness.agent.run_once() is AgentOutcome.IDLE
        assert harness.assets.publish_count == published


async def test_the_agent_heartbeats_the_lease_with_the_latest_progress(
    tmp_path: Path,
) -> None:
    plugin = _ProgressPlugin()
    async with _harness(tmp_path / "workspaces", plugins=(plugin,)) as harness:
        plugin.audit = harness.audit
        job = await harness.submit()

        assert await harness.agent.run_once() is AgentOutcome.COMPLETED

        beats = [
            event
            for event in harness.audit.events
            if event.job_id == job.job_id
            and event.event_type is AuditEventType.JOB_HEARTBEAT
        ]
        assert beats
        assert beats[0].detail == {"progress_percent": 42}


async def test_c_an_unserved_capability_is_rejected_before_any_download(
    tmp_path: Path,
) -> None:
    async with _harness(
        tmp_path / "workspaces", plugins=(_RejectingPlugin(),)
    ) as harness:
        job = await harness.submit(capability_id=OTHER)

        outcome = await harness.agent.run_once()

        assert outcome is AgentOutcome.REJECTED
        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.FAILED
        assert record.retryable is False
        assert record.failure_code is JobFailureCode.UNSUPPORTED_OPERATION
        assert harness.transfer.downloads == []
        assert harness.assets.publish_count == 0


async def test_d_a_plugin_failure_releases_the_job_and_clears_the_workspace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspaces"
    async with _harness(root, plugins=(_FlakyPlugin(),)) as harness:
        job = await harness.submit()

        outcome = await harness.agent.run_once()

        assert outcome is AgentOutcome.RELEASED
        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.QUEUED
        assert record.attempts == 1
        assert record.leased_by is None
        assert harness.transfer.downloads == list(job.input_keys)
        assert harness.assets.publish_count == 0
        # (f) the per-job workspace is gone after a released attempt too.
        assert not (root / job.job_id).exists()

        # A transient failure only costs one attempt: the next lease serves it.
        assert await harness.agent.run_once() is AgentOutcome.COMPLETED
        assert harness.jobs.records[job.job_id].status is JobStatus.COMPLETED
        assert harness.assets.assets[job.output_key] == (
            _expected_output(job.sources, "job-1", 7)
        )


async def test_d_a_failed_input_download_releases_the_job(tmp_path: Path) -> None:
    async with _harness(
        tmp_path / "workspaces", transfer_factory=_FailingDownloadTransfer
    ) as harness:
        job = await harness.submit()

        outcome = await harness.agent.run_once()

        assert outcome is AgentOutcome.RELEASED
        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.QUEUED
        assert record.attempts == 1
        assert record.leased_by is None
        assert harness.assets.publish_count == 0


async def test_d_an_input_over_the_scratch_bound_releases_the_job(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspaces"
    # Every seeded input is longer than this, so the first one trips the bound.
    async with _harness(root, max_input_bytes=4) as harness:
        job = await harness.submit()

        outcome = await harness.agent.run_once()

        assert outcome is AgentOutcome.RELEASED
        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.QUEUED
        assert record.attempts == 1
        assert harness.assets.publish_count == 0
        # Nothing oversized is left behind on the machine's scratch disk.
        assert not (root / job.job_id).exists()


async def test_d_a_plugin_rejection_during_execution_is_terminal(
    tmp_path: Path,
) -> None:
    async with _harness(
        tmp_path / "workspaces", plugins=(_RejectingExecutePlugin(),)
    ) as harness:
        job = await harness.submit()

        outcome = await harness.agent.run_once()

        assert outcome is AgentOutcome.REJECTED
        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.FAILED
        assert record.retryable is False
        assert record.failure_code is JobFailureCode.UNSUPPORTED_OPERATION
        assert harness.assets.publish_count == 0


async def test_e_draining_mid_job_still_completes_the_leased_work(
    tmp_path: Path,
) -> None:
    stop = asyncio.Event()
    plugin = _DrainingPlugin(stop=stop)
    async with _harness(tmp_path / "workspaces", plugins=(plugin,)) as harness:
        job = await harness.submit()

        await harness.agent.run_forever(stop)

        record = harness.jobs.records[job.job_id]
        assert record.status is JobStatus.COMPLETED
        assert harness.assets.assets[job.output_key] == (
            _expected_output(job.sources, "job-1", 7)
        )
        worker = await harness.registry.get("worker-a")
        assert worker is not None
        assert worker.status is WorkerStatus.DRAINING


async def test_e_draining_before_the_first_lease_never_leases(tmp_path: Path) -> None:
    stop = asyncio.Event()
    stop.set()
    async with _harness(tmp_path / "workspaces") as harness:
        job = await harness.submit()

        await harness.agent.run_forever(stop)

        assert harness.jobs.records[job.job_id].status is JobStatus.QUEUED
        assert harness.transfer.downloads == []
        worker = await harness.registry.get("worker-a")
        assert worker is not None
        assert worker.status is WorkerStatus.DRAINING


async def test_e_a_lease_taken_as_draining_begins_is_released(tmp_path: Path) -> None:
    plugin = _DrainOnValidatePlugin()
    async with _harness(tmp_path / "workspaces", plugins=(plugin,)) as harness:
        job = await harness.submit()

        await harness.agent.run_forever(plugin.stop)

        assert harness.jobs.records[job.job_id].status is JobStatus.QUEUED
        assert harness.transfer.downloads == []
        assert AuditEventType.JOB_RELEASED in harness.event_types(job.job_id)


async def test_the_poll_loop_survives_a_coordinator_transport_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    stop = asyncio.Event()
    routes: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        routes.append(request.url.path)
        if len(routes) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(httpx.codes.NO_CONTENT)

    async def sleep(delay: float) -> None:
        stop.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        agent = WorkerAgent(
            coordinator_url="http://coordinator",
            worker_id="worker-a",
            credential=lambda: CREDENTIAL,
            plugins=(DeterministicEchoPlugin(),),
            transfer=MemoryAssetTransfer(MemoryAssetStore()),
            http=http,
            workspace_root=tmp_path / "workspaces",
            min_poll_seconds=0.01,
            random=lambda: 1.0,
            sleep=sleep,
        )

        await agent.run_forever(stop)

    # The failed poll is treated as an idle one, and the drain still heartbeats.
    assert routes == ["/worker/v1/heartbeat", "/worker/v1/heartbeat"]
    warnings = [
        record for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warnings
    assert all("boom" not in record.getMessage() for record in warnings)
    assert warnings[0].outcome is AgentOutcome.IDLE


async def test_g_logs_never_carry_the_credential_or_a_grant_url(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    async with _harness(tmp_path / "workspaces") as harness:
        await harness.submit()

        assert await harness.agent.run_once() is AgentOutcome.COMPLETED

    assert caplog.records
    for record in caplog.records:
        rendered = [
            record.getMessage(),
            *(str(value) for value in record.__dict__.values()),
        ]
        for text in rendered:
            assert not any(marker in text for marker in GRANT_MARKERS), text


async def test_h_http_asset_transfer_streams_and_creates_once(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=b"source-bytes")
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transfer = HttpAssetTransfer(client)
        destination = tmp_path / "inputs" / "asset.bin"
        destination.parent.mkdir(parents=True)
        source = tmp_path / "output.txt"
        source.write_bytes(b"payload")

        written = await transfer.download("https://storage.invalid/read", destination)
        await transfer.upload("https://storage.invalid/write", source, "text/plain")

        assert written == len(b"source-bytes")
        assert destination.read_bytes() == b"source-bytes"
        upload = requests[-1]
        assert upload.method == "PUT"
        assert "x-goog-if-generation-match" not in upload.headers
        assert upload.headers["content-type"] == "text/plain"
        assert upload.read() == b"payload"


async def test_h_a_create_once_conflict_is_treated_as_published(tmp_path: Path) -> None:
    source = tmp_path / "output.txt"
    source.write_bytes(b"payload")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(412))
    ) as client:
        await HttpAssetTransfer(client).upload(
            "https://storage.invalid/write", source, "text/plain"
        )


@pytest.mark.parametrize("status_code", [404, 500])
async def test_h_a_failed_transfer_raises_without_naming_the_url(
    tmp_path: Path, status_code: int
) -> None:
    source = tmp_path / "output.txt"
    source.write_bytes(b"payload")
    destination = tmp_path / "asset.bin"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code))
    ) as client:
        transfer = HttpAssetTransfer(client)

        with pytest.raises(TransferError) as download_error:
            await transfer.download("https://storage.invalid/read", destination)
        with pytest.raises(TransferError) as upload_error:
            await transfer.upload("https://storage.invalid/write", source, "text/plain")

    assert "storage.invalid" not in str(download_error.value)
    assert "storage.invalid" not in str(upload_error.value)


async def test_h_a_download_stops_before_it_passes_the_byte_bound(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "asset.bin"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * 64)
        )
    ) as client:
        with pytest.raises(TransferError) as error:
            await HttpAssetTransfer(client).download(
                "https://storage.invalid/read", destination, max_bytes=16
            )

    # The chunk that would pass the bound is never written, so a machine with a
    # small scratch disk cannot be filled by one oversized input.
    assert destination.stat().st_size <= 16
    assert "storage.invalid" not in str(error.value)
