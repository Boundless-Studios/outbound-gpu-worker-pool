"""The execution plugin contract and the deterministic reference plugin.

A plugin is the only place model-specific work happens. It receives a validated
request and a workspace the agent already populated; it never sees a credential,
a signed URL, the coordinator, or another job's files. `validate` is the terminal
gate: it runs before any byte is downloaded, so an unsupported request costs
nothing and is never retried.

A plugin's manifest is the single source of truth for what the pool publishes:
`capability_schemas_from_plugins` is what a coordinator hands to
`WorkerPoolService`, so a capability can never be advertised without a plugin
that serves it.
"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from outbound_gpu_worker_pool.contracts import (
    DETERMINISTIC_ECHO_CAPABILITY,
    CapabilitySchema,
    JobPayloadValue,
    LeaseGrant,
    WorkerCapability,
)

DETERMINISTIC_ECHO_PLUGIN_ID = "deterministic-echo"
DETERMINISTIC_ECHO_PLUGIN_VERSION = "1"
DETERMINISTIC_ECHO_CONTRACT_VERSION = 1
DETERMINISTIC_ECHO_OUTPUT_NAME = "output.txt"
DETERMINISTIC_ECHO_CONTENT_TYPE = "text/plain"
DETERMINISTIC_ECHO_HEADER = b"deterministic-echo/v1\n"
DETERMINISTIC_ECHO_SCHEMA = CapabilitySchema(
    capability_id=DETERMINISTIC_ECHO_CAPABILITY,
    contract_version=DETERMINISTIC_ECHO_CONTRACT_VERSION,
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "seed": {"type": "integer", "minimum": 0},
            "label": {"type": "string", "maxLength": 64},
        },
    },
)
DEFAULT_ECHO_LABEL = "echo"
MAX_ECHO_LABEL_LENGTH = 64


class PluginRequestRejected(ValueError):
    """The request is not one this plugin can serve; the job must not be retried."""


@dataclass(frozen=True)
class CapabilityManifest:
    """What one plugin advertises to the coordinator."""

    plugin_id: str
    plugin_version: str
    capabilities: tuple[WorkerCapability, ...]
    schemas: tuple[CapabilitySchema, ...]


@dataclass(frozen=True)
class ValidatedRequest:
    """The typed, allowlisted inputs a plugin accepted from a lease."""

    capability_id: str
    contract_version: int
    inputs: dict[str, JobPayloadValue]
    seed: int | None


@dataclass
class ExecutionContext:
    """The isolated, per-job working set handed to one execution."""

    job_id: str
    lease_id: str
    deadline: datetime
    workspace: Path
    input_paths: dict[str, Path]
    cancel: asyncio.Event
    progress: Callable[[int], Awaitable[None]]


@dataclass(frozen=True)
class PluginHealth:
    healthy: bool
    detail: str = ""


@dataclass(frozen=True)
class PluginOutput:
    """The single immutable artifact one execution produced."""

    path: Path
    content_type: str
    model_id: str
    model_version: str
    seed: int | None = None
    diagnostics: dict[str, JobPayloadValue] = field(default_factory=dict)


class GpuExecutorPlugin(Protocol):
    def capabilities(self) -> CapabilityManifest: ...

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        """Return the typed request or raise PluginRequestRejected (terminal)."""
        ...

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput: ...

    async def cancel(self, job_id: str) -> bool: ...

    async def health(self) -> PluginHealth: ...


class DeterministicEchoPlugin:
    """Reference plugin: same inputs and payload produce identical bytes anywhere."""

    def capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            plugin_id=DETERMINISTIC_ECHO_PLUGIN_ID,
            plugin_version=DETERMINISTIC_ECHO_PLUGIN_VERSION,
            capabilities=(
                WorkerCapability(
                    capability_id=DETERMINISTIC_ECHO_CAPABILITY,
                    plugin_id=DETERMINISTIC_ECHO_PLUGIN_ID,
                    plugin_version=DETERMINISTIC_ECHO_PLUGIN_VERSION,
                ),
            ),
            schemas=(DETERMINISTIC_ECHO_SCHEMA,),
        )

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        if lease.capability_id != DETERMINISTIC_ECHO_CAPABILITY:
            raise PluginRequestRejected("unsupported capability")
        if lease.contract_version != DETERMINISTIC_ECHO_CONTRACT_VERSION:
            raise PluginRequestRejected("unsupported contract version")
        unknown = set(lease.payload) - {"seed", "label"}
        if unknown:
            raise PluginRequestRejected("payload carries unsupported keys")
        seed = lease.payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PluginRequestRejected("seed must be a non-negative integer")
        label = lease.payload.get("label", DEFAULT_ECHO_LABEL)
        if not isinstance(label, str) or len(label) > MAX_ECHO_LABEL_LENGTH:
            raise PluginRequestRejected(
                f"label must be a string of at most {MAX_ECHO_LABEL_LENGTH} characters"
            )
        return ValidatedRequest(
            capability_id=lease.capability_id,
            contract_version=lease.contract_version,
            inputs={"seed": seed, "label": label},
            seed=seed,
        )

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        # Key order, not arrival order: two workers given the same job must
        # produce the same bytes. A job with no inputs hashes the empty byte
        # string, so its digest is sha256(b"").
        digest = hashlib.sha256(
            b"".join(
                path.read_bytes() for _, path in sorted(context.input_paths.items())
            )
        ).hexdigest()
        body = (
            f"label={request.inputs['label']}\n"
            f"seed={request.inputs['seed']}\n"
            f"sha256={digest}\n"
        )
        output = context.workspace / DETERMINISTIC_ECHO_OUTPUT_NAME
        output.write_bytes(DETERMINISTIC_ECHO_HEADER + body.encode())
        await context.progress(100)
        return PluginOutput(
            path=output,
            content_type=DETERMINISTIC_ECHO_CONTENT_TYPE,
            model_id=DETERMINISTIC_ECHO_PLUGIN_ID,
            model_version=DETERMINISTIC_ECHO_PLUGIN_VERSION,
            seed=request.seed,
        )

    async def cancel(self, job_id: str) -> bool:
        return False

    async def health(self) -> PluginHealth:
        return PluginHealth(healthy=True)


def capability_schemas_from_plugins(
    plugins: Iterable[GpuExecutorPlugin],
) -> dict[str, CapabilitySchema]:
    """The coordinator publishes exactly what the enabled plugins declare."""
    schemas: dict[str, CapabilitySchema] = {}
    for plugin in plugins:
        for schema in plugin.capabilities().schemas:
            schemas[schema.capability_id] = schema
    return schemas
