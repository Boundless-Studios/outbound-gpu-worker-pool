"""The approved-workflow ComfyUI plugin (install the `comfy` extra).

A capability id maps to a versioned workflow template the operator installed on
the machine. A job never carries a graph, a node, a URL, or a path: it may fill a
typed allowlist of inputs the template declares, and it may bind the asset keys
its lease already granted to the template's declared image slots. Anything else
is a terminal rejection, so an unapproved workflow cannot reach the runtime.

The runtime itself is local by construction — the plugin refuses a `base_url`
that is not loopback or private — and nothing here logs or raises the prompt text
or the runtime address.
"""

import asyncio
import copy
import ipaddress
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx

from outbound_gpu_worker_pool.contracts import (
    CapabilitySchema,
    JobPayloadValue,
    LeaseGrant,
    WorkerCapability,
)
from outbound_gpu_worker_pool.plugins import (
    CapabilityManifest,
    ExecutionContext,
    PluginHealth,
    PluginOutput,
    PluginRequestRejected,
    ValidatedRequest,
)
from outbound_gpu_worker_pool.validation import validate_capability_id

COMFY_WORKFLOW_PLUGIN_ID = "comfy-workflow"
COMFY_WORKFLOW_PLUGIN_VERSION = "1"
DEFAULT_COMFY_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_COMFY_CLIENT_ID = "outbound-gpu-worker-pool"
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_COMFY_STARTUP_TIMEOUT_SECONDS = 180.0
STARTUP_COMMAND_TIMEOUT_SECONDS = 30.0
MAX_STARTUP_STDERR_LENGTH = 500
PACKAGED_TEMPLATES_DIRECTORY = Path(__file__).resolve().parent / "templates"
TEMPLATE_SUFFIX = ".template.json"
LOAD_IMAGE_CLASS_TYPE = "LoadImage"
IMAGES_INPUT_NAME = "images"
IMAGE_NODE_INPUT_NAME = "image"
SEED_INPUT_NAME = "seed"
FILENAME_PREFIX_INPUT_NAME = "filename_prefix"
OUTPUT_DIRECTORY = "output"
OUTPUT_PREFIX_ROOT = "ogwp"
OUTPUT_COLLECTION_NAMES = ("animated", "videos", "gifs", "images")
INPUT_KINDS = frozenset({"string", "integer", "number", "boolean"})
PROGRESS_SUBMITTED = 0
PROGRESS_RUNNING = 50
PROGRESS_COMPLETE = 100
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")

logger = logging.getLogger(__name__)


class ComfyRuntimeUnavailable(Exception):
    """The local ComfyUI runtime could not be confirmed up or brought up."""


@dataclass(frozen=True)
class TemplateInput:
    """One value a job may fill in one node of the template's graph.

    `minimum` and `maximum` bound a numeric value; for a string they bound its
    length, together with `max_length`.
    """

    name: str
    node_id: str
    input_name: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    max_length: int | None = None
    default: JobPayloadValue = None
    required: bool = False


@dataclass(frozen=True)
class ImageSlot:
    """A `LoadImage` node a job may bind one granted input asset to."""

    name: str
    node_id: str
    required: bool = False


@dataclass(frozen=True)
class ComfyTemplate:
    """One versioned workflow the operator installed and approved."""

    capability_id: str
    contract_version: int
    template_version: str
    graph: dict[str, JobPayloadValue]
    inputs: tuple[TemplateInput, ...]
    image_slots: tuple[ImageSlot, ...]
    output_node_id: str
    output_content_type: str
    model_id: str
    model_version: str


class TemplateRegistry:
    """The workflow templates one machine is approved to run, by capability id."""

    def __init__(self, templates: Iterable[ComfyTemplate]) -> None:
        self._templates = {template.capability_id: template for template in templates}

    @classmethod
    def from_directory(cls, directory: Path) -> "TemplateRegistry":
        """Load and validate every `*.template.json` in an operator-owned directory."""
        templates: dict[str, ComfyTemplate] = {}
        for path in sorted(Path(directory).glob(f"*{TEMPLATE_SUFFIX}")):
            try:
                template = _template_from_document(json.loads(path.read_text()))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name} is not a usable template: {exc}"
                ) from exc
            if template.capability_id in templates:
                raise ValueError(
                    f"{path.name} claims a capability another template already serves: "
                    f"{template.capability_id}"
                )
            templates[template.capability_id] = template
        return cls(templates.values())

    @property
    def templates(self) -> tuple[ComfyTemplate, ...]:
        return tuple(self._templates.values())

    def template(self, capability_id: str) -> ComfyTemplate | None:
        return self._templates.get(capability_id)


def input_schema(template: ComfyTemplate) -> dict[str, JobPayloadValue]:
    """The typed allowlist a coordinator publishes for one template."""
    properties: dict[str, JobPayloadValue] = {}
    required: list[JobPayloadValue] = []
    for entry in template.inputs:
        schema: dict[str, JobPayloadValue] = {"type": entry.kind}
        if entry.kind == "string":
            if entry.minimum is not None:
                schema["minLength"] = entry.minimum
            if entry.max_length is not None:
                schema["maxLength"] = entry.max_length
        else:
            if entry.minimum is not None:
                schema["minimum"] = entry.minimum
            if entry.maximum is not None:
                schema["maximum"] = entry.maximum
        properties[entry.name] = schema
        if entry.required:
            required.append(entry.name)
    properties[IMAGES_INPUT_NAME] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            slot.name: {"type": "string"} for slot in template.image_slots
        },
        "required": [slot.name for slot in template.image_slots if slot.required],
    }
    if any(slot.required for slot in template.image_slots):
        required.append(IMAGES_INPUT_NAME)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def capability_schemas(registry: TemplateRegistry) -> dict[str, CapabilitySchema]:
    """What an installed template directory publishes, without any local runtime."""
    return {
        template.capability_id: CapabilitySchema(
            capability_id=template.capability_id,
            contract_version=template.contract_version,
            input_schema=input_schema(template),
        )
        for template in registry.templates
    }


class ComfyWorkflowPlugin:
    """Runs one approved, locally installed workflow template per capability."""

    def __init__(
        self,
        registry: TemplateRegistry,
        *,
        client: httpx.AsyncClient,
        base_url: str = DEFAULT_COMFY_BASE_URL,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        client_id: str = DEFAULT_COMFY_CLIENT_ID,
        start_command: tuple[str, ...] | None = None,
        startup_timeout_seconds: float = DEFAULT_COMFY_STARTUP_TIMEOUT_SECONDS,
        startup_poll_interval_seconds: float = 2.0,
    ) -> None:
        _require_private_runtime(base_url)
        self._registry = registry
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._poll_interval_seconds = poll_interval_seconds
        self._client_id = client_id
        self._start_command = start_command
        self._startup_timeout_seconds = startup_timeout_seconds
        self._startup_poll_interval_seconds = startup_poll_interval_seconds
        self._prompts: dict[str, str] = {}

    def capabilities(self) -> CapabilityManifest:
        schemas = capability_schemas(self._registry)
        return CapabilityManifest(
            plugin_id=COMFY_WORKFLOW_PLUGIN_ID,
            plugin_version=COMFY_WORKFLOW_PLUGIN_VERSION,
            capabilities=tuple(
                WorkerCapability(
                    capability_id=capability_id,
                    plugin_id=COMFY_WORKFLOW_PLUGIN_ID,
                    plugin_version=COMFY_WORKFLOW_PLUGIN_VERSION,
                )
                for capability_id in schemas
            ),
            schemas=tuple(schemas.values()),
        )

    def validate(self, lease: LeaseGrant) -> ValidatedRequest:
        template = self._registry.template(lease.capability_id)
        if template is None:
            raise PluginRequestRejected("unsupported capability")
        if lease.contract_version != template.contract_version:
            raise PluginRequestRejected("unsupported contract version")
        declared = {entry.name: entry for entry in template.inputs}
        if set(lease.payload) - set(declared) - {IMAGES_INPUT_NAME}:
            raise PluginRequestRejected("payload carries unsupported keys")
        inputs: dict[str, JobPayloadValue] = {}
        for name, entry in declared.items():
            if name in lease.payload:
                inputs[name] = _checked_value(entry, lease.payload[name])
            elif entry.required:
                raise PluginRequestRejected(f"{name} is required")
            elif entry.default is not None:
                inputs[name] = entry.default
        inputs[IMAGES_INPUT_NAME] = _bound_images(template, lease)
        seed = inputs.get(SEED_INPUT_NAME)
        return ValidatedRequest(
            capability_id=lease.capability_id,
            contract_version=lease.contract_version,
            inputs=inputs,
            seed=seed if isinstance(seed, int) else None,
        )

    async def ensure_running(self) -> None:
        """Confirm the local runtime is up, starting it if the operator allows.

        A shared GPU box may run a helper that stops an idle ComfyUI to free the
        GPU for another product. The pool owns bringing it back up before it
        submits work, rather than assuming another process already did.
        """
        if await self._probe_up():
            return
        if self._start_command is None:
            raise ComfyRuntimeUnavailable(
                "the local runtime is not reachable and no start command is configured"
            )
        start = time.monotonic()
        await self._run_start_command()
        deadline = start + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            if await self._probe_up():
                logger.info(
                    "the agent started the local ComfyUI runtime",
                    extra={"seconds": round(time.monotonic() - start, 3)},
                )
                return
            await asyncio.sleep(self._startup_poll_interval_seconds)
        raise ComfyRuntimeUnavailable(
            "the local runtime did not become reachable within "
            f"{self._startup_timeout_seconds}s"
        )

    async def _probe_up(self) -> bool:
        try:
            response = await self._client.get(self._route("/system_stats"))
        except httpx.TransportError:
            return False
        return response.status_code == httpx.codes.OK

    async def _run_start_command(self) -> None:
        assert self._start_command is not None
        process = await asyncio.create_subprocess_exec(
            *self._start_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(), timeout=STARTUP_COMMAND_TIMEOUT_SECONDS
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("the ComfyUI start command timed out")
            return
        if process.returncode != 0:
            logger.warning(
                "the ComfyUI start command exited with status %s: %s",
                process.returncode,
                stderr[-MAX_STARTUP_STDERR_LENGTH:].decode(errors="replace"),
            )

    async def execute(
        self, context: ExecutionContext, request: ValidatedRequest
    ) -> PluginOutput:
        await self.ensure_running()
        template = self._registry.template(request.capability_id)
        if template is None:
            raise PluginRequestRejected("unsupported capability")
        graph = copy.deepcopy(template.graph)
        for entry in template.inputs:
            if entry.name in request.inputs:
                _node_inputs(graph, entry.node_id)[entry.input_name] = request.inputs[
                    entry.name
                ]
        images = request.inputs[IMAGES_INPUT_NAME]
        assert isinstance(images, dict)
        for slot in template.image_slots:
            key = images.get(slot.name)
            if not isinstance(key, str):
                _drop_slot(graph, slot)
                continue
            _node_inputs(graph, slot.node_id)[IMAGE_NODE_INPUT_NAME] = (
                await self._upload(context, slot.name, context.input_paths[key])
            )
        output_inputs = _node_inputs(graph, template.output_node_id)
        if FILENAME_PREFIX_INPUT_NAME in output_inputs:
            output_inputs[FILENAME_PREFIX_INPUT_NAME] = (
                f"{OUTPUT_PREFIX_ROOT}/{context.job_id}"
            )
        prompt_id = await self._submit(graph)
        self._prompts[context.job_id] = prompt_id
        try:
            await context.progress(PROGRESS_SUBMITTED)
            artifact = await self._await_artifact(context, prompt_id, template)
            output = await self._collect(
                context, template, artifact, prompt_id, request
            )
            await context.progress(PROGRESS_COMPLETE)
            return output
        finally:
            self._prompts.pop(context.job_id, None)

    async def cancel(self, job_id: str) -> bool:
        if job_id not in self._prompts:
            return False
        await self._interrupt()
        return True

    async def health(self) -> PluginHealth:
        response = await self._client.get(self._route("/system_stats"))
        if response.status_code == httpx.codes.OK:
            return PluginHealth(healthy=True)
        return PluginHealth(
            healthy=False,
            detail=f"the local runtime answered with status {response.status_code}",
        )

    async def _upload(
        self, context: ExecutionContext, slot_name: str, source: Path
    ) -> str:
        # Job-scoped so two concurrent jobs never overwrite each other's frame.
        filename = f"{context.job_id}-{slot_name}{source.suffix}"
        response = await self._client.post(
            self._route("/upload/image"),
            files={"image": (filename, source.read_bytes())},
            data={"type": "input", "overwrite": "true"},
        )
        _raise_for_status(response, "image upload")
        name = response.json().get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("the local runtime stored the image under no name")
        return name

    async def _submit(self, graph: dict[str, JobPayloadValue]) -> str:
        response = await self._client.post(
            self._route("/prompt"),
            json={"prompt": graph, "client_id": self._client_id},
        )
        _raise_for_status(response, "prompt submission")
        prompt_id = response.json().get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise RuntimeError("the local runtime accepted the prompt under no id")
        return prompt_id

    async def _await_artifact(
        self, context: ExecutionContext, prompt_id: str, template: ComfyTemplate
    ) -> dict[str, JobPayloadValue]:
        while True:
            if context.cancel.is_set():
                await self._interrupt()
                raise asyncio.CancelledError
            if datetime.now(UTC) >= context.deadline:
                await self._interrupt()
                raise TimeoutError(
                    "the local runtime did not finish before the execution deadline"
                )
            response = await self._client.get(self._route(f"/history/{prompt_id}"))
            _raise_for_status(response, "history poll")
            entry = response.json().get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError("the local runtime reported an execution error")
                if status.get("completed"):
                    artifact = _first_artifact(entry, template.output_node_id)
                    if artifact is None:
                        raise RuntimeError(
                            "the local runtime completed without an artifact"
                        )
                    return artifact
            await context.progress(PROGRESS_RUNNING)
            await asyncio.sleep(self._poll_interval_seconds)

    async def _collect(
        self,
        context: ExecutionContext,
        template: ComfyTemplate,
        artifact: dict[str, JobPayloadValue],
        prompt_id: str,
        request: ValidatedRequest,
    ) -> PluginOutput:
        filename = artifact.get("filename")
        subfolder = artifact.get("subfolder", "")
        if not isinstance(filename, str) or not PurePosixPath(filename).name:
            raise RuntimeError("the local runtime named no artifact file")
        directory = context.workspace / OUTPUT_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / PurePosixPath(filename).name
        async with self._client.stream(
            "GET",
            self._route("/view"),
            params={
                "filename": filename,
                "subfolder": subfolder if isinstance(subfolder, str) else "",
                "type": "output",
            },
        ) as response:
            _raise_for_status(response, "artifact download")
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
        return PluginOutput(
            path=destination,
            content_type=template.output_content_type,
            model_id=template.model_id,
            model_version=template.model_version,
            seed=request.seed,
            diagnostics={
                "prompt_id": prompt_id,
                "template_version": template.template_version,
            },
        )

    async def _interrupt(self) -> None:
        await self._client.post(self._route("/interrupt"))

    def _route(self, path: str) -> str:
        return f"{self._base_url}{path}"


def _template_from_document(document: dict[str, JobPayloadValue]) -> ComfyTemplate:
    graph = document["graph"]
    if not isinstance(graph, dict):
        raise ValueError("graph must be an object of node id to node")
    template = ComfyTemplate(
        capability_id=validate_capability_id(document["capability_id"]),
        contract_version=document["contract_version"],
        template_version=document["template_version"],
        graph=graph,
        inputs=tuple(
            TemplateInput(
                name=entry["name"],
                node_id=entry["node_id"],
                input_name=entry["input_name"],
                kind=entry["kind"],
                minimum=entry.get("minimum"),
                maximum=entry.get("maximum"),
                max_length=entry.get("max_length"),
                default=entry.get("default"),
                required=entry.get("required", False),
            )
            for entry in document["inputs"]
        ),
        image_slots=tuple(
            ImageSlot(
                name=entry["name"],
                node_id=entry["node_id"],
                required=entry.get("required", False),
            )
            for entry in document["image_slots"]
        ),
        output_node_id=document["output_node_id"],
        output_content_type=document["output_content_type"],
        model_id=document["model_id"],
        model_version=document["model_version"],
    )
    _validate_template(template)
    return template


def _validate_template(template: ComfyTemplate) -> None:
    names: set[str] = {IMAGES_INPUT_NAME}
    for declared in (*template.inputs, *template.image_slots):
        if declared.name in names:
            raise ValueError(f"two allowlist entries share the name {declared.name}")
        names.add(declared.name)
        if not isinstance(template.graph.get(declared.node_id), dict):
            raise ValueError(
                f"{declared.name} names a node the graph does not declare: "
                f"{declared.node_id}"
            )
    for entry in template.inputs:
        if entry.kind not in INPUT_KINDS:
            raise ValueError(
                f"input {entry.name} declares an unknown kind: {entry.kind}"
            )
    for slot in template.image_slots:
        node = template.graph[slot.node_id]
        assert isinstance(node, dict)
        if node.get("class_type") != LOAD_IMAGE_CLASS_TYPE:
            raise ValueError(
                f"image slot {slot.name} must bind a {LOAD_IMAGE_CLASS_TYPE} node"
            )
    if not isinstance(template.graph.get(template.output_node_id), dict):
        raise ValueError(
            "output_node_id names a node the graph does not declare: "
            f"{template.output_node_id}"
        )


def _checked_value(entry: TemplateInput, value: JobPayloadValue) -> JobPayloadValue:
    if entry.kind == "boolean":
        if not isinstance(value, bool):
            raise PluginRequestRejected(f"{entry.name} must be a boolean")
        return value
    if entry.kind == "string":
        if not isinstance(value, str):
            raise PluginRequestRejected(f"{entry.name} must be a string")
        length = len(value)
        if (entry.minimum is not None and length < entry.minimum) or (
            entry.max_length is not None and length > entry.max_length
        ):
            raise PluginRequestRejected(f"{entry.name} is outside its declared length")
        return value
    # A bool is an int in Python and is never a number here.
    if entry.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise PluginRequestRejected(f"{entry.name} must be an integer")
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PluginRequestRejected(f"{entry.name} must be a number")
    if (entry.minimum is not None and value < entry.minimum) or (
        entry.maximum is not None and value > entry.maximum
    ):
        raise PluginRequestRejected(f"{entry.name} is outside its declared range")
    return value


def _bound_images(
    template: ComfyTemplate, lease: LeaseGrant
) -> dict[str, JobPayloadValue]:
    requested = lease.payload.get(IMAGES_INPUT_NAME, {})
    if not isinstance(requested, dict):
        raise PluginRequestRejected("images must map a declared slot to a granted key")
    slots = {slot.name: slot for slot in template.image_slots}
    if set(requested) - set(slots):
        raise PluginRequestRejected("images names a slot the template does not declare")
    for slot in template.image_slots:
        if slot.required and slot.name not in requested:
            raise PluginRequestRejected(f"image slot {slot.name} is required")
    images: dict[str, JobPayloadValue] = {}
    for name, key in requested.items():
        if not isinstance(key, str) or key not in lease.input_keys:
            raise PluginRequestRejected(f"image slot {name} must bind a granted input")
        images[name] = key
    if set(lease.input_keys) - set(images.values()):
        raise PluginRequestRejected("the lease granted an input no image slot binds")
    return images


def _node_inputs(
    graph: dict[str, JobPayloadValue], node_id: str
) -> dict[str, JobPayloadValue]:
    node = graph[node_id]
    assert isinstance(node, dict)
    inputs = node["inputs"]
    assert isinstance(inputs, dict)
    return inputs


def _drop_slot(graph: dict[str, JobPayloadValue], slot: ImageSlot) -> None:
    """An unbound slot leaves neither its node nor a link into it."""
    graph.pop(slot.node_id, None)
    for node_id in graph:
        inputs = _node_inputs(graph, node_id)
        for name in [
            name
            for name, value in inputs.items()
            if _links_to(value, slot.node_id)
        ]:
            del inputs[name]


def _links_to(value: JobPayloadValue, node_id: str) -> bool:
    return isinstance(value, list) and bool(value) and value[0] == node_id


def _first_artifact(
    entry: dict[str, JobPayloadValue], node_id: str
) -> dict[str, JobPayloadValue] | None:
    outputs = entry.get("outputs", {})
    node = outputs.get(node_id, {}) if isinstance(outputs, dict) else {}
    if not isinstance(node, dict):
        return None
    for name in OUTPUT_COLLECTION_NAMES:
        values = node.get(name)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    return value
    return None


def _raise_for_status(response: httpx.Response, action: str) -> None:
    """httpx's own `raise_for_status` quotes the URL, so the status stands alone."""
    if response.status_code // 100 != 2:
        raise RuntimeError(f"{action} failed with status {response.status_code}")


def _require_private_runtime(base_url: str) -> None:
    host = urlsplit(base_url).hostname
    if host in LOOPBACK_HOSTS:
        return
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        raise ValueError(
            "base_url must address a loopback or private local runtime"
        ) from None
    if address.is_loopback or address.is_private or address in SHARED_ADDRESS_SPACE:
        return
    raise ValueError("base_url must address a loopback or private local runtime")
