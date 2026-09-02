"""ComfyUI approved-workflow plugin tests.

A workflow template belongs to the operator, never to the job: these cases prove a
lease can fill only the declared allowlist, that an unbound image slot leaves no
dangling node in the submitted graph, and that every runtime call goes to a fake
ComfyUI built on `httpx.MockTransport`. The end-to-end case runs the real
`WorkerAgent` against the in-memory coordinator harness from `test_agent`.
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from outbound_gpu_worker_pool import (
    AssetGrant,
    JobPayloadValue,
    JobStatus,
    JobSubmission,
    LeaseGrant,
)
from outbound_gpu_worker_pool.agent import AgentOutcome
from outbound_gpu_worker_pool.comfy import (
    PACKAGED_TEMPLATES_DIRECTORY,
    ComfyWorkflowPlugin,
    TemplateRegistry,
    capability_schemas,
    input_schema,
)
from outbound_gpu_worker_pool.plugins import ExecutionContext, PluginRequestRejected
from test_agent import _harness

H3_CAPABILITY = "video.minimax_h3.text_to_video.v1"
H3_CONDITIONING_NODE = "104"
H3_SEED_NODE = "15"
H3_SCHEDULER_NODE = "9"
H3_FIRST_FRAME_NODE = "200"
H3_OUTPUT_NODE = "92"
JOB_ID = "job-1"
PROMPT_ID = "prompt-1"
PROMPT_TEXT = "a paper lantern drifting over still water"
FRAME_KEY = "inputs/pool/frame.png"
OUTPUT_KEY = "outputs/pool/clip.mp4"
VIEW_BODY = b"mp4-bytes"
UPLOAD_FILENAME = re.compile(rb'name="image"; filename="([^"]+)"')


def _packaged() -> TemplateRegistry:
    return TemplateRegistry.from_directory(PACKAGED_TEMPLATES_DIRECTORY)


def _document(**overrides: JobPayloadValue) -> dict[str, JobPayloadValue]:
    """A minimal valid template document the loader cases mutate one field at a time."""
    return {
        "capability_id": "pool.test.workflow.v1",
        "contract_version": 1,
        "template_version": "1",
        "model_id": "test-model",
        "model_version": "1",
        "output_node_id": "3",
        "output_content_type": "video/mp4",
        "inputs": [
            {
                "name": "steps",
                "node_id": "1",
                "input_name": "steps",
                "kind": "integer",
                "minimum": 1,
                "maximum": 8,
                "default": 4,
            }
        ],
        "image_slots": [{"name": "first_frame", "node_id": "2", "required": False}],
        "graph": {
            "1": {
                "class_type": "Sampler",
                "inputs": {"steps": 4, "first_frame": ["2", 0]},
            },
            "2": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
            "3": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["1", 0], "filename_prefix": "ogwp/output"},
            },
        },
    } | overrides


def _template_directory(
    tmp_path: Path, *documents: tuple[str, dict[str, JobPayloadValue]]
) -> Path:
    directory = tmp_path / "templates"
    directory.mkdir(parents=True, exist_ok=True)
    for name, document in documents:
        (directory / f"{name}.template.json").write_text(json.dumps(document))
    return directory


def _lease(
    payload: dict[str, JobPayloadValue],
    *,
    input_keys: Iterable[str] = (),
    capability_id: str = H3_CAPABILITY,
    contract_version: int = 1,
) -> LeaseGrant:
    expires = datetime.now(UTC) + timedelta(minutes=20)
    return LeaseGrant(
        job_id=JOB_ID,
        claim_token="claim-1",
        lease_until=expires,
        execution_deadline_seconds=1200,
        capability_id=capability_id,
        contract_version=contract_version,
        request_digest="digest",
        idempotency_key="idem-1",
        input_keys=tuple(input_keys),
        output_key=OUTPUT_KEY,
        payload=payload,
        input_grants=(),
        output_grant=AssetGrant(
            key=OUTPUT_KEY,
            url="memory://upload/outputs/pool/clip.mp4",
            method="PUT",
            expires_at=expires,
            content_type="video/mp4",
        ),
    )


def _history(*, completed: bool = False, error: bool = False) -> dict[str, object]:
    status_str = "error" if error else ("success" if completed else "running")
    outputs = (
        {
            H3_OUTPUT_NODE: {
                "videos": [
                    {"filename": "clip.mp4", "subfolder": "ogwp", "type": "output"}
                ]
            }
        }
        if completed
        else {}
    )
    return {
        PROMPT_ID: {
            "status": {"completed": completed, "status_str": status_str},
            "outputs": outputs,
        }
    }


@dataclass
class _FakeComfy:
    """A local ComfyUI that records exactly what the plugin sent it."""

    histories: list[dict[str, object]] = field(default_factory=list)
    uploaded_names: list[str] = field(default_factory=list)
    graphs: list[dict[str, object]] = field(default_factory=list)
    views: list[dict[str, str]] = field(default_factory=list)
    interrupts: int = 0
    stats_status: int = httpx.codes.OK
    stored_as: str = "stored-frame.png"
    on_prompt: Callable[[], None] | None = None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/upload/image":
            match = UPLOAD_FILENAME.search(request.content)
            assert match is not None
            self.uploaded_names.append(match.group(1).decode())
            return httpx.Response(httpx.codes.OK, json={"name": self.stored_as})
        if path == "/prompt":
            self.graphs.append(json.loads(request.content)["prompt"])
            if self.on_prompt is not None:
                self.on_prompt()
            return httpx.Response(httpx.codes.OK, json={"prompt_id": PROMPT_ID})
        if path == f"/history/{PROMPT_ID}":
            remaining = self.histories
            entry = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return httpx.Response(httpx.codes.OK, json=entry)
        if path == "/view":
            self.views.append(dict(request.url.params))
            return httpx.Response(httpx.codes.OK, content=VIEW_BODY)
        if path == "/interrupt":
            self.interrupts += 1
            return httpx.Response(httpx.codes.OK)
        if path == "/system_stats":
            return httpx.Response(self.stats_status, json={})
        return httpx.Response(httpx.codes.NOT_FOUND)


@asynccontextmanager
async def _comfy(
    fake: _FakeComfy, registry: TemplateRegistry | None = None
) -> AsyncIterator[ComfyWorkflowPlugin]:
    async with httpx.AsyncClient(transport=fake.transport()) as client:
        yield ComfyWorkflowPlugin(
            registry if registry is not None else _packaged(),
            client=client,
            poll_interval_seconds=0.0,
        )


@dataclass
class _Run:
    context: ExecutionContext
    progress: list[int]


def _run(
    tmp_path: Path,
    *,
    remaining: timedelta = timedelta(minutes=20),
    cancel: asyncio.Event | None = None,
    input_paths: dict[str, Path] | None = None,
) -> _Run:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    progress: list[int] = []

    async def report(percent: int) -> None:
        progress.append(percent)

    return _Run(
        context=ExecutionContext(
            job_id=JOB_ID,
            lease_id="claim-1",
            deadline=datetime.now(UTC) + remaining,
            workspace=workspace,
            input_paths=input_paths or {},
            cancel=cancel if cancel is not None else asyncio.Event(),
            progress=report,
        ),
        progress=progress,
    )


def _frame(tmp_path: Path) -> Path:
    path = tmp_path / "inputs" / "frame.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"frame-bytes")
    return path


# --- the registry -----------------------------------------------------------


def test_a_directory_of_templates_loads_and_indexes_by_capability(
    tmp_path: Path,
) -> None:
    directory = _template_directory(tmp_path, ("workflow", _document()))

    registry = TemplateRegistry.from_directory(directory)

    template = registry.template("pool.test.workflow.v1")
    assert template is not None
    assert template.template_version == "1"
    assert template.output_content_type == "video/mp4"
    assert [entry.name for entry in template.inputs] == ["steps"]
    assert template.inputs[0].maximum == 8
    assert [slot.name for slot in template.image_slots] == ["first_frame"]
    assert registry.template("pool.unknown.workflow.v1") is None


def test_the_packaged_directory_ships_the_approved_h3_template() -> None:
    template = _packaged().template(H3_CAPABILITY)

    assert template is not None
    assert template.contract_version == 1
    assert template.model_id == "minimax-h3"
    assert template.model_version == "fl2va-int8"
    assert template.output_node_id == H3_OUTPUT_NODE
    assert [entry.name for entry in template.inputs] == [
        "prompt",
        "width",
        "height",
        "length",
        "steps",
        "seed",
        "fps",
    ]
    assert [slot.name for slot in template.image_slots] == ["first_frame"]
    assert template.graph[H3_FIRST_FRAME_NODE]["class_type"] == "LoadImage"
    assert template.graph[H3_CONDITIONING_NODE]["class_type"] == "MiniMaxH3ImageToVideo"


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        pytest.param(
            _document(
                image_slots=[
                    {"name": "first_frame", "node_id": "404", "required": False}
                ]
            ),
            "404",
            id="slot-names-an-absent-node",
        ),
        pytest.param(
            _document(
                inputs=[
                    {
                        "name": "steps",
                        "node_id": "404",
                        "input_name": "steps",
                        "kind": "integer",
                    }
                ]
            ),
            "404",
            id="input-names-an-absent-node",
        ),
        pytest.param(
            _document(output_node_id="404"),
            "404",
            id="output-names-an-absent-node",
        ),
        pytest.param(
            _document(
                image_slots=[{"name": "first_frame", "node_id": "1", "required": False}]
            ),
            "LoadImage",
            id="slot-is-not-a-load-image-node",
        ),
        pytest.param(
            _document(
                inputs=[
                    {
                        "name": "steps",
                        "node_id": "1",
                        "input_name": "steps",
                        "kind": "decimal",
                    }
                ]
            ),
            "decimal",
            id="unknown-input-kind",
        ),
        pytest.param(
            _document(capability_id="Not A Capability"),
            "capability_id",
            id="malformed-capability-id",
        ),
        pytest.param(_document(graph=[]), "graph", id="graph-is-not-an-object"),
    ],
)
def test_an_unusable_template_is_refused_by_file_name(
    tmp_path: Path, document: dict[str, JobPayloadValue], reason: str
) -> None:
    directory = _template_directory(tmp_path, ("broken", document))

    with pytest.raises(ValueError) as error:
        TemplateRegistry.from_directory(directory)

    assert "broken.template.json" in str(error.value)
    assert reason in str(error.value)


def test_two_templates_may_not_claim_one_capability(tmp_path: Path) -> None:
    directory = _template_directory(
        tmp_path, ("first", _document()), ("second", _document(template_version="2"))
    )

    with pytest.raises(ValueError) as error:
        TemplateRegistry.from_directory(directory)

    assert "pool.test.workflow.v1" in str(error.value)
    assert "second.template.json" in str(error.value)


# --- the published schema ---------------------------------------------------


def test_the_published_schema_is_the_declared_allowlist() -> None:
    template = _packaged().template(H3_CAPABILITY)
    assert template is not None

    schema = input_schema(template)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["prompt"] == {
        "type": "string",
        "minLength": 1,
        "maxLength": 2000,
    }
    assert schema["properties"]["width"] == {
        "type": "integer",
        "minimum": 256,
        "maximum": 1344,
    }
    assert schema["properties"]["images"] == {
        "type": "object",
        "additionalProperties": False,
        "properties": {"first_frame": {"type": "string"}},
        "required": [],
    }
    # `first_frame` is optional, so only the required inputs are named.
    assert schema["required"] == ["prompt"]


def test_capability_schemas_publish_one_entry_per_template() -> None:
    schemas = capability_schemas(_packaged())

    assert set(schemas) == {H3_CAPABILITY}
    assert schemas[H3_CAPABILITY].contract_version == 1
    assert schemas[H3_CAPABILITY].input_schema["additionalProperties"] is False


async def test_the_manifest_advertises_every_installed_template() -> None:
    async with _comfy(_FakeComfy()) as plugin:
        manifest = plugin.capabilities()

    assert manifest.plugin_id == "comfy-workflow"
    assert manifest.plugin_version == "1"
    assert [capability.capability_id for capability in manifest.capabilities] == [
        H3_CAPABILITY
    ]
    assert [schema.capability_id for schema in manifest.schemas] == [H3_CAPABILITY]


# --- validate ---------------------------------------------------------------


async def test_validate_applies_defaults_and_binds_the_granted_frame() -> None:
    async with _comfy(_FakeComfy()) as plugin:
        request = plugin.validate(
            _lease(
                {
                    "prompt": PROMPT_TEXT,
                    "steps": 8,
                    "images": {"first_frame": FRAME_KEY},
                },
                input_keys=(FRAME_KEY,),
            )
        )

    assert request.capability_id == H3_CAPABILITY
    assert request.inputs == {
        "prompt": PROMPT_TEXT,
        "width": 1344,
        "height": 768,
        "length": 56,
        "steps": 8,
        "seed": 0,
        "fps": 24,
        "images": {"first_frame": FRAME_KEY},
    }
    assert request.seed == 0


@pytest.mark.parametrize(
    ("payload", "input_keys"),
    [
        pytest.param({"prompt": PROMPT_TEXT, "quality": "high"}, (), id="unknown-key"),
        pytest.param({"prompt": 7}, (), id="wrong-type"),
        pytest.param({"prompt": PROMPT_TEXT, "steps": True}, (), id="bool-as-integer"),
        pytest.param({"prompt": PROMPT_TEXT, "steps": 41}, (), id="above-maximum"),
        pytest.param({"prompt": PROMPT_TEXT, "width": 8}, (), id="below-minimum"),
        pytest.param({"prompt": ""}, (), id="empty-required-string"),
        pytest.param({"prompt": "x" * 2001}, (), id="over-max-length"),
        pytest.param({"steps": 8}, (), id="missing-required-input"),
        pytest.param({"prompt": PROMPT_TEXT}, (FRAME_KEY,), id="unbound-granted-input"),
        pytest.param(
            {"prompt": PROMPT_TEXT, "images": {"first_frame": "inputs/pool/other.png"}},
            (FRAME_KEY,),
            id="slot-bound-to-a-key-the-lease-never-granted",
        ),
        pytest.param(
            {"prompt": PROMPT_TEXT, "images": {"last_frame": FRAME_KEY}},
            (FRAME_KEY,),
            id="undeclared-slot",
        ),
        pytest.param(
            {"prompt": PROMPT_TEXT, "images": "inputs/pool/frame.png"},
            (),
            id="images-is-not-an-object",
        ),
    ],
)
def test_a_request_outside_the_allowlist_is_rejected(
    payload: dict[str, JobPayloadValue], input_keys: tuple[str, ...]
) -> None:
    plugin = ComfyWorkflowPlugin(_packaged(), client=httpx.AsyncClient())

    with pytest.raises(PluginRequestRejected):
        plugin.validate(_lease(payload, input_keys=input_keys))


@pytest.mark.parametrize(
    ("capability_id", "contract_version"),
    [
        pytest.param("pool.unknown.workflow.v1", 1, id="unregistered-capability"),
        pytest.param(H3_CAPABILITY, 2, id="wrong-contract-version"),
    ],
)
def test_an_unserved_capability_or_contract_is_rejected(
    capability_id: str, contract_version: int
) -> None:
    plugin = ComfyWorkflowPlugin(_packaged(), client=httpx.AsyncClient())

    with pytest.raises(PluginRequestRejected):
        plugin.validate(
            _lease(
                {"prompt": PROMPT_TEXT},
                capability_id=capability_id,
                contract_version=contract_version,
            )
        )


def test_a_required_image_slot_must_be_bound(tmp_path: Path) -> None:
    directory = _template_directory(
        tmp_path,
        (
            "required-slot",
            _document(
                image_slots=[{"name": "first_frame", "node_id": "2", "required": True}]
            ),
        ),
    )
    plugin = ComfyWorkflowPlugin(
        TemplateRegistry.from_directory(directory), client=httpx.AsyncClient()
    )

    with pytest.raises(PluginRequestRejected):
        plugin.validate(_lease({}, capability_id="pool.test.workflow.v1"))


# --- execute ----------------------------------------------------------------


async def test_execute_fills_the_graph_and_publishes_the_runtime_artifact(
    tmp_path: Path,
) -> None:
    fake = _FakeComfy(histories=[_history(), _history(completed=True)])
    run = _run(tmp_path, input_paths={FRAME_KEY: _frame(tmp_path)})
    registry = _packaged()

    async with _comfy(fake, registry) as plugin:
        request = plugin.validate(
            _lease(
                {
                    "prompt": PROMPT_TEXT,
                    "steps": 8,
                    "images": {"first_frame": FRAME_KEY},
                },
                input_keys=(FRAME_KEY,),
            )
        )

        output = await plugin.execute(run.context, request)

    graph = fake.graphs[0]
    assert fake.uploaded_names == [f"{JOB_ID}-first_frame.png"]
    assert graph[H3_FIRST_FRAME_NODE]["inputs"]["image"] == fake.stored_as
    assert graph[H3_CONDITIONING_NODE]["inputs"]["prompt"] == PROMPT_TEXT
    assert graph[H3_CONDITIONING_NODE]["inputs"]["first_frame"] == [
        H3_FIRST_FRAME_NODE,
        0,
    ]
    assert graph[H3_CONDITIONING_NODE]["inputs"]["width"] == 1344
    assert graph[H3_SCHEDULER_NODE]["inputs"]["steps"] == 8
    assert graph[H3_SEED_NODE]["inputs"]["noise_seed"] == 0
    assert graph[H3_OUTPUT_NODE]["inputs"]["filename_prefix"] == f"ogwp/{JOB_ID}"
    assert fake.views == [
        {"filename": "clip.mp4", "subfolder": "ogwp", "type": "output"}
    ]
    assert output.path.read_bytes() == VIEW_BODY
    assert output.path.parent == run.context.workspace / "output"
    assert output.content_type == "video/mp4"
    assert output.model_id == "minimax-h3"
    assert output.model_version == "fl2va-int8"
    assert output.seed == 0
    assert output.diagnostics == {"prompt_id": PROMPT_ID, "template_version": "1"}
    assert run.progress == [0, 50, 100]
    # The installed template is the operator's; a job fills a copy of it.
    template = registry.template(H3_CAPABILITY)
    assert template is not None
    installed = template.graph[H3_OUTPUT_NODE]["inputs"]["filename_prefix"]
    assert installed != f"ogwp/{JOB_ID}"


async def test_an_unbound_slot_leaves_neither_its_node_nor_a_dangling_link(
    tmp_path: Path,
) -> None:
    fake = _FakeComfy(histories=[_history(completed=True)])
    run = _run(tmp_path)

    async with _comfy(fake) as plugin:
        request = plugin.validate(_lease({"prompt": PROMPT_TEXT}))

        await plugin.execute(run.context, request)

    graph = fake.graphs[0]
    assert H3_FIRST_FRAME_NODE not in graph
    assert "first_frame" not in graph[H3_CONDITIONING_NODE]["inputs"]
    assert fake.uploaded_names == []


async def test_a_runtime_execution_error_raises_for_the_agent_to_release(
    tmp_path: Path,
) -> None:
    fake = _FakeComfy(histories=[_history(error=True)])
    run = _run(tmp_path)

    async with _comfy(fake) as plugin:
        request = plugin.validate(_lease({"prompt": PROMPT_TEXT}))

        with pytest.raises(RuntimeError) as error:
            await plugin.execute(run.context, request)

    # A transient failure names neither the prompt nor the runtime address.
    assert PROMPT_TEXT not in str(error.value)
    assert "127.0.0.1" not in str(error.value)


async def test_a_cancelled_job_interrupts_the_runtime(tmp_path: Path) -> None:
    fake = _FakeComfy(histories=[_history()])
    cancel = asyncio.Event()
    cancel.set()
    run = _run(tmp_path, cancel=cancel)

    async with _comfy(fake) as plugin:
        request = plugin.validate(_lease({"prompt": PROMPT_TEXT}))

        with pytest.raises(asyncio.CancelledError):
            await plugin.execute(run.context, request)

    assert fake.interrupts == 1


async def test_a_passed_deadline_interrupts_the_runtime(tmp_path: Path) -> None:
    fake = _FakeComfy(histories=[_history()])
    run = _run(tmp_path, remaining=timedelta(seconds=-1))

    async with _comfy(fake) as plugin:
        request = plugin.validate(_lease({"prompt": PROMPT_TEXT}))

        with pytest.raises(TimeoutError):
            await plugin.execute(run.context, request)

    assert fake.interrupts == 1


async def test_cancel_interrupts_only_the_job_whose_prompt_is_running(
    tmp_path: Path,
) -> None:
    submitted = asyncio.Event()
    fake = _FakeComfy(
        histories=[_history(), _history(completed=True)], on_prompt=submitted.set
    )
    run = _run(tmp_path)
    cancelled: list[bool] = []

    async with _comfy(fake) as plugin:
        assert await plugin.cancel(JOB_ID) is False

        async def cancel_once_submitted() -> None:
            await submitted.wait()
            cancelled.append(await plugin.cancel(JOB_ID))

        request = plugin.validate(_lease({"prompt": PROMPT_TEXT}))
        watcher = asyncio.create_task(cancel_once_submitted())

        await plugin.execute(run.context, request)
        await watcher

        # The prompt is tracked for the length of the execution and no longer.
        assert cancelled == [True]
        assert await plugin.cancel(JOB_ID) is False

    assert fake.interrupts == 1


@pytest.mark.parametrize(
    ("status", "healthy"),
    [pytest.param(200, True, id="up"), pytest.param(503, False, id="down")],
)
async def test_health_reports_the_local_runtime(status: int, healthy: bool) -> None:
    fake = _FakeComfy(stats_status=status)

    async with _comfy(fake) as plugin:
        health = await plugin.health()

    assert health.healthy is healthy
    if not healthy:
        assert str(status) in health.detail


# --- the runtime address ----------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8188",
        "http://localhost:8188",
        "http://[::1]:8188",
        "http://192.168.1.50:8188",
        "http://10.4.0.9:8188",
        "http://100.64.1.2:8188",
    ],
)
def test_a_private_runtime_address_is_accepted(base_url: str) -> None:
    plugin = ComfyWorkflowPlugin(
        _packaged(), client=httpx.AsyncClient(), base_url=base_url
    )

    assert plugin.capabilities().plugin_id == "comfy-workflow"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://comfy.example.invalid:8188",
        "http://8.8.8.8:8188",
        "http://comfy-box:8188",
        "file:///etc/passwd",
    ],
)
def test_a_reachable_runtime_address_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError) as error:
        ComfyWorkflowPlugin(_packaged(), client=httpx.AsyncClient(), base_url=base_url)

    assert base_url not in str(error.value)


# --- end to end -------------------------------------------------------------


async def test_the_plugin_completes_one_h3_job_through_the_worker_agent(
    tmp_path: Path,
) -> None:
    fake = _FakeComfy(histories=[_history(), _history(completed=True)])

    async with httpx.AsyncClient(transport=fake.transport()) as client:
        plugin = ComfyWorkflowPlugin(
            _packaged(), client=client, poll_interval_seconds=0.0
        )
        async with _harness(tmp_path / "workspaces", plugins=(plugin,)) as harness:
            harness.assets.assets[FRAME_KEY] = b"frame-bytes"
            harness.assets.content_types[FRAME_KEY] = "image/png"
            record = await harness.service.submit(
                JobSubmission(
                    job_id=str(uuid4()),
                    idempotency_key="pool:h3-1",
                    capability_id=H3_CAPABILITY,
                    input_keys=(FRAME_KEY,),
                    output_key=OUTPUT_KEY,
                    payload={
                        "prompt": PROMPT_TEXT,
                        "images": {"first_frame": FRAME_KEY},
                    },
                    tenant_id="tenant-a",
                )
            )

            outcome = await harness.agent.run_once()

    assert outcome is AgentOutcome.COMPLETED
    assert harness.jobs.records[record.job_id].status is JobStatus.COMPLETED
    assert harness.assets.assets[OUTPUT_KEY] == VIEW_BODY
    graph = fake.graphs[0]
    assert fake.uploaded_names == [f"{record.job_id}-first_frame.png"]
    assert graph[H3_OUTPUT_NODE]["inputs"]["filename_prefix"] == f"ogwp/{record.job_id}"


async def test_the_plugin_completes_a_text_only_h3_job_with_no_granted_inputs(
    tmp_path: Path,
) -> None:
    fake = _FakeComfy(histories=[_history(), _history(completed=True)])

    async with httpx.AsyncClient(transport=fake.transport()) as client:
        plugin = ComfyWorkflowPlugin(
            _packaged(), client=client, poll_interval_seconds=0.0
        )
        async with _harness(tmp_path / "workspaces", plugins=(plugin,)) as harness:
            record = await harness.service.submit(
                JobSubmission(
                    job_id=str(uuid4()),
                    idempotency_key="pool:h3-text",
                    capability_id=H3_CAPABILITY,
                    input_keys=(),
                    output_key=OUTPUT_KEY,
                    payload={"prompt": PROMPT_TEXT},
                    tenant_id="tenant-a",
                )
            )

            outcome = await harness.agent.run_once()

    assert outcome is AgentOutcome.COMPLETED
    assert harness.jobs.records[record.job_id].status is JobStatus.COMPLETED
    assert harness.assets.assets[OUTPUT_KEY] == VIEW_BODY
    assert harness.transfer.downloads == []
    graph = fake.graphs[0]
    assert fake.uploaded_names == []
    assert H3_FIRST_FRAME_NODE not in graph
    assert graph[H3_CONDITIONING_NODE]["inputs"]["prompt"] == PROMPT_TEXT
