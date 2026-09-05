# outbound-gpu-worker-pool

Run GPU work on machines you trust without exposing them. A worker opens no port and holds
no bucket or database credential: it polls an authenticated coordinator over outbound HTTPS,
leases one job, downloads its inputs through job-scoped signed URLs, runs an approved plugin
in a per-job workspace, uploads the artifact through a create-once grant, and attests the
result. The coordinator verifies that attestation against the stored object before the job is
marked complete. PostgreSQL is the only durable state; there is no broker, no VPN, and no
inbound call to a worker, ever.

## Trust model

- **The coordinator never calls a worker.** Every lease, heartbeat, and commit is an outbound
  request the worker makes and the coordinator authenticates.
- **A worker holds one credential and nothing else.** No bucket key, no database URL, no
  provider secret. Its identity is per machine, so revoking one worker touches no other.
- **A worker never chooses what it touches.** URLs, keys, and payloads arrive inside a signed
  lease grant: exact object, single method, short-lived, with no list permission.
- **The coordinator verifies before it commits.** A completion is accepted only when the
  manifest's output key, idempotency key, and canonical request digest match the job row, the
  publication mode is `immutable_create_once`, and the stored object's size and sha256 match
  the attestation. Any mismatch fails the job terminally.
- **Nothing sensitive is written down.** Audit rows and logs carry ids, counts, digests,
  statuses, and durations — never a credential, a signed URL, a prompt, or asset bytes.

## Install

```bash
pip install outbound-gpu-worker-pool[coordinator]   # asyncpg + fastapi, the server side
pip install outbound-gpu-worker-pool[agent]         # httpx, the worker side
pip install outbound-gpu-worker-pool[comfy]         # the approved-workflow ComfyUI plugin
pip install outbound-gpu-worker-pool[gcs]           # Google Cloud Storage asset store
pip install outbound-gpu-worker-pool[google-auth]   # verify or mint Google identity tokens
```

The core package depends only on `pydantic`, so a worker machine installs no server code and
a coordinator installs no HTTP client.

## Quickstart

Coordinator and worker in one process, on in-memory stores. This is the whole round trip —
submit, lease, download, execute, publish, verify — and it runs as
`tests/test_readme_quickstart.py`, so it cannot drift from the library.

```python
import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

import httpx

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    JobSubmission,
    MemoryAssetStore,
    MemoryAssetTransfer,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerIdentity,
)
from outbound_gpu_worker_pool.agent import WorkerAgent
from outbound_gpu_worker_pool.coordinator import create_coordinator_app
from outbound_gpu_worker_pool.plugins import (
    DeterministicEchoPlugin,
    capability_schemas_from_plugins,
)
from outbound_gpu_worker_pool.service import WorkerPoolService


async def main() -> None:
    assets = MemoryAssetStore()
    await assets.write_once("inputs/hello.bin", b"hello worker pool", "application/octet-stream")

    plugins = (DeterministicEchoPlugin(),)
    service = WorkerPoolService(
        MemoryJobStore(),
        assets,
        MemoryWorkerRegistry(),
        MemoryAuditLog(),
        MemoryWorkerAuthenticator(
            {"worker-token": WorkerIdentity("worker-a", "static:worker-a", "static")}
        ),
        capability_schemas_from_plugins(plugins),
    )

    submitted = await service.submit(
        JobSubmission(
            job_id=str(uuid4()),
            idempotency_key="quickstart-1",
            capability_id=DETERMINISTIC_ECHO_CAPABILITY,
            input_keys=("inputs/hello.bin",),
            output_key="outputs/hello.txt",
            payload={"seed": 7, "label": "quickstart"},
        )
    )

    with tempfile.TemporaryDirectory() as workspace:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_coordinator_app(service)),
            base_url="http://coordinator",
        ) as http:
            agent = WorkerAgent(
                coordinator_url="http://coordinator",
                worker_id="worker-a",
                credential=lambda: "worker-token",
                plugins=plugins,
                transfer=MemoryAssetTransfer(assets),
                http=http,
                workspace_root=Path(workspace),
            )
            outcome = await agent.run_once()

    record = await service.get(submitted.job_id)
    print(f"{outcome} -> {record.status}")
    print((await assets.read_limited(record.output_key, 1_000_000)).decode())


asyncio.run(main())
```

In production the two halves are separate processes: replace the memory stores with
`PostgresJobStore`, `PostgresWorkerRegistry`, `PostgresAuditLog`, and `GcsAssetStore`, and
replace `MemoryAssetTransfer` with `HttpAssetTransfer`.

## Using it from your own application

The library ships **no product routes**. Job submission is a service call your own
authenticated handler makes, so tenancy, quotas, and authorization stay yours:

```python
record = await service.submit(submission)     # insert, or replay an equal submission
record = await service.get(job_id)            # never carries the lease claim token
cancelled = await service.cancel(job_id)      # queued or processing only
records = await service.list_for_tenant(tenant_id)
```

Worker administration is service calls too: `set_worker_status(worker_id, status)`,
`list_workers()`, and `audit_for_job(job_id)` for the redacted trail. What the library does
publish over HTTP is the authenticated worker surface: mount `create_worker_router(service)`
next to your own routes, or run `create_coordinator_app(service)` as its own deployment when
workers should reach a service that holds nothing else.

`PostgresJobStore.start()` applies the migrations in `migrations/001_worker_pool.sql` under an
advisory lock, creating `pool_jobs`, `pool_workers`, and `pool_audit_events`. Every table name
is a module constant so your own reporting can read them.

## Enrolling a worker

**Static tokens** (development, or a single machine). Generate the token and its digest, keep
the token on the worker machine only, and enroll the digest with the coordinator:

```bash
python -c "import secrets,hashlib; t=secrets.token_urlsafe(32); print(t, hashlib.sha256(t.encode()).hexdigest())"
```

Put `worker-id:<digest>` in the coordinator's `OGWP_WORKER_TOKENS` (comma separated for
several workers) and give the worker the token itself as `OGWP_WORKER_TOKEN`. The coordinator
stores digests only.

**Google OIDC** (one service account per machine — never a shared account, because the
identity subject is what revocation targets). The machine's identity token must carry a
verified `email` claim equal to the `identity_subject` of its registry row, so insert that row
before the first heartbeat:

```sql
INSERT INTO pool_workers (worker_id, identity_subject)
VALUES ('gpu-01', 'gpu-01@<project>.iam.gserviceaccount.com')
ON CONFLICT (worker_id) DO NOTHING;
```

Smoke-test the credential from the machine with
`gcloud auth print-identity-token --audiences=<audience>`. If the coordinator runs as a
private service, the worker's service account also needs permission to invoke it.

## Revoking a worker

Revocation is independent of the credential and takes effect on the worker's next request:

```python
await service.set_worker_status("gpu-01", WorkerStatus.REVOKED)
```

A revoked worker gets `403` on every `/worker/v1` route even with a valid credential, stays
revoked across heartbeats, and leaves every other worker untouched. `WorkerStatus.DRAINING`
is the softer form: the worker keeps its in-flight job but is granted no new lease.

## Running the coordinator

```bash
OGWP_DATABASE_URL=postgresql://... \
OGWP_ASSET_BUCKET=my-pool-assets \
OGWP_WORKER_AUTH=static \
OGWP_WORKER_TOKENS=gpu-01:<sha256 hex> \
python -m outbound_gpu_worker_pool.coordinator_main
```

| Variable | Values | Meaning |
|---|---|---|
| `OGWP_JOB_BACKEND` | `postgres` (default), `memory` | where jobs, workers, and audit rows live |
| `OGWP_DATABASE_URL` | connection URL | required for `postgres` |
| `OGWP_ASSET_BACKEND` | `gcs` (default), `memory` | which asset store mints grants |
| `OGWP_ASSET_BUCKET` | bucket name | required for `gcs` |
| `OGWP_SIGNING_SERVICE_ACCOUNT` | service account email | signs grant URLs when the runtime identity cannot |
| `OGWP_ALLOWED_READ_PREFIXES` | comma separated, default `inputs/,outputs/` | the only prefixes a read grant may address |
| `OGWP_ALLOWED_OUTPUT_PREFIXES` | comma separated, default `outputs/` | the only prefixes an upload grant may address |
| `OGWP_WORKER_AUTH` | `static` (default), `google_oidc` | how a worker credential is resolved; there is no `none` |
| `OGWP_WORKER_TOKENS` | `worker-id:<sha256 hex>[,...]` | required for `static`; digests only |
| `OGWP_WORKER_AUDIENCE` | audience string | required for `google_oidc` |
| `OGWP_WORKER_AUTO_ENROLL` | `false` (default), `true` | with `google_oidc`, admit a verified `gpu-worker-<id>@…` account that has no registry row yet and create its row on first heartbeat; enable only when something in front of the coordinator (for example Cloud Run IAM) already decides who may call it |
| `OGWP_CAPABILITY_PLUGINS` | comma separated, default `deterministic-echo` | the plugins whose schemas this coordinator publishes |
| `OGWP_PORT` | default `8080` | listen port |

## Running the agent

```bash
OGWP_WORKER_COORDINATOR_URL=https://<coordinator-host> \
OGWP_WORKER_ID=gpu-01 \
OGWP_WORKER_AUTH=static \
OGWP_WORKER_TOKEN=<token> \
OGWP_WORKER_PLUGINS=deterministic-echo \
OGWP_WORKER_WORKSPACE=/var/tmp/outbound-gpu-worker \
python -m outbound_gpu_worker_pool.agent_main
```

| Variable | Values | Meaning |
|---|---|---|
| `OGWP_WORKER_COORDINATOR_URL` | URL | required; the only host the agent talks to |
| `OGWP_WORKER_ID` | worker id | required; must match the enrolled row |
| `OGWP_WORKER_AUTH` | `static` (default), `google_oidc` | credential source |
| `OGWP_WORKER_TOKEN` | the token itself | required for `static` |
| `OGWP_WORKER_AUDIENCE` | audience string | required for `google_oidc`; a fresh identity token is minted per request |
| `OGWP_WORKER_PLUGINS` | comma separated, default `deterministic-echo` | the plugins this machine is approved to run (`comfy-workflow` below) |
| `OGWP_WORKER_CONCURRENCY` | default `1` | leases this machine advertises per capability |
| `OGWP_WORKER_WORKSPACE` | path, default `<tmpdir>/outbound-gpu-worker` | per-job workspaces are created and deleted under here |
| `OGWP_WORKER_MAX_INPUT_BYTES` | default `2147483648` (2 GiB) | the scratch bound: a single granted input larger than this aborts the download and releases the job |
| `OGWP_WORKER_GPU_MODEL` | free text | advertised to the registry |
| `OGWP_WORKER_VRAM_MB` | integer | advertised to the registry |

SIGTERM and SIGINT drain: the in-flight job finishes, a final draining heartbeat is sent, and
the process exits 0. A lease taken as draining begins is released back to the pool rather than
started.

To put the agent on a real Linux machine — one script, a systemd *user* unit, and a check that
proves the box exposes nothing new — see [`docs/operations.md`](docs/operations.md) and
`deploy/agent/`.

Writing your own plugin means implementing `GpuExecutorPlugin` — `capabilities()`,
`validate(lease)`, `execute(context, request)`, `cancel(job_id)`, `health()` — and naming it in
`OGWP_WORKER_PLUGINS`. `validate` is the terminal gate: it runs before a single byte is
downloaded. `DeterministicEchoPlugin` is the reference implementation and the round-trip probe.

## ComfyUI approved workflows

`comfy-workflow` runs ComfyUI on a machine you already trust, without letting a job decide
what runs there. **A job never carries a graph.** A capability id maps to a *workflow
template* — a versioned ComfyUI API-format graph the operator installed on the machine —
and the job may only fill the template's declared allowlist:

- a **typed input** per allowlist entry: a name, the node and node input it fills, its kind
  (`string`, `integer`, `number`, `boolean`), and its bounds and default. Anything outside the
  allowlist, of the wrong type, or out of range is a terminal rejection before a byte moves.
- an **image slot** per declared `LoadImage` node. The job binds a slot to an asset key the
  lease already granted (`{"images": {"first_frame": "inputs/…/frame.png"}}`); a slot bound to
  a key the coordinator did not grant, and a granted input no slot binds, are both rejected.
  An unbound optional slot is removed from the submitted graph along with the link into it.
  A slot may also declare `dependent_node_ids`: nodes that only make sense once the slot is
  bound (a scale/encode chain feeding a downstream node's optional input). Dropping the slot
  drops those nodes too, so a template author must give the downstream consumer a way to
  tolerate losing that input.

A template is a `*.template.json` file — `capability_id`, `contract_version`,
`template_version`, `model_id`, `model_version`, `output_node_id`, `output_content_type`,
`inputs`, `image_slots`, `graph` — and an operator installs a set of them by pointing
`OGWP_COMFY_TEMPLATES_DIR` at a directory of them. Every file is validated on load: the
capability id, that each referenced node (including a slot's `dependent_node_ids`) exists in
the graph, that a slot's node really is a `LoadImage`, that a dependent node is never another
slot's own node, that every input kind is known, and that no two templates claim one
capability. A bad file names itself and the worker refuses to start.

Three templates ship in the package and are the default set.

`image.flux2_klein.subject.v1`: a single still on the FLUX.2 Klein base 4B (fp8) with the
`qwen_3_4b` text encoder and `flux2-vae`, contract `1`, output `image/png`, fixed 1024x1024
canvas. Built for character subjects on a chroma-key green screen; the consumer keys the
backdrop out. It renders in about 20 s on an RTX 4090.

| Input | Kind | Range | Default |
|---|---|---|---|
| `prompt` | string | 1–2000 characters | required |
| `steps` | integer | 1–40 | 20 |
| `seed` | integer | ≥ 0 | 0 |

`image.flux2_klein.subject.v2`: the same base model, encoder, VAE, contract, output, and canvas
as v1, but conditions the render on up to three reference images through `ReferenceLatent`
instead of text alone — a subject that has to look like specific people or objects, not just
match a description. Each reference is optional and independent; any subset may be bound.

| Input | Kind | Range | Default |
|---|---|---|---|
| `prompt` | string | 1–2000 characters | required |
| `steps` | integer | 1–40 | 20 |
| `seed` | integer | ≥ 0 | 0 |
| `images.ref_1` | asset key | a granted input key | optional |
| `images.ref_2` | asset key | a granted input key | optional |
| `images.ref_3` | asset key | a granted input key | optional |

`video.minimax_h3.text_to_video.v1`, model `minimax-h3` / `fl2va-int8`, contract `1`, output
`video/mp4`.

| Input | Kind | Range | Default |
|---|---|---|---|
| `prompt` | string | 1–2000 characters | required |
| `width` | integer | 256–1344 | 1344 |
| `height` | integer | 256–1344 | 768 |
| `length` | integer | 17–161 | 56 |
| `steps` | integer | 4–40 | 20 |
| `seed` | integer | ≥ 0 | 0 |
| `fps` | integer | 8–30 | 24 |
| `images.first_frame` | asset key | a granted input key | optional |

**The runtime is local by construction.** `OGWP_COMFY_URL` must address loopback (`127.0.0.1`,
`localhost`, `::1`) or a private address (RFC1918, or `100.64.0.0/10` for a tailnet); the
plugin refuses to start against anything else, so an approved template cannot be pointed at a
runtime the operator does not own. Uploaded frames are named per job, the output prefix is
`ogwp/<job_id>`, and nothing the plugin logs or raises carries the prompt text or the runtime
address.

Run it with `pip install outbound-gpu-worker-pool[comfy]` and:

```bash
OGWP_WORKER_PLUGINS=comfy-workflow \
OGWP_COMFY_URL=http://127.0.0.1:8188 \
OGWP_COMFY_TEMPLATES_DIR=/etc/outbound-gpu-worker/templates \
python -m outbound_gpu_worker_pool.agent_main
```

| Variable | Values | Meaning |
|---|---|---|
| `OGWP_COMFY_URL` | default `http://127.0.0.1:8188` | the local ComfyUI; loopback or private only |
| `OGWP_COMFY_TEMPLATES_DIR` | path, default the packaged templates | the workflow templates this machine is approved to run |
| `OGWP_COMFY_START_COMMAND` | shell-split command, default unset | how to start the local runtime when it is down; unset means the agent fails the job instead |
| `OGWP_COMFY_STARTUP_TIMEOUT_SECONDS` | seconds, default `180` | how long to wait for the runtime to answer healthy after starting it |

A coordinator publishes the same templates' schemas with
`OGWP_CAPABILITY_PLUGINS=comfy-workflow`. It reads the packaged template directory for the
schemas only and never opens a connection to any runtime.

**The agent owns ComfyUI's availability, not the other way round.** Before every job it
probes `/system_stats`; if the runtime is down it runs `OGWP_COMFY_START_COMMAND` and waits
for health before submitting. This matters on a shared GPU machine, where another product's
helper may stop an idle ComfyUI to free the GPU: set `OGWP_COMFY_START_COMMAND` there (for
example `systemctl --user start comfyui`) so the pool brings the runtime back up itself
instead of failing every job with a connection error.

## Failure semantics

**Retryable — released, not failed.** A transport or execution failure, a lost lease, a
draining worker, or a temporary storage failure releases the job. It returns to the queue
without a further attempt being charged and another worker picks it up, until the attempt
budget is spent; then it settles as `failed` with `retryable=true` so an operator can decide.

**Terminal — failed on the spot.** A capability or contract version no plugin serves, a
payload a plugin rejects (`PluginRequestRejected`), or an attestation that does not match the
job row fails the job with `retryable=false`. These cost one attempt at most, because a retry
would fail identically.

Publication is create-once, so a retried attempt that already uploaded its artifact is a
replay rather than an overwrite: the upload's `412` is treated as already-published.

## Design

[`docs/design.md`](docs/design.md) has the architecture, the trust boundaries, the
alternatives that were rejected, and the rollout plan.
[`docs/operations.md`](docs/operations.md) is the other half: installing, enrolling, running,
verifying, draining, upgrading, and rotating a worker machine.

Status: alpha, slice 1 of the design's rollout — durable capability leases, the authenticated
coordinator API, job-scoped asset grants with verified immutable outputs, and the deterministic
reference plugin and agent.
