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
    WorkerEnrollment,
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

    # A trusted host approves both the identity and the house/tenant membership.
    await service.enroll_worker("worker-a", WorkerEnrollment("static:worker-a", None))

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

`PostgresJobStore.start()` applies the migrations in `migrations/001_worker_pool.sql`,
`migrations/002_worker_telemetry.sql`, and `migrations/003_worker_tenancy.sql` under an
advisory lock, creating `pool_jobs`, `pool_workers`, and `pool_audit_events`, adding the
`gpus` and `busy_job_id` columns to `pool_workers`, and adding `pool_workers.tenant_id`
and `pool_jobs.requirements`. Every migration is add-only and idempotent, so a replay
changes no column: nothing here ever drops a column a later start recreates. Every table name is a module constant so your own reporting can read them.

### Pool status routes

For an **administrator** surface, `create_pool_status_router(service, *, authorize_admin)`
publishes two read-only routes. The authorization dependency is mandatory and must return
literal `True` only for an authenticated pool administrator (or raise an HTTP error).
Returning a logged-in user object is not sufficient. These are global administrative views,
not tenant self-service endpoints; a `tenant_id` query parameter is a filter, not authorization.

```python
from fastapi import Depends
from outbound_gpu_worker_pool.routes import create_pool_status_router

# current_user is the host's existing authenticated-principal dependency.
async def authorize_pool_admin(user=Depends(current_user)) -> bool:
    return user.is_pool_admin is True

app.include_router(create_pool_status_router(service, authorize_admin=authorize_pool_admin))
```

- **`GET /pool/workers`** — `{"workers": [{worker_id, status, last_heartbeat_at, capabilities,
  gpus, busy_job_id, draining, tenant_id}, ...]}`. `?tenant_id=…` narrows to one tenant's
  machines (`tenant_id` is null for a house worker). `status` is `busy` when the worker's last heartbeat
  carried a `busy_job_id`, else `draining` when the registry has it draining, else `online`
  when the heartbeat is within 90 seconds, else `offline`. A worker never heard from, or not
  heard from in the last 24 hours, is omitted entirely rather than reported offline.
- **`GET /pool/queue`** — `{"queued": n, "processing": n, "by_capability": {capability_id:
  {"queued": n, "processing": n}}, "online_workers_by_capability": {capability_id: n}}`,
  bounded aggregate counts, never a job listing. The worker counts come from the registry's
  own rows (no job scan) and tell "queued with nothing online that serves this capability"
  from "queued, waiting its turn". They count capability only: tenancy and a job's
  `requirements` narrow further, so a nonzero count is not proof a *particular* job is
  leasable.

## Worker tenancy

A machine can belong to one tenant, so it only ever runs that tenant's work.

- **`pool_workers.tenant_id` null means the house pool** — the shared machines an
  operator runs. The coordinator's operator approves this membership **before**
  the first heartbeat. `OGWP_WORKER_TENANT` is only a consistency assertion: it must
  match the server-approved tenant, including an explicit null/house assignment.
  An unauthorized first assignment or a later change is refused with `409`; the
  worker cannot choose another tenant just by knowing its identifier.
- **The rule for leasing is exact-match, both ways.** A tenanted worker leases only
  that tenant's jobs; a house worker leases only untenanted ones. There is no
  spillover and no fallback: a tenant whose machine is offline sees their jobs
  wait rather than quietly consume the house pool, and house work never lands on
  somebody's own hardware.
- **A job's tenant is the host's to set** (`JobSubmission.tenant_id`), alongside
  `service.list_for_tenant(tenant_id)` and `GET /pool/workers?tenant_id=…` for
  showing one user their own machines.

## Hardware requirements

A job may also declare what the machine must offer, in `JobSubmission.requirements`
(`JobRequirements`), and only a worker that satisfies all of it may lease the job:

| Field | Meaning |
|---|---|
| `min_vram_mb` | the worker's advertised `vram_mb` must be at least this; a worker that advertises none matches only jobs with no `min_vram_mb` |
| `gpu_models` | any-of, matched case-insensitively against the worker's `gpu_model` |
| `labels` | all-of: every label must be one the worker advertises |

Empty requirements — the default — mean any worker serving the capability. This
composes with tenancy rather than replacing it: the right tenant with too little
VRAM still does not lease. Requirements and placement are matched in the lease
statement itself, so an ineligible worker never sees the job at all.

## Enrolling a worker

A valid credential alone does not authorize a new machine to join. First bind its worker ID,
full identity subject, and tenant in server-controlled policy. A host can pre-enroll it through
this **administrator-only service call** (never expose this on the worker API):

```python
from outbound_gpu_worker_pool import WorkerEnrollment

await service.enroll_worker(
    "gpu-01", WorkerEnrollment("static:gpu-01", "tenant-a")
)
```

For a standalone coordinator, `OGWP_WORKER_ENROLLMENTS` supplies approved first-heartbeat
bindings as JSON. An explicit `tenant_id` is required; `null` authorizes the house pool:

```json
{
  "gpu-01": {"identity_subject": "static:gpu-01", "tenant_id": "tenant-a"},
  "gpu-02": {"identity_subject": "gpu-worker-gpu-02@example-project.iam.gserviceaccount.com", "tenant_id": null}
}
```

The service constructor accepts the equivalent `enrollments: Mapping[str, WorkerEnrollment]`.
Existing registry bindings continue to work without this configuration. Both stores reject
identity or tenant replacement atomically, including concurrent registration. Use a new worker
ID and revoke the old one for a deliberate identity change; heartbeats are not a rotation API.


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
| `OGWP_WORKER_AUTO_ENROLL` | `false` (default), `true` | with `google_oidc`, allow only exact service-account identities in `OGWP_WORKER_ENROLLMENTS` to register on first heartbeat; requires a nonempty allowlist plus external IAM admission, never infers authorization from an account-name prefix |
| `OGWP_WORKER_ENROLLMENTS` | JSON object | server-approved worker IDs, full `identity_subject`, and explicit `tenant_id`; required for new first-heartbeat admissions, not existing registry rows |
| `OGWP_PRE_AUTH_LIMIT_PER_MINUTE` | default `1200` | process-wide admission budget before verification/database work |
| `OGWP_PER_SOURCE_LIMIT_PER_MINUTE` | default `120` | per-ASGI-peer admission budget; consider trusted proxy/NAT topology |
| `OGWP_AUTH_AUDIT_LIMIT_PER_MINUTE` | default `30` | shared sampling budget for authentication/rate-limit rejection audit rows |
| `OGWP_MAX_AUTH_CONCURRENCY` | default `16` | concurrent authentication operations; overload is rejected, not queued |
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
| `OGWP_WORKER_RUNTIME_VERSIONS` | default `{}` | JSON object of declared runtime/recipe versions, checked against job requirements |
| `OGWP_WORKER_MAX_INPUT_BYTES` | default `2147483648` (2 GiB) | the scratch bound: a single granted input larger than this aborts the download and releases the job |
| `OGWP_WORKER_TENANT` | tenant id, default unset | must match the server-approved tenant; unset requires explicit house-pool approval — see [Worker tenancy](#worker-tenancy) |
| `OGWP_WORKER_GPU_MODEL` | free text | advertised to the registry; also matched against a job's `gpu_models` requirement |
| `OGWP_WORKER_VRAM_MB` | integer | advertised to the registry; also matched against a job's `min_vram_mb` requirement |
| `OGWP_NVIDIA_SMI` | path | overrides binary resolution for per-GPU telemetry sampling; otherwise `nvidia-smi` on `PATH`, then `/usr/lib/wsl/lib/nvidia-smi` |

Every heartbeat also samples per-GPU utilization and memory with `nvidia-smi` (a 3 second
timeout, run off the event loop) and reports the job it is currently executing, if any. Neither
can delay or fail a heartbeat: a missing or failing `nvidia-smi` logs one warning and reports no
GPUs rather than retrying or raising. See [Pool status routes](#pool-status-routes) for how a
host surfaces this.

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
`video/mp4`. Both frames are optional: bind none for text-to-video, `first_frame` alone to
start the clip on an image, or both to pin the clip's first and last frames (a frame-pinned
camera move between two images). An unbound frame is dropped from the graph, so a job that
sends no `last_frame` submits exactly the graph it did before the slot existed.

| Input | Kind | Range | Default |
|---|---|---|---|
| `prompt` | string | 1–2000 characters | required |
| `width` | integer | 256–1344 | 1344 |
| `height` | integer | 256–1344 | 768 |
| `length` | integer | 17–362 | 56 |
| `steps` | integer | 4–40 | 20 |
| `seed` | integer | ≥ 0 | 0 |
| `fps` | integer | 8–30 | 24 |
| `images.first_frame` | asset key | a granted input key | optional |
| `images.last_frame` | asset key | a granted input key | optional |

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

**A machine profile can share the agent with other templates.** Setting
`MVS_WORKER_PROFILE=h3-24gb-standard` on the `agent_main` entrypoint opts into a mixed mode:

- Startup runs the standard profile check against the running ComfyUI (models, nodes, VRAM and
  the optional last-frame input). A failed check starts no agent.
- The worker registers the recipe, configuration and feature versions the check found, in place
  of `OGWP_WORKER_RUNTIME_VERSIONS`.
- `video.minimax_h3.text_to_video.v1` is served only by the profile plugin. It re-checks the
  runtime before each job and attests the runtime versions and effective inputs it executed.
- Every other approved template (Flux, dialogue) stays on `comfy-workflow`. That plugin's
  registry has the profile's capability removed, and the agent refuses to start if any other
  plugin still claims it.

`MVS_WORKER_PROFILE=flux2-klein-subject-12gb` mixes the same way: its capability
`image.flux2_klein.subject.12gb.v1` runs only through the profile plugin. Only these two
profiles can be mixed; the bounded 16GB recipe is refused. Without the
variable the entrypoint behaves exactly as before. `build_agent_from_env` never reads the
variable, so callers that wire their own profile plugin are unaffected.

**One runtime can serve several profiles.** Give `MVS_WORKER_PROFILE` a comma-separated
list, for example `h3-24gb-standard,flux2-klein-subject-12gb`.

- Every profile is checked at startup, and a failure in any one of them starts no agent.
- Each capability runs only through its own profile plugin.
- The first profile registers the plain `magic_recipe` and `magic_config` keys. A
  single-profile worker registers exactly the same keys, so jobs pinned before keep matching.
- Each further profile registers `magic_recipe@<profile>`, `magic_config@<profile>` and
  `magic_profile@<profile>`, plus its features under the same suffix. A job for that profile
  pins the namespaced keys, and its plugin attests them.
- The coordinator attests whichever recipe a job pinned. It compares the effective inputs
  that the profile declares (`MachineProfile.attested_inputs`).

Profiles are entries in `machine_profiles.PROFILES`. Each entry gives the profile's
capability, its template directory, a fingerprinted description, its attested inputs and its
launch-flag check. Adding a profile does not change any existing recipe version.

### 3D engine profile

`mesh3d-24gb-standard` is an engine profile (`kind='engine'`). It runs AISmith-3D's
`aismith-engine` CLI instead of a ComfyUI template, and serves
`mesh.trellis2.image_to_mesh.v1` and `mesh.blender.clean.v1`. Its recipe lives in
`engine_recipes.py`:

- the engine release `engine-v0.1.0` (commit `f7466926`), with the sources-lock and
  weights-lock digests at that commit;
- CUDA architecture 8.9 and Blender 4.2.x LTS;
- a 24,000 MB minimum;
- per capability, its attested inputs, deadline and weight sets:
  - mesh: `resolution` and `seed`, 1,800 s;
  - clean: `quality_tier` or `target_triangles`, 600 s.

The recipe version fingerprints that description, including the `qualified` flag.

**The profile is qualified, on a stated scope.** It was measured on one RTX 4090 on
2026-11-05 and 2026-11-06:
- the front view only, mesh resolution 1024, and the standard clean tier without view
  projection;
- mesh engine peak about 6.9 GB, wall time up to 3:27;
- clean engine peak about 3 GB, up to 49 s.

Mesh output is not deterministic: the same spec and seed gave a different mesh.

The 24 GB minimum, the deadlines and `mixed_mode=False` (the profile never shares a runtime
with H3) are decisions recorded in the recipe's provenance. A smaller card gets its own
profile after a run on such a card.

The `aismith-engine` plugin (`engine.py`) re-runs `inspect` before each job. It fails
closed if any of these has drifted:
- the engine commit or a lock digest;
- a file of any weight set the profile's stages load. Sets no stage loads, such as the
  phase-2 rig checkpoint, are not required;
- the GPU architecture or VRAM;
- the Blender series;
- offline mode.

Enrolment runs `inspect --full`.

The refusal paths for an unqualified engine profile stay in place, tested against a
test-only unqualified copy:
- enrolment and the plugin constructors;
- the coordinator's 422 `profile_not_qualified` on registration;
- `submit`, leasing and completion.

A coordinator that cannot import `machine_profiles` refuses every engine capability.

### Running the 3D engine worker

The engine runs on its own worker identity, never beside H3 or any other profile. On a
shared 4090, stop the H3 worker before you start the engine worker. Set
`MVS_WORKER_PROFILE=mesh3d-24gb-standard` alone, and the agent entrypoint
(`python -m outbound_gpu_worker_pool.agent_main`) starts an engine worker:
- startup runs `inspect --full` against the recipe, and any drift starts no agent;
- the worker registers only `mesh.trellis2.image_to_mesh.v1` and `mesh.blender.clean.v1`,
  with the engine's runtime versions, at concurrency 1;
- it refuses the engine profile listed with any other profile, any `OGWP_WORKER_PLUGINS`
  other than `aismith-engine`, and `OGWP_WORKER_CONCURRENCY` other than 1.

The plugin has two transports. The worker chooses one from the environment:

| Transport | Selected by | Use |
| --- | --- | --- |
| `http` | `OGWP_ENGINE_URL` set | The hardened worker. The plugin talks to an `aismith-engine serve` sidecar |
| `local` | `OGWP_ENGINE_URL` unset | Development and tests. The plugin runs the CLI as a subprocess in its own process group |

| Variable | Meaning |
| --- | --- |
| `OGWP_ENGINE_URL` | The sidecar runner, for example `http://127.0.0.1:8190`. Loopback or private addresses only, as for `OGWP_COMFY_URL` |
| `MVS_ENGINE_COMMAND` | The `local` engine command (default `aismith-engine`), for example `/opt/aismith/.venv/bin/aismith-engine` |
| `OGWP_WORKER_WORKSPACE` | The root for per-job directories. A job's engine work directory is `<root>/<job>/engine` |
| `MVS_ENGINE_INSPECT_TIMEOUT_SECONDS` | The bound on each per-job `inspect`, default 600. The job's own deadline also bounds it |
| `MVS_ENGINE_FULL_INSPECT_TIMEOUT_SECONDS` | The bound on the startup `inspect --full`, default 1800 |
| `MVS_ENGINE_KILL_GRACE_SECONDS` | `local` only: the time between SIGTERM and SIGKILL of the stage's process group, default 10 |

Stage timeouts are not configurable. Each stage runs for at most the smaller of its profile
deadline and the lease's.

The engine reads its runtime roots itself: `TRELLIS_REFINER_DIR`, `BLENDER_EXE`,
`U2NET_HOME`, `UNIRIG_DIR` and `AISMITH_DATA_DIR`. With `local`, the worker passes its
environment through unchanged and always sets `AISMITH_OFFLINE=1`, which the CLI also sets.
With `http`, the sidecar's own environment holds them.

```bash
OGWP_WORKER_COORDINATOR_URL=https://studio.example/pool \
OGWP_WORKER_ID=ilya-desktop-3d OGWP_WORKER_AUTH=static OGWP_WORKER_TOKEN=... \
MVS_WORKER_PROFILE=mesh3d-24gb-standard \
OGWP_ENGINE_URL=http://127.0.0.1:8190 \
python -m outbound_gpu_worker_pool.agent_main
```

**The hardened sidecar.** The `agent` container holds the worker credential and the
egress proxy, but no GPU. The `engine` container holds the GPU, the read-only model stores
and the engine runtimes, and has no credentials. It runs `aismith-engine serve` on
loopback. The agent runs with `network_mode: service:engine`, the same pattern ComfyUI
uses, so the two share only a loopback network.

Over `http`, each job runs `GET /inspect` and the same drift check as `local`, then:
1. `PUT` each input to `/jobs/<job>/inputs/<name>`.
2. `POST /jobs/<job>/run` with the deadline. The plugin also enforces the deadline itself.
3. Check the result exactly as for `local`: status, stage, schema and engine identity.
4. `GET` the output. Its GLB magic, size and sha256 are checked on the fetched bytes and
   against `X-Sha256` and `Content-Length`.
5. `DELETE /jobs/<job>`, always.

A lease cancellation or a deadline sends `POST /jobs/<job>/cancel` first. Enrolment runs
`GET /inspect?full=1`. The plugin never sends runner requests through the worker's egress
proxy.

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

## Security and upgrades

See [`SECURITY.md`](SECURITY.md) for the trust boundary, private reporting guidance, and
secrets-handling policy, and [`docs/security-upgrade.md`](docs/security-upgrade.md) before
upgrading an existing coordinator. This hardening release intentionally rejects unapproved
first-heartbeat enrollment and requires explicit administrator authorization for status routes.

CI scans every reachable revision and raw Git object with checksum-pinned Gitleaks on pull
requests, pushes to `main`, and a weekly schedule. It includes public branches, tags, and retained
PR heads; output contains finding metadata only. This does not replace GitHub push protection,
private reporting, or credential rotation after a genuine exposure.

## Design

[`docs/design.md`](docs/design.md) has the architecture, the trust boundaries, the
alternatives that were rejected, and the rollout plan.
[`docs/operations.md`](docs/operations.md) is the other half: installing, enrolling, running,
verifying, draining, upgrading, and rotating a worker machine.

Status: alpha, slice 1 of the design's rollout — durable capability leases, the authenticated
coordinator API, job-scoped asset grants with verified immutable outputs, and the deterministic
reference plugin and agent.

### Binding a machine recipe

Hosts may set `JobRequirements.worker_ids` and `runtime_versions` to bind a job
to an authorized worker and its declared configuration. These constraints are
checked atomically during every lease/re-lease in both stores, in addition to
tenant, capability and VRAM checks. Worker status includes declared runtime
versions and VRAM for host-side discovery.

`machine_profiles` provides an opt-in `h3-16gb-lightx8` recipe (8 steps,
512×512/four requested seconds, W4A8 + LightX2V 768p, Kitchen over Sage), and
`h3-24gb-standard` retains the standard graph. `flux2-klein-subject-12gb` serves
subject images on a 12GB card: the unchanged Klein v2 graph under its own capability
`image.flux2_klein.subject.12gb.v1`, fixed at 1024×1024, exactly 20 steps and at most three
references, with a 12,000 MB minimum. It refuses `--gpu-only` and `--highvram`. Its evidence
is an RTX 4090 with VRAM held so 11,914 MiB (and 10,900 MiB) stayed free. On that card,
0, 1 and 3 references gave output byte-identical to v2, peaking at 10,218 MiB. Two
references and a physical 12GB card have not been measured. Standard template loading
advertises neither the 16GB graph nor the 12GB subject graph. An explicit profile checks models, nodes and runtime
before advertising; its executor rechecks the runtime before each render.
