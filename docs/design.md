# Outbound GPU worker pool — architecture

## Objective

Let several products share a pool of trusted GPU machines — a few team-owned desktops, and
optionally hosted providers behind the same contract — **without exposing a desktop, a
ComfyUI instance, or a private network to the internet**. The control plane must stay cheap
when idle, support heterogeneous capabilities, and let hosted providers be added behind the
same worker and plugin contract.

## Decision

Build an **outbound-only, pull-based** pool around a durable Postgres job store and a
pluggable asset store.

- Do not expose ComfyUI or a desktop HTTP endpoint.
- Do not run a message broker on a desktop and do not make workers reachable from the
  control plane.
- Each worker agent initiates authenticated HTTPS requests to a narrow coordinator API,
  leases compatible work, downloads inputs through job-scoped signed URLs, executes a local
  plugin, uploads immutable outputs through create-once grants, and commits an attested
  result.
- The Postgres job record is the source of truth. Any existing push transport (for example
  Cloud Tasks waking a CPU worker) keeps working for non-pool jobs; pool jobs are pulled.

## System architecture

### 1. Control plane — the coordinator

A private HTTP surface, separate from any browser-facing API:

| Route | Purpose |
|---|---|
| `POST /worker/v1/heartbeat` | register or refresh a worker and its capabilities |
| `POST /worker/v1/lease` | lease one compatible job (`204` when nothing is claimable) |
| `POST /worker/v1/jobs/{job_id}/heartbeat` | renew the lease, report progress |
| `POST /worker/v1/jobs/{job_id}/complete` | commit an attested output manifest |
| `POST /worker/v1/jobs/{job_id}/fail` | terminal or retryable failure |
| `POST /worker/v1/jobs/{job_id}/release` | give the job back (drain, transient error) |
| `GET /worker/v1/capabilities/schema` | typed input schema per capability |

Requirements:

- Not invokable without worker authentication.
- Never accepts shell commands, workflow graphs, filesystem paths, URLs, or code. Request
  bodies are closed, bounded DTOs.
- Returns only a validated operation DTO and short-lived asset grants.
- Applies per-worker and global rate limits.
- Records an audit event for lease, heartbeat, release, completion, failure, rejection,
  authentication failure, and administrative status changes. Audit rows carry ids,
  counts, digests, and truncated reasons — never credentials, URLs, prompts, or bytes.
- Scales to zero when no worker is polling; cold-start latency is acceptable.

### 2. Durable capability queue

Each claimable job declares:

- capability id (for example `video.minimax.image_to_video.v1`,
  `comfy.workflow.<approved-id>.v1`, `trellis.image_to_3d.v1`) and contract version;
- input asset keys and one output key;
- an immutable request digest and an idempotency key;
- priority, attempt budget, and execution deadline;
- tenant or project ownership;
- cancellation state.

Each worker advertises a stable worker id, approved plugin ids and versions, GPU model and
VRAM, runtime versions, concurrency per capability, health, and an optional cost class
(`local`, `hosted`, `premium`).

Lease selection is atomic (`FOR UPDATE SKIP LOCKED`), capability-aware, and oldest-first
within priority. Expired leases become claimable again. A replaced claim token cannot
complete, heartbeat, fail, or release. Retry counts and deadlines live on the job, never in
a worker process.

### 3. Worker agent — outbound only

A small supervised process on each GPU machine:

- initiates outbound HTTPS only; opens no port; ComfyUI (if any) listens only on loopback or
  a private container network;
- holds no bucket credential and no database access;
- cannot enumerate jobs or assets outside its active lease;
- polls with jitter and exponential idle backoff; renews leases while executing;
- drains gracefully for maintenance; enforces configured concurrency;
- deletes temporary job data after completion or terminal failure;
- emits structured logs and metrics without prompts, tokens, signed URLs, or asset bytes.

Authentication: one dedicated least-privilege identity per machine. Prefer keyless
workload identity federation; if a long-lived credential is unavoidable, isolate it per
worker, rotate it, and make revocation independent of the credential. Never share a
credential across the pool.

### 4. Plugin contract

The agent loads approved plugins; the control plane never learns provider internals.

```
capabilities() -> CapabilityManifest
validate(LeaseGrant) -> ValidatedRequest      # terminal gate, runs before any download
execute(ExecutionContext, ValidatedRequest) -> PluginOutput
cancel(job_id) -> bool
health() -> PluginHealth
```

`ExecutionContext` contains only job and lease identifiers, the execution deadline, the
already-downloaded input paths, a temporary workspace, a cancellation signal, and a
progress callback. `OutputManifest` contains the output key, content type, byte length,
sha256, idempotency key, request digest, plugin id and version, model id and version, seed,
`publication_mode=immutable_create_once`, and optional safe diagnostics.

Reference plugins: a deterministic echo (ships here, used for canaries), an approved-workflow
ComfyUI plugin mapping a capability id to a versioned local workflow template with a typed
input allowlist, and a hosted-provider plugin that runs without a local GPU.

### 5. Asset boundary

Canonical assets stay in the product's bucket. For remote execution the coordinator:

- resolves canonical keys and issues short-lived signed `GET` URLs for exact input objects;
- issues create-once upload grants for the exact output key (`ifGenerationMatch=0`);
- verifies, before marking a job complete: output key, idempotency key, request digest,
  publication attestation, object existence, byte length, and sha256;
- returns the original immutable output on a replay of the same idempotency key and
  terminally rejects a reused key with a different digest.

Grants expire near the lease deadline and cannot list the bucket. The plugin never receives a
general storage credential.

### 6. Product integration

1. The product API authorizes the caller and writes the durable job.
2. The UI receives the job id and polls or subscribes to status.
3. A compatible worker leases the job, executes, uploads, and commits with its lease token.
4. The product publishes the completed asset into its own model.

No product request is held open for the duration of generation.

## Trust boundaries

| Boundary | Rules |
|---|---|
| Internet → product | authenticated; tenant authorization before job creation; users never choose worker ids, bucket paths, provider endpoints, or raw workflow graphs |
| Coordinator → worker | the coordinator never calls into a machine; every poll and commit is authenticated; lease tokens are job-specific, expiring, and bound to the worker |
| Worker → local runtime | ComfyUI on loopback or isolated network; only approved versioned templates; no custom nodes installed by a job; containers unprivileged with no host Docker socket; egress denied by default where practical |
| Worker → assets | exact-object, short-lived grants; no list permission; immutable publication; bounded and cleaned scratch |

## Alternatives considered

- **Message broker shared between the control plane and desktops** — rejected: an always-on
  broker, public/private connectivity, credentials, upgrades, monitoring, and a second
  durability model. Reconsider only if measured throughput exceeds the Postgres lease model.
- **Public endpoint or tunnel per GPU desktop** — rejected: even with mesh VPNs or mTLS a
  push endpoint increases exposed surface and couples availability to individual machines.
  Tunnels remain an operator-only maintenance tool, never the job transport.
- **Hosted provider only** — useful as a plugin and overflow path, not the unified solution;
  it cannot preserve custom local workflows and may cost materially more.

## Failure semantics

Retryable: worker disappears or lease expires; transient provider or transport failure;
temporary storage failure; worker draining before execution; provider rate limit. The worker
releases the job; the coordinator requeues it until the attempt budget is spent, then fails
it with `retryable=true` so an operator can resubmit.

Terminal: invalid or unsupported capability or contract version; digest or idempotency
conflict; authorization failure; invalid inputs; unsafe workflow input; output key,
checksum, or attestation mismatch.

## Observability and operations

Metrics: queue depth and oldest-job age by capability and priority; active, idle, draining,
stale, and offline workers; lease acquisition, renewal, expiry, reassignment; execution
duration and failure class by plugin, model, and worker; transfer duration and bytes;
retries and idempotent replays; hosted-provider cost; rejected authentication attempts.

Operator actions: drain or revoke one worker; disable one capability; cancel one job or a
tenant's queued jobs; globally disable execution; inspect a redacted audit trail; rotate
worker credentials; distinguish retryable infrastructure failure from terminal request
failure.

## Rollout

1. **Queue and identity foundations** — capability fields, worker registry, lease and
   heartbeat endpoints, audit log, per-worker identities, a fake worker with a deterministic
   plugin. *Shipped in this repository.*
2. **Safe worker agent and ComfyUI plugin** — packaged supervised agent, loopback-only
   ComfyUI, approved-workflow registry with typed inputs, first machine enrolled.
3. **Asset-grant hardening** — scoped grants through the asset store, immutable publication,
   proven idempotent retry.
4. **Multi-worker pool and hosted provider** — enrollment of remaining machines, routing,
   fairness, health, drain controls, hosted overflow plugin.
5. **Production non-destructive canary** — deployed with generation disabled; health,
   lease, signed-asset, and deterministic probes; dashboards and alerts.
6. **Controlled enablement** — one allowlisted operation for admins, real generation into
   an isolated project, then expansion by capability after a soak.

## Non-goals

Exposing ComfyUI publicly; general remote shell or arbitrary workflow execution; sharing a
desktop filesystem; moving canonical job state into a broker; enabling generation in the
first infrastructure change; byte-identical output across different GPUs or providers;
untrusted third-party worker operators; replacing object storage as canonical asset
storage.
