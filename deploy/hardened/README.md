# Isolated GPU deployment

This profile runs the agent and ComfyUI in separate non-root containers. Only
the trusted host supervisor can access Docker. Both workloads have read-only
code, no personal home or control sockets, zero capabilities, seccomp, bounded
memory/CPU/PIDs/scratch, and an isolated internal bridge with no host gateway.
ComfyUI binds loopback in that namespace; the agent shares its network namespace,
not its filesystem or process namespace. There are no published ports.

The agent alone receives a worker-only credential. ComfyUI gets only a curated
read-only model directory; custom nodes and external API nodes are disabled.
All enabled templates must work with the pinned built-in node set. Only approved
safe model files belong in that directory; resolve symlinks into a curated tree
instead of mounting broad model-download caches or personal storage.

An HTTPS CONNECT relay accepts only exact configured hostnames on port 443,
rejects nonpublic DNS answers and connects to the validated address. Workloads
have no default external route or external DNS resolver. Set the allowlist to
the coordinator, token issuer and exact asset hosts; do not use wildcards.
This bounds network access, not individual URLs or data sent to approved hosts.

GPU JIT compilation needs a C compiler/Python headers in the immutable runtime
image and an executable cache. Only the bounded 512 MiB `/jit` tmpfs is
executable; media scratch stays `noexec`. Both disappear when the container is
recreated. This does not grant host code-write or privileged execution access.

The isolated agent reads bounded loopback ComfyUI memory telemetry through
`OGWP_WORKER_GPU_TELEMETRY=comfy`. It does not need a GPU device or `nvidia-smi`.
Utilization is unavailable through that endpoint and is reported as unknown,
not zero. The default sampler for other deployments remains `nvidia-smi`.

## Installation and verification

Requires Docker Engine 28+ (isolated bridge gateway mode), Compose v2, Python 3
on the host, and a tested WSL GPU driver. `wsl.yaml` exposes only `/dev/dxg` and
read-only WSL driver libraries. Other GPU platforms require a separately reviewed
device profile; the supplied launcher fails closed on unsupported hosts.

1. Build and review two immutable images. The agent image needs the installed
   `agent,comfy,google-auth` extras, this directory's `egress_proxy.py` at
   `/opt/ogwp/egress_proxy.py`, and the approved templates at
   `/opt/ogwp/templates`. ComfyUI and its Python environment belong at
   `/opt/comfy` and `/opt/comfy/venv`. Normalize virtualenv interpreter symlinks
   to paths inside the image. Never copy host credentials, Git history or a
   personal home into either build context. Record code/dependency/model hashes.
2. Place this directory at `~/.local/share/ogwp-hardened/hardened`. Keep its
   parent private (0700). Create `profile.env` in the parent with required
   `OGWP_AGENT_IMAGE` and `OGWP_COMFY_IMAGE` full immutable digests,
   `OGWP_EGRESS_HOSTS`, `OGWP_MODELS_DIR`, `OGWP_AGENT_ENV`, and
   `OGWP_WORKER_CREDENTIAL`. Set memory/CPU budgets for the machine, up to the reviewed ceilings of
   28 GiB/8 CPUs for ComfyUI, 1 GiB/2 CPUs for the agent, and 128 MiB/0.5 CPU for the relay. No production
   values belong in this repository. Place `model-manifest.json` beside the profile,
   containing an array of `{file, bytes, sha256}` records for every curated model.
   Startup hashes the exact tree before launching ComfyUI; missing, extra, changed
   or symlinked files fail closed. This checks operator-approved bytes, not model
   provenance, and trusts the host administrator against later modification. Docker-local `sha256:<64 hex>` image IDs
   are supported; registry tags alone are rejected.
3. The private agent environment contains the coordinator URL, worker identity,
   auth mode/audience and approved plugin names. The profile forces concurrency
   one, scratch/template locations, proxy configuration, and an empty startup
   command. Never put cloud admin, database or deployment credentials there.
   For the supplied Google OIDC configuration, mount a dedicated service-account
   key at `/run/credentials/worker.json`. The non-root container must be able to
   read the file: a read-only copy inside a 0700 host directory works without
   making its parent accessible to other host users. Keep the original private.
   Recreate the agent when rotating an atomically replaced bind-mounted key.
4. Drain and disable the old host agent and its ability to start host ComfyUI.
   Do not run old and new agents with the same identity simultaneously. Keep
   unrelated desktop services outside this deployment.
5. Install `ogwp-hardened.service` into `~/.config/systemd/user`, reload the user
   manager and enable/start it. Enable user lingering through the host's normal
   administration path if not already configured. The supervisor starts the
   runtime/proxy, verifies configuration before starting the agent, runs active
   denial probes, and checks for drift every 30 seconds. Failure drains/stops
   workloads; restarts recreate containers from the approved configuration.
   Docker itself has no automatic restart policy that bypasses this gate.
6. Run a synthetic image/video generation and the plugin's attestation/cancel/
   lease tests. Inspect `verify.py` output and run `probe.py` inside both
   containers with `OGWP_PROBE_ROLE=agent` or `comfy`. The probes check effective
   UID, seccomp, capabilities, forbidden filesystem writes, network denial,
   external DNS denial, and blocked proxy destinations without printing secrets.

Stop with `systemctl --user stop ogwp-hardened`. This drains the agent before
stopping ComfyUI. Do not auto-fallback to the old unsandboxed services if the
gate fails. Investigate the failure and restore only a verified profile.

## What this boundary does not promise

The host/Docker administrator is trusted; the supervisor deliberately controls
Docker, and its files must never be writable from workloads. Container isolation
shares the host kernel and GPU driver. It cannot guarantee safety against a
kernel/driver vulnerability, and ordinary consumer GPUs do not provide a hard
per-container VRAM quota. Concurrency and memory limits reduce resource pressure;
use a dedicated GPU host/VM for stronger separation from personal workloads.
This profile does not certify host firewall/VPN/Windows settings or unrelated
listeners. It makes no new service reachable from those networks. Audit other
machines independently, keep drivers/engine patched and never attach public CI
to a GPU or desktop host.

References: [Docker isolation and port mapping](https://docs.docker.com/engine/network/port-publishing/),
[Docker security](https://docs.docker.com/engine/security/).
