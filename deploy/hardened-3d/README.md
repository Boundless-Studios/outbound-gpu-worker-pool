# Isolated 3D engine deployment

This profile runs the `mesh3d-24gb-standard` worker the way the H3 hardened profile
(`ogwp-hardened`) runs ComfyUI, with the AISmith-3D engine sidecar in ComfyUI's place.
It is a separate Compose project, `ogwp-hardened-3d`, with three non-root containers:

| Service | Holds | Network |
| --- | --- | --- |
| `engine` | the GPU (`/dev/dxg` plus read-only WSL driver libraries), the read-only model stores and the baked-in engine runtimes. It runs `aismith-engine serve --host 127.0.0.1 --port 8190` | one internal, gateway-isolated network; DNS `127.0.0.1` |
| `agent` | the 3D worker's credential and nothing else. It has no GPU, no models and no runtimes | the engine's network namespace only (`network_mode: service:engine`) |
| `egress` | the HTTPS CONNECT relay, which allows only the exact hosts in the allowlist | the workload network and one outbound network |

The agent reaches the runner at `http://127.0.0.1:8190` (`OGWP_ENGINE_URL`). The plugin
never sends runner requests through the proxy. The runner has no authentication, so it
listens on loopback inside the shared namespace. The probe checks that it does not answer
on the workload address. The two containers share no filesystem, process namespace or
credentials. The engine has no credential, no proxy setting and no route out. Hugging Face,
transformers and the engine all run offline.

Each container runs as UID 65532, with a read-only root filesystem, every capability dropped
and `no-new-privileges`. Each has the default seccomp profile, no published ports and a
private cgroup namespace. PIDs, memory and CPU are bounded, swap is disabled, and the only
writable space is bounded tmpfs. Docker has no restart policy. A systemd user unit runs
`run.sh`, the gated supervisor, which recreates the stack from the approved profile and
verifies it before the agent starts. It keeps verifying every 30 seconds after that.

`verify.py` enforces the H3 gate's rules for this layout:
- the GPU device and the model mounts are only in `engine`, and the credential is only in `agent`;
- the agent shares only the engine's network namespace;
- `egress` bridges exactly the workload network and one outbound network;
- each service runs its approved image digest;
- the engine runs the reviewed `serve` command line, its `/health` healthcheck passes, its
  offline environment is set and its engine commit is the approved one;
- the agent runs the engine profile alone: one plugin (`aismith-engine`), concurrency 1, a
  loopback runner URL and no local engine command.

`probe.py` checks the same properties from inside each container.

### Writable paths in the engine

| Path | Size | What it holds |
| --- | --- | --- |
| `/scratch` | 4 GiB, noexec | The runner's work root (`/scratch/engine`) and ComfyUI's input, output and temp directories, which the image symlinks into scratch. Also the engine's data directory and `HOME`. The runner takes images up to 20 MiB and meshes up to 256 MiB per input |
| `/jit` | 1 GiB, the only executable tmpfs | The Triton, torch and CUDA JIT caches and `TMPDIR`. Drop it if a smoke run shows it is unused |
| `/opt/trellis2-refiner/models` | 16 MiB | The engine writes its weight-hash cache (`.weights-verified.json`) at the root of the TRELLIS model directory. A read-only mount there would fail `inspect --full` at enrolment. So this directory is a tmpfs, and the three weight-lock subtrees are bound read-only beneath it: `visualbruno`, `microsoft/TRELLIS-image-large` and `facebook` |
| `.../models/microsoft/TRELLIS.2-4B` | 1 MiB | ComfyUI-Trellis2 copies a helper config here on every load. It is not a model |

### Model verification

The supervisor hashes the mounted stores before the engine starts (`verify.py
--models-only`). It checks them against `model-manifest-3d.json`, beside the profile. The
manifest lists every file of the five weight sets the mesh stage loads, each with its store,
path, size and sha256. The trellis store covers only the three subtrees above. The rembg store
is the directory mounted at `U2NET_HOME/models`. The gate fails closed on a missing, extra,
resized, changed or symlinked file, on a special file, and on any format other than
`.safetensors`, `.json` or `.onnx`. Pickle checkpoints never mount.

The engine's own `inspect --full` still runs at enrolment, and the plugin checks the cached
`inspect` before every job. Neither replaces the host gate. The engine container is the party
under test, so its self-report cannot also be its gate. Generate the manifest from the reviewed
lock at the engine tag, so that the two checks agree by construction:

```bash
python3 verify.py --manifest-from-lock <engine-export>/vendor/weights.lock.json > ../model-manifest-3d.json
```

The host administrator is trusted against later modification. The mounts are read-only to the
container.

### Resource bounds

| Container | Default | Ceiling `verify.py` enforces |
| --- | --- | --- |
| `engine` | 24 GiB, 8 CPUs, 1,024 PIDs, 1 GiB shm | At most 28 GiB and at least 16 GiB, 8 CPUs, 1,024 PIDs, 2 GiB shm |
| `agent` | 2 GiB, 2 CPUs, 256 PIDs | The same as the default |
| `egress` | 128 MiB, 0.5 CPU, 256 PIDs | The same as the default |

The engine's figures come from qualification. Mesh RSS peaked at about 9 GB. Up to 4 GiB of
scratch, 1 GiB of JIT and 1 GiB of shm are charged to the same memory cgroup, which makes
roughly 15 GiB at the worst point. The 16 GiB floor is there to stop a configuration that
would OOM mid-job. The PID ceiling is higher than H3's because the controller counts threads,
and torch, onnxruntime and Blender each start a thread per visible core. Set
`OGWP_ENGINE_MEMORY` and `OGWP_ENGINE_CPUS` in the profile, within those ceilings.

The agent's download bound is `OGWP_WORKER_MAX_INPUT_BYTES=268435456`, the runner's own mesh
limit. Its 1 GiB scratch holds one lease's inputs and the fetched GLB.

The agent has no GPU and no `nvidia-smi`, so its heartbeat GPU telemetry is empty. That data
is unknown, not zero.

## Time-sharing the 4090 with H3

The card runs either `ogwp-hardened` (H3) or `ogwp-hardened-3d`, never both. The recipe's
`mixed_mode=False` records that decision. Switch only with `switch.sh`:

```bash
switch.sh --to 3d            # dry run: runs the gate and prints the plan
switch.sh --to 3d --apply    # drain H3, start the 3D worker
switch.sh --to h3 --apply    # drain the 3D worker, start H3
```

`switch_gate.py` reads only box-local evidence: systemd, `docker ps`, `docker exec` into the
source's own runtime, and `nvidia-smi`. It refuses unless all of these hold:
- the target is not running: its unit is inactive and none of its containers run;
- every one of 5 GPU samples is at most 5% utilization;
- the source is quiet:
  - from H3: the ComfyUI queue is empty and its history shows no event for 10 minutes;
  - from 3D: the engine runner reports `busy: false` on 3 samples.

Missing or malformed evidence also refuses. If the source runtime is broken, stop its unit by
hand first. A stopped source needs only the idle-GPU check.

With `--apply`, the switch takes these steps:
1. `systemctl --user disable --now <source>`. The supervisor drains its agent: 25 minutes for
   H3, 35 for the 3D worker.
2. It confirms that no source container is running. If one is, it does not start the target.
3. `systemctl --user enable --now <target>`.
4. It waits up to 30 minutes for the target's supervisor to print its verified-isolation line.
5. Post-checks: exactly one unit is active and enabled, with its three containers running.
   The target's `verify.py` must pass.

A failure leaves the GPU idle for investigation. The switch never falls back to the source.

The 3D launcher refuses to start (exit 3, not retried) while H3 is active. It drains and stops
if H3 appears beside it. Its unit is ordered after `ogwp-hardened.service`, so an H3 unit
still enabled at boot wins, and the 3D unit refuses. **Operator follow-up:** add the mirror
check to H3's `run.sh`, which is not in this repository yet.

## Enrolment

The 3D worker is a second worker identity: for example `ilya-desktop-3d`, beside H3's own.
It needs:
- its own credential: a dedicated service-account key, or a static token in the private agent
  env file;
- its own `OGWP_WORKER_ID`;
- the coordinator's registration for `mesh3d-24gb-standard`.

Never reuse H3's credential or ID: the coordinator would see two agents with one identity
when the card switches.

The private env file (`OGWP_AGENT_ENV`) holds only these: the coordinator URL, the worker ID,
the auth mode and audience, and the tenant. Compose overrides the profile, plugin, engine URL,
concurrency, proxy, workspace and input bound. At startup the agent runs `GET
/inspect?full=1`. It refuses to register on any drift from the recipe.

## What a deploy needs

1. **The engine image** (`Dockerfile.engine`), built from a prepared, reviewed context:
   - `engine/`: an AISmith-3D `git archive` at tag `engine-v0.2.0`, which adds `serve`.
     Pass its full commit as `--build-arg ENGINE_COMMIT=...`. `inspect` reports that commit
     because the image has no `.git`.
   - `engine-venv/`: the engine's frozen Linux virtualenv.
   - `trellis2-refiner/`: the ComfyUI 0.33.0 tree with `custom_nodes/ComfyUI-Trellis2`,
     `.venv-cu128` (torch 2.8 cu128) and `native-contract.json`, without `models/`.
   - `blender/`: Blender 4.2.23 LTS for linux-x64, checked against its published sha256.
   - A base image by digest that provides Python 3.12, such as Ubuntu 24.04.

   The build copies each venv's interpreter into the venv. This is needed because the engine
   resolves symlinks when it picks the refiner interpreter. The build fails if it finds Git
   metadata, host paths in `.pth` files, a bundled model directory, or the wrong engine,
   ComfyUI, Blender or torch version.
2. **The agent image** (`Dockerfile.agent`): H3's pattern, from an export of this repository
   at a commit with the engine HTTP transport (6f43b02 or later). The engine plugin is already
   in the package, so the only differences from H3 are these:
   - the extras are `agent,google-auth`;
   - it has no templates;
   - the relay comes from this directory. `egress_proxy.py` is a byte-identical copy of the H3
     relay.
3. **`profile-3d.env`**, beside this directory (mode 0600, parent 0700), holding:
   - `OGWP_AGENT_IMAGE` and `OGWP_ENGINE_IMAGE`: full immutable digests;
   - `OGWP_ENGINE_COMMIT`;
   - `OGWP_EGRESS_HOSTS`: the coordinator, the token issuer and the exact asset hosts, without
     wildcards;
   - `OGWP_TRELLIS_MODELS_DIR`: the curated store containing `visualbruno/`, `microsoft/` and
     `facebook/`;
   - `OGWP_REMBG_MODELS_DIR`: the directory holding `birefnet-general/` and
     `birefnet-general-lite/`;
   - `OGWP_AGENT_ENV` and `OGWP_WORKER_CREDENTIAL`;
   - optionally `OGWP_ENGINE_MEMORY` and `OGWP_ENGINE_CPUS`.

   Curate the stores. Copy the exact lock files out of a Hugging Face download, with no
   `.cache/` metadata and no symlinks. Copying them to ext4 also avoids rehashing about 10 GB
   over `/mnt/k` at every start.
4. **`model-manifest-3d.json`**, beside the profile, from `--manifest-from-lock`.
5. Install this directory at `~/.local/share/ogwp-hardened-3d/hardened-3d` and the unit in
   `~/.config/systemd/user`. Keep it disabled until the first `switch.sh --to 3d --apply`.

### Needs operator input before the first deploy

- **The recipe pin.** `engine_recipes.MESH3D_RECIPE` pins `engine-v0.1.0` (`f7466926`), but
  `serve` first ships in 0.2.0. Moving the pins means the tag, the commit and both lock digests
  from `inspect --full`. That changes `recipe_version`, which is a recorded decision, not part
  of this change. Until then the agent refuses the 0.2.0 engine as drift.
- **The Python ABI.** Both venvs must be Python 3.12 builds, to match the image's interpreter.
  Rebuild them in a 3.12 container if the box's venvs use another minor version.
- **CuMesh, o-voxel and flash-attn.** Copy them from the qualifying venv, where they were built
  for sm_89. A rebuild without a GPU at build time needs `TORCH_CUDA_ARCH_LIST=8.9` and
  `FORCE_CUDA=1`. It also needs the CUDA 12.8 toolkit in a separate build stage, not in the
  runtime image. `native-contract.json` must still say `cuda_architecture: 8.9`.
- **Blender.** Its X and GL client libraries are installed for headless use. Confirm a
  `clean` smoke run in the image.
- **Smoke checks.** Before enrolment, confirm the nested tmpfs and read-only mounts on the
  box with `probe.py` (`weight_cache_writable`, `models_write_denied`). Confirm too that a
  mesh and a clean job complete. Writes outside the paths above surface as read-only errors.
- **H3's launcher** needs the mirror of the never-both check.

## What this boundary does not promise

The same limits as the H3 profile apply. The host and Docker administrator is trusted. Containers
share the host kernel and GPU driver. Consumer GPUs have no per-container VRAM quota. This
profile does not certify host firewall, VPN or Windows settings.
