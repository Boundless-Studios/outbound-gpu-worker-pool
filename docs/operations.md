# Running a worker machine

How to put the agent on a Linux GPU box, prove the box exposes nothing new, and keep it
running. The first target is Ubuntu 24.04 (including WSL2), where ComfyUI is a native venv
install rather than a container.

Everything here runs as an unprivileged user. There is no root step, no Docker, and no port
to open: the agent dials out to the coordinator and nothing ever dials in.

## Prerequisites

- **Python 3.13 or newer.** Ubuntu 24.04 ships 3.12 as `python3`, so install 3.13 (for
  example from the deadsnakes PPA) and point the installer at it with
  `OGWP_PYTHON=python3.13`.
- **A systemd user manager.** `systemctl --user status` must answer. On WSL2 this needs
  `systemd=true` under `[boot]` in `/etc/wsl.conf` and a `wsl --shutdown`.
- **ComfyUI bound to loopback, with the H3 weights installed**, if this machine serves
  `comfy-workflow`. Start it with `--listen 127.0.0.1`; the plugin refuses any base URL that
  is not loopback or a private address, and `check-exposure.sh` fails the box if the runtime
  is bound anywhere else.
- **A per-machine credential**: a static token, or a service account dedicated to this
  machine for Google OIDC. Never share one across machines — revocation targets the identity,
  so a shared credential cannot be revoked for one box.
- `git`, and `iproute2` for `ss`.

## Install

```bash
git clone https://github.com/Boundless-Studios/outbound-gpu-worker-pool
cd outbound-gpu-worker-pool
OGWP_PYTHON=python3.13 ./deploy/agent/install.sh
```

It creates `~/.local/share/outbound-gpu-worker/` with a venv, a `workspace/` for per-job
scratch, a `templates/` copy of the packaged workflow templates for you to curate, and an
`agent.env` template at mode 0600. It installs the systemd **user** unit into
`~/.config/systemd/user/outbound-gpu-worker.service`. It is safe to rerun.

`OGWP_HOME` moves the install, but the unit's paths are literal, so a non-default home means
editing `EnvironmentFile`, `ExecStart`, and `ReadWritePaths` in the installed unit — the
installer says so when you use one.

## Enroll

Enrollment is a coordinator-side act. The machine only ever holds its own credential.

**Static token.** Generate the pair on the machine, keep the token here, hand the digest to
whoever runs the coordinator:

```bash
python3 -c "import secrets,hashlib; t=secrets.token_urlsafe(32); print(t, hashlib.sha256(t.encode()).hexdigest())"
```

The digest goes in the coordinator's `OGWP_WORKER_TOKENS` as `<worker-id>:<digest>`. The token
goes in this machine's `agent.env` as `OGWP_WORKER_TOKEN`. The coordinator never stores the
token itself.

**Google OIDC.** Create one service account for this machine and insert its registry row
before the first heartbeat:

```sql
INSERT INTO pool_workers (worker_id, identity_subject)
VALUES ('gpu-01', 'gpu-01@<project>.iam.gserviceaccount.com')
ON CONFLICT (worker_id) DO NOTHING;
```

Then set `OGWP_WORKER_AUTH=google_oidc`, `OGWP_WORKER_AUDIENCE=<audience>`, and
`GOOGLE_APPLICATION_CREDENTIALS=<key file>` in `agent.env`. Smoke-test it with
`gcloud auth print-identity-token --audiences=<audience>`.

## Run

Fill in `~/.local/share/outbound-gpu-worker/agent.env` — every variable is listed there,
commented, with what it means — and then:

```bash
systemctl --user enable --now outbound-gpu-worker
loginctl enable-linger "$USER"      # so the agent keeps running after you log out
systemctl --user status outbound-gpu-worker
```

The unit restarts the agent on failure after 10 seconds. It writes nowhere but its own home:
`ProtectSystem=strict` and `ProtectHome=read-only` with a single `ReadWritePaths`.

## Verify exposure

```bash
./deploy/agent/check-exposure.sh
```

It prints every listening TCP socket on the machine and the ComfyUI bind it found, then exits
non-zero if ComfyUI listens on anything but loopback, or if the agent itself holds a listening
socket. The agent must hold none: it is outbound only. Run it after the first start and after
any ComfyUI upgrade, since a ComfyUI launcher flag is the usual way a box accidentally starts
listening on `0.0.0.0`.

An external port scan from another host is the other half of the same check. This one tells
you what that scan will find, from inside the box.

## Drain and stop

```bash
systemctl --user stop outbound-gpu-worker
```

Stopping sends SIGTERM, which the agent treats as a drain, not a kill: the in-flight job runs
to completion, a final draining heartbeat tells the coordinator to grant this worker no new
lease, and the process exits 0. A lease granted just as the drain begins is released back to
the queue rather than started, so no job is lost. `TimeoutStopSec=1500` gives a long job the
room to finish; only past that does systemd escalate.

To stop taking work without stopping the process, the coordinator side sets
`WorkerStatus.DRAINING` for this worker.

## Upgrade

```bash
OGWP_REF=<sha> ./deploy/agent/install.sh
systemctl --user restart outbound-gpu-worker
```

Reinstalling touches only the venv: `agent.env` and any template you curated are left alone.
Pin a sha rather than tracking `main` on a machine you care about. The restart drains first,
so the upgrade waits for the in-flight job.

## Logs

```bash
journalctl --user -u outbound-gpu-worker -f
journalctl --user -u outbound-gpu-worker --since -1h
```

Records carry job ids, worker ids, byte counts, sha256 digests, outcomes, and durations. They
never carry a credential, a signed URL, a prompt, or asset bytes — a failure is reported by
exception type for exactly that reason, so a log line will not tell you *what* a third-party
library complained about, only that it did.

## When the coordinator is unreachable

Nothing to do. The agent logs `worker could not reach the coordinator`, backs off
exponentially up to a minute with jitter, and keeps retrying. It holds no queue and loses no
work: a lease it never got is a lease another worker takes, and a job it was running is
released back to the queue by lease expiry on the coordinator side. Do not restart the
service to "reconnect" — that only costs you the in-flight job's progress.

If it stays unreachable, check outbound HTTPS from the machine
(`curl -sS -o /dev/null -w '%{http_code}\n' <coordinator-url>/health`) before suspecting the
agent. A `403` on every route means the worker was revoked, not that the network is down.

## Rotate or revoke the credential

**Revoke** — takes effect on the worker's next request, and is independent of the credential:

```python
await service.set_worker_status("gpu-01", WorkerStatus.REVOKED)
```

The machine then gets `403` on every `/worker/v1` route even while holding a valid token. No
other worker is touched.

**Rotate a static token.** Generate a new token and digest, add the new digest to the
coordinator's `OGWP_WORKER_TOKENS` alongside the old one, put the new token in `agent.env`,
`systemctl --user restart outbound-gpu-worker`, confirm a heartbeat lands, then drop the old
digest. Overlapping the two is what keeps the rotation from needing a maintenance window.

**Rotate a Google service account key.** Write the new key file, point
`GOOGLE_APPLICATION_CREDENTIALS` at it, restart, confirm a heartbeat, then delete the old key
in the cloud project. The registry row does not change, because the identity subject — the
service account — did not.

If a credential may have leaked, revoke first and rotate second. Revocation does not depend on
the credential, so it works even while the leaked one is still valid.
