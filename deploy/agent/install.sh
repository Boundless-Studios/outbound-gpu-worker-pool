#!/usr/bin/env bash
# Install or upgrade the outbound worker agent for the user running this script.
#
# No root, no Docker, no listener: everything lands under $OGWP_HOME and the
# agent runs as a systemd *user* service. Rerunning is the upgrade path
# (`OGWP_REF=<sha> ./install.sh && systemctl --user restart outbound-gpu-worker`);
# it never overwrites the env file or a template the operator has curated.
#
#   OGWP_HOME    where the agent lives  (default ~/.local/share/outbound-gpu-worker)
#   OGWP_REF     git ref to install     (default main)
#   OGWP_PYTHON  interpreter to build the venv with, must be 3.13+ (default python3)
set -euo pipefail

OGWP_HOME="${OGWP_HOME:-$HOME/.local/share/outbound-gpu-worker}"
OGWP_REF="${OGWP_REF:-main}"
OGWP_PYTHON="${OGWP_PYTHON:-python3}"

REPOSITORY="https://github.com/Boundless-Studios/outbound-gpu-worker-pool"
REQUIREMENT="outbound-gpu-worker-pool[agent,comfy,google-auth] @ git+${REPOSITORY}@${OGWP_REF}"
SOURCE_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME="outbound-gpu-worker.service"
UNIT_DIRECTORY="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
DEFAULT_HOME="$HOME/.local/share/outbound-gpu-worker"
VENV="$OGWP_HOME/venv"
ENVIRONMENT_FILE="$OGWP_HOME/agent.env"
TEMPLATES="$OGWP_HOME/templates"
WORKSPACE="$OGWP_HOME/workspace"

# The directory of `*.template.json` files to seed $TEMPLATES from: the
# installed package if it is importable, otherwise this checkout.
packaged_templates() {
    local directory
    if directory="$("$VENV/bin/python" -c \
        'import outbound_gpu_worker_pool.comfy as c; print(c.PACKAGED_TEMPLATES_DIRECTORY)' \
        2>/dev/null)"; then
        printf '%s' "$directory"
        return 0
    fi
    directory="$SOURCE_DIRECTORY/../../src/outbound_gpu_worker_pool/templates"
    if [ -d "$directory" ]; then
        printf '%s' "$directory"
        return 0
    fi
    return 1
}

mkdir -p "$OGWP_HOME" "$WORKSPACE" "$TEMPLATES" "$UNIT_DIRECTORY"

if [ ! -d "$VENV" ]; then
    "$OGWP_PYTHON" -m venv "$VENV"
fi

# Prefer the venv's own pip, and install nothing outside the venv.
PATH="$VENV/bin:$PATH"
export PATH VIRTUAL_ENV="$VENV"
pip install --upgrade "$REQUIREMENT"

if templates="$(packaged_templates)"; then
    for template in "$templates"/*.template.json; do
        [ -e "$template" ] || continue
        # A rerun must never undo the operator's curation of this directory.
        destination="$TEMPLATES/$(basename "$template")"
        [ -e "$destination" ] || cp "$template" "$destination"
    done
else
    echo "note: no packaged templates found to seed $TEMPLATES" >&2
fi

if [ ! -f "$ENVIRONMENT_FILE" ]; then
    # It carries the worker token, so it is never readable by anyone else.
    (
        umask 077
        cat >"$ENVIRONMENT_FILE" <<ENVIRONMENT
# Environment for the outbound GPU worker agent.
#
# systemd reads this file literally: one KEY=value per line, no shell
# expansion, no command substitution, no \`export\`. Write absolute paths.
# Uncomment and fill what this machine needs, then:
#   systemctl --user restart outbound-gpu-worker

# --- the coordinator, and who this machine is --------------------------------

# Required. The only host the agent ever talks to, over outbound HTTPS.
#OGWP_WORKER_COORDINATOR_URL=https://coordinator.example

# Required. Must equal the worker_id enrolled with the coordinator.
#OGWP_WORKER_ID=gpu-01

# How this machine's credential is produced: static (default) or google_oidc.
#OGWP_WORKER_AUTH=static

# Required for static auth: the token itself. The coordinator stores only its
# sha256 digest, so this file is the only copy — treat it as a secret.
#OGWP_WORKER_TOKEN=

# Required for google_oidc: the audience a fresh identity token is minted for.
# One service account per machine, never shared across the pool.
#OGWP_WORKER_AUDIENCE=

# The service account key file google_oidc mints from, when this machine has no
# attached identity. Keep it 0600 and owned by this user.
#GOOGLE_APPLICATION_CREDENTIALS=

# --- what this machine is approved to run ------------------------------------

# Comma separated; default deterministic-echo. Use comfy-workflow on a GPU box.
#OGWP_WORKER_PLUGINS=comfy-workflow

# Leases advertised per capability. Keep at 1 unless the GPU can really overlap.
#OGWP_WORKER_CONCURRENCY=1

# Where per-job workspaces are created and deleted. Left unset, the agent uses a
# directory under the unit's private /tmp, which is also fine.
#OGWP_WORKER_WORKSPACE=$WORKSPACE

# The scratch bound. One granted input larger than this aborts the download and
# releases the job rather than filling the disk. Default 2147483648 (2 GiB).
#OGWP_WORKER_MAX_INPUT_BYTES=2147483648

# Advertised to the registry; free text and an integer, for operator reporting.
#OGWP_WORKER_GPU_MODEL=rtx-4090
#OGWP_WORKER_VRAM_MB=24576

# --- the local ComfyUI (comfy-workflow only) ---------------------------------

# Must be loopback or a private address; the plugin refuses anything else, so an
# approved template cannot be pointed at a runtime you do not own.
#OGWP_COMFY_URL=http://127.0.0.1:8188

# The approved workflow templates this machine may run. Left unset, the packaged
# template directory inside the venv is used instead of your curated copy.
#OGWP_COMFY_TEMPLATES_DIR=$TEMPLATES
ENVIRONMENT
    )
    chmod 600 "$ENVIRONMENT_FILE"
fi

cp "$SOURCE_DIRECTORY/$UNIT_NAME" "$UNIT_DIRECTORY/$UNIT_NAME"

if command -v systemctl >/dev/null 2>&1 && systemctl --user daemon-reload 2>/dev/null; then
    echo "reloaded the systemd user manager"
fi

if [ "$OGWP_HOME" != "$DEFAULT_HOME" ]; then
    echo
    echo "warning: OGWP_HOME is $OGWP_HOME but $UNIT_NAME points at $DEFAULT_HOME."
    echo "         Edit EnvironmentFile, ExecStart and ReadWritePaths in"
    echo "         $UNIT_DIRECTORY/$UNIT_NAME before starting the service."
fi

cat <<NEXT

Installed the agent into $OGWP_HOME.

Next:
  1. Fill in $ENVIRONMENT_FILE (coordinator URL, worker id, credential).
  2. Curate $TEMPLATES if this machine runs comfy-workflow.
  3. systemctl --user enable --now outbound-gpu-worker
  4. loginctl enable-linger $(id -un)      # so the agent survives logout
  5. deploy/agent/check-exposure.sh        # prove the box exposes nothing new

Logs: journalctl --user -u outbound-gpu-worker -f
NEXT
