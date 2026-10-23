#!/usr/bin/env bash
# Fail-closed teardown, shared by run.sh's exit trap and systemd's ExecStopPost.
# Agent SIGTERM stops leasing, completes or releases its current lease and drains;
# the runtime and relay stop only after it.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${OGWP_PROFILE_FILE:-$HERE/../profile.env}"
compose=(docker compose --env-file "$PROFILE" -f "$HERE/compose.yaml")
if [[ -e /dev/dxg ]]; then
  compose+=(-f "$HERE/wsl.yaml")
fi
"${compose[@]}" stop --timeout 1500 agent || true
"${compose[@]}" stop --timeout 30 comfy egress || true
