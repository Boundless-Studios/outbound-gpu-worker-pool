#!/usr/bin/env bash
# Trusted host launcher for the 3D engine worker. The Docker socket is NEVER
# exposed to a workload. The GPU is time-shared with H3 (`ogwp-hardened`): this
# launcher refuses to start, and stops, whenever H3 is running. Switch with
# switch.sh, which drains one before starting the other.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${OGWP_PROFILE_FILE:-$HERE/../profile-3d.env}"
OTHER_PROJECT=ogwp-hardened
OTHER_UNIT=ogwp-hardened.service
compose=(docker compose --env-file "$PROFILE" -f "$HERE/compose.yaml")
if [[ -e /dev/dxg ]]; then
  compose+=(-f "$HERE/wsl.yaml")
else
  echo "This profile currently verifies WSL GPU isolation; supply a reviewed native GPU profile before using another host." >&2
  exit 2
fi

other_running() {
  systemctl --user is-active --quiet "$OTHER_UNIT" && return 0
  [[ -n "$(docker ps -q --filter "label=com.docker.compose.project=$OTHER_PROJECT" --filter status=running)" ]]
}

# Never both: refuse before touching the GPU (exit 3 is not retried by the unit).
if other_running; then
  echo "H3 ($OTHER_PROJECT) is running; the GPU runs one worker at a time. Use switch.sh --to 3d." >&2
  exit 3
fi

cleanup() {
  # Agent SIGTERM stops leasing, completes/releases its current lease and drains.
  # A mesh lease may run for its full 30-minute deadline.
  "${compose[@]}" stop --timeout 2100 agent || true
  "${compose[@]}" stop --timeout 30 engine egress || true
}
trap cleanup EXIT
trap 'exit 0' TERM INT

# Verify model bytes before the engine can load them. The host administrator remains trusted.
python3 "$HERE/verify.py" --profile "$PROFILE" --models-only

# Recreate from the approved profile; never resume drifted container settings.
"${compose[@]}" up -d --force-recreate --wait --wait-timeout 300 egress engine
"${compose[@]}" up --no-start --force-recreate --no-deps agent
python3 "$HERE/verify.py" --profile "$PROFILE" --allow-stopped-agent
engine_id="$("${compose[@]}" ps -q engine)"
docker exec -i -e OGWP_PROBE_ROLE=engine "$engine_id" /opt/aismith/.venv/bin/python - < "$HERE/probe.py"
"${compose[@]}" start agent
python3 "$HERE/verify.py" --profile "$PROFILE"
agent_id="$("${compose[@]}" ps -q agent)"
docker exec -i -e OGWP_PROBE_ROLE=agent "$agent_id" python - < "$HERE/probe.py"
echo "GPU isolation verified; 3D worker is running under the gated supervisor."
while sleep 30; do
  result="$(python3 "$HERE/verify.py" --profile "$PROFILE")" || { printf '%s\n' "$result" >&2; exit 1; }
  if other_running; then
    echo "H3 ($OTHER_PROJECT) started beside the 3D worker; draining." >&2
    exit 1
  fi
done
