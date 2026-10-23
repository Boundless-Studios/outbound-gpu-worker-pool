#!/usr/bin/env bash
# Trusted host launcher. The Docker socket is NEVER exposed to a workload.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${OGWP_PROFILE_FILE:-$HERE/../profile.env}"
compose=(docker compose --env-file "$PROFILE" -f "$HERE/compose.yaml")
if [[ -e /dev/dxg ]]; then
  compose+=(-f "$HERE/wsl.yaml")
else
  echo "This profile currently verifies WSL GPU isolation; supply a reviewed native GPU profile before using another host." >&2
  exit 2
fi
cleanup() {
  # Agent SIGTERM stops leasing, completes/releases its current lease and drains.
  "${compose[@]}" stop --timeout 1500 agent || true
  "${compose[@]}" stop --timeout 30 comfy egress || true
}
trap cleanup EXIT
trap 'exit 0' TERM INT

# Verify model bytes before ComfyUI can load them. The host administrator remains trusted.
python3 "$HERE/verify.py" --profile "$PROFILE" --models-only

# Recreate from the approved profile; never resume drifted container settings.
"${compose[@]}" up -d --force-recreate --wait --wait-timeout 180 egress comfy
"${compose[@]}" up --no-start --force-recreate --no-deps agent
python3 "$HERE/verify.py" --profile "$PROFILE" --allow-stopped-agent
comfy_id="$("${compose[@]}" ps -q comfy)"
docker exec -i -e OGWP_PROBE_ROLE=comfy "$comfy_id" /opt/comfy/venv/bin/python - < "$HERE/probe.py"
"${compose[@]}" start agent
python3 "$HERE/verify.py" --profile "$PROFILE"
agent_id="$("${compose[@]}" ps -q agent)"
docker exec -i -e OGWP_PROBE_ROLE=agent "$agent_id" python - < "$HERE/probe.py"
echo "GPU isolation verified; worker is running under the gated supervisor."
while sleep 30; do
  result="$(python3 "$HERE/verify.py" --profile "$PROFILE")" || { printf '%s\n' "$result" >&2; exit 1; }
done
