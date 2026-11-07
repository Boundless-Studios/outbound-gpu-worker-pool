#!/usr/bin/env bash
# Time-share the GPU between H3 (ogwp-hardened) and the 3D worker
# (ogwp-hardened-3d). Never runs both: it drains and stops the running one,
# confirms it is gone, then starts the other and waits for its supervisor to
# report verified isolation. Dry run unless --apply.
#
#   switch.sh --to 3d [--apply]
#   switch.sh --to h3 [--apply]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
H3_DIR="${OGWP_H3_DIR:-$HOME/.local/share/ogwp-hardened}"
D3_DIR="${OGWP_3D_DIR:-$HOME/.local/share/ogwp-hardened-3d}"
TIMEOUT="${OGWP_SWITCH_TIMEOUT_SECONDS:-1800}"

usage() { echo "usage: $0 --to h3|3d [--apply]" >&2; exit 2; }
to="" apply=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --to) [[ $# -ge 2 ]] || usage; to="$2"; shift 2 ;;
    --apply) apply=1; shift ;;
    *) usage ;;
  esac
done
case "$to" in
  3d) from=h3 ;;
  h3) from=3d ;;
  *) usage ;;
esac

unit() { case "$1" in h3) echo ogwp-hardened.service ;; 3d) echo ogwp-hardened-3d.service ;; esac; }
project() { case "$1" in h3) echo ogwp-hardened ;; 3d) echo ogwp-hardened-3d ;; esac; }
banner() {
  case "$1" in
    h3) echo "GPU isolation verified; worker is running under the gated supervisor." ;;
    3d) echo "GPU isolation verified; 3D worker is running under the gated supervisor." ;;
  esac
}
verify_target() {
  case "$1" in
    h3) python3 "$H3_DIR/hardened/verify.py" --profile "$H3_DIR/profile.env" ;;
    3d) python3 "$D3_DIR/hardened-3d/verify.py" --profile "$D3_DIR/profile-3d.env" ;;
  esac
}
running_containers() {
  docker ps -q --filter "label=com.docker.compose.project=$(project "$1")" --filter status=running
}

echo "Gate: switching the GPU from $from to $to." >&2
if ! python3 "$HERE/switch_gate.py" gate --to "$to"; then
  echo "Gate refused; nothing changed." >&2
  exit 1
fi

if [[ $apply -ne 1 ]]; then
  cat >&2 <<EOF
Dry run: the gate passes. With --apply this would:
  1. systemctl --user disable --now $(unit "$from")   (drains its agent first)
  2. confirm no $(project "$from") container is running
  3. systemctl --user enable --now $(unit "$to")
  4. wait up to ${TIMEOUT}s for: $(banner "$to")
  5. postcheck (one worker active and enabled) and verify.py for $to
EOF
  exit 0
fi

# 1. Drain-stop the source and keep it from returning at boot. The unit's stop
#    timeout covers the agent's drain; this blocks until it has stopped.
systemctl --user disable --now "$(unit "$from")"

# 2. Never both: refuse to start the target while anything of the source runs.
if [[ -n "$(running_containers "$from")" ]] || systemctl --user is-active --quiet "$(unit "$from")"; then
  echo "$from is still running after its stop; not starting $to. Investigate before retrying." >&2
  exit 1
fi

# 3. Start the target under its gated supervisor.
since="$(date '+%Y-%m-%d %H:%M:%S')"
systemctl --user reset-failed "$(unit "$to")" 2>/dev/null || true
systemctl --user enable --now "$(unit "$to")"

# 4. Wait for the supervisor's verified-isolation line. No fallback on failure:
#    a failed gate leaves the GPU idle for the operator to investigate.
deadline=$((SECONDS + TIMEOUT))
until journalctl --user -u "$(unit "$to")" --since "$since" -o cat --no-pager | grep -qxF "$(banner "$to")"; do
  if systemctl --user is-failed --quiet "$(unit "$to")" || ! systemctl --user is-active --quiet "$(unit "$to")"; then
    echo "$to failed to start under its gate; the GPU is idle. See: journalctl --user -u $(unit "$to")" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "$to did not report verified isolation within ${TIMEOUT}s; stopping it." >&2
    systemctl --user stop "$(unit "$to")" || true
    exit 1
  fi
  sleep 10
done

# 5. Post-checks: exactly one worker, enabled for boot, and its gate passing.
python3 "$HERE/switch_gate.py" postcheck --to "$to"
verify_target "$to"
echo "Switched the GPU to $to; $from is drained, stopped and disabled." >&2
