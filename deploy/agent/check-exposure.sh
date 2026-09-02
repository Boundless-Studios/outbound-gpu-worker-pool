#!/usr/bin/env bash
# Prove this machine exposes nothing new. Run it on the box, as the user the
# agent runs as, after `systemctl --user start outbound-gpu-worker`.
#
# The design's promise is that a worker opens no port and that ComfyUI is
# reachable only from the machine itself. This is that promise, checked from
# the inside — an external port scan is the other half, and this one tells you
# what a scan would find before you run it.
#
# Exit 0: every listening socket is either loopback or something the operator
#         already runs; the agent holds none.
# Exit 1: ComfyUI is bound off loopback, or the agent is listening.
# Exit 2: the check could not run.
set -euo pipefail

COMFY_PORTS="8188 8189"
AGENT_PATTERN="outbound_gpu_worker_pool.agent_main"

if ! command -v ss >/dev/null 2>&1; then
    echo "check-exposure.sh needs ss (install iproute2)" >&2
    exit 2
fi

is_loopback() {
    case "$1" in
    127.*|::1|"[::1]"|localhost) return 0 ;;
    *) return 1 ;;
    esac
}

is_comfy_port() {
    local port
    for port in $COMFY_PORTS; do
        [ "$1" = "$port" ] && return 0
    done
    return 1
}

echo "== listening TCP sockets =="
listening="$(ss -ltnp 2>/dev/null || true)"
printf '%s\n' "$listening"
echo

# `ss -l` data rows always start with LISTEN, so this drops the header without
# depending on `-H`, which older iproute2 does not have.
sockets="$(printf '%s\n' "$listening" | awk '$1 == "LISTEN" { print $4 "|" $6 }')"

agent_pids="$(pgrep -f "$AGENT_PATTERN" 2>/dev/null || true)"
failures=0
comfy_found=0

while IFS='|' read -r address_port process; do
    [ -n "$address_port" ] || continue
    port="${address_port##*:}"
    address="${address_port%:*}"

    if is_comfy_port "$port"; then
        comfy_found=1
        echo "ComfyUI bind: $address:$port"
        if ! is_loopback "$address"; then
            echo "FAIL: ComfyUI listens on $address:$port, which is not loopback." >&2
            echo "      Start it with --listen 127.0.0.1 so only this machine reaches it." >&2
            failures=$((failures + 1))
        fi
    fi

    for pid in $agent_pids; do
        case "$process" in
        *"pid=$pid,"*)
            echo "FAIL: the worker agent (pid $pid) holds a listening socket on $address:$port." >&2
            echo "      The agent is outbound only; it must never listen." >&2
            failures=$((failures + 1))
            ;;
        esac
    done
done <<EOF
$sockets
EOF

if [ "$comfy_found" -eq 0 ]; then
    echo "ComfyUI bind: none — nothing listens on ports $COMFY_PORTS."
fi

if [ -z "$agent_pids" ]; then
    echo "worker agent: not running, so its sockets could not be checked."
else
    echo "worker agent: running as pid(s) $(echo "$agent_pids" | tr '\n' ' ')"
fi

echo
if [ "$failures" -ne 0 ]; then
    echo "exposure check FAILED with $failures problem(s)."
    exit 1
fi
echo "exposure check passed: no new reachable service on this machine."
