#!/usr/bin/env bash
# systemd ExecStartPre: after a reboot the user unit can start before dockerd
# answers. Waiting here (bounded) keeps those early starts from using up the
# unit's restart burst and leaving the worker failed once Docker is ready.
set -uo pipefail
deadline=$((SECONDS + ${OGWP_DOCKER_WAIT_SECONDS:-600}))
until docker info >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Docker did not become ready in time." >&2
    exit 1
  fi
  sleep "${OGWP_DOCKER_WAIT_INTERVAL:-5}"
done
