#!/usr/bin/env bash
# Canonical host-side Docker Compose scheduler trigger.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="${SCHEDULER_TRIGGER_LOCK_FILE:-/run/lock/betting-nightly.lock}"

log_event() {
    local event="$1"
    shift
    printf '%s event=%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$event" "$*"
}

if ! command -v flock >/dev/null 2>&1; then
    log_event scheduler_trigger_failed "reason=flock_unavailable"
    exit 78
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log_event scheduler_trigger_overlap_skipped "lock_file=$LOCK_FILE"
    exit 75
fi

cd "$PROJECT_ROOT"
started_at="$(date +%s)"
log_event scheduler_trigger_started "project_root=$PROJECT_ROOT"

on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        local elapsed=$(( $(date +%s) - started_at ))
        log_event scheduler_trigger_failed "exit_code=$exit_code elapsed_seconds=$elapsed"
    fi
    exit "$exit_code"
}
trap on_exit EXIT

docker compose up -d --wait --wait-timeout 300 api
docker compose run --rm --no-deps scheduler

elapsed=$(( $(date +%s) - started_at ))
log_event scheduler_trigger_completed "elapsed_seconds=$elapsed"
trap - EXIT
