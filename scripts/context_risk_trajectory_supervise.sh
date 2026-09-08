#!/usr/bin/env bash
# Run under a detached process with stdin=/dev/null and a fresh dedicated log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?phase name required}"
shift
[[ "$mode" =~ ^[a-z][a-z0-9_]*$ ]] || exit 2
[[ "$#" -gt 0 ]] || exit 2
root="${EPM_CONTEXT_RISK_TRAJECTORY_PROCESS_ROOT:?fresh run root required}"
launch_id="${EPM_CONTEXT_RISK_LAUNCH_ID:?unique launch id required}"
deadline="${EPM_CONTEXT_RISK_PROCESS_TIMEOUT_SECONDS:?whole-process deadline required}"
[[ "$launch_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$deadline" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$root"
prefix="$root/${mode}_${launch_id}_process"
for suffix in pid worker.pid exit.json; do
  if [[ -e "$prefix.$suffix" ]]; then
    echo "Process evidence already exists: $prefix.$suffix" >&2
    exit 2
  fi
done
sentinel_root="${EPM_CONTEXT_RISK_SENTINEL_DIR:-/workspace/logs}"
sentinel="$sentinel_root/issue-2670-epm_progress-trajectory-${launch_id}-${mode}.json"
if [[ -d "$sentinel_root" && ( -e "$sentinel" || -e "$sentinel.processed" ) ]]; then
  echo "Refusing existing write-once phase sentinel before worker launch" >&2
  exit 126
fi
worker_pid=''
group_state() {
  local listing
  if ! listing=$(ps -eo pgid=,stat=); then
    echo "Cannot enumerate worker process group" >&2
    return 2
  fi
  awk -v group="$worker_pid" '$1 == group && $2 !~ /^[ZX]/ { live=1 } END { exit !live }' \
    <<< "$listing"
}
finish() {
  rc=$?
  trap - EXIT TERM INT
  # The group can outlive its leader. Ignore reparented zombies, which
  # cannot retain a CUDA context, but verify every live descendant drains.
  cleanup=not_started
  if [[ -n "$worker_pid" ]]; then
    cleanup=no_live_members
    if group_state; then
      cleanup=terminated_descendants
      if ! kill -TERM -- "-$worker_pid"; then
        echo "TERM raced with worker group exit; verifying drainage" >&2
      fi
      for ((i=0; i<30; i++)); do
        if group_state; then sleep 1; else break; fi
      done
      if group_state; then
        cleanup=killed_descendants
        if ! kill -KILL -- "-$worker_pid"; then
          echo "KILL raced with worker group exit; verifying drainage" >&2
        fi
        for ((i=0; i<10; i++)); do
          if group_state; then sleep 1; else break; fi
        done
      fi
    fi
    if group_state; then
      echo "Worker group $worker_pid still has live members after cleanup" >&2
      cleanup=failed_live_members
      if [[ "$rc" == 0 ]]; then rc=125; fi
    else
      group_rc=$?
      if [[ "$group_rc" != 1 ]]; then
        echo "Worker group cleanup could not be verified" >&2
        cleanup=failed_verification
        if [[ "$rc" == 0 ]]; then rc=125; fi
      fi
    fi
  fi
  if [[ "$rc" == 0 && -d "$sentinel_root" ]]; then
    if [[ -e "$sentinel" || -e "$sentinel.processed" ]]; then
      echo "Refusing existing write-once phase sentinel" >&2
      rc=126
    elif ! printf '{"sentinel_schema_version":1,"kind":"epm:progress","version":1,"task_id":2670,"gate":"phase","by":"codex-trajectory-supervisor","note":"Trajectory phase %s completed with exit0 and owned descendants drained; scientific completion is assessed separately."}\n' "$mode" > "$sentinel.tmp"; then
      echo "Phase sentinel write failed" >&2
      rc=126
    elif ! mv "$sentinel.tmp" "$sentinel"; then
      echo "Phase sentinel installation failed" >&2
      rc=126
    fi
  fi
  printf '{"mode":"%s","supervisor_pid":%s,"worker_pid":%s,"exit_code":%s,"cleanup":"%s","finished_unix":%s}\n' \
    "$mode" "$$" "${worker_pid:-null}" "$rc" "$cleanup" "$(date +%s)" > "$prefix.exit.json.tmp"
  mv "$prefix.exit.json.tmp" "$prefix.exit.json"
  if [[ "$rc" == 0 ]]; then
    echo "[phase=done] trajectory phase $mode exited0 and descendants drained"
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf '%s\n' "$$" > "$prefix.pid.tmp"
mv "$prefix.pid.tmp" "$prefix.pid"
export UV_NO_SYNC=1
echo "[phase=$mode] supervisor pid=$$ root=$root utc=$(date -u +%FT%TZ)"
setsid timeout --kill-after=60s "$deadline" \
  "$@" &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$prefix.worker.pid.tmp"
mv "$prefix.worker.pid.tmp" "$prefix.worker.pid"
echo "[worker-start] mode=$mode pid=$worker_pid"
wait "$worker_pid"
