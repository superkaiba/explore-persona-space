#!/usr/bin/env bash
set -euo pipefail

export PATH="/root/.local/bin:${PATH}"
export UV_NO_SYNC=1

repo=/workspace/explore-persona-space
mode=${1:-production}
required_ancestor=883467ec025b4efa3f4b2ce48bea82ab496e201a

case "$mode" in
  smoke)
    out=/workspace/issue2094-joint-broad-qualitative-smoke
    pid_file=/workspace/logs/issue-2094-jointqual-smoke.pid
    extra_args=(--smoke)
    hard_stop_hours=0.5
    ;;
  production)
    out=/workspace/issue2094-joint-broad-qualitative
    pid_file=/workspace/logs/issue-2094-jointqual.pid
    extra_args=(--write-sentinel)
    hard_stop_hours=4
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

cd "$repo"
if [ -f ./.env ]; then
  set -a
  . ./.env
  set +a
fi

git merge-base --is-ancestor "$required_ancestor" HEAD
test -z "$(git status --short --untracked-files=no)"
mkdir -p /workspace/logs

printf '%s\n' "$$" > "${pid_file}.tmp"
mv "${pid_file}.tmp" "$pid_file"

exec /root/eps-venv/bin/python scripts/issue2094_natural_corrected.py \
  --profile broad_joint \
  --out "$out" \
  --batch-size 1 \
  --hard-stop-hours "$hard_stop_hours" \
  "${extra_args[@]}"
