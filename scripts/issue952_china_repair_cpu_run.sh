#!/usr/bin/env bash
# Dedicated CPU continuation; immutable GPU/judge inputs, no new inference.
set -euo pipefail
umask 077
export PATH="/root/.local/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072

expected_sha="${1:?Pass the reviewed code SHA}"
data_revision="${2:?Pass the verified terminal GPU data revision}"
judge_revision="${3:?Pass the verified complete judge archive revision}"
for revision in "$expected_sha" "$data_revision" "$judge_revision"; do
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Every revision must be an immutable 40-character commit SHA" >&2
    exit 86
  fi
done

cd /workspace/explore-persona-space
test "$(git rev-parse HEAD)" = "$expected_sha"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked source is dirty; refusing CPU continuation" >&2
  exit 86
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

run_root=/workspace/issue952-china-repair-v2-cpu
judge_prefix=issue952_position_divergence/followups/china_refusal_wording_withholding_v2/attempt1/judge
mkdir -p "$run_root/logs" /workspace/logs
printf '%s\n' "$$" > /workspace/logs/issue-952-china-repair-cpu.pid
uv run --no-sync python -c \
  'from scripts.issue952_china_repair_analysis import require_cpu_lane; require_cpu_lane(True)'
test -x /usr/bin/time

for phase in stage load pilot full export; do
  phase_log="$run_root/logs/${phase}-$(date -u +%Y%m%dT%H%M%S).log"
  echo "[phase=$phase] code_sha=$expected_sha data_revision=$data_revision judge_revision=$judge_revision log=$phase_log"
  /usr/bin/time -v uv run --no-sync python -m scripts.issue952_china_repair_analysis \
    "$phase" --root "$run_root" --data-revision "$data_revision" \
    --judge-revision "$judge_revision" --judge-prefix "$judge_prefix" \
    --cpu-lane --threads 8 > "$phase_log" 2>&1
  echo "Completed CPU/$phase rc=0"
done
test -s "$run_root/analysis/full/done.json"
test -s "$run_root/export.json"
echo "[phase=done] Full CPU analysis and verified export finished; owner must persist logs/receipt and verify teardown"
