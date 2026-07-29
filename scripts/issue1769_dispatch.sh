#!/usr/bin/env bash
# Issue #1769 pod-side dispatcher (GPU phases only): P0 artifact checks ->
# G0 pilot gate -> G1 grid (sharded across every visible GPU, one model copy
# per GPU via a per-worker CUDA_VISIBLE_DEVICES pin) -> finalize (coverage +
# manifest + upload verify) -> git commit/push of the phase-G metadata ->
# results sentinel -> [phase=done].
#
# The J (judge, Batch API) and A (analysis) phases run OFF-pod on the VM
# (plan §4 DAG) — every upload is sequenced BEFORE [phase=done] so the pod
# can terminate immediately after.
#
# Pod-side code NEVER shells out to scripts/task.py (CLAUDE.md): progress is
# [phase=...] log lines + the end-of-run sentinel JSON under /workspace/logs/
# (poll_pipeline.py contract). The [phase=done] token is RESERVED for the
# single terminal line below — every python phase is redirected to its own
# log so a stray token never reaches this main log.
#
# Usage:
#   bash scripts/issue1769_dispatch.sh --all      # full production run
#   bash scripts/issue1769_dispatch.sh --smoke    # --tiny CPU smoke (local-mirror)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional in the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

ISSUE=1769
LOG_DIR="${LOG_DIR:-/workspace/logs}"
if [ ! -d /workspace ]; then LOG_DIR="${REPO_ROOT}/logs"; fi
mkdir -p "$LOG_DIR"
# Pid-file launch contract (pod-side-reporting.md): rewritten on EVERY
# (re)launch of this dispatcher.
echo $$ > "$LOG_DIR/issue-${ISSUE}.pid"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

SMOKE=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --all) ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

GIT_SHA="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[phase=dispatch] issue${ISSUE} dispatcher starting (smoke=${SMOKE}, sha=${GIT_SHA}, branch=${GIT_BRANCH})"

TINY_FLAG=()
KIND="epm:results"
OUT_ROOT="eval_results/issue_1769/phase_g"
BULK_ROOT="/workspace/eps-issue-1769"
if [ ! -d /workspace ]; then BULK_ROOT="data/issue_1769/bulk"; fi
if [ "$SMOKE" -eq 1 ]; then
  TINY_FLAG=(--tiny)
  KIND="epm:smoke-result"
  OUT_ROOT="data/issue_1769/tiny_smoke/out"
  BULK_ROOT="data/issue_1769/tiny_smoke/bulk"
fi
DRIVER=(uv run python scripts/issue1769_run.py "${TINY_FLAG[@]}" --out-root "$OUT_ROOT" --bulk-root "$BULK_ROOT")

# GPU width: derived from nvidia-smi in BOTH modes (never a smoke-narrowed
# pin — smoke on a CPU host naturally runs 1 worker; production fails loud
# on 0 GPUs). Wave size == visible GPU count (dispatcher-wave gotcha).
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
if [ "$SMOKE" -eq 0 ] && [ "$NGPU" -lt 1 ]; then
  echo "[phase=dispatch_failed] production run requires >= 1 GPU (nvidia-smi found $NGPU)" >&2
  exit 3
fi
WORKERS="$NGPU"
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

KILL_HALT=""

# ── P0: artifact checks (pod-side re-run; CPU, before any model load) ──
echo "[phase=p0]"
"${DRIVER[@]}" --phase p0 > "$LOG_DIR/issue-${ISSUE}-p0.log" 2>&1 || {
  echo "[phase=p0_failed] see $LOG_DIR/issue-${ISSUE}-p0.log" >&2
  tail -n 60 "$LOG_DIR/issue-${ISSUE}-p0.log" >&2
  exit 4
}

# ── G0: pilot timing gate (1 GPU — gate runs before the fan-out) ─────
echo "[phase=pilot]"
set +e
env CUDA_VISIBLE_DEVICES=0 "${DRIVER[@]}" --phase pilot > "$LOG_DIR/issue-${ISSUE}-pilot.log" 2>&1
PILOT_RC=$?
set -e
if [ "$PILOT_RC" -eq 7 ]; then
  # Designed artifact-routed HALT (pilot_gate_report.json + rc=7) — routed
  # on the artifact like a kill criterion, never an anonymous crash.
  echo "[phase=pilot_gate_halt] G0 refused the grid (see $OUT_ROOT/pilot_gate_report.json)"
  KILL_HALT="pilot_gate"
elif [ "$PILOT_RC" -ne 0 ]; then
  echo "[phase=pilot_failed] rc=$PILOT_RC" >&2
  tail -n 60 "$LOG_DIR/issue-${ISSUE}-pilot.log" >&2
  exit 5
fi

# ── G1: grid, sharded by (trait x arm) across every visible GPU ──────
if [ -z "$KILL_HALT" ]; then
  echo "[phase=grid] fan-out across $WORKERS worker(s)"
  PIDS=()
  for g in $(seq 0 $((WORKERS - 1))); do
    # Per-worker CVD pin in the LAUNCHER env (CVD-clobber gotcha): one model
    # copy per GPU; the smoke's single CPU worker takes the same path.
    CUDA_VISIBLE_DEVICES="$g" "${DRIVER[@]}" --phase grid --shard "$g" --n-shards "$WORKERS" \
      > "$LOG_DIR/issue-${ISSUE}-grid-shard${g}.log" 2>&1 &
    PIDS+=($!)
  done
  GRID_FAIL=0
  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      GRID_FAIL=1
      echo "[phase=grid_shard_failed] shard $i — inner log tail follows" >&2
      tail -n 120 "$LOG_DIR/issue-${ISSUE}-grid-shard${i}.log" >&2
    fi
  done
  if [ "$GRID_FAIL" -ne 0 ]; then exit 6; fi

  # ── finalize: coverage + manifest + upload verify ────────────────
  echo "[phase=finalize]"
  "${DRIVER[@]}" --phase finalize > "$LOG_DIR/issue-${ISSUE}-finalize.log" 2>&1 || {
    echo "[phase=finalize_failed] see $LOG_DIR/issue-${ISSUE}-finalize.log" >&2
    tail -n 60 "$LOG_DIR/issue-${ISSUE}-finalize.log" >&2
    exit 8
  }
fi

# ── commit the phase-G metadata to the issue branch (production only:
# the smoke's out-root lives under data/ by design and mutating git state
# from a smoke is out of scope — the commit body is the #1415-proven
# commit_push_verify shape) ──────────────────────────────────────────
if [ "$SMOKE" -eq 0 ]; then
  echo "[phase=commit_results]"
  git add -- "$OUT_ROOT"
  if ! git diff --cached --quiet; then
    git commit -m "issue1769 phase-G metadata (pilot + cells manifest + p0; kill_halt=${KILL_HALT:-none})"
  else
    echo "[phase=commit_results] nothing new to commit (resume re-run)"
  fi
  if ! git push origin "$GIT_BRANCH"; then
    echo "[phase=commit_results] push failed once; retrying" >&2
    sleep 15
    git push origin "$GIT_BRANCH"
  fi
  AHEAD="$(git rev-list --count "origin/${GIT_BRANCH}..HEAD")"
  if [ "$AHEAD" -ne 0 ]; then
    echo "[phase=push_verify_failed] ${AHEAD} commit(s) unpushed on ${GIT_BRANCH}" >&2
    exit 86
  fi
fi

# ── results sentinel (poll_pipeline.py contract) ─────────────────────
echo "[phase=sentinel] writing results sentinel"
uv run python - "$KIND" "$ISSUE" "$GIT_SHA" "$OUT_ROOT" "$SMOKE" "$KILL_HALT" <<'PY'
import json
import sys
import time
from pathlib import Path

kind, issue, git_sha, out_root, smoke, kill_halt = sys.argv[1:7]
logs_dir = Path("/workspace/logs")
if not logs_dir.is_dir():
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
kind_slug = kind.replace(":", "_")
path = logs_dir / f"issue-{issue}-{kind_slug}-{int(time.time())}.json"

eval_paths = sorted(
    str(p)
    for pat in (
        "p0_artifact_check.json",
        "pilot.json",
        "pilot_gate_report.json",
        "cells_manifest.json",
    )
    for p in Path(out_root).glob(pat)
)
summary = "issue-1769 phase-G GPU run complete (P0 + G0 pilot + 600-cell grid + finalize)"
if kill_halt:
    summary = (
        f"issue-1769 phase-G HALTED by the pre-registered G0 pilot gate; "
        f"see {out_root}/pilot_gate_report.json"
    )
note = {
    "summary": summary,
    "kill_halt": kill_halt or None,
    "eval_paths": eval_paths,
    "cells_metadata_dir": f"{out_root}/cells",
    "reproducibility_card": {
        "adapter_paths": "n/a (no training in this experiment)",
        "wandb_url": "n/a (no training metrics)",
        "hf_artifact_prefixes": ["issue1769_prefill_decode/raw_completions/"],
        "eval_json_paths": eval_paths,
        "seeds": {"generation_seed_base": 42, "bootstrap": 0},
        "git_commit": git_sha,
    },
    "smoke": bool(int(smoke)),
}
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "note": json.dumps(note, indent=2),
    "task_id": int(issue),
    "by": "issue1769_dispatch",
    "smoke": bool(int(smoke)),
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote sentinel {path}")
PY

echo "[phase=done]"
