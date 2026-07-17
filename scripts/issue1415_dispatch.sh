#!/usr/bin/env bash
# Issue #1415 pod-side dispatcher (GPU phases ONLY): pair bank -> phase-1
# driver --pilot -> phase-1 full (incremental HF uploads inside the driver) ->
# pair-bank HF upload -> git commit/push of the phase-1 metadata -> results
# sentinel -> [phase=done].
#
# CPU phases (null battery, map transport, judge, logit lens over the saved
# tensors) run OFF-pod later (round-B spec) — this dispatcher sequences every
# upload BEFORE [phase=done] so the pod can terminate immediately after.
#
# Pod-side code NEVER shells out to scripts/task.py (CLAUDE.md): progress is
# [phase=...] log lines + the end-of-run sentinel JSON under /workspace/logs/
# (poll_pipeline.py contract, .claude/rules/pod-side-reporting.md). The
# [phase=done] token is RESERVED for the single terminal line below — every
# python phase is redirected to its own log so the driver's internal
# [phase=done] never reaches this main log.
#
# Usage:
#   bash scripts/issue1415_dispatch.sh            # full production run
#   bash scripts/issue1415_dispatch.sh --smoke    # --tiny CPU smoke (local-mirror)
#   bash scripts/issue1415_dispatch.sh --replicate           # l14-behavioral-replication:
#       driver --replicate-l14 for seed bases 43 then 44 (fresh baselines +
#       fixed L14/a4 steered cells; frozen parent deltas; pilot/K1/K2 skipped)
#   bash scripts/issue1415_dispatch.sh --replicate --smoke   # tiny CPU replication smoke
#       (requires a prior --smoke run: the tiny parent bulk provides the deltas)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional in the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

ISSUE=1415
LOG_DIR="${LOG_DIR:-/workspace/logs}"
if [ ! -d /workspace ]; then LOG_DIR="${REPO_ROOT}/logs"; fi
mkdir -p "$LOG_DIR"
# Pid-file launch contract (pod-side-reporting.md): the poller's primary
# liveness probe; rewritten on EVERY (re)launch of this dispatcher.
echo $$ > "$LOG_DIR/issue-${ISSUE}.pid"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

SMOKE=0
REPLICATE=0
EXTRA_ARGS=()
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --replicate) REPLICATE=1 ;;
    *) EXTRA_ARGS+=("$a") ;;
  esac
done

# Result-push verification contract (#1205) + per-file artifact-presence
# assert (#1325), shared by the phase-1 and replication branches.
# $1 = commit message; remaining args = git-add paths.
commit_push_verify() {
  local msg="$1"
  shift
  local committed=""
  git add -- "$@"
  if ! git diff --cached --quiet; then
    git commit -m "$msg"
    committed="$(git diff --name-only HEAD~1..HEAD)"
  else
    echo "[phase=commit_results] nothing new to commit (resume re-run)"
  fi
  if ! git push origin "$GIT_BRANCH"; then
    echo "[phase=commit_results] push failed once; retrying" >&2
    sleep 15
    git push origin "$GIT_BRANCH"
  fi
  local ahead
  ahead="$(git rev-list --count "origin/${GIT_BRANCH}..HEAD")"
  if [ "$ahead" -ne 0 ]; then
    echo "[phase=push_verify_failed] ${ahead} commit(s) unpushed on ${GIT_BRANCH}" >&2
    exit 86
  fi
  if [ -n "$committed" ]; then
    local missing=0 f
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      if [ -z "$(git ls-tree -r "origin/${GIT_BRANCH}" --name-only -- "$f")" ]; then
        echo "[phase=artifact_assert_failed] missing from pushed tree: $f" >&2
        missing=1
      fi
    done <<< "$committed"
    if [ "$missing" -ne 0 ]; then exit 87; fi
  fi
}

GIT_SHA="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[phase=dispatch] issue${ISSUE} dispatcher starting (smoke=${SMOKE}, sha=${GIT_SHA}, branch=${GIT_BRANCH})"

TINY_FLAG=()
KIND="epm:results"
OUT_ROOT="eval_results/issue_1415/phase1"
if [ "$SMOKE" -eq 1 ]; then
  TINY_FLAG=(--tiny)
  KIND="epm:smoke-result"
  OUT_ROOT="data/issue_1415/tiny_smoke/out"
fi

# ── l14-behavioral-replication branch (--replicate) ─────────────────
# Seed bases 43 then 44, sequentially on the one GPU (the two seeds share the
# model + delta staging inside each driver invocation; single-GPU intent —
# nothing to shard). No pair-bank build (the driver fetches + sha-gates the
# FROZEN parent bank), no pilot (throughput measured: 2.31 s/sample), no
# K1/K2 (no ceiling arm). Judging runs OFF-pod afterwards
# (issue1415_judge.py --replication <seed>).
if [ "$REPLICATE" -eq 1 ]; then
  REP_SEEDS=(43 44)
  REP_OUT_ROOTS=()
  for SEED in "${REP_SEEDS[@]}"; do
    if [ "$SMOKE" -eq 1 ]; then
      REP_OUT_ROOTS+=("data/issue_1415/tiny_smoke/out_rep${SEED}")
    else
      REP_OUT_ROOTS+=("eval_results/issue_1415/phase1_rep${SEED}")
    fi
  done
  for SEED in "${REP_SEEDS[@]}"; do
    echo "[phase=replicate_rep${SEED}] fresh baselines + L14/a4 steered cells (seed base ${SEED})"
    uv run python scripts/issue1415_run_phase1.py --replicate-l14 --seed-base "$SEED" \
      ${TINY_FLAG[@]+"${TINY_FLAG[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
      > "$LOG_DIR/issue-${ISSUE}-replicate-rep${SEED}.log" 2>&1
  done

  if [ "$SMOKE" -eq 0 ]; then
    echo "[phase=commit_results] committing replication metadata to ${GIT_BRANCH}"
    commit_push_verify \
      "issue-1415 l14-behavioral-replication pod run: rep43+rep44 cell metadata + manifests" \
      "${REP_OUT_ROOTS[@]}"
  fi

  echo "[phase=sentinel] writing results sentinel"
  uv run python - "$KIND" "$ISSUE" "$GIT_SHA" "$SMOKE" "${REP_OUT_ROOTS[@]}" <<'PY'
import json
import sys
import time
from pathlib import Path

kind, issue, git_sha, smoke = sys.argv[1:5]
out_roots = sys.argv[5:]
logs_dir = Path("/workspace/logs")
if not logs_dir.is_dir():
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
kind_slug = kind.replace(":", "_")
path = logs_dir / f"issue-{issue}-{kind_slug}-{int(time.time())}.json"

eval_paths = sorted(str(p) for root in out_roots for p in Path(root).glob("phase1_manifest.json"))
note = {
    "summary": (
        "issue-1415 l14-behavioral-replication GPU run complete: fresh baselines + fixed "
        "L14/alpha=4 steered cells (both arms) at seed bases 43 + 44 (frozen parent deltas; "
        "judging runs OFF-pod via issue1415_judge.py --replication <seed>)"
    ),
    "followup_label": "l14-behavioral-replication",
    "eval_paths": eval_paths,
    "cells_metadata_dirs": [f"{root}/cells" for root in out_roots],
    "reproducibility_card": {
        "adapter_paths": "n/a (no training in this experiment)",
        "wandb_url": "n/a (no training metrics)",
        "hf_artifact_prefixes": [
            "raw_completions/issue_1415/gen_rep43/",
            "raw_completions/issue_1415/gen_rep44/",
            "analysis_tensors/issue_1415/activations/ (frozen parent deltas, REUSED not re-run)",
            "data/issue_1415/pair_bank.json (frozen parent bank, pairs-content sha-gated)",
        ],
        "eval_json_paths": eval_paths,
        "seeds": {
            "generation_seed_bases": {
                "rep43": {"label": 43, "effective_seed_base": 43000},
                "rep44": {"label": 44, "effective_seed_base": 44000},
            },
            "per_draw_seed": "effective_seed_base + draw_index (disjoint from the parent 42..51)",
        },
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
    "by": "issue1415_dispatch",
    "smoke": bool(int(smoke)),
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote sentinel {path}")
PY

  echo "[phase=done]"
  exit 0
fi

if [ "$SMOKE" -eq 0 ]; then
  echo "[phase=pair_bank] building the 28-pair context bank"
  uv run python scripts/issue1415_pair_bank.py \
    > "$LOG_DIR/issue-${ISSUE}-pair-bank.log" 2>&1
fi

echo "[phase=pilot] phase-1 pilot timing gate"
uv run python scripts/issue1415_run_phase1.py --pilot \
  ${TINY_FLAG[@]+"${TINY_FLAG[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  > "$LOG_DIR/issue-${ISSUE}-pilot.log" 2>&1

echo "[phase=phase1_full] phase-1 full run (1b/1a/K1/1c+K2/1d/1e + incremental HF uploads)"
set +e
uv run python scripts/issue1415_run_phase1.py \
  ${TINY_FLAG[@]+"${TINY_FLAG[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  > "$LOG_DIR/issue-${ISSUE}-phase1.log" 2>&1
PHASE1_RC=$?
set -e

# Kill-criteria / gate HALTs are DOMAIN outcomes, not crashes: route on the
# ARTIFACT (gotchas.md: wrap-script route-on-artifact) — the driver writes the
# report BEFORE exiting rc=4 (K1) / rc=5 (K2) / rc=7 (pilot timing gate).
# Anything else non-zero is a real crash.
KILL_HALT=""
if [ "$PHASE1_RC" -ne 0 ]; then
  if [ "$PHASE1_RC" -eq 4 ] && grep -q '"fired": true' "$OUT_ROOT/k1_report.json" 2>/dev/null; then
    KILL_HALT="k1_abort"
    echo "[phase=kill_halt] K1 fired (ceiling shows no separation) — sweep aborted, reporting"
  elif [ "$PHASE1_RC" -eq 5 ] && grep -q '"fired": true' "$OUT_ROOT/k2_report.json" 2>/dev/null; then
    KILL_HALT="k2_halt"
    echo "[phase=kill_halt] K2 fired (coherence collapse on pilot+first-5) — 1c halted, reporting"
  elif [ "$PHASE1_RC" -eq 7 ] && grep -q '"fired": true' "$OUT_ROOT/pilot_gate_report.json" 2>/dev/null; then
    KILL_HALT="pilot_gate"
    echo "[phase=kill_halt] pilot timing gate refused the full sweep — reporting (see pilot_gate_report.json + plan §13 descope ladder)"
  else
    echo "[phase=phase1_failed] driver exited rc=${PHASE1_RC} with no matching kill-report" >&2
    exit "$PHASE1_RC"
  fi
fi

if [ "$SMOKE" -eq 0 ]; then
  echo "[phase=upload_pair_bank] pair bank -> HF data repo"
  uv run python - <<'PY' > "$LOG_DIR/issue-${ISSUE}-upload-bank.log" 2>&1
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue1415_run_phase1 as drv

drv._hf_upload(
    Path("data/issue_1415/pair_bank.json"), "data/issue_1415/pair_bank.json"
)
PY

  echo "[phase=commit_results] committing phase-1 metadata to ${GIT_BRANCH}"
  commit_push_verify \
    "issue-1415 phase-1 pod run: cell metadata, alpha selections, manifest" \
    eval_results/issue_1415/phase1
fi

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
        "alpha_selection_1c.json",
        "alpha_selection_1d.json",
        "phase1_manifest.json",
        "pilot.json",
        "pilot_gate_report.json",
        "k1_report.json",
        "k2_report.json",
        "steered_canonical_index.json",
    )
    for p in Path(out_root).glob(pat)
)
summary_line = "issue-1415 phase-1 GPU run complete (pair bank + 1b/1a/K1/1c+K2/1d/1e)"
if kill_halt:
    report_by_halt = {
        "k1_abort": "k1_report.json",
        "k2_halt": "k2_report.json",
        "pilot_gate": "pilot_gate_report.json",
    }
    summary_line = (
        f"issue-1415 phase-1 HALTED by pre-registered gate ({kill_halt}); "
        f"see {out_root}/{report_by_halt[kill_halt]}"
    )
note = {
    "summary": summary_line,
    "kill_halt": kill_halt or None,
    "eval_paths": eval_paths,
    "cells_metadata_dir": f"{out_root}/cells",
    "reproducibility_card": {
        "adapter_paths": "n/a (no training in this experiment)",
        "wandb_url": "n/a (no training metrics)",
        "hf_artifact_prefixes": [
            "raw_completions/issue_1415/",
            "analysis_tensors/issue_1415/activations/",
            "analysis_tensors/issue_1415/activations_steered/",
            "data/issue_1415/pair_bank.json",
        ],
        "eval_json_paths": eval_paths,
        "seeds": {"generation_seed_base": 42, "null_battery": [1415, 1416]},
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
    "by": "issue1415_dispatch",
    "smoke": bool(int(smoke)),
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"wrote sentinel {path}")
PY

echo "[phase=done]"
