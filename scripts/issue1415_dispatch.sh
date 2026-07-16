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
EXTRA_ARGS=()
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    *) EXTRA_ARGS+=("$a") ;;
  esac
done

GIT_SHA="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "[phase=dispatch] issue${ISSUE} dispatcher starting (smoke=${SMOKE}, sha=${GIT_SHA}, branch=${GIT_BRANCH})"

TINY_FLAG=()
if [ "$SMOKE" -eq 1 ]; then TINY_FLAG=(--tiny); fi

if [ "$SMOKE" -eq 0 ]; then
  echo "[phase=pair_bank] building the 28-pair context bank"
  uv run python scripts/issue1415_pair_bank.py \
    > "$LOG_DIR/issue-${ISSUE}-pair-bank.log" 2>&1
fi

echo "[phase=pilot] phase-1 pilot timing gate"
uv run python scripts/issue1415_run_phase1.py --pilot \
  ${TINY_FLAG[@]+"${TINY_FLAG[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  > "$LOG_DIR/issue-${ISSUE}-pilot.log" 2>&1

echo "[phase=phase1_full] phase-1 full run (1b/1a/1c/1d + incremental HF uploads)"
uv run python scripts/issue1415_run_phase1.py \
  ${TINY_FLAG[@]+"${TINY_FLAG[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  > "$LOG_DIR/issue-${ISSUE}-phase1.log" 2>&1

COMMITTED_FILES=""
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
  git add -- eval_results/issue_1415/phase1
  if ! git diff --cached --quiet; then
    git commit -m "issue-1415 phase-1 pod run: cell metadata, alpha selections, manifest"
    COMMITTED_FILES="$(git diff --name-only HEAD~1..HEAD)"
  else
    echo "[phase=commit_results] nothing new to commit (resume re-run)"
  fi
  # Result-push verification contract (#1205): bare push, rev-list proof,
  # one retry, then per-file ls-tree artifact-presence assert (#1325).
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
  if [ -n "$COMMITTED_FILES" ]; then
    MISSING=0
    while IFS= read -r f; do
      [ -z "$f" ] && continue
      if [ -z "$(git ls-tree -r "origin/${GIT_BRANCH}" --name-only -- "$f")" ]; then
        echo "[phase=artifact_assert_failed] missing from pushed tree: $f" >&2
        MISSING=1
      fi
    done <<< "$COMMITTED_FILES"
    if [ "$MISSING" -ne 0 ]; then exit 87; fi
  fi
fi

echo "[phase=sentinel] writing results sentinel"
KIND="epm:results"
OUT_ROOT="eval_results/issue_1415/phase1"
if [ "$SMOKE" -eq 1 ]; then
  KIND="epm:smoke-result"
  OUT_ROOT="data/issue_1415/tiny_smoke/out"
fi
uv run python - "$KIND" "$ISSUE" "$GIT_SHA" "$OUT_ROOT" "$SMOKE" <<'PY'
import json
import sys
import time
from pathlib import Path

kind, issue, git_sha, out_root, smoke = sys.argv[1:6]
logs_dir = Path("/workspace/logs")
if not logs_dir.is_dir():
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
kind_slug = kind.replace(":", "_")
path = logs_dir / f"issue-{issue}-{kind_slug}-{int(time.time())}.json"

eval_paths = sorted(
    str(p)
    for pat in ("alpha_selection_1c.json", "alpha_selection_1d.json", "phase1_manifest.json")
    for p in Path(out_root).glob(pat)
)
note = {
    "summary": "issue-1415 phase-1 GPU run complete (pair bank + 1b/1a/1c/1d)",
    "eval_paths": eval_paths,
    "cells_metadata_dir": f"{out_root}/cells",
    "reproducibility_card": {
        "adapter_paths": "n/a (no training in this experiment)",
        "wandb_url": "n/a (no training metrics)",
        "hf_artifact_prefixes": [
            "raw_completions/issue_1415/",
            "analysis_tensors/issue_1415/activations/",
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
