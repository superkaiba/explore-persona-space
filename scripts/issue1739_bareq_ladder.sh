#!/usr/bin/env bash
# R2.5 / R2.75 — bare-query round L-LADDER (one box, all three behaviors).
#
# The committed bare-query round ran at ONE labeled budget: `--budget` defaults
# to the whole train table, so `bareq_score_done.json` records a single
# max-budget read per behavior and BOTH of its variant columns
# (`variants: [context_end, prefix_end]` — the round emits the prefix-arm and
# the bare-query arm in one pass). What is missing is the LABEL-BUDGET axis:
# how the bare-query read scales with L. This box sweeps `--budget` over each
# behavior's L ladder, reusing the gap-1 box's staging chain verbatim.
#
# RUN-1 CRASH FIX (2026-08-01, all 9 rungs, ~1 s each, exit 1). The scorer's
# rail-3 guard requires `out_root.resolve().name == "bareq_map"` and refuses any
# other leaf ("refusing to write bare-query results outside this round's own
# dir"); run 1 passed `<base>/L<budget>`, so every rung died at post-parse
# validation before touching a store. The out-root is now
# `<base>/L<budget>/bareq_map` — the guard is CORRECT and stays untouched; it
# was the composed argv that was wrong. The fix is pinned by an argv dry-run
# that calls the scorer's own `_assert_outputs_safe` on this exact path
# (tests/test_issue1739_bareq_ladder_argv.py) rather than trusting a re-read.
#
# NOTE on R2.5 (prefix-end L scaling) — the OTHER reading of R2.5, the fits.py
# prefix_end L-ladder on the TRAIN setting, is ALREADY COMMITTED: the main grid
# holds 810 plain cells per behavior covering variant=prefix_end x budgets
# {250, 2500, max} x U {250, 5000, full} x its regimes. No box is spent on it;
# this box covers the bare-query surface, where only the max budget existed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

BEHAVIORS="${EPM_I1739_BEHAVIORS:-evil sycophancy hallucination}"
OUT_BASE="eval_results/issue_1739/r275_query_scaling"
ACCT="$OUT_BASE/bareq_ladder_invocations.json"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$OUT_BASE" "$LOG_DIR"

upload_out_base() {
  uv run python - "$OUT_BASE" <<'PYEOF' || echo "[bqladder] WARNING: upload leg failed" >&2
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

root = Path(sys.argv[1])
if not any(root.rglob("*")):
    print("[bqladder] nothing to upload", flush=True)
    raise SystemExit(0)
hub.retry_transient(
    lambda: HfApi().upload_folder(
        folder_path=str(root),
        path_in_repo="issue1739_maxood/r275_query_scaling",
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        # BOTH forms are required: `**/*.json` is fnmatch-ed against the
        # folder-RELATIVE path, so it matches only files at least one directory
        # deep — a bare `**/` pattern set silently drops every ROOT-level file.
        # Run 1 lost `bareq_ladder_invocations.json` (the round's own rc
        # accounting) to exactly that; the `*.<ext>` twins carry the root level.
        allow_patterns=[
            "*.json",
            "*.jsonl",
            "*.log",
            "**/*.json",
            "**/*.jsonl",
            "**/*.log",
        ],
    ),
    what="bareq-ladder upload",
)
print("[bqladder] HF upload done", flush=True)
PYEOF
}
trap upload_out_base EXIT

echo "[bqladder] behaviors='$BEHAVIORS' $(date -u +%FT%TZ)"

echo "[bqladder] stage train + extraction + wcrung stores $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
uv run python scripts/issue1739_wcrung_arms_run.py \
  --behaviors $BEHAVIORS \
  --store-root data/issue_1739/hf_dl \
  --stage-only

echo "[bqladder] defensive explicit wcrung stage (idempotent) $(date -u +%FT%TZ)"
uv run python - <<'PYEOF'
import argparse
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from scripts.issue1739_wcrung_arms_run import stage_wcrung_store

p = stage_wcrung_store(argparse.Namespace(store_root=Path("data/issue_1739/hf_dl")))
print(f"[bqladder] wcrung store at {p}", flush=True)
PYEOF

echo "[bqladder] stage bare capture store + committed contrast + queries $(date -u +%FT%TZ)"
uv run python scripts/issue1739_bareq_score_prestage.py

printf '[]' > "$ACCT.parts"
FATAL=0
for B in $BEHAVIORS; do
  case "$B" in
    evil) BUDGETS="250 2500 8000"; LEGS="1 2" ;;
    *) BUDGETS="250 2500 16000"; LEGS="1" ;;
  esac
  for L in $BUDGETS; do
    tag="${B}_L${L}"
    log="$LOG_DIR/$tag.log"
    echo "[bqladder] $tag $(date -u +%FT%TZ)"
    set +e
    # shellcheck disable=SC2086
    uv run python scripts/issue1739_bareq_score.py \
      --behaviors "$B" --legs $LEGS \
      --budget "$L" --draw 0 --seed 0 \
      --null-shuffle-seeds 8 --device cpu \
      --store-root data/issue_1739/hf_dl \
      --bareq-store data/issue_1739/hf_dl/bareq_capture_store \
      --query-manifest eval_results/issue_1739/bareq_map/bareq_queries.json \
      --out-root "$OUT_BASE/L$L/bareq_map" \
      --force-own-pool-frozen > "$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      FATAL=1
      echo "[bqladder] $tag rc=$rc FAILED — tail:" >&2
      tail -25 "$log" >&2
    else
      echo "[bqladder] $tag rc=0"
    fi
    uv run python - "$ACCT.parts" "$tag" "$B" "$L" "$rc" <<'PYEOF'
import json
import sys

path, tag, behavior, budget, rc = sys.argv[1:6]
rows = json.load(open(path))
rows.append({"tag": tag, "behavior": behavior, "budget_l": int(budget), "rc": int(rc)})
json.dump(rows, open(path, "w"), indent=1)
PYEOF
  done
done
mv "$ACCT.parts" "$ACCT"
echo "[bqladder] invocation accounting -> $ACCT"
cat "$ACCT"

if [ "$FATAL" -ne 0 ]; then
  echo "[bqladder] FAILING: at least one budget rung failed" >&2
  exit 1
fi
echo "[bqladder] done rc=0 $(date -u +%FT%TZ)"
