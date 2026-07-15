#!/usr/bin/env bash
# Issue #1310 dispatcher: focused per-character context->dialogue map
# (LABELED SCRIPT FORMAT — per-turn (X,Y) points; the run-1 rebuild).
# stage -> gen (base+instruct, multi-character `<LABEL>:` scenes) -> attribute
# (base+instruct, deterministic line-prefix parse -> per-turn pairs) -> extract
# (base+instruct, one-per-GPU when >=2 GPUs) -> fit -> figures -> upload ->
# results sentinel -> [phase=done].
#
# ONE code path for smoke / production — the tiny-cell subset threads through
# every phase via the size flags below.
#
# Sizing: production generates all N_PROMPTS_PER_PERSONA (=300) shared scenarios
# per persona x model; at ~8 target turns/scene each persona clears ~1.5k-2.4k
# per-turn points (adequate power vs 3584 dims). Battery = 20 settings x 18
# situations = 360 combos (headroom over 300).
#
# Modes:
#   bash scripts/issue1310_dispatch.sh --smoke   # CPU VM: tiny model + stub gen
#   bash scripts/issue1310_dispatch.sh           # production
#
# Pod-side contract: [phase=...] log lines; the terminal [phase=done] is
# emitted ONLY here, after the final sentinel write. NEVER shells task.py.
set -euo pipefail

# RunPod/GCE-safe repo-root resolution (#1310 sibling of the #825 WORKLOAD_ROOT bug).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
# GCE lane has NO .env (tokens ride the startup-script env) — conditional only.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export TQDM_DISABLE=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

ISSUE=1310
RUN_START_EPOCH="$(date +%s)"
export RUN_START_EPOCH
MODE="production"
for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="smoke" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" || LOG_DIR="$(pwd)/logs"; mkdir -p "$LOG_DIR"

run_extract_both() {
  # run_extract_both <data_dir> <extra_flags...>
  local data_dir="$1"; shift
  local extra="$*"
  local ngpu
  ngpu="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
  if [ "$ngpu" -ge 2 ]; then
    echo "[phase=p2_extract_base] capture base (GPU 0, concurrent)"
    echo "[phase=p2_extract_instruct] capture instruct (GPU 1, concurrent)"
    set +e
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1310_extract_store.py \
      --model base --data-dir "$data_dir" --equivalence-check --resume $extra \
      > "$LOG_DIR/issue-1310-extract-base.log" 2>&1 &
    local pb=$!
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue1310_extract_store.py \
      --model instruct --data-dir "$data_dir" --equivalence-check --resume $extra \
      > "$LOG_DIR/issue-1310-extract-instruct.log" 2>&1 &
    local pi=$!
    wait "$pb"; local rc_b=$?
    wait "$pi"; local rc_i=$?
    set -e
    tail -5 "$LOG_DIR/issue-1310-extract-base.log" "$LOG_DIR/issue-1310-extract-instruct.log" || true
    if [ "$rc_b" -ne 0 ] || [ "$rc_i" -ne 0 ]; then
      echo "[i1310] extract failed (base rc=$rc_b instruct rc=$rc_i)" >&2; exit 1
    fi
  else
    for m in base instruct; do
      echo "[phase=p2_extract_$m] capture $m (single GPU / CPU)"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python \
        scripts/issue1310_extract_store.py --model "$m" --data-dir "$data_dir" \
        --equivalence-check --resume $extra
    done
  fi
}

run_pipeline() {
  # run_pipeline <data_dir> <out_dir> <fig_dir> <n_prompts> <gen_flags>
  #              <attr_flags> <extract_flags> <fit_flags>
  local data_dir="$1" out_dir="$2" fig_dir="$3" n_prompts="$4" \
        gen_flags="$5" attr_flags="$6" extract_flags="$7" fit_flags="$8"

  for m in base instruct; do
    echo "[phase=p1_gen_$m] story generation ($m)"
    uv run python scripts/issue1310_gen_stories.py --model "$m" --data-dir "$data_dir" \
      ${n_prompts:+--n-prompts "$n_prompts"} $gen_flags
    echo "[phase=p1_attr_$m] attribution + pair build ($m)"
    uv run python scripts/issue1310_attribute.py --model "$m" --data-dir "$data_dir" \
      --out-dir "$out_dir" $attr_flags
  done

  run_extract_both "$data_dir" $extract_flags

  echo "[phase=p3_fits] fit battery + swap + ceiling"
  uv run python scripts/issue1310_fit.py --data-dir "$data_dir" --out-dir "$out_dir" $fit_flags

  echo "[phase=p4_figures] figures"
  uv run python scripts/issue1310_figures.py --results-dir "$out_dir" --fig-dir "$fig_dir"
}

if [ "$MODE" = "smoke" ]; then
  # CPU-VM smoke: tiny random-init Qwen2 (real tokenizer) + stub generation.
  # Scratch dirs — committed eval_results/figures are never touched.
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-1310-smoke}"
  rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
  TINY="$SMOKE_ROOT/tiny_model"
  uv run python scripts/issue1310_extract_store.py --make-tiny-model "$TINY"
  run_pipeline "$SMOKE_ROOT/data" "$SMOKE_ROOT/eval" "$SMOKE_ROOT/figs" 6 \
    "--stub-gen --skip-upload" "--mock-judge --audit-n 8" \
    "--tiny-model-dir $TINY --batch-size 2" "--smoke --null-draws 3 --n-boot 50"
  echo "[i1310] smoke complete under $SMOKE_ROOT"
  exit 0
fi

# ---------------- production ----------------
run_pipeline "data/issue_${ISSUE}" "eval_results/issue_${ISSUE}" \
  "figures/issue_${ISSUE}" "" "" "--audit-n 200" "--batch-size 8" ""

echo "[phase=p5_upload] uploads (text/JSON unconditional; tensors batched)"
uv run python - "$ISSUE" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate import hub

issue = int(sys.argv[1])
prefix = "issue1310_char_map"
repo = "superkaiba1/explore-persona-space-data"
data = Path(f"data/issue_{issue}")

# pairs + eval JSONs (incl. attribution audits) — text path, one folder commit.
if (data / "pairs").is_dir():
    hub._upload(
        data / "pairs",
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=f"{prefix}/raw_completions/pairs",
    )
hub._upload(
    Path(f"eval_results/issue_{issue}"),
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/eval_results",
)
# Span-summary stores (batched folder commit per model).
for m in ("base", "instruct"):
    d = data / "store" / m
    if d.is_dir():
        hub._upload(
            d,
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=f"{prefix}/analysis_tensors/store_{m}",
        )
print("[i1310-p5] uploads complete")
PY

echo "[phase=p5_git] committing eval JSONs + figures to the issue branch"
git add "eval_results/issue_${ISSUE}" "figures/issue_${ISSUE}" || true
if git commit -m "task #${ISSUE}: eval results + figures (pod run)" >/dev/null 2>&1; then
  # #1205: verify the push landed (never swallow); retry once, then fail loud.
  git push origin "issue-${ISSUE}"
  behind="$(git rev-list --count "origin/issue-${ISSUE}..HEAD")"
  if [ "$behind" != "0" ]; then
    git push origin "issue-${ISSUE}"
    behind="$(git rev-list --count "origin/issue-${ISSUE}..HEAD")"
    [ "$behind" = "0" ] || { echo "[i1310] FATAL: results push did not land ($behind ahead)" >&2; exit 87; }
  fi
  echo "[i1310] results push verified (0 ahead of origin/issue-${ISSUE})"
else
  echo "[i1310] nothing to commit"
fi

echo "[phase=p6_sentinel] writing results sentinel"
uv run python - "$ISSUE" "$LOG_DIR" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

issue, log_dir = int(sys.argv[1]), Path(sys.argv[2])
out = Path(f"eval_results/issue_{issue}")
run_start = int(os.environ.get("RUN_START_EPOCH", "0"))
gpu_hours = round((time.time() - run_start) / 3600.0, 3) if run_start else None
summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}

pp = summary.get("per_persona", {})
r2_headline = {
    persona: {m: (entry.get("r2_headline") if entry else None) for m, entry in models.items()}
    for persona, models in pp.items()
}
swap = {
    m: (v or {}).get("delta_r2_char") for m, v in (summary.get("swap_specificity") or {}).items()
}
eval_numbers = {
    "per_persona_r2_headline": r2_headline,
    "swap_delta_r2_char": swap,
    "assistant_ceiling": summary.get("assistant_ceiling"),
    "attribution": {
        m: (a or {}).get("attribution_precision") for m, a in (summary.get("attribution") or {}).items()
    },
}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out.glob("*.json")),
    "reproducibility_card": {
        "wandb_url": "n/a - no training in this task (no WandB runs)",
        "hf_hub_url": (
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
            "tree/main/issue1310_char_map"
        ),
        "worktree_path": ".claude/worktrees/issue-1310",
        "final_commit_sha": commit,
        "gpu_hours_used": gpu_hours,
        "gpu_hours_budgeted": 3,
        "plan_deviations": [],
    },
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": issue,
    "by": "issue1310_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
path = log_dir / f"issue-{issue}-epm_results-{int(time.time())}.json"
path.write_text(json.dumps(sentinel, indent=2))
print(f"[i1310-p6] sentinel written: {path}")
PY

echo "[phase=done]"
