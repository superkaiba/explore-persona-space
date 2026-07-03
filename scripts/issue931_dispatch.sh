#!/usr/bin/env bash
# Issue #931 dispatcher: P0 pairs -> P1 gen+attr -> P2 capture (3 regimes)
# -> P2' chat-store staging (overlapped) -> P3 fits (+G1, G1b, MLP) -> P4
# similarity + figures -> uploads -> results sentinel -> [phase=done].
#
# ONE code path for smoke / canary / production — the tiny-cell subset
# threads through every phase via the size flags below (Step 6d.0
# PASS_CANARY: the pod-side canary re-runs the SAME chain with the REAL
# vLLM + bf16 7B model on 3 novels / 20 stories before production).
#
# Modes:
#   bash scripts/issue931_dispatch.sh --smoke    # CPU VM: tiny model + stub gen
#   bash scripts/issue931_dispatch.sh --canary-only
#   bash scripts/issue931_dispatch.sh            # production (canary first)
#
# Pod-side contract: [phase=...] log lines; the terminal [phase=done] is
# emitted ONLY here, after the final sentinel write. NEVER shells task.py.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
# GCE lane has NO .env (tokens ride the startup-script env) — conditional only.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export TQDM_DISABLE=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

ISSUE=931
MODE="production"
for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="smoke" ;;
    --canary-only) MODE="canary" ;;
    --no-canary) NO_CANARY=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done
NO_CANARY="${NO_CANARY:-0}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" || LOG_DIR="$(pwd)/logs"; mkdir -p "$LOG_DIR"

run_pipeline() {
  # run_pipeline <tag> <data_dir> <out_dir> <fig_dir> <max_novels> <n_prompts>
  #              <n_articles> <max_anchors> <gen_flags> <model_flags>
  #              <gate_flags> <chat_flags> <audit_flags> <chat_dir>
  local tag="$1" data_dir="$2" out_dir="$3" fig_dir="$4" max_novels="$5" \
        n_prompts="$6" n_articles="$7" max_anchors="$8" gen_flags="$9" \
        model_flags="${10}" gate_flags="${11}" chat_flags="${12}" \
        audit_flags="${13}" chat_dir="${14}"

  echo "[phase=p0_pairs] ($tag) building pairs"
  set +e
  uv run python scripts/issue931_build_pairs.py \
    --data-dir "$data_dir" --out-dir "$out_dir" \
    ${max_novels:+--max-novels "$max_novels"} \
    --n-articles "$n_articles" --max-anchors "$max_anchors" $gate_flags
  p0_rc=$?
  set -e
  if [ "$p0_rc" -ne 0 ]; then
    # HF `datasets` can SIGABRT (rc=134, PyGILState_Release) at interpreter
    # shutdown AFTER all outputs are written (gotchas.md) — route on the
    # ARTIFACTS, not the rc: tolerate ONLY when every P0 output exists.
    for f in pairs_armA.jsonl windows_armA.jsonl prompt_battery.json \
             pairs_meta.json pairs_armC.jsonl articles_armC.jsonl; do
      if [ ! -s "$data_dir/pairs/$f" ]; then
        echo "($tag) P0 FAILED rc=$p0_rc (missing $f)" >&2
        exit "$p0_rc"
      fi
    done
    echo "($tag) tolerated P0 rc=$p0_rc — artifacts complete (datasets finalize SIGABRT)"
  fi

  echo "[phase=p1_gen] ($tag) story generation"
  uv run python scripts/issue931_gen_stories.py \
    --battery "$data_dir/pairs/prompt_battery.json" --data-dir "$data_dir" \
    ${n_prompts:+--n-prompts "$n_prompts"} $gen_flags

  echo "[phase=p1_attr] ($tag) attribution + audit"
  uv run python scripts/issue931_attribute.py \
    --stories "$data_dir/stories/stories_seed42.jsonl" \
    --data-dir "$data_dir" --out-dir "$out_dir" $audit_flags

  # P2' chat-store staging (STAGE-ONLY — no GPU fit) overlaps the captures.
  echo "[phase=p2prime_chatstage] ($tag) chat-store staging (background)"
  ( uv run python scripts/issue931_fit_cells.py --cells chat_ref --stage-only \
      --data-dir "$data_dir" --out-dir "$out_dir" \
      --chat-store-dir "$chat_dir" $chat_flags \
      > "$LOG_DIR/issue-931-$tag-chatstage.log" 2>&1 \
      && echo "($tag) chat staging complete" ) &
  CHAT_STAGE_PID=$!

  for regime in armA armB armC; do
    echo "[phase=p2_extract_$(echo "$regime" | tr 'A-Z' 'a-z')] ($tag) capture $regime"
    uv run python scripts/issue931_extract_store.py --regime "$regime" \
      --data-dir "$data_dir" --equivalence-check --resume $model_flags
  done

  echo "[phase=p2prime_join] ($tag) waiting for chat staging"
  if ! wait "$CHAT_STAGE_PID"; then
    echo "($tag) WARNING: chat staging exited non-zero; artifact check follows" >&2
  fi
  ls "$chat_dir"/instruct_chat_s_shard*.pt >/dev/null

  echo "[phase=p3_fits] ($tag) fit battery + gates"
  uv run python scripts/issue931_fit_cells.py --cells all --mlp --g1b \
    --data-dir "$data_dir" --out-dir "$out_dir" --chat-store-dir "$chat_dir" $gate_flags

  echo "[phase=p4_similarity] ($tag) transfer matrix + subspace/CKA"
  uv run python scripts/issue931_similarity.py \
    --data-dir "$data_dir" --out-dir "$out_dir" --chat-store-dir "$chat_dir" \
    --save-maps $gate_flags

  echo "[phase=p4_figures] ($tag) figures"
  uv run python scripts/issue931_figures.py --results-dir "$out_dir" --fig-dir "$fig_dir"
}

if [ "$MODE" = "smoke" ]; then
  # CPU-VM smoke: tiny random-init Qwen2 (real tokenizer) + stub generation +
  # fabricated chat store. Scratch dirs — committed eval_results/figures are
  # never touched.
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-931-smoke}"
  rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
  TINY="$SMOKE_ROOT/tiny_model"
  uv run python scripts/issue931_extract_store.py --make-tiny-model "$TINY"
  run_pipeline smoke "$SMOKE_ROOT/data" "$SMOKE_ROOT/eval" "$SMOKE_ROOT/figs" \
    2 8 6 2 "--stub-gen --skip-upload" "--tiny-model-dir $TINY --batch-size 2" \
    "--smoke" "--fabricate-chat-smoke" "--mock-judge --audit-n 8" \
    "$SMOKE_ROOT/data/chat_store"
  echo "[i931] smoke complete under $SMOKE_ROOT"
  exit 0
fi

# Canary + production SHARE the staged chat store (one ~21 GB download).
CHAT_DIR="data/issue_${ISSUE}/chat_store"

if [ "$MODE" = "canary" ] || { [ "$MODE" = "production" ] && [ "$NO_CANARY" != "1" ]; }; then
  echo "[phase=canary] 3-novel/20-story real-path canary"
  CANARY_ROOT="data/issue_931_canary"
  rm -rf "$CANARY_ROOT" "eval_results/issue_${ISSUE}/canary"
  run_pipeline canary "$CANARY_ROOT" "eval_results/issue_${ISSUE}/canary" \
    "$CANARY_ROOT/figs" 3 20 20 3 "--skip-upload" "--batch-size 4" "--smoke" \
    "--stage-chat-store" "--audit-n 20" "$CHAT_DIR"
  echo "[i931] canary complete"
  if [ "$MODE" = "canary" ]; then exit 0; fi
fi

# ---------------- production ----------------
run_pipeline prod "data/issue_${ISSUE}" "eval_results/issue_${ISSUE}" \
  "figures/issue_${ISSUE}" "" "" 600 6 "" "--batch-size 8" "" \
  "--stage-chat-store" "--audit-n 200" "$CHAT_DIR"

echo "[phase=p5_upload] uploads (text/JSON unconditional; tensors batched)"
uv run python - "$ISSUE" <<'PY'
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate import hub

issue = int(sys.argv[1])
prefix = "issue931_story_map"
repo = "superkaiba1/explore-persona-space-data"
data = Path(f"data/issue_{issue}")

# pairs_meta + gates + eval JSONs (text path, one folder commit each).
hub._upload(
    data / "pairs",
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/raw_completions/pairs_meta",
)
hub._upload(
    Path(f"eval_results/issue_{issue}"),
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/eval_results",
)
# Span-summary stores + preds + full-map tensors (batched folder commits).
for sub in ("armA", "armB", "armC", "preds", "maps"):
    d = data / "store" / sub
    if d.is_dir():
        hub._upload(
            d,
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=f"{prefix}/analysis_tensors/{sub}",
        )
print("[i931-p5] uploads complete")
PY

echo "[phase=p5_git] committing eval JSONs + figures to the issue branch"
git add "eval_results/issue_${ISSUE}" "figures/issue_${ISSUE}" || true
if git commit -m "task #${ISSUE}: eval results + figures (pod run)" >/dev/null 2>&1; then
  git push origin "issue-${ISSUE}" || echo "[i931] WARNING: git push failed (HF copies uploaded)" >&2
else
  echo "[i931] nothing to commit"
fi

echo "[phase=p6_sentinel] writing results sentinel"
uv run python - "$ISSUE" "$LOG_DIR" <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

issue, log_dir = int(sys.argv[1]), Path(sys.argv[2])
out = Path(f"eval_results/issue_{issue}")


def _read(name):
    p = out / name
    return json.loads(p.read_text()) if p.exists() else None


def _l19(cell):
    d = _read(f"cells_{cell}.json")
    if not d:
        return None
    return d["r2_per_layer_obs"][d["headline_layer"]]


tm = _read("transfer_matrix.json") or {"rows": [], "headline_layer": 19}
primary = {
    f"{r['direction']}({r['x_recipe']})": round(r["fraction_of_ceiling"], 4)
    for r in tm["rows"]
    if r.get("power_matched") and r.get("application") == "recentered"
    and r.get("layer") == tm.get("headline_layer") and r.get("tier") == "primary"
}
eval_numbers = {
    "r2_L19": {c: _l19(c) for c in (
        "armA_within", "armA_within_lastpos", "armB_within", "armB_within_lastpos",
        "armC_sep", "armC_prevmean", "chat_ref")},
    "transfer_fraction_primary_recentered_matched": primary,
    "delta_char": {a: (_read(f"delta_char_{a}.json") or {}).get("delta_r2_char")
                   for a in ("armA", "armB")},
    "g1_gate": (_read("g1_gate_931.json") or {}).get("pass"),
    "g1b_parity": (_read("mlp_parity.json") or {}).get("abs_delta"),
    "attribution_precision": ((_read("attribution_audit.json") or {}).get("audit") or {}).get(
        "precision"),
}
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out.glob("*.json")),
    "reproducibility_card": {
        "wandb_url": "n/a - no training in this task (no WandB runs)",
        "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue931_story_map",
        "worktree_path": ".claude/worktrees/issue-931",
        "final_commit_sha": commit,
        "gpu_hours_used": None,
        "gpu_hours_budgeted": 6,
        "plan_deviations": [],
    },
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": issue,
    "by": "issue931_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
path = log_dir / f"issue-{issue}-epm_results-{int(time.time())}.json"
path.write_text(json.dumps(sentinel, indent=2))
print(f"[i931-p6] sentinel written: {path}")
PY

echo "[phase=done]"
