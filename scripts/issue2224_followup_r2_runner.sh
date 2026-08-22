#!/bin/bash
# issue #2224 — follow-up round 2 (POD-side, detached; proposer-9b-cheap):
# seed-137 replication train+gen on the 18 DECIDING cells only.
# {exact_dp__top, prompttoken_dp__top, random__shared} x {lmsys, ultrachat}
# x {evil, hallucination, sycophancy}. Single-knob replication: --seed 137
# plus isolation routing (--out-root ..._seed137, --hf-prefix-suffix _seed137,
# seed-137 eval-questions dir). Seed-42 artifacts are never read or written
# (fresh out-root; suffixed HF prefixes; --cells runs never emit the parent's
# .done_4b4/.done_4b5 sentinels by driver design).
#
# Phases: stage (train mixes + pre-generated seed-137 panel from HF)
#   -> train (18 LoRA cells, CVD fan-out across all GPUs)
#   -> eval  (vLLM multi-LoRA gen; 6 base pseudo-cells auto-ride-along)
#   -> upload (bulk postft_eval + eval questions to the _seed137 prefixes).
# The judge chain runs VM-SIDE after harvest (API-bound, off-pod) — commands
# in the round report; NOT here (#664 idle-GPU class).
#
# Launch (orchestrator, detached; pod intent lora-7b x4 H100):
#   ssh <pod> 'cd /workspace/explore-persona-space && \
#     setsid nohup bash scripts/issue2224_followup_r2_runner.sh \
#       > /workspace/logs/issue-2224-fu-r2-round.log 2>&1 < /dev/null & \
#     echo $! > /workspace/logs/issue-2224-fu-r2-round.pid'
#
# Pod-side: NO shared-VM thread-cap prefix (dedicated GPUs keep full width);
# NEVER shells out to scripts/task.py (sentinel JSON is the poller interface).
set -uo pipefail
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
REPO="${EPS_POD_REPO:-/workspace/explore-persona-space}"
cd "$REPO" || exit 1
set -a; [ -f .env ] && source .env; set +a

CELLS="lmsys__evil__exact_dp__top,lmsys__evil__prompttoken_dp__top,lmsys__evil__random__shared,\
lmsys__hallucination__exact_dp__top,lmsys__hallucination__prompttoken_dp__top,lmsys__hallucination__random__shared,\
lmsys__sycophancy__exact_dp__top,lmsys__sycophancy__prompttoken_dp__top,lmsys__sycophancy__random__shared,\
ultrachat__evil__exact_dp__top,ultrachat__evil__prompttoken_dp__top,ultrachat__evil__random__shared,\
ultrachat__hallucination__exact_dp__top,ultrachat__hallucination__prompttoken_dp__top,ultrachat__hallucination__random__shared,\
ultrachat__sycophancy__exact_dp__top,ultrachat__sycophancy__prompttoken_dp__top,ultrachat__sycophancy__random__shared"
SEED=137
OUT_ROOT="data/issue_2224/screening_ft_seed137"
EVAL_Q_DIR="data/issue_2224/eval_questions_seed137"
SUFFIX="_seed137"

LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
SENTINEL="$LOGDIR/issue-2224-fu-r2-round.json"
PHASES_FILE="$LOGDIR/issue-2224-fu-r2-phases.jsonl"
: > "$PHASES_FILE"
# Clear any STALE prior-run sentinel at launch (stale-artifact false-DONE
# class, #779/#825; CLAUDE.md § Monitoring).
rm -f "$SENTINEL"
CURRENT_PHASE=init
export SENTINEL PHASES_FILE

write_sentinel() {
  local rc=$?
  ISSUE_RC="$rc" CURRENT_PHASE="$CURRENT_PHASE" uv run python - <<'PY'
import json, os
phases = []
p = os.environ["PHASES_FILE"]
if os.path.exists(p):
    for line in open(p):
        line = line.strip()
        if line:
            phases.append(json.loads(line))
rc = int(os.environ["ISSUE_RC"])
failing = next((ph["phase"] for ph in phases if ph["rc"] != 0), None)
out = {
    "issue": 2224,
    "round": "followup-r2",
    "status": "done" if rc == 0 and failing is None else "failed",
    "rc": rc,
    "failing_phase": failing,
    "current_phase": os.environ["CURRENT_PHASE"],
    "phases": phases,
}
with open(os.environ["SENTINEL"], "w") as f:
    json.dump(out, f, indent=1)
print("sentinel written:", os.environ["SENTINEL"])
PY
}
trap write_sentinel EXIT

# Per-worker pattern (workflow_lint --check-phase-done-reserved): each child's
# stdout redirects to its OWN log ON the invocation line — the sweep script
# emits `[phase=done]` terminals that must never reach this dispatcher's main
# log (poll false-done class, #545/#920). phase_start/phase_end keep the
# suite4a bookkeeping (main-log progress lines + phases.jsonl + fail-fast).
phase_start() { CURRENT_PHASE="$1"; echo "[phase=$1] start $(date -u +%FT%TZ)"; }
phase_end() {
  local name="$1" rc="$2"
  echo "{\"phase\": \"$name\", \"rc\": $rc}" >> "$PHASES_FILE"
  echo "[phase=$name] rc=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    echo "[phase=$name] FAILED — tail:"; tail -15 "$LOGDIR/issue-2224-fu-r2-$name.log"
    exit "$rc"
  fi
}

# 1) Stage inputs: 18 train mixes (manifest-recorded absolute paths) + the
#    VM-generated seed-137 eval-question panel (local-first / HF-fallback).
phase_start stage
uv run python scripts/issue2224_followup_r2_stage.py \
  > "$LOGDIR/issue-2224-fu-r2-stage.log" 2>&1
phase_end stage $?

# 2) Train: 18 per-cell LoRA finetunes, CVD-sharded across all visible GPUs.
#    Adapters upload per cell to issue2224_screening/adapters_seed137/<cell>.
phase_start train
uv run python scripts/issue2224_finetune_sweep.py --phase train \
  --cells "$CELLS" --seed "$SEED" --out-root "$OUT_ROOT" --hf-prefix-suffix "$SUFFIX" \
  > "$LOGDIR/issue-2224-fu-r2-train.log" 2>&1
phase_end train $?

# 3) Eval gen: vLLM multi-LoRA over the seed-137 panel (base pseudo-cells for
#    the 6 (corpus, trait) blocks auto-ride-along per driver design).
phase_start eval
uv run python scripts/issue2224_finetune_sweep.py --phase eval \
  --cells "$CELLS" --seed "$SEED" --out-root "$OUT_ROOT" \
  --eval-questions-dir "$EVAL_Q_DIR" --hf-prefix-suffix "$SUFFIX" \
  > "$LOGDIR/issue-2224-fu-r2-eval.log" 2>&1
phase_end eval $?

# 4) Upload: ONE bulk commit each — postft_eval -> raw_completions/
#    postft_eval_seed137, questions -> eval_questions_seed137 (name-set verified).
phase_start upload
uv run python scripts/issue2224_finetune_sweep.py --phase upload \
  --out-root "$OUT_ROOT" --eval-questions-dir "$EVAL_Q_DIR" --hf-prefix-suffix "$SUFFIX" \
  > "$LOGDIR/issue-2224-fu-r2-upload.log" 2>&1
phase_end upload $?

# Judge chain (pilot gate -> Batch waves) runs VM-side after harvest; git
# commits of eval_results/ land in the orchestrator's completion turn.
echo "ROUND DONE $(date -u +%FT%TZ)"
