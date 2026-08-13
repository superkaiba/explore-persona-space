#!/bin/bash
# issue #2224 — follow-up round 2 VM-side judge chain (API-bound, off-pod):
# harvest seed-137 generations -> judge-pilot (rule-26 gate) -> judge (Batch
# waves) -> seed-comparison. Runs AFTER the pod runner
# (issue2224_followup_r2_runner.sh) completes its upload phase.
#
# EVERY seed-isolation flag is EXPLICIT here (fu-r2 review r2 blocker: with
# --seed 137 and any parent-default dir, judged_current sha-mismatches and the
# judge OVERWRITES the parent's committed selection_finetune/<cid>/
# trait_scores.json, and _check_pilot is satisfied by the PARENT's passing
# pilot reports). The sweep's assert_seed_isolation guard refuses those
# invocations mechanically; this runner pins the correct ones.
#
# Launch (VM, detached):
#   cd <WT> && mkdir -p data/issue_2224/judge_fu_r2 && \
#     setsid nohup bash scripts/issue2224_followup_r2_judge.sh \
#       > data/issue_2224/judge_fu_r2/round.log 2>&1 < /dev/null & \
#     echo $! > data/issue_2224/judge_fu_r2/round.pid
set -uo pipefail
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
WT=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2224
# CROSS-WORKTREE dependency: the persona_vectors clone (rubric source —
# issue778_lib.load_trait_data) lives in issue-816's worktree, same as the
# parent judge_ft/runner.sh. The rubric sha lands in each seed-137
# trait_scores.json and the compare script asserts cross-seed rubric identity,
# so a drifted clone fails loud there rather than silently re-instrumenting.
PV=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-816/external/persona_vectors
cd "$WT" || exit 1
set -a; [ -f .env ] && source .env; [ -f ../../../.env ] && source ../../../.env; set +a

CELLS="lmsys__evil__exact_dp__top,lmsys__evil__prompttoken_dp__top,lmsys__evil__random__shared,\
lmsys__hallucination__exact_dp__top,lmsys__hallucination__prompttoken_dp__top,lmsys__hallucination__random__shared,\
lmsys__sycophancy__exact_dp__top,lmsys__sycophancy__prompttoken_dp__top,lmsys__sycophancy__random__shared,\
ultrachat__evil__exact_dp__top,ultrachat__evil__prompttoken_dp__top,ultrachat__evil__random__shared,\
ultrachat__hallucination__exact_dp__top,ultrachat__hallucination__prompttoken_dp__top,ultrachat__hallucination__random__shared,\
ultrachat__sycophancy__exact_dp__top,ultrachat__sycophancy__prompttoken_dp__top,ultrachat__sycophancy__random__shared"
SEED=137
OUT_ROOT="data/issue_2224/screening_ft_seed137"
EVAL_Q_DIR="data/issue_2224/eval_questions_seed137"
JUDGE_ROOT="data/issue_2224/judge_postft_seed137"
SCORES_DIR="eval_results/issue_2224/followup_r2/selection_finetune_seed137"
PILOT_DIR="eval_results/issue_2224/followup_r2/judge_pilots"
# Rule-26(b) waivers: NONE pre-granted for the seed-137 pilots. If the pilot
# trips on an explained content-drop, append documented
# `--waive-parse-fail <wave>:<arm>` flags to the judge_pilot invocation below
# (the parent needed 3; each waiver's explanation goes in the dispatch marker).

LOGDIR="$WT/data/issue_2224/judge_fu_r2"
mkdir -p "$LOGDIR"
SENTINEL="$LOGDIR/round-sentinel.json"
PHASES_FILE="$LOGDIR/phases.jsonl"
: > "$PHASES_FILE"
# Clear any STALE prior-run sentinel at launch (stale-artifact false-DONE class).
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
    "round": "fu-r2-judge",
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

# Per-worker pattern (workflow_lint --check-phase-done-reserved): every child's
# stdout redirects to its OWN log ON the invocation line — the sweep emits
# `[phase=done]` terminals that must never reach this dispatcher's main log.
# Shared-VM thread caps on every python phase (#847; judge is API-bound but the
# reduce/aggregation steps run numpy on this shared box).
CAPS=(env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
      NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2)
phase_start() { CURRENT_PHASE="$1"; echo "[phase=$1] start $(date -u +%FT%TZ)"; }
phase_end() {
  local name="$1" rc="$2"
  echo "{\"phase\": \"$name\", \"rc\": $rc}" >> "$PHASES_FILE"
  echo "[phase=$name] rc=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    echo "[phase=$name] FAILED — tail:"; tail -15 "$LOGDIR/phase-$name.log"
    exit "$rc"
  fi
}

# 1) Harvest the pod-uploaded seed-137 generations from HF into the isolated
#    out-root (idempotent; fail-loud when the pod upload has not landed).
phase_start harvest
"${CAPS[@]}" uv run python scripts/issue2224_followup_r2_stage.py --harvest-postft \
  --out-root "$OUT_ROOT" > "$LOGDIR/phase-harvest.log" 2>&1
phase_end harvest $?

# 2) Pilot gate (rule 26; ~45k-call trait waves + coherence wave => REQUIRED).
phase_start judge_pilot
"${CAPS[@]}" uv run python scripts/issue2224_finetune_sweep.py --phase judge-pilot \
  --seed "$SEED" --cells "$CELLS" --pv-root "$PV" \
  --out-root "$OUT_ROOT" --eval-questions-dir "$EVAL_Q_DIR" \
  --judge-root "$JUDGE_ROOT" --trait-scores-dir "$SCORES_DIR" \
  --pilot-report-dir "$PILOT_DIR" > "$LOGDIR/phase-judge_pilot.log" 2>&1
phase_end judge_pilot $?

# 3) Production judge (Batch waves; per-cell trait_scores.json checkpoint/resume).
# --judge-threshold-base 1 FORCES the Batch API on every production wave: the
# tier-scaled default routed 2,500-draw waves SYNC on this org, and the ~50 min
# of sync exposure on the shared VM got the phase earlyoom-SIGTERM'd (rc=143).
# The pilot phase above deliberately keeps the default (sync at ~100-200 draws).
phase_start judge
"${CAPS[@]}" uv run python scripts/issue2224_finetune_sweep.py --phase judge \
  --seed "$SEED" --cells "$CELLS" --pv-root "$PV" \
  --out-root "$OUT_ROOT" --eval-questions-dir "$EVAL_Q_DIR" \
  --judge-root "$JUDGE_ROOT" --trait-scores-dir "$SCORES_DIR" \
  --pilot-report-dir "$PILOT_DIR" --judge-threshold-base 1 \
  > "$LOGDIR/phase-judge.log" 2>&1
phase_end judge $?

# 4) Seed-137 vs seed-42 deciding-contrast comparison (parent machinery verbatim).
phase_start compare
"${CAPS[@]}" uv run python scripts/issue2224_followup_r2_compare.py \
  --seed137-scores-dir "$SCORES_DIR" > "$LOGDIR/phase-compare.log" 2>&1
phase_end compare $?

# Git commits of eval_results/ outputs land in the orchestrator's completion turn.
echo "ROUND DONE $(date -u +%FT%TZ)"
