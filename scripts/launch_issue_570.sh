#!/bin/bash
# ── Issue #570 top-level launcher glue (committed copy, round 4) ─────────────
# Provenance: byte-faithful to the experimenter-composed pod launcher
# pod-570:/workspace/launch_issue_570.sh (md5 2e844c45c8034bc315fbe991be369bd3,
# fetched 2026-06-11 — the launcher that ran the round-4 incident run), now
# committed so the glue is reproducible. Deltas vs the pod copy (4): this
# header, the idempotent Step-0.5 precheck skip, the SECONDS resume note, and
# the terminal [phase=done] emission — each marked "round-4" below.
#
# Round-4 resume semantics (after the rescue-coverage-assert incident): this
# glue is IDEMPOTENT end to end — a relaunch re-walks every phase in order and
# each phase no-ops on its own completed artifacts:
#   - b-hat: skipped when data/issue543_ratio_survival/bhat.json exists (below);
#   - phase-1 cells: run_issue543_ratio.py returns early on an existing
#     phase1_result.json, and (round 4) RESUMES from a completed-but-uncommitted
#     training (final adapter + callback_stop_record.json present, e.g. after a
#     downstream coverage-assert crash) instead of retraining;
#   - ladders: eval_issue570_ladder.py skips a seed/variant whose
#     phase1_ladder.json + phase1_pick_record.json already exist (--force to
#     re-ladder);
#   - G1' verdicts: CPU, recomputed from the pick records (idempotent overwrite);
#   - phase-2 / evals / absorption / alignment: per-cell result-JSON idempotency
#     in the reviewed scripts.
# The coverage assert itself is the branch-aware shared form
# (_issue543_common.assert_ladder_coverage_steps): lowest retained step <=
# max(25, stop - 60) — valid on both the 5e-6 ramp (stop ~95-110) and the
# pre-registered lr 2e-6 rescue ramp (stop ~195).
#
# Sequencing per plan v2 §4.1/§7 G1′ (v2 ordering): ALL 3 seeds Phase-1 @ 5e-6
# in parallel (GPUs 0/1/2) → 3 ladders → G1′ verdict (→ registered rescue once
# if ≥2/3 seeds lack an eligible checkpoint) → pre-SFT evals → 6 Phase-2 cells
# → post-SFT evals → absorption (2 arms) → Betley/ARC grid → results sentinel.
# All science logic lives in the reviewed scripts on branch issue-570;
# this file only sequences documented CLIs.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && . ./.env; set +a
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export EPM_OUTPUT_ROOT=/tmp/issue_570_results
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline
LOGD=/workspace/logs
mkdir -p "$LOGD" "$EPM_OUTPUT_ROOT"
echo $$ > "$LOGD/issue-570-run.pid"
SECONDS=0
WORKTREE_PATH="/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-570"
SEEDS=(42 137 256)
log() { echo "[$(date -u +%FT%TZ)] [glue] $*"; }
trap 'log "glue exiting rc=$?"' EXIT
log "launcher start HEAD=$(git rev-parse --short HEAD) pid=$$"

# ── Step 0.5 pre-probe: NON-GATING, GPU 3, concurrent with Phase 1 ──────────
# round-4: idempotent — skipped when the preprobe outputs already exist.
if [ -d eval_results/issue_570/preprobe/seed256 ]; then
  log "[phase=step05] hub-precheck outputs exist — skipping (resume)"
  echo OK > "$LOGD/issue-570-step05.status"
  STEP05_PID=""
else
  log "[phase=step05] hub-precheck start (GPU 3, non-gating)"
  ( uv run python scripts/eval_issue570_ladder.py --hub-precheck --gpu 3 \
      > "$LOGD/issue-570-step05-precheck.log" 2>&1 \
      && echo OK > "$LOGD/issue-570-step05.status" \
      || echo FAIL > "$LOGD/issue-570-step05.status" ) &
  STEP05_PID=$!
fi

# ── Step 0: b-hat measure (the #543 driver runs this automatically; the #570
#    per-phase invocation path must run it explicitly — plan §7 Step-0 check).
#    Idempotent: skipped when bhat.json already exists. ────────────────────────
if [ ! -f data/issue543_ratio_survival/bhat.json ]; then
  log "[phase=measure_bhat] b-hat measure (GPU 0)"
  uv run python scripts/run_issue543_ratio.py --measure-bhat --issue-ns 570 --gpu 0 \
    > "$LOGD/issue-570-bhat.log" 2>&1 || { log "FATAL: measure-bhat failed"; exit 2; }
fi
log "b-hat record: $(head -c 300 data/issue543_ratio_survival/bhat.json 2>/dev/null || echo MISSING)"

phase1_round() {  # $1 = label, rest = extra args
  local label="$1"; shift
  local pids=() rc=0 i S
  for i in 0 1 2; do
    S=${SEEDS[$i]}
    log "[phase=phase1] label=$label seed=$S gpu=$i start"
    uv run python scripts/run_issue543_ratio.py --arm r50 --seed "$S" --phase phase1 \
      --issue-ns 570 --phase1-save-limit 40 --gpu "$i" "$@" \
      > "$LOGD/issue-570-p1-${label}-s${S}.log" 2>&1 &
    pids+=($!)
  done
  for i in 0 1 2; do
    if ! wait "${pids[$i]}"; then
      log "phase1[$label] seed=${SEEDS[$i]} FAILED — see issue-570-p1-${label}-s${SEEDS[$i]}.log"
      rc=1
    fi
  done
  return $rc
}

ladder_round() {  # $1 = label, rest = extra args
  local label="$1"; shift
  local pids=() rc=0 i S
  for i in 0 1 2; do
    S=${SEEDS[$i]}
    log "[phase=ladder] label=$label seed=$S gpu=$i start"
    uv run python scripts/eval_issue570_ladder.py --seed "$S" --gpu "$i" "$@" \
      > "$LOGD/issue-570-ladder-${label}-s${S}.log" 2>&1 &
    pids+=($!)
  done
  for i in 0 1 2; do
    if ! wait "${pids[$i]}"; then
      log "ladder[$label] seed=${SEEDS[$i]} FAILED — see issue-570-ladder-${label}-s${SEEDS[$i]}.log"
      rc=1
    fi
  done
  return $rc
}

log "[phase=phase1] all 3 seeds @ lr 5e-6 (registered ramp)"
phase1_round 5e6 --phase1-save-steps 5 || { log "FATAL: phase1@5e-6 failed"; exit 2; }
log "[phase=ladder] all 3 seeds @ 5e-6"
ladder_round 5e6 || { log "FATAL: ladder@5e-6 failed"; exit 2; }

[ -n "$STEP05_PID" ] && { wait "$STEP05_PID" || true; }
log "step0.5 status: $(cat "$LOGD/issue-570-step05.status" 2>/dev/null || echo unknown) (non-gating)"

# ── G1′ verdict (CPU; refuses partial seed sets) ─────────────────────────────
log "[phase=gate_verdict] G1-prime verdict"
uv run python scripts/eval_issue570_ladder.py --g1-verdict \
  > "$LOGD/issue-570-g1-verdict.log" 2>&1 \
  || { log "FATAL: g1-verdict failed"; exit 2; }
VERDICT=$(uv run python -c "import json; print(json.load(open('eval_results/issue_570/g1_verdict.json'))['verdict'])")
log "G1-prime verdict: $VERDICT"

IV=""
if [ "$VERDICT" = "rescue" ]; then
  IV="rescue_lr2e6"
  log "[phase=rescue] registered ONE all-seed rescue: lr 2e-6, save_steps 3, install-variant $IV"
  phase1_round rescue --phase1-save-steps 3 --phase1-lr 2e-6 --install-variant "$IV" \
    || { log "FATAL: rescue phase1 failed"; exit 2; }
  ladder_round rescue --install-variant "$IV" \
    || { log "FATAL: rescue ladder failed"; exit 2; }
  uv run python scripts/eval_issue570_ladder.py --g1-verdict --install-variant "$IV" \
    > "$LOGD/issue-570-g1-verdict-rescue.log" 2>&1 \
    || { log "FATAL: rescue g1-verdict failed"; exit 2; }
  log "rescue verdict recorded; proceeding with picks/fallbacks as found (plan §7: no further phase-1 re-runs)"
fi
IVFLAG=()
[ -n "$IV" ] && IVFLAG=(--install-variant "$IV")
P1DIR="phase1${IV:+_$IV}"

# ── Resolve per-seed picked checkpoints from pick records ────────────────────
declare -A PICKED
INCLUDED=()
for S in "${SEEDS[@]}"; do
  REC="eval_results/issue_570/${P1DIR}/seed${S}/phase1_pick_record.json"
  D=$(REC_PATH="$REC" uv run python -c "import json,os; r=json.load(open(os.environ['REC_PATH'])); print(r.get('picked_local_dir') or '')" 2>/dev/null || echo "")
  if [ -n "$D" ] && [ -d "$D" ]; then
    PICKED[$S]="$D"; INCLUDED+=("$S")
    log "seed $S picked checkpoint: $D"
  else
    log "seed $S has NO usable pick (record=$REC) — excluded from downstream cells"
  fi
done
if [ "${#INCLUDED[@]}" -lt 2 ]; then
  log "FATAL: <2 seeds with usable picks (K2 territory, plan §7) — aborting before Phase 2"
  exit 2
fi
SEEDS_CSV=$(IFS=,; echo "${INCLUDED[*]}")
log "included seeds: $SEEDS_CSV"

# ── Pre-SFT full evals (parallel over included seeds) ───────────────────────
log "[phase=pre_eval] pre-SFT full evals on picked checkpoints"
pids=(); g=0; rc=0
for S in "${INCLUDED[@]}"; do
  uv run python scripts/eval_issue543.py --arm r50 --seed "$S" --phase phase1 --issue-ns 570 \
    --adapter-path "${PICKED[$S]}" "${IVFLAG[@]}" --gpu "$g" \
    > "$LOGD/issue-570-preeval-s${S}.log" 2>&1 &
  pids+=($!); g=$((g+1))
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] || { log "FATAL: a pre-SFT eval failed (see issue-570-preeval-s*.log)"; exit 2; }

# ── Phase 2: two erasure arms × included seeds (one arm per round) ───────────
phase2_arm() {  # $1 = arm, rest = extra args (corpus flag for org_em)
  local arm="$1"; shift
  local pids=() g=0 rc=0 S
  for S in "${INCLUDED[@]}"; do
    log "[phase=phase2] arm=$arm seed=$S gpu=$g start"
    uv run python scripts/run_issue543_ratio.py --arm r50 --seed "$S" --phase phase2 \
      --issue-ns 570 --variant "$arm" --phase2-lr 5e-6 \
      --phase2-start-adapter "${PICKED[$S]}" "${IVFLAG[@]}" --gpu "$g" "$@" \
      > "$LOGD/issue-570-p2-${arm}-s${S}.log" 2>&1 &
    pids+=($!); g=$((g+1))
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
log "[phase=phase2] aligned arm (org_benign, default good corpus = #557 parity)"
phase2_arm org_benign || { log "FATAL: phase2 org_benign failed"; exit 2; }
log "[phase=phase2] misaligned arm (org_em)"
phase2_arm org_em --phase2-corpus-hf-path issue376_em/v1/bad_medical_advice_6k.jsonl \
  || { log "FATAL: phase2 org_em failed"; exit 2; }

# ── Post-SFT full evals (one arm per round) ──────────────────────────────────
posteval_arm() {
  local arm="$1"
  local pids=() g=0 rc=0 S
  for S in "${INCLUDED[@]}"; do
    uv run python scripts/eval_issue543.py --arm r50 --seed "$S" --phase phase2 --issue-ns 570 \
      --variant "$arm" "${IVFLAG[@]}" --gpu "$g" \
      > "$LOGD/issue-570-posteval-${arm}-s${S}.log" 2>&1 &
    pids+=($!); g=$((g+1))
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}
log "[phase=post_eval] org_benign"
posteval_arm org_benign || { log "FATAL: post-eval org_benign failed"; exit 2; }
log "[phase=post_eval] org_em"
posteval_arm org_em || { log "FATAL: post-eval org_em failed"; exit 2; }

# ── Absorption guard per arm (manifest from pick records + phase2 results) ──
absorption_arm() {  # $1 = arm
  local arm="$1"
  local V="${arm}${IV:+_$IV}"
  local M="$LOGD/issue-570-absorb-manifest-${V}.json"
  ARM_ENV="$arm" V_ENV="$V" M_ENV="$M" P1DIR_ENV="$P1DIR" SEEDS_ENV="$SEEDS_CSV" \
    uv run python - <<'PYEOF'
import json, os
v, out, p1dir = os.environ["V_ENV"], os.environ["M_ENV"], os.environ["P1DIR_ENV"]
seeds = [int(s) for s in os.environ["SEEDS_ENV"].split(",")]
sets = [{"name": "base", "kind": "base", "adapter_source": None}]
for s in seeds:
    rec = json.load(open(f"eval_results/issue_570/{p1dir}/seed{s}/phase1_pick_record.json"))
    sets.append({"name": f"pre_seed{s}", "kind": "pre", "seed": s,
                 "adapter_source": {"local_path": rec["picked_local_dir"]}})
for s in seeds:
    rj = f"eval_results/issue_570/{v}/seed{s}/phase2_result.json"
    assert os.path.exists(rj), f"missing {rj}"
    sets.append({"name": f"post_{v}_seed{s}", "kind": "post", "variant": v, "seed": s,
                 "adapter_source": {"local_result_json": rj}})
json.dump(sets, open(out, "w"), indent=2)
print(f"absorption manifest {out}: {len(sets)} sets")
PYEOF
  local corpus=()
  [ "$arm" = "org_em" ] && corpus=(--corpus-hf-path issue376_em/v1/bad_medical_advice_6k.jsonl)
  uv run python scripts/probe_issue557_absorption.py --issue-ns 570 \
    --variants "$V" --seeds "$SEEDS_CSV" --adapter-set-manifest "$M" \
    "${corpus[@]}" --gpu 0 > "$LOGD/issue-570-absorb-${V}.log" 2>&1
}
gpu_settle() {  # round-4 hotfix: wait out vLLM worker teardown before next engine init
  local t=0
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ "$t" -lt 120 ]; do
    sleep 5; t=$((t+5))
  done
  log "gpu_settle: compute-apps drained after ${t}s (cap 120s); settling 15s more"
  sleep 15
}
log "[phase=absorption] org_benign"
absorption_arm org_benign || { log "FATAL: absorption org_benign failed"; exit 2; }
gpu_settle
log "[phase=absorption] org_em"
absorption_arm org_em || { log "FATAL: absorption org_em failed"; exit 2; }
gpu_settle

# ── Betley + ARC-C manipulation check (sequential merge→eval→delete) ─────────
log "[phase=alignment] Betley + ARC grid (--default-grid)"
# round-5 hotfix: offline env — every grid artifact is local/cached (base
# tokenizer+model cached, adapters local, ARC-C in-git; Betley judge is the
# Anthropic API, unaffected). Kills the is_base_mistral Hub probe that 429'd
# run 4 (org-wide HF rate limit fires even on fully-cached tokenizer loads).
# Offline cache-resolution verified on pod-570 pre-launch (OFFLINE_OK).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/eval_issue570_alignment.py --default-grid --gpu 0 \
  > "$LOGD/issue-570-alignment.log" 2>&1 \
  || { log "FATAL: alignment grid failed"; exit 2; }

# ── Step-7 results sentinel ──────────────────────────────────────────────────
# NOTE (round-4 resume): SECONDS counts THIS launcher invocation only. On a
# resume relaunch the experimenter must add the prior invocation's realized
# wall-hours (from the prior issue-570-run.log timestamps) to gpu_hours_used —
# either by editing GPU_HOURS here before redeploy or via a --plan-deviation
# note — so the Step-7 contract reflects cumulative pod-GPU-hours.
# round-4 resume: +6.5 pod-GPU-h realized across runs 1-3 (per the NOTE above)
# round-5 resume: +2.0 pod-GPU-h realized in run 4 (08:13:54Z->08:43:20Z = 0.49h x 4 GPUs) -> 8.5
GPU_HOURS=$(awk -v s="$SECONDS" 'BEGIN{printf "%.2f", s/3600*4 + 8.5}')
DEV=()
[ -n "$IV" ] && DEV+=(--plan-deviation "G1-prime registered rescue fired (install-variant rescue_lr2e6) :: >=2/3 seeds lacked an eligible clean-form checkpoint at 5e-6")
log "[phase=rollup] results sentinel (gpu_hours_used=$GPU_HOURS pod-GPU-hours)"
uv run python scripts/run_issue543_ratio.py --results-sentinel --issue-ns 570 \
  --gpu-hours-used "$GPU_HOURS" --gpu-hours-budgeted 17.0 \
  --worktree-path "$WORKTREE_PATH" \
  "${DEV[@]}" > "$LOGD/issue-570-results-sentinel.log" 2>&1 \
  || { log "FATAL: results sentinel failed"; exit 2; }
# round-4: terminal [phase=done] in THIS (main) log — poll_pipeline tails
# issue-570-run.log and declares done only when the most recent [phase=...]
# token is done; --results-sentinel's own [phase=done] lands in its redirected
# log, so without this line a graceful completion reads as dead.
log "[phase=done]"
log "DONE — wall $(awk -v s="$SECONDS" 'BEGIN{printf "%.2f h", s/3600}'), included seeds: $SEEDS_CSV, install_variant: ${IV:-none}"
