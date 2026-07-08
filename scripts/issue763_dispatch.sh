#!/usr/bin/env bash
# Issue #763 dispatch — the --workload-cmd driver (GCP / RunPod lane).
#
# UNIFIED end-to-end: this script runs the SAME python entrypoint chain in both
# real and --smoke modes; --smoke threads through to each phase (tiny slice:
# 1 behavior x 3 contexts x 5 probes, CPU, no API/HF). One launch covers ALL
# phases (build pools -> source-side baseline -> generate -> capture ->
# PV extract -> judge -> fit -> figures -> upload). No separate smoke/sweep
# code path (the PASS_UNIFIED smoke-architecture contract).
#
#   bash scripts/issue763_dispatch.sh                # full run (5 behaviors x 50 ctx)
#   bash scripts/issue763_dispatch.sh --smoke        # tiny offline verification (no pod/GPU/API)
#
# Phases:
#   build_pools         author + freeze + HF-upload the 5 eliciting pools (CPU/API)
#   source_side_baseline base-model propensity read (predicts low_dynamic_range; CPU/API)
#   generate            on-policy completions per (context x probe)            (GPU/vLLM)
#   capture             matched-probe teacher-forced v0(C,B) at all 28 layers  (GPU)
#   pv_extract          faithful persona-vector r_B (baseline arm)             (GPU/API)
#   judge               E0(C,B) via Sonnet (rubrics verbatim) + structural fmt (CPU/API)
#   fit                 GLM (primary) / ridge / PV LOCO + nulls + ceiling      (CPU)
#   figures             the §6 hero grid + exploratory dump                    (CPU)
#   upload              raw completions + v0/r_B analysis tensors -> HF        (CPU, real only)

set -euo pipefail

# REPO_ROOT defaults to the GCP workload root or the RunPod path; honored if
# pre-exported by the dispatch (#641: GCE startup exports it). The bash
# per-command `REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue763_dispatch.sh`
# form supersedes for that one command (belt-and-suspenders).
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [ ! -d "$REPO_ROOT/scripts" ]; then
  # Fall back to the dir this script lives in (local VM smoke).
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

SMOKE=""
# FROM_PHASE: "" = phase-1 (gate exit); "pv_capture" = GPU resume after the
# off-pod PV judge; "fit" = fit-only resume — E0 + PV inputs are DONE + on HF, so
# skip straight to the vectorized fit -> figures -> upload. Fit device defaults
# to CPU (task #763 r6 USER OVERRIDE: the prior serial fit idled a 1xH100 ~16h);
# EPM_FIT_DEVICE=cuda routes the batched fits to GPU (PM cutover directive
# 2026-07-02) — the vectorization is unchanged either way.
FROM_PHASE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke" ;;
    --from-phase)
      shift
      FROM_PHASE="${1:-}"
      ;;
    --from-phase=*) FROM_PHASE="${1#--from-phase=}" ;;
  esac
  shift || true
done
if [ -n "$FROM_PHASE" ] && [ "$FROM_PHASE" != "pv_capture" ] && [ "$FROM_PHASE" != "fit" ]; then
  echo "[issue763.dispatch] FATAL: unknown --from-phase '$FROM_PHASE' (only 'pv_capture' | 'fit')" >&2
  exit 2
fi

# Load credentials for the judge / HF (set -a && source .env && set +a per
# research-project-structure.md — never bare load_dotenv in a heredoc).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[issue763.dispatch] REPO_ROOT=$REPO_ROOT SMOKE='${SMOKE}' commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --smoke = 1 behavior x 3 contexts x 5 probes, fully offline (CPU, no API/HF).
# The real run = 5 behaviors x 50 contexts x ~60 probes on the GPU.
if [ -n "$SMOKE" ]; then
  SMOKE_MODEL="${EPM_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  BEH="deception"

  echo "[phase=build_pools]"
  uv run python scripts/issue763_build_probe_pools.py --smoke --behaviors "$BEH"

  echo "[phase=source_side_baseline]"
  uv run python scripts/issue763_source_side_baseline.py --smoke --behaviors "$BEH" \
    --no-vllm --mock-judge --model-name "$SMOKE_MODEL"

  echo "[phase=generate]"
  uv run python scripts/issue763_generate_completions.py --smoke --behaviors "$BEH" \
    --n-contexts 3 --no-vllm --model-name "$SMOKE_MODEL"

  echo "[phase=capture]"
  uv run python scripts/issue763_capture_v0_matched.py --smoke --behaviors "$BEH" \
    --device cpu --model-name "$SMOKE_MODEL" --batch-size 4 --check-equivalence

  echo "[phase=pv_extract]"
  uv run python scripts/issue763_extract_pv_rb.py --smoke --behaviors "$BEH" \
    --mock --device cpu --model-name "$SMOKE_MODEL"

  echo "[phase=judge]"
  uv run python scripts/issue763_judge_e0.py --smoke --behaviors "$BEH" --mock-judge

  # OPT-IN LIVE-BATCH judge gate (task #763 r3): the --mock-judge smoke above
  # never submits a real Anthropic Batch, so it cannot catch a malformed batch
  # request shape (the empty-system-block 400 that quarantined all 8000 graded
  # requests). EPM_LIVE_BATCH_SMOKE=1 runs ~5 REAL graded requests through the
  # forced batch path (~$0.01). Default OFF so the offline --smoke stays fully
  # free/offline (PASS_UNIFIED). Run pre-launch for any judged production run.
  if [ "${EPM_LIVE_BATCH_SMOKE:-0}" = "1" ]; then
    echo "[phase=live_batch_judge_smoke]"
    uv run python scripts/issue763_live_batch_smoke.py
  fi

  echo "[phase=fit]"
  uv run python scripts/issue763_fit_predictors.py --smoke --behaviors "$BEH"

  echo "[phase=figures]"
  uv run python scripts/issue763_plot.py --smoke --behaviors "$BEH"

  echo "[phase=done]"
  exit 0
fi

# ── FIT RESUME (--from-phase fit; CPU by default, GPU via EPM_FIT_DEVICE) ─────
# task #763 r6 USER OVERRIDE: v0(C,B) + graded E0(C,B) + the PV baseline are DONE
# and staged from HF; this resume skips every earlier GPU phase and runs the
# vectorized fit (analysis.issue_763_vectorized; ~1000h serial null -> minutes),
# then figures + upload. The fit device defaults to CPU (cpu-mid lane); the PM
# GPU-cutover directive (2026-07-02) makes it overridable — launch with
# EPM_FIT_DEVICE=cuda on a GPU lane to route the batched fits to the GPU (the
# on-instance exactness gate re-validates the GPU path vs the CPU-serial oracles).
# The fit stages E0 + pv_rb + pv_shards from HF (issue763_fit_predictors.
# _stage_fit_inputs_from_hf, fail-loud if absent) and v0 lazily (_load_v0).
if [ "$FROM_PHASE" = "fit" ]; then
  echo "[issue763.dispatch] fit resume (device=${EPM_FIT_DEVICE:-cpu}); commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

  echo "[phase=fit]"
  uv run python scripts/issue763_fit_predictors.py

  echo "[phase=figures]"
  uv run python scripts/issue763_plot.py

  echo "[phase=upload]"
  uv run python scripts/issue763_upload.py

  echo "[phase=finalize]"
  echo "[phase=done]"
  exit 0
fi

# ── REAL RUN — TWO-PHASE pod-cycling (off-pod PV judge) ───────────────────────
# #763 BLOCKER pv-judge-not-off-pod: the PV judge (and the E0 judge) route through
# eval.batch_judge's DEADLINE-BOUNDED poll (potentially hours). Running them on a
# live GPU pod would hold the GPU through the poll (the #664 spend-leak class), so
# this dispatcher SPLITS at a pod-cycling gate:
#
#   PHASE 1 (this script, no --from-phase): build_pools -> source_side_baseline ->
#     generate (E0 completions) -> capture (v0) -> pv_extract generate (rollouts)
#     -> upload-progress (raw completions + rollouts -> HF, pre-teardown) ->
#     EMIT GATE pv_phase1_done + EXIT. NO --phase judge here, and NO --device cuda
#     after the gate, so the GPU pod can be STOPPED.
#
#   ORCHESTRATOR (SKILL.md Step 6d.4 gate handler, OFF-pod on the VM): on
#     gate=pv_phase1_done it `pod.py stop`s the pod, runs the OFF-pod judge on the
#     VM — `issue763_extract_pv_rb.py --phase judge` (fetches the rollouts from HF,
#     batch-judges, uploads the keep-flags to HF) — then `pod.py resume`s and
#     re-dispatches `issue763_dispatch.sh --from-phase pv_capture`.
#
#   PHASE 2 (--from-phase pv_capture): pv_extract capture (GPU, fetches the
#     keep-flags from HF) -> judge (E0, OFF-pod batch_judge but AFTER the last
#     --device cuda phase, so the GPU block has closed) -> fit -> figures ->
#     final upload (epm:results) -> [phase=done].

if [ "$FROM_PHASE" != "pv_capture" ]; then
  # ── PHASE 1 (GPU live) ──
  # Build pools is normally run OFF-pod before provision (CPU/API) so the frozen
  # pools are HF-uploaded for the git-clone-only lane to snapshot_download. On the
  # pod we stage them: if absent, snapshot_download the HF inputs mirror; if the
  # builder must run here (API present), run it.
  echo "[phase=build_pools]"
  uv run python scripts/issue763_stage_pools.py || \
    uv run python scripts/issue763_build_probe_pools.py

  echo "[phase=source_side_baseline]"
  uv run python scripts/issue763_source_side_baseline.py

  echo "[phase=generate]"
  uv run python scripts/issue763_generate_completions.py

  echo "[phase=capture]"
  uv run python scripts/issue763_capture_v0_matched.py --device cuda

  # PV rollout generation (vLLM batched) — GPU LIVE; writes rollouts to disk.
  echo "[phase=pv_extract_generate]"
  uv run python scripts/issue763_extract_pv_rb.py --device cuda --phase generate

  # Upload raw completions + analysis-input ROLLOUTS BEFORE the GPU pod is
  # released (Upload Policy: raw completions + plan-referenced analysis inputs
  # must land on HF before pod termination; #763 BLOCKER pv-rollouts-not-uploaded
  # — pv_rollouts/ is now in the upload iteration). NON-final epm:upload-progress
  # sentinel (#763 CONCERN premature-results-sentinel: no epm:results yet).
  echo "[phase=upload_progress]"
  uv run python scripts/issue763_upload.py --progress-only

  # GATE: the PV (and E0) judges run OFF-pod. Emit a BLOCKING gate sentinel and
  # EXIT — the orchestrator stops the pod, runs the off-pod judge on the VM,
  # resumes, and re-dispatches `--from-phase pv_capture`. There is NO --device
  # cuda and NO --phase judge AFTER this point in phase 1.
  echo "[phase=pv_phase1_done]"
  uv run python scripts/issue763_upload.py --emit-gate pv_phase1_done
  echo "[issue763.dispatch] phase-1 complete; parked at gate=pv_phase1_done (off-pod judge)"
  exit 0
fi

# ── PHASE 2 (--from-phase pv_capture; GPU live for capture, then OFF-pod tail) ──
# The OFF-pod judge already ran on the VM and uploaded the keep-flags to HF; the
# resumed pod fetches them in --phase capture via snapshot_download.

# PV teacher-forced capture of the KEPT rollouts -> r_B (GPU). LAST --device cuda
# phase — every phase below is GPU-free.
echo "[phase=pv_extract_capture]"
uv run python scripts/issue763_extract_pv_rb.py --device cuda --phase capture

# E0 judge — OFF-pod (eval.batch_judge), AFTER the last --device cuda phase so the
# GPU block has closed. GPU-FREE by construction (judge = API, format = code).
echo "[phase=judge]"
uv run python scripts/issue763_judge_e0.py

echo "[phase=fit]"
uv run python scripts/issue763_fit_predictors.py

echo "[phase=figures]"
uv run python scripts/issue763_plot.py

# FINAL upload (raw completions + v0 + r_B analysis tensors + pv_judge keep-flags)
# AFTER fit + figures exist, then the END-OF-RUN epm:results sentinel (#763
# CONCERN premature-results-sentinel: epm:results appears only after every primary
# deliverable — E0_matched_by_behavior.json / matched_predictor_results.json /
# figures — is on disk).
echo "[phase=upload]"
uv run python scripts/issue763_upload.py

echo "[phase=finalize]"
echo "[phase=done]"
