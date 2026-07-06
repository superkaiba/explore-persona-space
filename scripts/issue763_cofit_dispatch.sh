#!/usr/bin/env bash
# Issue #763 `neutral-contrast-and-cofit` dispatch — the --workload-cmd driver.
#
# UNIFIED end-to-end (PASS_UNIFIED smoke-architecture contract): --smoke runs
# the SAME python entrypoint chain as the real run on a tiny offline slice
# (1 behavior, CPU 0.5B model, mock judge, tiny perms); every phase's cell list
# derives from the same --behaviors/--smoke threading. No separate smoke path.
#
#   bash scripts/issue763_cofit_dispatch.sh --smoke                # offline verification
#   bash scripts/issue763_cofit_dispatch.sh --phase-group A        # GPU capture-7b leg
#   bash scripts/issue763_cofit_dispatch.sh --phase-group B        # VM off-pod judge leg
#   bash scripts/issue763_cofit_dispatch.sh --phase-group C        # GPU eval-lane co-fit leg
#
# Three sequential production groups (plan §4.1/§4.2/§9 — no GPU held through
# the Batch-API judge poll):
#   A (capture-7b, 1x A100-80): neutral_generate (vLLM) -> capture_arm_means
#     (batched TF + parity smoke) -> capture_context_side (c0 prompt shards) ->
#     progress upload -> EMIT GATE cofit_phaseA_done + EXIT (pod terminates).
#   B (VM, 0 GPU): neutral_judge (eval.batch_judge, ~5k calls) ->
#     assemble_directions (r_C / r_neutral + cos integrity AFTER the parity
#     gate) -> directions upload.
#   C (eval lane, EPM_FIT_DEVICE=cuda): cofit_predictors (manifest + 8-method
#     co-fit + selection-symmetric nulls + nonlinear block) -> figures ->
#     FINAL upload (epm:results).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
if [ ! -d "$REPO_ROOT/scripts" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

SMOKE=""
PHASE_GROUP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke" ;;
    --phase-group)
      shift
      PHASE_GROUP="${1:-}"
      ;;
    --phase-group=*) PHASE_GROUP="${1#--phase-group=}" ;;
  esac
  shift || true
done
if [ -z "$SMOKE" ] && [ "$PHASE_GROUP" != "A" ] && [ "$PHASE_GROUP" != "B" ] && [ "$PHASE_GROUP" != "C" ]; then
  echo "[issue763.cofit] FATAL: pass --smoke or --phase-group A|B|C" >&2
  exit 2
fi

# Smoke scoping (review r1 C1(iii)): --smoke arms EPM_ISSUE763_SMOKE_SCOPE=1 so
# every WRITE-target path in the python entrypoints rebinds under smoke_scope/
# — mock artifacts can never land at (or clobber) canonical production paths.
# Production groups explicitly UNSET it (each entrypoint also fails loud on
# env-set-without---smoke via ensure_smoke_scope).
if [ -n "$SMOKE" ]; then
  export EPM_ISSUE763_SMOKE_SCOPE=1
else
  unset EPM_ISSUE763_SMOKE_SCOPE || true
fi

# Credentials (judge / HF) — set -a && source .env (never bare load_dotenv in a heredoc).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "[issue763.cofit] REPO_ROOT=$REPO_ROOT SMOKE='${SMOKE}' GROUP='${PHASE_GROUP}' commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [ -n "$SMOKE" ]; then
  SMOKE_MODEL="${EPM_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  BEH="deception"

  # Residue sentinel: any file CREATED OR MODIFIED under the canonical
  # (non-smoke_scope) round paths during the smoke fails the run at the end —
  # the mechanical proof that the smoke scoping holds (review r1 C1(iii)).
  RESIDUE_MARK="$(mktemp /tmp/issue763_smoke_residue.XXXXXX)"
  RESIDUE_DIRS="data/issue_763/neutral_rollouts data/issue_763/neutral_judge \
data/issue_763/pv_rollouts data/issue_763/pv_judge_v2 data/issue_763/pv_artifacts \
eval_results/issue_763/neutral-contrast-and-cofit eval_results/issue_763/pv_shards \
figures/issue_763"

  # Upstream mock artifacts through the SAME Phase-1 scripts (rollouts +
  # canonical keep-flags + rb shard at smoke dims) so the round phases consume
  # producer-shaped inputs end to end.
  echo "[phase=pv_prereqs_smoke]"
  uv run python scripts/issue763_extract_pv_rb.py --smoke --behaviors "$BEH" \
    --mock --device cpu --model-name "$SMOKE_MODEL"

  echo "[phase=neutral_generate]"
  uv run python scripts/issue763_extract_pv_rb.py --phase neutral_generate --smoke \
    --mock --device cpu --model-name "$SMOKE_MODEL" --behaviors "$BEH"

  echo "[phase=neutral_judge]"
  uv run python scripts/issue763_extract_pv_rb.py --phase neutral_judge --smoke --mock \
    --behaviors "$BEH"

  echo "[phase=capture_arm_means]"
  uv run python scripts/issue763_extract_pv_rb.py --phase capture_arm_means --smoke \
    --device cpu --model-name "$SMOKE_MODEL" --batch-size 4 --behaviors "$BEH"

  echo "[phase=capture_context_side]"
  uv run python scripts/issue763_capture_v0_matched.py --span prompt --smoke \
    --device cpu --model-name "$SMOKE_MODEL" --batch-size 4 --behaviors "$BEH"

  echo "[phase=assemble_directions]"
  uv run python scripts/issue763_extract_pv_rb.py --phase assemble_directions --smoke \
    --behaviors "$BEH"

  echo "[phase=cofit]"
  uv run python scripts/issue763_cofit_predictors.py --smoke --behaviors "$BEH" --force

  echo "[phase=figures]"
  uv run python scripts/issue763_cofit_plot.py --smoke --behaviors "$BEH"

  echo "[phase=upload]"
  echo "[issue763.cofit] smoke: uploads are LOG-ONLY (offline; the real groups upload)"

  echo "[phase=residue_check]"
  RESIDUE="$(find $RESIDUE_DIRS -path '*smoke_scope*' -prune -o -type f -newer "$RESIDUE_MARK" -print 2>/dev/null || true)"
  rm -f "$RESIDUE_MARK"
  if [ -n "$RESIDUE" ]; then
    echo "[issue763.cofit] FATAL: smoke wrote/modified CANONICAL production paths:" >&2
    echo "$RESIDUE" >&2
    exit 3
  fi
  echo "[issue763.cofit] residue check clean: no canonical path touched by the smoke"

  echo "[phase=done]"
  exit 0
fi

if [ "$PHASE_GROUP" = "A" ]; then
  # ── PHASE GROUP A (GPU capture-7b) ──
  echo "[phase=neutral_generate]"
  uv run python scripts/issue763_extract_pv_rb.py --phase neutral_generate --device cuda

  echo "[phase=capture_arm_means]"
  uv run python scripts/issue763_extract_pv_rb.py --phase capture_arm_means --device cuda

  echo "[phase=capture_context_side]"
  uv run python scripts/issue763_capture_v0_matched.py --span prompt --device cuda

  # Rollout text + per-rollout means + c0 shards land on HF BEFORE the pod is
  # released (Upload Policy; the neutral judge runs OFF-pod from the HF mirror).
  echo "[phase=upload_progress]"
  uv run python scripts/issue763_cofit_upload.py --progress-only

  echo "[phase=cofit_phaseA_done]"
  uv run python scripts/issue763_cofit_upload.py --emit-gate cofit_phaseA_done
  echo "[issue763.cofit] phase A complete; parked at gate=cofit_phaseA_done (off-pod judge)"
  exit 0
fi

if [ "$PHASE_GROUP" = "B" ]; then
  # ── PHASE GROUP B (VM, 0 GPU — Batch-API judge + direction assembly) ──
  echo "[phase=neutral_judge]"
  uv run python scripts/issue763_extract_pv_rb.py --phase neutral_judge

  echo "[phase=assemble_directions]"
  uv run python scripts/issue763_extract_pv_rb.py --phase assemble_directions

  echo "[phase=upload_directions]"
  uv run python scripts/issue763_cofit_upload.py --directions-only

  echo "[issue763.cofit] phase B complete (r_C + r_neutral assembled + uploaded)"
  exit 0
fi

# ── PHASE GROUP C (eval lane; batched fits on EPM_FIT_DEVICE, default cuda) ──
echo "[phase=cofit]"
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cuda}" uv run python scripts/issue763_cofit_predictors.py

echo "[phase=figures]"
uv run python scripts/issue763_cofit_plot.py

echo "[phase=upload]"
uv run python scripts/issue763_cofit_upload.py

echo "[phase=finalize]"
echo "[phase=done]"
