#!/usr/bin/env bash
# Issue #920: sequential GPU-instance phase driver (plan §3.5 dispatcher).
#
# Phases (each checkpoints + resumes; a re-run skips completed phases):
#   gen_b -> extract (G1 equivalence gate FIRST, then set A + set B)
#         -> fits (G2 gate + K3 anchor gate inside) -> dv1 nulls (G2-null gate)
#         -> results sentinel.
# The post-release cpu-mid phase (`issue920_nulls_figures.py --cpu-aggregation`)
# is dispatched SEPARATELY by the orchestrator (plan §10 workload commands).
#
# Resume predicates key on each phase's POST-GATE/POST-UPLOAD `*_done.json`
# marker (written by the phase script at the very END of its success path —
# after K3 for fits, after the Hub upload for gen/extract/nulls), never on the
# intermediate artifacts alone: a retry after a failed gate or a failed upload
# must RE-ENTER the phase, not skip it (round-1 blocker
# `k3-resume-bypasses-anchor-gate`).
#
# Pod-side contract: [phase=...] log lines; the token [phase=done] is RESERVED
# for the SINGLE terminal line at the bottom of this script (phase scripts end
# with non-reserved tokens like [phase=fits_complete] — #545 false-done class).
# Sentinels under /workspace/logs/issue-920-*.json (the python scripts write
# their own per-phase progress sentinels; issue920_results_sentinel.py writes
# the final epm:results payload). Pod-side code NEVER shells out to task.py.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
export PATH="/root/.local/bin:$PATH"

# EPM_I920_SMOKE=1: the SAME dispatcher + SAME entrypoints at tiny N (smoke IS the
# production path — the subset threads through EVERY phase: gen writes a
# 50-ctx x 1-probe mock store, extract consumes IT for set B + the HF set-A bucket,
# fits enumerate cells FROM the store, nulls read the fits' stores; nothing
# re-enumerates a full registered grid). Outputs go to a SCRATCH root so smoke
# never overwrites committed eval_results/figures.
SMOKE="${EPM_I920_SMOKE:-0}"
OUT_ROOT="$REPO_ROOT"
GEN_FLAGS=()
EXTRACT_FLAGS=(--gpu --equiv-gate-first --probe-set both --batch-probes 8)
FIT_FLAGS=()
NULL_FLAGS=(--gpu-null-only)
EXPECT_N=50
if [ "$SMOKE" = "1" ]; then
  OUT_ROOT="${EPM_I920_SMOKE_ROOT:-/tmp/i920_dispatch_smoke}"
  mkdir -p "$OUT_ROOT"
  GEN_FLAGS=(--smoke --mock-engine --n-probes 1 --no-upload
    --out-dir "$OUT_ROOT/data/issue_920/gen_b")
  EXTRACT_FLAGS=(--smoke --model Qwen/Qwen2.5-0.5B-Instruct --n-probes 1
    --equiv-gate-first --equiv-gate-dry --probe-set both --batch-probes 8 --no-upload
    --out-root "$OUT_ROOT/data/issue_920" --gen-b-dir "$OUT_ROOT/data/issue_920/gen_b")
  FIT_FLAGS=(--store-root "$OUT_ROOT/data/issue_920"
    --eval-out "$OUT_ROOT/eval_results/issue_920"
    --preds-out "$OUT_ROOT/data/issue_920/preds" --skip-anchor-gate)
  NULL_FLAGS=(--gpu-null-only --n-draws 5 --g2-draws 3 --no-upload
    --store-root "$OUT_ROOT/data/issue_920"
    --preds-dir "$OUT_ROOT/data/issue_920/preds"
    --null-out "$OUT_ROOT/data/issue_920/null_matrices"
    --eval-out "$OUT_ROOT/eval_results/issue_920")
fi

# Credentials: uv run does NOT auto-load .env — source at entry, fail loud.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
: "${HF_TOKEN:?HF_TOKEN missing after .env load — refusing to launch}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

mkdir -p /workspace/logs "$OUT_ROOT/data/issue_920" \
  "$OUT_ROOT/eval_results/issue_920" "$OUT_ROOT/figures/issue_920"
if [ "$SMOKE" != "1" ]; then
  # plan §6.5 globs read /workspace/eval_results + /workspace/data — mirror via symlink
  if [ -d /workspace ] && [ ! -e /workspace/eval_results/issue_920 ]; then
    mkdir -p /workspace/eval_results
    ln -sfn "$REPO_ROOT/eval_results/issue_920" /workspace/eval_results/issue_920
  fi
  if [ -d /workspace ] && [ ! -e /workspace/data/issue_920 ]; then
    mkdir -p /workspace/data
    ln -sfn "$REPO_ROOT/data/issue_920" /workspace/data/issue_920
  fi
fi
[ -f /workspace/logs/issue-920-start-ts ] || date +%s > /workspace/logs/issue-920-start-ts

echo "[phase=gen_b] set-B greedy completions (vLLM, own process)"
GEN_DONE=$(ls "$OUT_ROOT"/data/issue_920/gen_b/*.json 2>/dev/null | wc -l || true)
if [ -f "$OUT_ROOT/data/issue_920/gen_b_done.json" ] && [ "$GEN_DONE" -ge "$EXPECT_N" ]; then
  echo "[phase=gen_b] phase-done marker + $EXPECT_N per-context files present — skip (resume)"
else
  # NOTE: the phase script has its own per-context resume, so this re-entry is
  # cheap when the crash was late (e.g. at the upload step).
  uv run python scripts/issue920_gen_completions_b.py ${GEN_FLAGS[@]+"${GEN_FLAGS[@]}"}
fi

echo "[phase=extract] G1 gate + 55-family extraction, sets A+B (HF, own process)"
EXT_A=$(ls "$OUT_ROOT"/data/issue_920/summaries_setA/*.pt 2>/dev/null | wc -l || true)
EXT_B=$(ls "$OUT_ROOT"/data/issue_920/summaries_setB/*.pt 2>/dev/null | wc -l || true)
if [ -f "$OUT_ROOT/data/issue_920/extract_done.json" ] \
  && [ "$EXT_A" -ge "$EXPECT_N" ] && [ "$EXT_B" -ge "$EXPECT_N" ]; then
  echo "[phase=extract] phase-done marker + both stores complete — skip (resume)"
else
  uv run python scripts/issue920_extract_summaries.py "${EXTRACT_FLAGS[@]}"
fi

echo "[phase=fits] batched LOFO fit battery (G2 gate + K3 anchor gate inside)"
# The resume predicate REQUIRES the post-K3 fits_done marker — the eval JSON +
# preds are written BEFORE the K3 anchor gate, so their presence alone must
# never skip this phase (a K3 FAIL would otherwise be bypassed on retry).
if [ -f "$OUT_ROOT/data/issue_920/preds/fits_done.json" ] \
  && [ -f "$OUT_ROOT/eval_results/issue_920/map_skill_by_cell.json" ] \
  && [ -f "$OUT_ROOT/data/issue_920/preds/pooled_heldout_predictions.pt" ]; then
  echo "[phase=fits] post-K3 fit-done marker + outputs present — skip (resume)"
else
  EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cuda}" uv run python scripts/issue920_fit_lofo.py \
    "${FIT_FLAGS[@]}"
fi

echo "[phase=nulls_gpu] DV-1 perm-refit null battery (G2-null gate inside)"
if [ -f "$OUT_ROOT/data/issue_920/null_matrices/dv1_done.json" ] \
  && [ -f "$OUT_ROOT/data/issue_920/null_matrices/dv1_null_skills.pt" ]; then
  echo "[phase=nulls_gpu] phase-done marker + dv1_null_skills.pt present — skip (resume)"
else
  # Internal resume: an existing dv1_null_skills.pt skips the battery recompute;
  # the upload + done marker re-run.
  EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cuda}" uv run python scripts/issue920_nulls_figures.py \
    "${NULL_FLAGS[@]}"
fi

echo "[phase=results_sentinel] composing the epm:results payload"
uv run python scripts/issue920_results_sentinel.py --eval-out "$OUT_ROOT/eval_results/issue_920"

echo "[phase=done] issue #920 GPU pipeline complete"
