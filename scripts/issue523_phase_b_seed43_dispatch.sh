#!/usr/bin/env bash
# Issue #523 — Phase B seed-43 retrain dispatcher.
#
# Thin wrapper around scripts/i474_phase23_dispatch.sh: exports the #523-
# specific env vars (seed=43, HF subfolder prefix=i523_loc, WandB project,
# epochs=1-only) and invokes the parent dispatcher with --arm=loc.
#
# This IS the unified smoke/sweep pattern from #474 — smoke (--smoke=A1)
# runs the SAME dispatcher with --smoke-only, sweep (no --smoke flag) runs
# the full waves filtered to loc-arm only. Per plan v2 §4.10 architectural
# parity.
#
# Plan v2 §4 Phase B mapping:
#   * seed swap (42 → 43): EPM_TRAIN_SEED=43
#   * adapters/i523_loc_<cond>_ep1_seed43/: EPM_HF_PATH_TEMPLATE +
#     EPM_OUTPUT_DIR_PREFIX
#   * WandB project issue_523_seed43: WANDB_PROJECT
#   * epoch 1 only: EPM_TRAIN_EPOCHS=1
#
# After the training waves, runs i474_phase4_eval.py against the seed-43
# adapters with the same env overrides + writes the per-cell diagonal
# implant report to eval_results/issue_523/seed43_per_cell_implant.json.
#
# Usage:
#   bash scripts/issue523_phase_b_seed43_dispatch.sh                       # full sweep
#   bash scripts/issue523_phase_b_seed43_dispatch.sh --smoke=A1            # A1 smoke only
#   bash scripts/issue523_phase_b_seed43_dispatch.sh --smoke=A1 --no-upload  # smoke, no HF upload
#   bash scripts/issue523_phase_b_seed43_dispatch.sh --no-eval             # train only, skip Phase 4

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# ── #523-specific environment overrides ──
# Read by scripts/i474_phase23_train.py + scripts/i474_phase4_eval.py +
# scripts/i474_phase2_smoke_check.py + scripts/i474_check_adapter_hf_presence.py
# (all four files patched to honor these env vars with #474 defaults).
export EPM_TRAIN_SEED=43
export EPM_TRAIN_EPOCHS=1
export EPM_HF_PATH_TEMPLATE='adapters/i523_loc_{cid}_ep{ep}_seed43'
export EPM_OUTPUT_DIR_PREFIX='adapters/i523_loc'
export EPM_RUN_NAME_PREFIX='i523'
# Per the #474 recipe + .claude/rules/upload-policy.md: never upload
# merged checkpoint dirs, delete after verified HF upload.
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export EPM_PERSIST_ADAPTER_HF_REPO='superkaiba1/explore-persona-space'
export WANDB_PROJECT='issue_523_seed43'
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_523
mkdir -p "$LOG_DIR"

# Parse our local flags BEFORE forwarding to the parent dispatcher.
SMOKE_ONLY=""
SMOKE_COND=""
NO_UPLOAD=0
NO_EVAL=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --smoke=*)
            SMOKE_COND="${arg#--smoke=}"
            SMOKE_ONLY=1
            ;;
        --no-upload)
            NO_UPLOAD=1
            ;;
        --no-eval)
            NO_EVAL=1
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

echo "[phase=preflight] issue523 phase B seed43 dispatcher $(date -Iseconds)"
echo "[phase=preflight]   EPM_TRAIN_SEED=$EPM_TRAIN_SEED"
echo "[phase=preflight]   EPM_TRAIN_EPOCHS=$EPM_TRAIN_EPOCHS"
echo "[phase=preflight]   EPM_HF_PATH_TEMPLATE=$EPM_HF_PATH_TEMPLATE"
echo "[phase=preflight]   EPM_OUTPUT_DIR_PREFIX=$EPM_OUTPUT_DIR_PREFIX"
echo "[phase=preflight]   WANDB_PROJECT=$WANDB_PROJECT"
echo "[phase=preflight]   smoke_only=$SMOKE_ONLY smoke_cond=$SMOKE_COND no_eval=$NO_EVAL"

# Marker assert (delegate to parent dispatcher's assertion via --arm=loc).
# Parent's tokenization assertion fires before any training subprocess.

# ── Dispatcher invocation ──
# Plan v2 §4 Phase B notes that the seed-43 leg is loc-arm ONLY (the pos-arm
# was explicitly de-prioritized in #502; the headline lives on loc_ep1).
PARENT_ARGS=(--arm=loc)
if [ "$SMOKE_ONLY" = "1" ]; then
    # The parent's --smoke-only flag trains A1 with the REAL recipe and runs
    # the A1 → C1 bystander check. The plan §4 Phase B smoke gate IS the
    # parent's A_loc smoke gate (diagonal ≥ 0.80 AND C1 bystander on-policy
    # ※ emission < 0.30) per byte-identical inheritance.
    PARENT_ARGS+=(--smoke-only)
    # SMOKE_COND is informational; the parent always uses A1 for smoke. If a
    # caller passes --smoke=B1 they get the same A1 cell — flag and continue.
    if [ -n "$SMOKE_COND" ] && [ "$SMOKE_COND" != "A1" ]; then
        echo "[phase=preflight] WARNING: --smoke=$SMOKE_COND ignored; parent uses A1." >&2
    fi
fi

# Forward any additional args (e.g. --resume).
PARENT_ARGS+=("${EXTRA_ARGS[@]}")

echo "[phase=dispatch] invoking i474_phase23_dispatch.sh ${PARENT_ARGS[*]}"
bash "$(dirname "$0")/i474_phase23_dispatch.sh" "${PARENT_ARGS[@]}"

# When --no-eval (or --smoke-only), skip Phase 4.
if [ "$NO_EVAL" = "1" ] || [ "$SMOKE_ONLY" = "1" ]; then
    echo "[phase=done] training-only path complete (no_eval=$NO_EVAL smoke_only=$SMOKE_ONLY)"
    exit 0
fi

# ── Phase 4 cross-eval against #474's frozen R_test (50 questions) ──
# Reuse i474_phase4_eval.py — it honors EPM_HF_PATH_TEMPLATE for downloads
# AND EPM_PHASE4_OUTPUT_ROOT for output directory redirect.
echo "[phase=cross_eval] running i474_phase4_eval.py against seed-43 adapters"
export EPM_PHASE4_OUTPUT_ROOT='eval_results/issue_523/seed43_cross_eval'
mkdir -p eval_results/issue_523/seed43_cross_eval/loc_ep1
uv run python scripts/i474_phase4_eval.py \
    --arm loc \
    --checkpoint-epoch 1 \
    > "$LOG_DIR/cross_eval.log" 2>&1

# Merge per-shard / per-cell roll-ups into G_logprob_matrix.json
echo "[phase=cross_eval_merge] merging G_partial → G_logprob_matrix.json"
uv run python scripts/i474_phase4_merge.py \
    --arm loc \
    --checkpoint-epoch 1 \
    > "$LOG_DIR/cross_eval_merge.log" 2>&1

# ── Per-cell diagonal-implant report ──
# Reads the freshly written G_logprob_matrix.json and emits a compact
# 16-row JSON of diagonal g_logprob per source LoRA — the
# seed43_implant_per_cell.png figure inputs (plan §6).
echo "[phase=implant_report] writing seed43_per_cell_implant.json"
uv run python - <<'PYEOF'
import json
from pathlib import Path

src = Path("eval_results/issue_523/seed43_cross_eval/loc_ep1/G_logprob_matrix.json")
dst = Path("eval_results/issue_523/seed43_per_cell_implant.json")
d = json.loads(src.read_text())
G = d["G"]
conds = d["conditions"]
rows = []
n_converged = 0
for c in conds:
    diag = G[c][c]
    g_lp = diag["g_logprob"]
    converged = g_lp >= -0.80  # log-prob threshold; -0.80 ≈ argmax-marker
    if converged:
        n_converged += 1
    rows.append(
        {
            "cond_id": c,
            "g_logprob_diag": g_lp,
            "delta_g_diag": diag.get("delta_g"),
            "emission_rate_diag": diag.get("emission_recompute_rate"),
            "converged_ge_0_80": converged,
        }
    )
out = {
    "schema_version": 1,
    "issue": 523,
    "phase": "phase_b_diagonal_implant",
    "n_cells": len(conds),
    "n_converged_ge_0_80": n_converged,
    "fraction_converged": n_converged / max(len(conds), 1),
    "rows": rows,
    "source_matrix": str(src),
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=2))
print(f"wrote {dst} ({n_converged}/{len(conds)} cells converged)")
PYEOF

echo "[phase=done] issue523 phase B seed43 complete $(date -Iseconds)"
