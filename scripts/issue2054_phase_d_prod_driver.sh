#!/usr/bin/env bash
# issue-2054 Phase-D production driver (pod-side, 1x H100, eval intent;
# pod-2054-d1) — the cell-(c) leg: splice + capture on ONE GPU pod.
#
# Stages (sequential; each inner invocation runs redirected to its OWN child
# log so no inner [phase=done] token can reach this dispatcher's main log —
# the reserved terminal line below is emitted ONLY by this dispatcher):
#
#   stage_scaffolds — idempotent skip-if-present staging of the 4 char-variant
#       ADMITTED scaffold pools via scripts/issue2054_stage_scaffolds.py (the
#       r14 Phase-B prerequisite stager; manifest-driven, verified vs
#       kept.json). Dest = phase_d's --scaffolds-dir default
#       (data/issue_2054/scaffolds/), so phase_d resolves them by default.
#       The Phase-C on-policy ANSWER POOL is deliberately NOT staged here:
#       issue2054_phase_d.py self-stages it per variant from HF
#       issue2054_lattice/on_policy/{model}/{variant}/ (the r18 re-wire,
#       commits a971fe1487 + 5913c85e0d; fail-loud, sha-pinned into its
#       resume regime) — a duplicate driver-side stage would just shadow it.
#   phase_d — the cell-(c) splice (STORY-authored answers re-framed through
#       the CHAT template), --form chat, both model halves via the --variants
#       default (8 op-variants); CPU, minutes; fail-loud HF mirror to
#       issue2054_lattice/cell_c/ is phase_d's own upload step (M2).
#       The shared fold map is COMMITTED at
#       eval_results/issue_2054/shared_fold_map.json (verified below, never
#       staged — phase_d's --fold-map default resolves it).
#   capture x2 — teacher-forced layer-19 capture over the phase_d output,
#       MODEL-MATCHED like the on_policy (d) cells: the spliced (c) text is
#       captured through the model that AUTHORED the answer
#       (issue2054_phase_d._ANSWER_MODEL_FROM_TAIL: variant tail `_op` ->
#       qwen2.5-7b-instruct, `_op_base` -> qwen2.5-7b), so each (c) cell
#       pairs with the (d) cell of the SAME (character, model, answer_form).
#       Sequential on the single GPU (8 cells x ~8k rows = ~64k rows total,
#       under ~1 GPU-h at the measured a/b/d throughput). Capture uploads its
#       own npz store per invocation (fail-loud, M2) to
#       issue2054_lattice/activations/ — the same stream-reduced per-cell
#       npz + digest convention as the a/b/d cells; this driver surfaces its
#       rc and never discards the upload outcome.
#
# Conventions (modeled on issue2054_phase_b_prod_driver.sh +
# issue2054_capture_prod_driver.sh): set -uo pipefail, repo-root cd,
# conditional .env sourcing, [phase=...] breadcrumbs, per-stage child logs,
# end-of-run poll_pipeline.py results sentinel + the reserved [phase=done]
# terminal line, NO task.py shellouts. Idempotent re-entry: stage_scaffolds
# skips when the 4 pools are present; phase_d + capture resume completed
# regime-matching units via their own done sidecars (C9/M6).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_D1_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
START_EPOCH=$(date +%s)
FOLD_MAP="eval_results/issue_2054/shared_fold_map.json"
CELL_C_DIR="data/issue_2054/cell_c/"
# Model-matched capture split (source: issue2054_phase_d._ANSWER_MODEL_FROM_TAIL
# + DEFAULT_VARIANTS; pinned by tests/test_issue2054_phase_d_prod_driver.py).
VARIANTS_INSTRUCT="char_helios_op,char_wren_op,char_dana_op,char_vex_op"
VARIANTS_BASE="char_helios_op_base,char_wren_op_base,char_dana_op_base,char_vex_op_base"

echo "[phase=phase_d_prod] driver start $(date -u +%FT%TZ)"

halt() {  # halt <stage> <rc> <child log>
  local stage="$1" rc="$2" log="$3"
  echo "[phase=phase_d_prod] HALT ${stage} rc=${rc} (tail follows)"
  tail -30 "$log" || true
  exit "$rc"
}

# --- preflight: committed fold map + a visible GPU ---------------------------
if [ ! -s "$FOLD_MAP" ]; then
  echo "[phase=phase_d_prod] HALT shared fold map missing/empty: $FOLD_MAP" \
    "(partial clone? run: git sparse-checkout add eval_results/issue_2054)"
  exit 2
fi
if ! nvidia-smi -L > /dev/null 2>&1; then
  # Capture silently degrades to CPU when cuda is unavailable (a ~64k-row
  # CPU crawl, not a crash) — refuse up front instead.
  if [ "${ISSUE2054_D1_ALLOW_CPU:-0}" != "1" ]; then
    echo "[phase=phase_d_prod] HALT no GPU visible (capture would silently" \
      "crawl on CPU); set ISSUE2054_D1_ALLOW_CPU=1 to override"
    exit 2
  fi
  echo "[phase=phase_d_prod] WARN no GPU visible; proceeding (ISSUE2054_D1_ALLOW_CPU=1)"
fi

# --- stage 1: scaffolds (idempotent, skip-if-present) ------------------------
echo "[phase=phase_d_prod stage=stage_scaffolds] start $(date -u +%FT%TZ)"
NEED_STAGE=0
for v in char_helios char_wren char_dana char_vex; do
  [ -s "data/issue_2054/scaffolds/${v}/scaffolds_${v}.jsonl" ] || NEED_STAGE=1
done
if [ "$NEED_STAGE" -eq 1 ]; then
  uv run python scripts/issue2054_stage_scaffolds.py \
    --variants char_helios,char_wren,char_dana,char_vex \
    > "$LOG_DIR/issue-2054-d1-stage-scaffolds.log" 2>&1
  rc=$?
  echo "[phase=phase_d_prod stage=stage_scaffolds] rc=${rc} $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    halt stage_scaffolds "$rc" "$LOG_DIR/issue-2054-d1-stage-scaffolds.log"
  fi
else
  echo "[phase=phase_d_prod stage=stage_scaffolds] skip — all 4 pools present $(date -u +%FT%TZ)"
fi

# --- stage 2: phase_d splice (--form chat; production defaults) --------------
echo "[phase=phase_d_prod stage=phase_d] start $(date -u +%FT%TZ)"
uv run python scripts/issue2054_phase_d.py --form chat \
  > "$LOG_DIR/issue-2054-d1-phase-d.log" 2>&1
rc=$?
echo "[phase=phase_d_prod stage=phase_d] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  halt phase_d "$rc" "$LOG_DIR/issue-2054-d1-phase-d.log"
fi

# --- stage 3: cell-(c) capture, model-matched, sequential on the 1 GPU -------
run_capture() {  # run_capture <model_slug> <variants_csv> <child log>
  local model="$1" variants="$2" log="$3"
  uv run python scripts/issue2054_capture.py \
    --input-dir "$CELL_C_DIR" --phase cell_c --form chat \
    --model "$model" --variants "$variants" \
    > "$log" 2>&1
}

echo "[phase=phase_d_prod stage=capture model=qwen2.5-7b-instruct] start $(date -u +%FT%TZ)"
run_capture qwen2.5-7b-instruct "$VARIANTS_INSTRUCT" "$LOG_DIR/issue-2054-d1-capture-instruct.log"
rc=$?
echo "[phase=phase_d_prod stage=capture model=qwen2.5-7b-instruct] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  halt capture_instruct "$rc" "$LOG_DIR/issue-2054-d1-capture-instruct.log"
fi

echo "[phase=phase_d_prod stage=capture model=qwen2.5-7b] start $(date -u +%FT%TZ)"
run_capture qwen2.5-7b "$VARIANTS_BASE" "$LOG_DIR/issue-2054-d1-capture-base.log"
rc=$?
echo "[phase=phase_d_prod stage=capture model=qwen2.5-7b] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  halt capture_base "$rc" "$LOG_DIR/issue-2054-d1-capture-base.log"
fi

# --- stage 4: end-of-run results sentinel, then the reserved terminal line ---
write_sentinel() {  # write_sentinel <path> <elapsed seconds>
  uv run python - "$1" "$2" <<'PYEOF'
import json
import subprocess
import sys
import time
from pathlib import Path

out, elapsed = Path(sys.argv[1]), int(sys.argv[2])
sha = (
    subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    or "unknown"
)


def _read(p: str) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}


pd = _read("data/issue_2054/cell_c/phase_d_digest__chat.json")
cap_paths = {
    m: f"data/issue_2054/activations/capture_digest__cell_c__chat__{m}.json"
    for m in ("qwen2.5-7b-instruct", "qwen2.5-7b")
}
caps = {m: _read(p) for m, p in cap_paths.items()}
note = {
    "eval_numbers": {
        "phase_d_n_total_out": pd.get("n_total_out"),
        "phase_d_variants_below_floor": pd.get("variants_below_floor"),
        "capture_n_total_ok": {m: c.get("n_total_ok") for m, c in caps.items()},
    },
    "eval_paths": ["data/issue_2054/cell_c/phase_d_digest__chat.json", *cap_paths.values()],
    "reproducibility_card": {
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "hf_data_paths": ["issue2054_lattice/cell_c/", "issue2054_lattice/activations/"],
        "capture_diagnostics_dir": "eval_results/issue_2054/capture_diagnostics/",
        "notes": "cell-(c) splice+capture leg: no adapters trained, no WandB runs",
    },
    "wandb_url": "n/a (capture-only run; no training — see reproducibility_card)",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": sha,
    "gpu_hours_used": round(elapsed / 3600.0, 2),
    "gpu_hours_budgeted": 2.0,
    "plan_deviations": "none",
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 2054,
    "by": "issue2054_phase_d_prod_driver",
    "gate": "phase_d_prod",
    "blocks_pipeline": False,
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note, indent=2, sort_keys=True),
}
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
tmp.replace(out)
print(f"[sentinel] wrote {out}", flush=True)
PYEOF
}

ELAPSED=$(( $(date +%s) - START_EPOCH ))
SENTINEL="$LOG_DIR/issue-2054-epm_results-$(date +%s).json"
write_sentinel "$SENTINEL" "$ELAPSED"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[phase=phase_d_prod] HALT sentinel write rc=${rc} (results ARE persisted on HF;"
  echo "  the sentinel is the poller handoff — do not re-run compute for this)"
  exit "$rc"
fi

echo "[phase=phase_d_prod] driver_rc=0 elapsed_s=${ELAPSED} $(date -u +%FT%TZ)"
echo "[phase=done]"
