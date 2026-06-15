#!/usr/bin/env bash
# Task #623 dispatch driver — pod-side, training-free persona-vector decomposition.
#
# Runs the headline arm on 1 GPU (GCP lora-7b / A100-80G): forward-pass +
# vLLM generation only, then uploads intermediate analysis tensors to HF before
# pod termination. The off-pod CPU analysis (phase 6, issue623_analyze.py) runs
# on the VM AFTER the pod terminates (per CLAUDE.md "CPU-only phases don't hold
# GPU pods").
#
# Phases (each emits a [phase=<name>] log line; smoke = sweep with a smaller
# cell list via --personas / --layers / --n-questions threaded into every phase):
#   p0_preflight   tolerant preflight (parse --json; tolerate behind-origin/main)
#   persona_resolve  -> panel_prompts.json
#   vector_extract   -> persona centroids (Method A + B)
#   sycophancy_trait -> trait vector (last-token + response-avg)
#   steering_probe   -> K2 HALT gate + headline-layer selection
#   syc_i_load       -> syc_i.json (reused #612 base rates, NO new generation)
#   upload           -> centroids + trait + raw completions to HF data repo
#   done             -> terminal marker
#
# Launch (via the backend router; GCP auto lora-7b lane, ~3h << 24h fence):
#   uv run python scripts/dispatch_issue.py --issue 623 --intent lora-7b \
#     --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue623_dispatch.sh'
#
# Smoke (tiny slice, exit 0 + artifact per phase):
#   bash scripts/issue623_dispatch.sh --smoke \
#     --personas satirist,journalist,assistant --layers 21 --n-questions 6
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export TQDM_DISABLE=1                                   # GCE startup-script bufio guard (#607)
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"  # #628 fork guard
export WANDB_PROJECT="${WANDB_PROJECT:-issue623}"

# Defaults (full sweep).
PERSONAS=""              # empty => full 36-persona panel
LAYERS="7 14 21 27"
N_QUESTIONS_TRAIT="40"   # paper 40, split 20 extraction / 20 eval
N_VECTOR_QUESTIONS=""    # empty => all 240 extraction questions
GPU_ID="0"
SKIP_UPLOAD="0"
SKIP_PREFLIGHT="0"
SMOKE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE="1"; shift ;;
    --personas) PERSONAS="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --n-questions) N_QUESTIONS_TRAIT="$2"; N_VECTOR_QUESTIONS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --skip-upload) SKIP_UPLOAD="1"; shift ;;
    --skip-preflight) SKIP_PREFLIGHT="1"; shift ;;
    *) echo "[driver] unknown arg: $1" >&2; exit 2 ;;
  esac
done

DATA_DIR="data/persona_vectors/issue623"
EVAL_DIR="eval_results/issue_623"
LOGS_DIR="${ISSUE623_LOGS_DIR:-/workspace/logs}"
RUN_LOGS="$LOGS_DIR/issue623_driver"
mkdir -p "$RUN_LOGS" "$DATA_DIR" "$EVAL_DIR"

fail() { echo "[driver] FATAL: $*" >&2; exit "${2:-1}"; }

# Optional per-phase override args.
PERSONA_ARG=()
[[ -n "$PERSONAS" ]] && PERSONA_ARG=(--personas "$PERSONAS")
VEC_NQ_ARG=()
[[ -n "$N_VECTOR_QUESTIONS" ]] && VEC_NQ_ARG=(--n-questions "$N_VECTOR_QUESTIONS")

echo "[driver] issue 623 dispatch: smoke=$SMOKE personas='${PERSONAS:-ALL}' layers='$LAYERS' gpu=$GPU_ID"

# ── p0_preflight (tolerant; parse --json, tolerate behind-origin/main #552) ──
if [[ "$SKIP_PREFLIGHT" == "0" ]]; then
  echo "[driver] [phase=p0_preflight] tolerant preflight"
  uv run python - <<'PY' || fail "preflight (non-git-check) errors" 2
import json, re, subprocess, sys
proc = subprocess.run(
    ["uv", "run", "python", "-m", "explore_persona_space.orchestrate.preflight", "--json"],
    capture_output=True, text=True,
)
m = re.search(r"\{.*\}", proc.stdout, re.S)
if not m:
    print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
    sys.exit(1)
report = json.loads(m.group(0))
behind = re.compile(r"behind origin/(main|issue-623)|git fetch origin failed")
real = [e for e in report.get("errors", []) if not behind.search(str(e))]
if real:
    print("preflight errors:", real, file=sys.stderr)
    sys.exit(1)
print("preflight OK (git behind-origin tolerated on issue branches)")
PY
else
  echo "[driver] [phase=p0_preflight] SKIPPED (--skip-preflight)"
fi

# ── phase 1: persona resolve ──
echo "[driver] [phase=persona_resolve] resolving panel prompts"
uv run python scripts/issue623_persona_resolve.py \
  --output "$DATA_DIR/panel_prompts.json" "${PERSONA_ARG[@]}" \
  >"$RUN_LOGS/persona_resolve.log" 2>&1 \
  || fail "persona_resolve failed — see $RUN_LOGS/persona_resolve.log"

# ── phase 2: persona panel vectors (Method AB) ──
echo "[driver] [phase=vector_extract] extracting persona centroids (Method AB)"
# shellcheck disable=SC2086
uv run python scripts/issue623_persona_panel_vectors.py \
  --panel-prompts "$DATA_DIR/panel_prompts.json" \
  --method AB --layers $LAYERS --gpu-id "$GPU_ID" \
  --output-dir "$DATA_DIR" "${PERSONA_ARG[@]}" "${VEC_NQ_ARG[@]}" \
  >"$RUN_LOGS/vector_extract.log" 2>&1 \
  || fail "vector_extract failed — see $RUN_LOGS/vector_extract.log"

# ── phases 3 + 4: sycophancy trait vector + steering probe (K2 HALT) ──
echo "[driver] [phase=sycophancy_trait] extracting sycophancy trait vector + steering probe"
# shellcheck disable=SC2086
uv run python scripts/issue623_extract_sycophancy_vector.py \
  --layers $LAYERS --gpu-id "$GPU_ID" \
  --n-questions "$N_QUESTIONS_TRAIT" \
  --output-dir "$DATA_DIR/sycophancy_trait" \
  --steering-output "$EVAL_DIR/steering_probe.json" \
  --steering-effect-output "$EVAL_DIR/steering_effect_by_layer.json" \
  >"$RUN_LOGS/sycophancy_trait.log" 2>&1 \
  || fail "sycophancy_trait / steering_probe failed (or K2 HALT) — see $RUN_LOGS/sycophancy_trait.log"

# ── phase 5: behavioral DV syc_i (reuse #612 base rates) ──
echo "[driver] [phase=syc_i_load] resolving behavioral DV syc_i (reuse #612)"
uv run python scripts/issue623_behavioral_dv.py \
  --panel-prompts "$DATA_DIR/panel_prompts.json" \
  --output "$EVAL_DIR/syc_i.json" \
  >"$RUN_LOGS/syc_i_load.log" 2>&1 \
  || fail "syc_i_load failed — see $RUN_LOGS/syc_i_load.log"

# ── upload intermediate analysis tensors + raw completions to HF ──
if [[ "$SKIP_UPLOAD" == "0" ]]; then
  echo "[driver] [phase=upload] uploading centroids + trait + raw completions to HF"
  uv run python scripts/issue623_upload.py \
    --persona-vectors-dir "$DATA_DIR" \
    >"$RUN_LOGS/upload.log" 2>&1 \
    || fail "upload failed — see $RUN_LOGS/upload.log"
else
  echo "[driver] [phase=upload] SKIPPED (--skip-upload)"
fi

echo "[driver] [phase=done] issue 623 dispatch complete"
