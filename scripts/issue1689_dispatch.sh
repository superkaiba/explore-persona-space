#!/usr/bin/env bash
# Issue #1689 phase dispatcher — invoked via
#   `dispatch_issue.py launch --workload-cmd 'bash scripts/issue1689_dispatch.sh <phase>' --repo-branch issue-1689`.
#
# Emits pod-side sentinel JSONs at /workspace/logs/issue-1689-*.json
# that poll_pipeline.py drains (per CLAUDE.md pod-side reporting contract).
# Prints [phase=<name>] log lines the poller matches; [phase=done] on
# graceful completion + writes the epm:results payload sentinel.
#
# Phases: corpus → render → haiku_u2 → onpolicy → capture → fit_cells →
#         fit_ladder → analyze
#
# Smoke: bash scripts/issue1689_dispatch.sh --smoke  (runs full pipeline
# at tiny n=5 slice through every phase - the PASS_UNIFIED architecture).
set -euo pipefail

# Self-resolve REPO_ROOT: lane-agnostic per the #825 crash-fix precedent
# (never bare REPO_ROOT="$WORKLOAD_ROOT" under set -u).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"

# Conditional dotenv sourcing (RunPod pushes .env; GCE lane has none).
# See gotchas.md § "source ./.env CONDITIONALLY".
if [ -f ./.env ]; then
    set -a
    . ./.env
    set +a
fi

ISSUE_NUM=1689
ISSUE_SLUG=speaker_lattice
DATA_ROOT="${DATA_ROOT:-data/issue_${ISSUE_NUM}}"
EVAL_ROOT="${EVAL_ROOT:-eval_results/issue_${ISSUE_NUM}}"
STORE_ROOT="${STORE_ROOT:-analysis_tensors/issue_${ISSUE_NUM}/store}"
FIG_ROOT="${FIG_ROOT:-figures/issue_${ISSUE_NUM}}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$DATA_ROOT" "$EVAL_ROOT" "$STORE_ROOT" "$FIG_ROOT"
# LOG_DIR may not be writable off-pod; skip mkdir if unwritable.
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp"

SMOKE=""
PHASE=""
if [ "${1:-}" = "--smoke" ]; then
    SMOKE="--smoke"
    PHASE="all"
elif [ -n "${1:-}" ]; then
    PHASE="$1"
    if [ "${2:-}" = "--smoke" ]; then
        SMOKE="--smoke"
    fi
else
    PHASE="all"
fi

MODEL_INSTRUCT="Qwen/Qwen2.5-7B-Instruct"
MODEL_BASE="Qwen/Qwen2.5-7B"

run_phase_corpus() {
    echo "[phase=corpus]"
    local out="$DATA_ROOT/two_turn_lmsys.jsonl"
    if [ -n "$SMOKE" ]; then
        # Write a tiny synthetic fixture; the real corpus phase uses
        # scripts/issue1689_gen_corpus.py.
        mkdir -p "$(dirname "$out")"
        uv run python -c "
import json
from pathlib import Path
rows = [
    {'conv_id': f'smoke-{i}', 'u1': f'Hello {i}?', 'a1': f'Reply to {i}.',
     'u2_lmsys': f'Follow-up {i}.'}
    for i in range(5)
]
Path('$out').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
print(f'[corpus] wrote 5 smoke rows to $out')
"
    else
        uv run python scripts/issue1689_gen_corpus.py \
            --out "$out" --n 3800
    fi
}

run_phase_render() {
    echo "[phase=render]"
    local corpus="$DATA_ROOT/two_turn_lmsys.jsonl"
    local out="$DATA_ROOT/rendered.jsonl"
    uv run python scripts/issue1689_render_conditions.py \
        --in "$corpus" --out "$out" --conditions all $SMOKE
}

run_phase_haiku_u2() {
    echo "[phase=haiku_u2]"
    local rendered_dir="$DATA_ROOT/rendered.jsonl"
    for cond in user_haiku_chat user_haiku_naturalistic user_haiku_story; do
        local input="$rendered_dir/${cond}.jsonl"
        local out="$DATA_ROOT/haiku_u2/${cond}.jsonl"
        uv run python scripts/issue1689_haiku_u2_gen.py \
            --in "$input" --out "$out" --condition "$cond" $SMOKE
        [ -z "$SMOKE" ] || break  # smoke: one condition only
    done
}

run_phase_onpolicy() {
    echo "[phase=onpolicy]"
    local rendered_dir="$DATA_ROOT/rendered.jsonl"
    local conds_smoke="assistant_chat"
    local conds_full="assistant_chat assistant_naturalistic assistant_story helios_chat helios_naturalistic helios_story wren_chat wren_naturalistic wren_story dana_chat dana_naturalistic dana_story user_onpolicy_chat user_onpolicy_naturalistic user_onpolicy_story"
    local conds="$conds_full"
    [ -n "$SMOKE" ] && conds="$conds_smoke"
    # Both models iterated in smoke AND full modes — H5 (base vs instruct
    # parity) requires the base arm; a single-model smoke leaves the base
    # branch unexercised (round-2 concern dispatch-base-model-missing).
    local models_smoke="$MODEL_INSTRUCT"
    local models_full="$MODEL_BASE $MODEL_INSTRUCT"
    local models="$models_full"
    [ -n "$SMOKE" ] && models="$models_smoke"  # smoke: instruct only (1 model × 1 cond); full: BOTH
    # NOTE: only the smoke leg is restricted to instruct-only to bound smoke wall-time;
    # full mode iterates both MODEL_BASE and MODEL_INSTRUCT per plan §5 H5.
    for cond in $conds; do
        for model in $models; do
            local input="$rendered_dir/${cond}.jsonl"
            local out="$DATA_ROOT/onpolicy/${cond}_$(basename "$model").jsonl"
            local stats="$EVAL_ROOT/onpolicy_stats/${cond}_$(basename "$model").json"
            uv run python scripts/issue1689_gen_onpolicy.py \
                --in "$input" --out "$out" \
                --stats-out "$stats" \
                --condition "$cond" --model "$model" $SMOKE
        done
    done
}

run_phase_capture() {
    echo "[phase=capture]"
    local rendered_dir="$DATA_ROOT/rendered.jsonl"
    # R13 fix: capture the FULL 21-condition lattice per plan §4/§5, not a
    # smoke-slice. The stale "full path drives via a per-cell loop in
    # production" comment described no such loop; production silently ran
    # smoke-scale (assistant_chat only) and Phase D fit_ladder crashed on
    # missing cells (assistant_naturalistic/L14.pt) after ~13h of upstream
    # compute.
    local conds_smoke="assistant_chat"
    local conds_full="assistant_chat assistant_naturalistic assistant_story dana_chat dana_naturalistic dana_story helios_chat helios_naturalistic helios_story wren_chat wren_naturalistic wren_story user_haiku_chat user_haiku_naturalistic user_haiku_story user_lmsys_chat user_lmsys_naturalistic user_lmsys_story user_onpolicy_chat user_onpolicy_naturalistic user_onpolicy_story"
    local conds="$conds_full"
    [ -n "$SMOKE" ] && conds="$conds_smoke"
    # Same both-models discipline as run_phase_onpolicy: H5 (base vs instruct
    # parity) needs both arms captured. Smoke stays instruct-only to bound wall.
    local models_smoke="$MODEL_INSTRUCT"
    local models_full="$MODEL_BASE $MODEL_INSTRUCT"
    local models="$models_full"
    [ -n "$SMOKE" ] && models="$models_smoke"
    for cond in $conds; do
        for model in $models; do
            # Idempotent skip-if-exists: capture is expensive (LoRA-adapter-free
            # forward passes for prefix/context/answer at 4 layers), and a
            # partial capture (e.g. assistant_chat × both models already
            # completed under R11) is byte-identical to what this rerun would
            # produce (same code path, same inputs). Skip populated cells so a
            # resume can complete the missing 40/42 cells without re-doing
            # the 2 already captured. See save_cell() in issue1689_capture.py:
            # model_slug = model_name.replace("/", "_") → "Qwen_Qwen2.5-7B" /
            # "Qwen_Qwen2.5-7B-Instruct".
            local model_slug
            model_slug=$(echo "$model" | tr '/' '_')
            local cell_dir="${STORE_ROOT}/${model_slug}/${cond}"
            if [ -d "$cell_dir" ] && [ -n "$(ls -A "$cell_dir" 2>/dev/null)" ]; then
                echo "[capture] SKIP ${cell_dir} (already populated)"
                continue
            fi
            local input="$DATA_ROOT/onpolicy/${cond}_$(basename "$model").jsonl"
            [ ! -f "$input" ] && input="$rendered_dir/${cond}.jsonl"
            local skip_upload_flag=""
            [ -n "$SMOKE" ] && skip_upload_flag="--skip-upload"
            uv run python scripts/issue1689_capture.py \
                --in "$input" --out-root "$STORE_ROOT" \
                --condition "$cond" --model "$model" \
                $SMOKE $skip_upload_flag
        done
    done
}

run_phase_fit_cells() {
    echo "[phase=fit_cells]"
    # R13 fix: iterate the FULL 42-cell (2 models × 21 conditions) lattice, not
    # the single hardcoded (Qwen_Qwen2.5-7B-Instruct, assistant_chat) cell.
    # Idempotent skip-if-exists: fit_cells writes one JSON per cell; a resume
    # skips the 1 completed cell (heldout_r2_Qwen_Qwen2.5-7B-Instruct_assistant_chat.json
    # under R11) and processes the remaining 41.
    local models_full="Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct"
    local models_smoke="Qwen_Qwen2.5-7B-Instruct"
    local models="$models_full"
    local conds_full="assistant_chat assistant_naturalistic assistant_story dana_chat dana_naturalistic dana_story helios_chat helios_naturalistic helios_story wren_chat wren_naturalistic wren_story user_haiku_chat user_haiku_naturalistic user_haiku_story user_lmsys_chat user_lmsys_naturalistic user_lmsys_story user_onpolicy_chat user_onpolicy_naturalistic user_onpolicy_story"
    local conds_smoke="assistant_chat"
    local conds="$conds_full"
    if [ -n "$SMOKE" ]; then
        models="$models_smoke"
        conds="$conds_smoke"
    fi
    for model_slug in $models; do
        for cond in $conds; do
            local cell="${model_slug}/${cond}"
            local out="$EVAL_ROOT/percell/heldout_r2_${model_slug}_${cond}.json"
            if [ -f "$out" ]; then
                echo "[fit_cells] SKIP ${out} (already exists)"
                continue
            fi
            uv run python scripts/issue1689_fit_cells.py \
                --store-root "$STORE_ROOT" --cell "$cell" --out "$out" $SMOKE
        done
    done
}

run_phase_fit_ladder() {
    echo "[phase=fit_ladder]"
    # Iterate BOTH models in full mode (H5 parity); smoke stays instruct-only.
    # --all-layers loops over CAPTURE_LAYERS=(14,18,19,26) per plan §6 exploratory dump;
    # headline layer stays 19.
    local models_smoke="Qwen_Qwen2.5-7B-Instruct"
    local models_full="Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct"
    local models="$models_full"
    [ -n "$SMOKE" ] && models="$models_smoke"
    # In smoke mode we run only the headline layer (19) for wall-time; full mode iterates all 4.
    local all_layers_flag="--all-layers"
    [ -n "$SMOKE" ] && all_layers_flag=""
    for model_slug in $models; do
        local out="$EVAL_ROOT/ladder/ladder_${model_slug}_L19.json"
        uv run python scripts/issue1689_fit_ladder.py \
            --store-root "$STORE_ROOT" --model-slug "$model_slug" \
            --layer 19 --out "$out" $all_layers_flag $SMOKE
    done
}

run_phase_analyze() {
    echo "[phase=analyze]"
    local model_slug="Qwen_Qwen2.5-7B-Instruct"
    local ladder="$EVAL_ROOT/ladder/ladder_${model_slug}_L19.json"
    uv run python scripts/issue1689_analyze.py \
        --ladder-json "$ladder" --out-figs "$FIG_ROOT" \
        --out-manifest "$EVAL_ROOT/manifest.json" \
        --model "$model_slug" $SMOKE
}

# Dispatch on phase argument.
case "$PHASE" in
    corpus)   run_phase_corpus ;;
    render)   run_phase_render ;;
    haiku_u2) run_phase_haiku_u2 ;;
    onpolicy) run_phase_onpolicy ;;
    capture)  run_phase_capture ;;
    fit_cells|fit) run_phase_fit_cells ;;
    fit_ladder|ladder) run_phase_fit_ladder ;;
    analyze)  run_phase_analyze ;;
    all|--smoke|"")
        run_phase_corpus
        run_phase_render
        run_phase_haiku_u2
        run_phase_onpolicy
        run_phase_capture
        run_phase_fit_cells
        run_phase_fit_ladder
        run_phase_analyze
        ;;
    *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac

# Emit epm:results sentinel per pod-side reporting contract.
SENTINEL="$LOG_DIR/issue-${ISSUE_NUM}-results.json"
cat > "$SENTINEL" <<EOF
{
  "issue": ${ISSUE_NUM},
  "phase_run": "$PHASE",
  "smoke": $([ -n "$SMOKE" ] && echo "true" || echo "false"),
  "eval_dir": "$EVAL_ROOT",
  "fig_dir": "$FIG_ROOT",
  "status": "done"
}
EOF
echo "[phase=done]"
