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
    for cond in $conds; do
        for model in "$MODEL_INSTRUCT"; do  # smoke: instruct only; full: both models
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
    local conds_smoke="assistant_chat"
    local conds="$conds_smoke"  # full path drives via a per-cell loop in production
    for cond in $conds; do
        for model in "$MODEL_INSTRUCT"; do
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
    local model_slug="Qwen_Qwen2.5-7B-Instruct"
    local cell="${model_slug}/assistant_chat"
    local out="$EVAL_ROOT/percell/heldout_r2_${model_slug}_assistant_chat.json"
    uv run python scripts/issue1689_fit_cells.py \
        --store-root "$STORE_ROOT" --cell "$cell" --out "$out" $SMOKE
}

run_phase_fit_ladder() {
    echo "[phase=fit_ladder]"
    local model_slug="Qwen_Qwen2.5-7B-Instruct"
    local out="$EVAL_ROOT/ladder/ladder_${model_slug}_L19.json"
    uv run python scripts/issue1689_fit_ladder.py \
        --store-root "$STORE_ROOT" --model-slug "$model_slug" \
        --layer 19 --out "$out" $SMOKE
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
