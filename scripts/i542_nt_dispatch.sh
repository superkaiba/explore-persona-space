#!/usr/bin/env bash
# Genuine-near-twin (follow-up `genuine-near-twin-negatives`) pod/VM-side driver.
#
# Wraps the per-phase `scripts/i542_dispatch.py` invocations for the two
# proximity arms (nt_close + xfam_long) + the seed-43 repl_nt noise-floor arm,
# at the LONGER (band-stop-OFF, 1-epoch, bystander-headroom early-stop) budget.
# Plan v4 §13. Architecturally UNIFIED with smoke (PASS_UNIFIED): the SAME
# phase entrypoints, subprocess shape, [phase=...] logging, sentinel writer,
# and teardown as the full sweep -- `--smoke` flows `--smoke` through to every
# i542_dispatch.py call and rebinds the *_smoke artifact roots.
#
# Phase order (serial, 1× A100-80):
#   p0prime  fetch (v1 freeze @ i542 pin + parent clouds) -> contexts
#            (APPEND near-twins via --regen-nt-twins) -> checks -> responses
#            (4 near-twin contexts) -> clouds (4 near-twin reduced clouds) ->
#            manip_check (Gate 1 / K2'': ratio<=0.5, ABORT-and-REPORT on fail)
#   train    build 36 mixes (16 nt_close + 16 xfam_long + 4 repl_nt) then train
#            (band-stop log-only + bystander-headroom early-stop)
#   eval     four-float slot cross-eval vs the parent base slots + V2 parity
#   gate     K1'' non-saturated-anchor gate (>5/16 out of [5,12] in either arm
#            -> ABORT-and-REPORT the realized landing table)
#   assemble per-arm G tensors + upload (mixes / responses / G npz / clouds)
# The VM-side CPU analysis (`--phase analyze --steps nt`) runs OFF-pod after
# the pod terminates (the partial-correlation read + hero figures are zero-GPU).
#
# REPO_ROOT default is the RunPod path; on the GCP lane the orchestrator
# threads REPO_ROOT="$WORKLOAD_ROOT" (gotchas.md #599 -- the GCE startup script
# clones to $WORKLOAD_ROOT). Pod-side code NEVER shells out to scripts/task.py
# (CLAUDE.md): progress flows via the per-phase sentinels i542_dispatch.py
# writes to /workspace/logs (drained by backend_poll.py / poll_pipeline.py).
#
# CVD_PIN_EXEMPT: serial single-GPU run (--gpu-id 0, no backgrounded parallel
# cells), so the per-cell CUDA_VISIBLE_DEVICES co-location trap does not apply.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
# Route trainer + WandB tempfiles OFF the FUSE mount on RunPod (gotchas.md
# D-state zombie mitigation); GCP ignores a non-MooseFS path harmlessly.
export WANDB_DIR="${WANDB_DIR:-/tmp/issue_542_wandb}"
mkdir -p "$WANDB_DIR"

SMOKE=0
ARMS=("nt_close" "xfam_long" "repl_nt")
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --arm=*) ARMS=("${arg#--arm=}") ;;
        *) ;;
    esac
done
SMOKE_FLAG=()
[ "$SMOKE" -eq 1 ] && SMOKE_FLAG=("--smoke")

echo "[phase=preflight] === i542 near-twin dispatcher $(date -Iseconds) arms=${ARMS[*]} smoke=$SMOKE REPO_ROOT=$REPO_ROOT ==="

# Marker-token assert at launch (defense-in-depth; the dispatcher's _tokenizer()
# asserts it in-process too). set -a/source .env keeps HF_TOKEN available.
set -a
[ -f "$REPO_ROOT/.env" ] && source "$REPO_ROOT/.env"
set +a
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
assert tok.encode(' ※', add_special_tokens=False) == [83399], 'marker token id drift'
assert tok.convert_tokens_to_ids('<|im_end|>') == 151645, '<|im_end|> id drift'
print('marker token id OK: 83399; <|im_end|> OK: 151645')
"

dispatch() {
    echo "[phase=dispatch] i542_dispatch.py $*"
    uv run python scripts/i542_dispatch.py "$@" "${SMOKE_FLAG[@]}"
}

# --- p0prime: fetch -> contexts(append near-twins) -> checks -> responses ->
#     clouds -> manip_check (Gate 1). The manip_check ABORTS the run on FAIL
#     (set -e propagates the SystemExit), which is the deliberate
#     abort-and-report fallback (the failed-manipulation sentinel is written
#     by the dispatcher before it exits).
dispatch --phase p0prime --arm nt_close \
    --steps fetch,contexts,checks,responses,clouds,manip_check

# --- train + eval per arm (serial, 1 GPU). ---
for arm in "${ARMS[@]}"; do
    dispatch --phase train --arm "$arm" --gpu-id 0
    dispatch --phase eval --arm "$arm" --gpu-id 0
done

# --- K1'' non-saturated-anchor gate (CPU). Reads nt_close + xfam_long
#     diagonals; >5/16 out of band in either arm -> ABORT-and-REPORT (the
#     dispatcher raises after writing the landing table + sentinel). Only run
#     when BOTH proximity arms were trained this invocation. ---
if [[ " ${ARMS[*]} " == *" nt_close "* && " ${ARMS[*]} " == *" xfam_long "* ]]; then
    dispatch --phase gate --steps k1prime_nt
fi

# --- assemble per-arm G tensors + upload. ---
for arm in "${ARMS[@]}"; do
    dispatch --phase assemble --arm "$arm" --steps arms
done
# Upload all mixes / responses / G npz / clouds in one pass (idempotent resume).
if [ "$SMOKE" -eq 0 ]; then
    dispatch --phase assemble --arm nt_close --steps upload
fi

# End-of-run results sentinel with the reproducibility card (training task).
# poll_pipeline.py drains this; it MUST carry _SENTINEL_REQUIRED_KEYS
# (sentinel_schema_version / kind / version / note) + the per-cell adapter
# paths + WandB run names (workflow.yaml § markers epm:results). Pod-side code
# NEVER shells out to task.py -- the orchestrator posts the marker from this
# sentinel. The intermediate per-phase [phase=done] lines above are mid-run
# noise the poller survives via PID/sentinel corroboration (#545); the
# AUTHORITATIVE terminal [phase=done] is the final echo below.
if [ "$SMOKE" -eq 0 ]; then
    EPM_RESULTS_ARMS="${ARMS[*]}" uv run python scripts/i542_nt_results_sentinel.py
fi

echo "[phase=done] i542 near-twin dispatcher complete $(date -Iseconds)"
