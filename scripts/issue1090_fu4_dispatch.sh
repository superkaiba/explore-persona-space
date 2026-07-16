#!/usr/bin/env bash
# #1090 fu4-extended-dose-lr GPU-lane dispatcher: stage/verify frozen mixes ->
# 9 retrains (3 cells x lr {1e-5,3e-5,1e-4}) @ epochs=15 / save_steps=5 ->
# Tier-1 ladder (all rungs, judge@300 / structural) -> dose-select vs
# (0.60, 0.85) -> Tier-2 generation (trained arm; base reused from fu3) ->
# tf-margin at the selected rung -> per-run upload -> sentinel -> [phase=done].
#
# All science lives in scripts/issue1090_fu4.py (the unified smoke/full
# driver); this wrapper owns env setup, the WandB project pin, and the SINGLE
# terminal [phase=done] line (reserved token — the poller's done signal; the
# python driver never emits it). Mirrors scripts/issue1090_fu2_dispatch.sh.
#
# Usage (the FULL pod/GCE-side entrypoint; smokes invoke the python driver
# directly with --smoke):
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1090_fu4_dispatch.sh \
#     --manifest eval_results/issue_1090/fu4-extended-dose-lr/cell_manifest_fu4.json \
#     [driver args...]
#
# fu5 round (plan v7 `fu5-finish-impolite-bare-and-formatting-rank`): SAME
# wrapper — "$@" forwards --round + the fu5 manifest verbatim (D2 item 3).
# Run the pod-side K5 rank-seam smoke FIRST (plan §7 K5; never fan out on a
# failed K5), then dispatch:
#   uv run python scripts/issue1090_fu4.py --full --round fu5 --phase k5-rank-smoke \
#     --allow-unpinned-gpu --out-root /workspace/data/issue_1090/fu5_k5smoke
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1090_fu4_dispatch.sh --round fu5 \
#     --manifest eval_results/issue_1090/finish-impolite-bare-and-formatting-rank/cell_manifest_fu5.json
#
# THIS usage line governs the launch composer (plan §10's sketches drift from
# the shipped CLI — code-review v16 Minor 3):
#   * the manifest MUST already be COMMITTED to the issue-1090 branch before
#     dispatch (the GCP lane is git-clone-only; an uncommitted manifest kills
#     all workers at _load_manifest). Produce it VM-side first:
#       uv run python scripts/issue1090_fu4.py --full [--round fu5] --phase stage \
#         --manifest-out <deliverables_dir>/cell_manifest_<round>.json
#   * the driver REQUIRES --smoke|--full (this wrapper defaults --full) and
#     the stage phase takes --manifest-out, not --manifest.
#   * a full-mode worker REFUSES to train without its manifest sha pin.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional inside the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

export WANDB_PROJECT="${WANDB_PROJECT:-issue1090}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "[phase=dispatch] issue1090_fu4 dispatcher starting (args: $*)"

# Default to --full; a caller passing --smoke gets the tiny-real path through
# the SAME dispatcher.
MODE="--full"
for arg in "$@"; do
    if [ "$arg" = "--smoke" ] || [ "$arg" = "--full" ]; then MODE=""; fi
done

rc=0
# shellcheck disable=SC2086
uv run python scripts/issue1090_fu4.py $MODE --phase dispatch "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=driver_failed] issue1090_fu4 driver exited rc=$rc" >&2
    exit "$rc"
fi

echo "[phase=done]"
