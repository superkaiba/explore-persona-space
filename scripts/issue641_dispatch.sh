#!/usr/bin/env bash
# Issue #641 (Phase 2) — matched-dose install-resistance dose curves for EM.
#
# Single blocking workload entrypoint for the unified backend router's
# --workload-cmd contract. Drives the GPU pipeline phases in sequence on the
# worker (GCP A100 auto-lane); the CPU-only P4 aggregate runs OFF-POD on the VM
# after upload + terminate (plan §9/§10) and is NOT part of this workload.
#
# Phase order (each a separate blocking `uv run` process; per CLAUDE.md
# checkpoint-per-phase, every phase persists its outputs the moment it finishes):
#   1. base-propensity (P0)            — base-model harmful-advice propensity read
#   2. run / Arm A (P1+P2+P3)          — 6 sources x 2 seeds, dose ladder
#   3. select-neutral                  — mechanical matched-neutral pick (post-P0)
#   4. run / Arm B (P1+P2+P3)          — teacher vs matched neutral, fixed dose
#
# The Arm-B matched neutral is resolved at runtime by `select-neutral`, which
# writes eval_results/issue_641/base_propensity/matched_neutral.json
# ({persona_key, gap, within_floor, pool, ...}); this wrapper reads persona_key
# from it and threads it into the Arm-B `--sources sp_teacher_ho,<neutral>` call.
set -euo pipefail

# REPO_ROOT threaded from dispatch as REPO_ROOT="$WORKLOAD_ROOT" so the GCE
# clone path (/workspace/eps-issue-641) wins over any RunPod default; fall back
# to the RunPod path only when neither is set.
cd "${REPO_ROOT:-/workspace/explore-persona-space}"

# Disable tqdm progress bars (#607: vLLM \r-progress bars overflow the GCE
# metadata-runner's bounded bufio.Scanner -> SIGPIPE -> VM zombie). The startup
# script already redirects workload output to a log file, but this is belt-and-
# suspenders for any tty-like fd inherited by a workload sub-process.
export TQDM_DISABLE=1

DISPATCH="scripts/issue641_dose_curves.py"
EVAL_ROOT="${I641_EVAL_ROOT:-eval_results/issue_641}"
LADDER="50,100,150,250,375,560"

echo "[issue641] starting at $(date -u +%FT%TZ) repo=$(pwd)"

# 1. P0 — base-model harmful-advice propensity (seed 42 only, plan §10).
echo "[issue641] base-propensity at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase base-propensity --seeds 42

# 2. Arm A — 6 #537 EM source contexts x 2 seeds, dose ladder.
echo "[issue641] run Arm A at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase run \
    --sources icl_k2,wc_short_advice,sp_doctor,reph_imp,sp_ph1,wc_short_code \
    --seeds 42,1042 --max-steps 560 --save-steps 25 --save-total-limit 30 \
    --ladder "$LADDER" --probes 8 --samples 5

# 3. select-neutral — mechanical matched-neutral pick from the P0 read.
echo "[issue641] select-neutral at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase select-neutral

# Resolve the chosen neutral persona_key for the Arm-B --sources arg.
NEUTRAL="$(uv run python -c "import json,os; p=os.path.join('${EVAL_ROOT}','base_propensity','matched_neutral.json'); print(json.load(open(p))['persona_key'])")"
if [ -z "$NEUTRAL" ]; then
    echo "[issue641] FATAL: matched_neutral.json missing persona_key" >&2
    exit 1
fi
echo "[issue641] matched neutral = ${NEUTRAL}"

# 4. Arm B — teacher (sp_teacher_ho) vs matched neutral, fixed matched dose x 2 seeds.
echo "[issue641] run Arm B (sp_teacher_ho,${NEUTRAL}) at $(date -u +%FT%TZ)"
uv run python "$DISPATCH" --phase run \
    --sources "sp_teacher_ho,${NEUTRAL}" \
    --seeds 42,1042 --max-steps 560 --save-steps 25 --save-total-limit 30 \
    --ladder "$LADDER" --probes 8 --samples 5

echo "[issue641] GPU pipeline complete at $(date -u +%FT%TZ) (P4 aggregate runs off-pod on the VM)"
