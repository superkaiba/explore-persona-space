#!/usr/bin/env bash
# Issue #2588-larger submit wrapper (fellows cluster "charmander").
#
# Renders + submits ONE sbatch job for ONE panel model (ANY key in
# issue2588_panel_common.PANEL); the job runs the model's arms sequentially
# (arm a then arm b where both exist). One key per invocation — no sbatch
# loops.
#
# Exact submit form — feed the HF token via stdin so it never lands in shell
# history or on disk:
#
#   read -r HF_TOKEN; export HF_TOKEN; bash scripts/issue2588x_submit.sh q38fn
#
# The token reaches the job through sbatch's DEFAULT environment export (this
# script passes NO --export flag; sbatch defaults to --export=ALL). The token
# is never written to any file, and nothing here touches /home.

set -euo pipefail

# Overridable for the functional test only (a stub sbatch on PATH, BASE
# pointed at a tmp dir whose repo/ symlinks the real tree, PY at the test
# interpreter). Production submits use the defaults.
BASE="${EPS2588X_BASE:-/workspace/superkaiba/eps2588x}"
LOGS="$BASE/logs"
PY="${EPS2588X_PY:-$BASE/venv/bin/python}"
export PYTHONPATH="$BASE/repo/src:$BASE/repo/scripts"

KEY="${1:-}"
if [ -z "$KEY" ]; then
  echo "usage: bash scripts/issue2588x_submit.sh <model_key>" >&2
  echo "known keys (panel registry): $("$PY" -c 'import issue2588_panel_common as PC; print(" ".join(PC.PANEL))')" >&2
  exit 2
fi
: "${HF_TOKEN:?HF_TOKEN must be in the submitting shell env (read -r HF_TOKEN; export HF_TOKEN)}"

# Single source of truth: validate the key and read the tensor-parallel width
# from the panel registry (PC.PANEL[key].tp_gpus). The job body re-asserts the
# same value against the registry before launching anything.
TP="$("$PY" -c '
import sys

import issue2588_panel_common as PC

k = sys.argv[1]
if k not in PC.PANEL:
    msg = "unknown 2588x model key: " + k + " (known: " + " ".join(PC.PANEL) + ")"
    print(msg, file=sys.stderr)
    raise SystemExit(2)
print(PC.PANEL[k].tp_gpus)
' "$KEY")"

# Cap profile (issue #2659 truncation rerun): EPS_CAP_PROFILE selects the
# max_new_tokens table in issue2588_panel_common.CAP_PROFILES (default "v1"
# the original caps, "long" the rerun table). It reaches the job through
# sbatch's default environment export exactly like HF_TOKEN. The job body
# validates the name on its first python call (panel_common raises at import
# on an unknown profile, before any GPU work). Non-v1 jobs are named
# eps2588x-<profile>-<key> so the queue shows which profile a job runs.
EPS_CAP_PROFILE="${EPS_CAP_PROFILE:-v1}"
export EPS_CAP_PROFILE
# Single-phase mode: EPS_PHASE=<phase> (default all) makes the job run ONE
# run_cell invocation of that phase on the model's FIRST arm only. Use case:
# --phase g2-anchor is cell-independent and must be published once per cap
# profile before any long-profile fits phase can pass its sentinel await.
# The phase rides the job name so the queue shows it.
EPS_PHASE="${EPS_PHASE:-all}"
export EPS_PHASE
JOB_NAME="eps2588x-${KEY}"
if [ "$EPS_CAP_PROFILE" != "v1" ]; then
  JOB_NAME="eps2588x-${EPS_CAP_PROFILE}-${KEY}"
fi
if [ "$EPS_PHASE" != "all" ]; then
  JOB_NAME="${JOB_NAME}-${EPS_PHASE}"
fi

# HF capture forward batch (rows per teacher-forced forward, EPS_CAPTURE_BS)
# + per-GPU headroom (GiB, EPS_CAPTURE_HEADROOM_GIB). Eager attention
# materialises (B, heads, T, T) scores, so the wide / long-context MoE rows
# take smaller batches, and DeepSeek-V4-Flash keeps 40 GiB free (job 62667:
# 113.7 GiB allocated + 21.8 GiB fragmentation on GPU 1 at HR=24, and
# DeepseekV4 has _supports_sdpa=False so eager scores are unavoidable).
# Every other registry key takes the defaults BS=8 HR=24. Both reach the job
# through sbatch's default environment export like HF_TOKEN.
BS=8
HR=24
case "$KEY" in
  q38fn)      BS=8 ;;
  q35_397b)   BS=4 ;;
  dsv4_flash) BS=2; HR=40 ;;
  glm53)      BS=4 ;;
  dsv4_pro)   BS=1 ;;
esac
export EPS_CAPTURE_BS="$BS"
export EPS_CAPTURE_HEADROOM_GIB="$HR"
CPUS=$(( 8 * TP ))
if [ "$CPUS" -gt 64 ]; then CPUS=64; fi
MEM=$(( 128 * TP ))

mkdir -p "$LOGS"

sbatch \
  --gres="gpu:${TP}" \
  --cpus-per-task="$CPUS" \
  --mem="${MEM}G" \
  --job-name="$JOB_NAME" \
  --output="${LOGS}/%x-%j.out" \
  "$(dirname "$0")/issue2588x_cell_job.sh" "$KEY" "$TP"
