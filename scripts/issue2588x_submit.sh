#!/usr/bin/env bash
# Issue #2588-larger submit wrapper (fellows cluster "charmander").
#
# Renders + submits ONE sbatch job for ONE extension model; the job runs the
# model's arms sequentially (arm a then arm b; glm53 is arm b only). One key
# per invocation — no sbatch loops.
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

KEY="${1:?usage: bash scripts/issue2588x_submit.sh <model_key>  (one of: q38fn q35_397b dsv4_flash glm53 dsv4_pro q3_32b qwq_32b q25_32b o3_32b_t)}"
: "${HF_TOKEN:?HF_TOKEN must be in the submitting shell env (read -r HF_TOKEN; export HF_TOKEN)}"

BASE=/workspace/superkaiba/eps2588x
LOGS="$BASE/logs"

# Per-model tensor-parallel width. MUST mirror PanelModel.tp_gpus in
# scripts/issue2588_panel_common.py — tests/test_issue2588x_submit_table.py
# pins this table to the registry, and the job body re-asserts the value
# against the registry before launching anything.
case "$KEY" in
  q38fn)      TP=4; BS=8 ;;
  q35_397b)   TP=4; BS=4 ;;
  dsv4_flash) TP=2; BS=4 ;;
  glm53)      TP=8; BS=4 ;;
  dsv4_pro)   TP=8; BS=2 ;;
  # Same-width (h=5120) column extension, 2026-09-02: dense bf16 ~65 GB, one GPU each.
  q3_32b)     TP=1 ;;
  qwq_32b)    TP=1 ;;
  q25_32b)    TP=1 ;;
  o3_32b_t)   TP=1 ;;
  *) echo "unknown 2588x model key: $KEY (expected q38fn|q35_397b|dsv4_flash|glm53|dsv4_pro|q3_32b|qwq_32b|q25_32b|o3_32b_t)" >&2; exit 2 ;;
esac

# HF capture forward batch (rows per teacher-forced forward). Eager attention
# materialises (B, heads, T, T) scores, so the wide / long-context MoE rows
# take smaller batches; the job script reads EPS_CAPTURE_BS (default 8) and
# it reaches the job through sbatch's default --export=ALL like HF_TOKEN.
export EPS_CAPTURE_BS="${BS:-8}"
CPUS=$(( 8 * TP ))
if [ "$CPUS" -gt 64 ]; then CPUS=64; fi
MEM=$(( 128 * TP ))

mkdir -p "$LOGS"

sbatch \
  --gres="gpu:${TP}" \
  --cpus-per-task="$CPUS" \
  --mem="${MEM}G" \
  --job-name="eps2588x-${KEY}" \
  --output="${LOGS}/%x-%j.out" \
  "$(dirname "$0")/issue2588x_cell_job.sh" "$KEY" "$TP"
