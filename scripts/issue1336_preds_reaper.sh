#!/usr/bin/env bash
# Pod-side preds janitor for the #1336 rigid-decomposition run.
#
# WHY. issue1336_metric_ladder.py writes a per-pair fp16 preds npz UNCONDITIONALLY
# (np.savez at metric_ladder.py:1331 — no flag disables it; --preds-dir only
# redirects). Each file is ~223 KB per ROW per pair, so it scales linearly in the
# surface's n: 294 MB on chat/gsm8k_test1319 (n=1319), and ~3.9 GB apiece on
# lmsys23k (n~17.7k). Across 7 pairs x 8 surfaces that projects to ~128 GB, on a
# 240 GB disk that ALSO needs up to ~87 GB of staged turnstores per surface. The
# main launcher's between-surface reap clears only the turnstores, so without this
# the run hits its own fail-loud disk assert around surface 7-8 and aborts.
#
# WHAT IS DISCARDED AND WHY THAT IS SAFE. Only the large intermediate TENSOR. The
# round's actual deliverable — the per-pair R2 summaries incl. the orth_tiers block
# — lives in the pair JSON (~45 KB each) and is never touched. The npz is
# regenerable: re-run the launcher for that surface (stage from the HF turnstores,
# which are already persisted, then ~101 s per battery). Declared as a discard with
# that regen recipe in the task marker, per the upload policy.
#
# SAFETY. A file is deleted ONLY when (a) its pair JSON already exists — i.e. the
# battery that wrote it completed — and (b) it has not been modified for >= 15 min,
# so the in-flight writer can never be the target. The manifest JSON is kept.
# Exits when the main run's sentinel lands, or when the main pid is gone.
set -uo pipefail

PREDS=/workspace/explore-persona-space/data/issue_1336/metric_ladder_preds
PAIRS=/workspace/eval_results/issue_1336_rigid/metric_ladder
SENTINEL=/workspace/logs/issue-1336-rigid-done.json
MAINPID_FILE=/workspace/logs/issue-1336-rigid.pid
LOG=/workspace/logs/issue-1336-preds-reaper.log

echo $$ > /workspace/logs/issue-1336-preds-reaper.pid
echo "[reaper] start $(date -u +%FT%TZ) preds=$PREDS pairs=$PAIRS" >> "$LOG"

while true; do
  freed=0
  if [ -d "$PREDS" ]; then
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      base=$(basename "$f")                       # ladpreds_<unit>.npz
      unit="${base#ladpreds_}"; unit="${unit%.npz}"
      if [ -f "$PAIRS/pair_${unit}.json" ]; then
        sz=$(du -m "$f" 2>/dev/null | cut -f1)
        rm -f "$f" && freed=$((freed + ${sz:-0}))
        echo "[reaper] $(date -u +%FT%TZ) discarded $base (${sz:-?}MB; pair JSON present)" >> "$LOG"
      fi
    done < <(find "$PREDS" -maxdepth 1 -name 'ladpreds_*.npz' -mmin +15 2>/dev/null)
  fi
  [ "$freed" -gt 0 ] && echo "[reaper] $(date -u +%FT%TZ) freed ${freed}MB; df: $(df -h /workspace | tail -1)" >> "$LOG"

  if [ -f "$SENTINEL" ]; then
    echo "[reaper] $(date -u +%FT%TZ) main run sentinel present — final sweep done, exiting" >> "$LOG"
    break
  fi
  p=$(cat "$MAINPID_FILE" 2>/dev/null)
  if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then
    echo "[reaper] $(date -u +%FT%TZ) main pid $p gone — exiting" >> "$LOG"
    break
  fi
  sleep 300
done
