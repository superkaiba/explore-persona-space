#!/usr/bin/env bash
# Issue #1901 inline round `encbaseline` — pod-side phase chain.
# Detached launcher: stage -> embed(bge) -> score(bge) -> embed(e5) -> score(e5).
# Fail-fast (set -e): a phase failure stops the chain rather than scoring
# against a partial pool.
set -euo pipefail

cd /workspace/explore-persona-space

S=scripts/issue1901_encoder_semantic_baseline.py
OUT=eval_results/issue_1901/encbaseline
N_CHUNKS="${N_CHUNKS:-112}"

mkdir -p "$OUT"

echo "=== [encbaseline] stage (${N_CHUNKS} chunks) $(date -u +%FT%TZ) ==="
uv run python "$S" stage --n-chunks "$N_CHUNKS"

for EMB in bge e5; do
  echo "=== [encbaseline] embed ${EMB} $(date -u +%FT%TZ) ==="
  uv run python "$S" embed --embedder "$EMB" --batch 128

  echo "=== [encbaseline] fit-score ${EMB} $(date -u +%FT%TZ) ==="
  uv run python "$S" fit-score \
    --embedder "$EMB" \
    --solver-gate \
    --eval-out "${OUT}/results_${EMB}.json"
done

echo "=== [encbaseline] CHAIN COMPLETE $(date -u +%FT%TZ) ==="
