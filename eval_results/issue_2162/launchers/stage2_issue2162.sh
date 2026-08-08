#!/bin/bash
# Corrective staging: the va_anchors_*.pt shards (missing from the original
# staging list) + hardlink into the anchors dir both loaders read.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
uv run python - <<'EOF'
from explore_persona_space.orchestrate.hub import stage_hub_prefix

files = stage_hub_prefix(
    "superkaiba1/explore-persona-space-data",
    "issue2162_ctxinfo/analysis_tensors/anchors",
    "/workspace/issue2162_stage",
)
n = len(files)
print(f"STAGED issue2162_ctxinfo/analysis_tensors/anchors: {n} files (expected 16)", flush=True)
assert n == 16, f"expected 16 va_anchors shards, got {n}"
EOF
ln -f /workspace/issue2162_stage/issue2162_ctxinfo/analysis_tensors/anchors/va_anchors_*.pt \
      /workspace/issue2162_stage/issue2162_ctxinfo/raw_completions/anchors/
echo "LINKED $(ls /workspace/issue2162_stage/issue2162_ctxinfo/raw_completions/anchors/va_anchors_*.pt | wc -l) va_anchors shards into anchors dir"
echo "STAGE2 COMPLETE"
