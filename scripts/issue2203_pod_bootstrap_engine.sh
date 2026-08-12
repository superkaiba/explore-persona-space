#!/usr/bin/env bash
# Issue #2203 — 32B pod bootstrap: deliver the paper's capping engine (§9, Fix D).
#
# external/assistant-axis is git-UNTRACKED on EVERY branch (plan §12), so the
# standard repo clone does NOT deliver it. This clones safety-research/
# assistant-axis at the pinned SHA into <repo>/external/assistant-axis and gates
# the 32B phase on the pod-side --import-check (which loads the paper's
# steering.py file-scoped — torch+typing only, avoiding the package __init__'s
# plotly/sklearn imports; see paper_engine.py). Run BEFORE any 32B generation.
#
# No GPU launches here (a CPU-only import gate), so the CVD-pin lint is N/A.
set -euo pipefail

PINNED_SHA="a98961956072224eaf244eb289d6c01700b63795"
REPO_URL="https://github.com/safety-research/assistant-axis.git"

# Repo root = the parent of this script's scripts/ dir (pod checkout layout).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/external/assistant-axis"

echo "[phase=bootstrap_engine] repo_root=$REPO_ROOT dest=$DEST pinned=$PINNED_SHA"

if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$REPO_ROOT/external"
    echo "[phase=bootstrap_engine] cloning $REPO_URL -> $DEST"
    git clone "$REPO_URL" "$DEST"
fi

git -C "$DEST" fetch --quiet origin "$PINNED_SHA" 2>/dev/null || git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$PINNED_SHA"
HEAD_SHA="$(git -C "$DEST" rev-parse HEAD)"
if [ "$HEAD_SHA" != "$PINNED_SHA" ]; then
    echo "[phase=bootstrap_engine] FATAL: HEAD $HEAD_SHA != pinned $PINNED_SHA" >&2
    exit 1
fi
echo "[phase=bootstrap_engine] checkout OK ($HEAD_SHA)"

# Import gate: the file-scoped paper-engine load must resolve on the pod.
uv run python "$REPO_ROOT/scripts/issue2203_phase3.py" --import-check
echo "[phase=bootstrap_engine] ok"
