#!/bin/bash
# Apply local patches to submodule dependencies (rllm + verl).
# Run after: git submodule update --init --recursive
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCHES_DIR="$REPO_ROOT/patches"
RLLM_DIR="$REPO_ROOT/terminal-bench-rl/external/rllm"
VERL_DIR="$RLLM_DIR/verl"

echo "Applying rllm patches..."
git -C "$RLLM_DIR" apply --check "$PATCHES_DIR/rllm.patch" 2>/dev/null && \
  git -C "$RLLM_DIR" apply "$PATCHES_DIR/rllm.patch" || \
  echo "  (already applied or conflicts — skipping)"

echo "Applying verl patches..."
git -C "$VERL_DIR" apply --check "$PATCHES_DIR/verl.patch" 2>/dev/null && \
  git -C "$VERL_DIR" apply "$PATCHES_DIR/verl.patch" || \
  echo "  (already applied or conflicts — skipping)"

echo "Done."
