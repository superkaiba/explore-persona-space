#!/usr/bin/env bash
# Issue #1482 G1-reconciliation probe: same capture code (bc95d8f2 lineage),
# same texts, parent machine class (A100) — discriminates cross-GPU numerics
# from code drift behind the phase-C G1 gate failure. See
# scripts/issue1482_g1probe_stage.py for context.
set -euo pipefail
REPO_ROOT="${WORKLOAD_ROOT:-$PWD}"
cd "$REPO_ROOT"
OUT=data/issue_1482/kresample
PREFIX=issue1482_kresample
uv run python scripts/issue1482_kresample.py --phase b0 --hf-prefix "$PREFIX" --out "$OUT" --skip-upload
uv run python scripts/issue1482_g1probe_stage.py --hf-prefix "$PREFIX" --out "$OUT"
uv run python scripts/issue1482_kresample.py --phase b2 --hf-prefix "$PREFIX" --out "$OUT" --skip-upload
uv run python scripts/issue1482_g1probe_stage.py --hf-prefix "$PREFIX" --out "$OUT" --upload
echo "[phase=done]"
