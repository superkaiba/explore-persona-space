#!/usr/bin/env bash
# Issue #816 pod-side driver — Persona Vectors' 3 NON-prediction experiments.
#
# UNIFIED smoke = sweep: the SAME driver runs the 1-cell smoke and the full
# sweep; smoke is this script with EPM_I816_SMOKE=1 (or the env knobs below),
# which threads --cells 1 / tiny slices into the Python fan-out dispatcher, which
# in turn threads --cells into EVERY per-cell entrypoint (probe/steering/
# preventative/screening). No divergent smoke path.
#
# Sequences (all phases emit [phase=<name>] structured JSON the poller parses;
# the SINGLE terminal [phase=done] fires only on graceful completion):
#   step 0  setup      : clone safety-research/persona_vectors@b8e0f04 + unzip dataset.zip
#   step 0b preflight  : REQUIRED §12 assert — every consumed #778 finetune JSON present
#   step 1  dispatch   : Phase-A 8-GPU fan-out (probe + Exp-2 + Exp-4 + Exp-5 capture)
#   step 2  upload     : Exp-4 adapters (HF model) + Exp-5 tensors + raw gens (HF data)
#   (Phase B judge + Phase C null battery run OFF-POD on the VM after pod release)
#
# Env knobs (all optional; defaults = full production sweep):
#   EPM_I816_SMOKE=1     -> tiny slice (--cells 1, few q/rollouts, capped steps)
#   EPM_I816_TRAITS      -> space-separated trait list (default: evil sycophancy hallucination)
#   EPM_I816_PHASES      -> which phases (default: probe steering preventative screening)
#   EPM_I816_N_RANDOM_DIRS  -> Exp-4 random dirs per trait (default 10; 20 if wall-clock permits)
#   EPM_I816_N_SAMPLES   -> Exp-5 samples/dataset (default 500)
#   EPM_I816_SKIP_UPLOAD=1  -> skip step 2 (smoke/debug)
#
# REPO_ROOT resolves via ${REPO_ROOT:-...}; the GCE startup script exports
# REPO_ROOT=$WORKLOAD_ROOT before running the workload (#641), and RunPod runs
# from the clone dir, so the default only matters for a bare local invocation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

ISSUE=816
SLUG="persona_vectors"
PV_SHA="b8e0f044fe2410a6fad579f38324f03f13b4e917"
PV_REPO="https://github.com/safety-research/persona_vectors.git"
EXTERNAL_ROOT="external/persona_vectors"
LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"

# Load credentials at entry (uv run does NOT auto-load .env). Source into the
# shell env so every uv subprocess inherits (the canonical heredoc-safe idiom).
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

TRAITS="${EPM_I816_TRAITS:-evil sycophancy hallucination}"
PHASES="${EPM_I816_PHASES:-probe steering preventative screening}"
N_RANDOM_DIRS="${EPM_I816_N_RANDOM_DIRS:-10}"
N_SAMPLES="${EPM_I816_N_SAMPLES:-500}"
EXTRA=""
if [ "${EPM_I816_SMOKE:-0}" = "1" ]; then
  TRAITS="${EPM_I816_TRAITS:-evil}"
  EXTRA="--cells 1 --n-questions 2 --n-rollouts 1 --max-steps 2 --n-samples 3"
fi

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# ── step 0: pod-side external-repo clone + unzip ───────────────────────────────
log_phase setup "cloning $PV_REPO @ $PV_SHA"
if [ ! -f "$EXTERNAL_ROOT/dataset.zip" ] && [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  mkdir -p external
  if [ ! -d "$EXTERNAL_ROOT/.git" ]; then
    rm -rf "$EXTERNAL_ROOT"
    git clone "$PV_REPO" "$EXTERNAL_ROOT"
  fi
  git -C "$EXTERNAL_ROOT" fetch --depth 1 origin "$PV_SHA"
  git -C "$EXTERNAL_ROOT" checkout "$PV_SHA"
fi
GOT_SHA="$(git -C "$EXTERNAL_ROOT" rev-parse HEAD)"
if [ "$GOT_SHA" != "$PV_SHA" ]; then
  echo "FATAL: external persona_vectors HEAD $GOT_SHA != pinned $PV_SHA" >&2
  exit 1
fi
if [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  log_phase setup "unzip dataset.zip"
  ( cd "$EXTERNAL_ROOT" && unzip -q -o dataset.zip )
fi
test -d "$EXTERNAL_ROOT/dataset" || { echo "FATAL: dataset/ missing after unzip" >&2; exit 1; }
test -f "$EXTERNAL_ROOT/data_generation/trait_data_eval/evil.json" \
  || { echo "FATAL: trait eval JSON missing" >&2; exit 1; }
log_phase setup "external inputs staged (sha=$GOT_SHA)"

# ── step 0b: REQUIRED §12 preflight — every consumed #778 finetune JSON present ─
# Assert BEFORE any Exp-4/Exp-5 cell runs; abort fail-loud on a miss (Exp-4
# coef-0 baseline + Exp-5 regression y-axis). Only relevant when those phases run.
if printf '%s' "$PHASES" | grep -qE 'preventative|screening'; then
  log_phase preflight "asserting reused #778 finetune JSONs present"
  # shellcheck disable=SC2086
  uv run python - "$TRAITS" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import issue816_lib as ilib

traits = tuple(sys.argv[1].split())
root = Path("eval_results/issue_778")
# Exp-4 consumes the diagonal (steer trait T on dataset that induces T); Exp-5's
# y-axis consumes the full family x trait grid. Assert both supersets present.
families = tuple(ilib.lib.FAMILIES)
ilib.assert_778_consumer_paths(root, traits=traits, families=traits)  # Exp-4 diagonal
# Exp-5 needs the full 24-cell grid per trait; assert the misaligned_2 rows exist
# for every (family) x (trait) the regression will read.
import itertools
missing = []
for trait, fam, ver in itertools.product(traits, families, ilib.lib.VERSIONS):
    p = root / f"finetune_{trait}_{fam}_{ver}.json"
    if not p.exists():
        missing.append(str(p))
if missing:
    raise FileNotFoundError(
        "Exp-5 y-axis: reused #778 finetune scores missing (§12 preflight):\n  "
        + "\n  ".join(missing[:20]) + (f"\n  ... (+{len(missing)-20} more)" if len(missing) > 20 else "")
    )
print(f"[preflight] all reused #778 finetune JSONs present ({len(traits)} traits)")
PY
  log_phase preflight "reused #778 consumer paths verified"
fi

# ── step 1: Phase-A 8-GPU fan-out dispatch ─────────────────────────────────────
log_phase dispatch "start phases=$PHASES traits=$TRAITS"
# shellcheck disable=SC2086
uv run python scripts/issue816_dispatch.py \
  --phases $PHASES \
  --traits $TRAITS \
  --external-root "$EXTERNAL_ROOT" \
  --dataset-root "$EXTERNAL_ROOT/dataset" \
  --out-root eval_results/issue_816/v3 \
  --ckpt-root checkpoints/issue_816 \
  --cache-dir data/issue_816/hf_dl \
  --n-random-dirs "$N_RANDOM_DIRS" \
  --n-samples "$N_SAMPLES" \
  $EXTRA

# Between-phase cleanup to bound peak footprint (multi-phase contract).
uv run python scripts/clean_experiment_downloads.py "$ISSUE" --incremental --apply || true

# ── step 2: upload (Exp-4 adapters + Exp-5 tensors + raw generations) ──────────
UPLOAD_SUMMARY="{}"
if [ "${EPM_I816_SKIP_UPLOAD:-0}" != "1" ]; then
  log_phase upload "start"
  UPLOAD_SUMMARY="$(uv run python scripts/issue816_upload.py --issue "$ISSUE" --slug "$SLUG" --out-root eval_results/issue_816/v3)"
else
  log_phase upload "SKIPPED (EPM_I816_SKIP_UPLOAD=1)"
fi

# ── end-of-run sentinel + terminal phase line ──────────────────────────────────
uv run python scripts/issue816_write_sentinel.py \
  --issue "$ISSUE" \
  --slug "$SLUG" \
  --upload-summary "$UPLOAD_SUMMARY" \
  --logs-dir "$LOGS_DIR"

log_phase done "issue-816 pod Phase-A complete (judge + null battery run off-pod on the VM)"
