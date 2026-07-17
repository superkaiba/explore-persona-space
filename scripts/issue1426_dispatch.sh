#!/usr/bin/env bash
# Issue #1426 pod-side dispatch (plan §4.3b + §10 launch row):
#   headroom assert → primary pipeline (gate → G → P → B → F1 → MLP → F2/F3 →
#   figures, incremental uploads inside the driver) → in-provision cap16k stage
#   (skip-check → local stage copy → invocation 2 extract --force-regen-16k
#   --hf-stage-suffix _16k → invocation 3 restricted f1 at the PRIMARY run's
#   frozen layers → frozen-layer diagnostic → scoped fit_results_16k upload) →
#   finalize (the ONE epm:results sentinel + [phase=done], AFTER the cap16k
#   stage so the poller's terminal signal is true end-of-workload).
#
# SMOKE=1 runs the SAME sequence on CPU against the tiny-real fixture model
# (issue1426_tiny_e2e_fixture.py), faking ONLY the vLLM boundary
# (--synthetic-completions) and the Hub boundary (--no-upload) — the
# smoke/production unification contract (plan §4.7).
#
# Production (GCP capture-7b lane):
#   uv run python scripts/dispatch_issue.py --issue 1426 --intent capture-7b \
#     --time-budget-hours 12 --repo-branch issue-1426 \
#     --workload-cmd "bash scripts/issue1426_dispatch.sh"
#
# CPU tiny-e2e smoke (VM; fixture model built first):
#   uv run python scripts/issue1426_tiny_e2e_fixture.py --out /tmp/issue-1426-smoke/tiny_model
#   SMOKE=1 bash scripts/issue1426_dispatch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  # Shared-VM smoke: thread caps are mandatory (code-style.md #847); pod/GCE
  # production launches never carry them (dedicated GPUs keep full width).
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-1426-smoke}"
  OUT_DIR="${OUT_DIR:-$SMOKE_ROOT/data}"
  EVAL_OUT="${EVAL_OUT:-$SMOKE_ROOT/eval}"
  FIGURES_DIR="${FIGURES_DIR:-$SMOKE_ROOT/figures}"
  LOG_DIR="${LOG_DIR:-$SMOKE_ROOT/logs}"
  MODEL_DIR="${MODEL_DIR:-$SMOKE_ROOT/tiny_model}"
  DRIVER_FLAGS=(--smoke --device cpu --model "$MODEL_DIR" --synthetic-completions
    --contexts "${SMOKE_CONTEXTS:-6}" --probes "${SMOKE_PROBES:-4}" --no-upload
    --log-dir "$LOG_DIR")
  PRIMARY_FIT_FLAGS=(--layers 0 1 --n-perms 10 --n-boot 50)
  CAP16K_PERMS=10 CAP16K_BOOT=50
  NEED_GB_START=2 NEED_GB_STAGE=1
else
  OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/issue_1426}"
  EVAL_OUT="${EVAL_OUT:-$REPO_ROOT/eval_results/issue_1426}"
  FIGURES_DIR="${FIGURES_DIR:-$REPO_ROOT/figures/issue_1426}"
  DRIVER_FLAGS=(--gpu)
  PRIMARY_FIT_FLAGS=()
  CAP16K_PERMS=50 CAP16K_BOOT=2000     # plan §4.3b invocation 3
  NEED_GB_START=60 NEED_GB_STAGE=15    # plan §9 out-root mount binding
fi
OUT_16K="${OUT_16K:-${OUT_DIR}_cap16k}"
EVAL_16K="$EVAL_OUT/cap16k"
DRIVER="$REPO_ROOT/scripts/issue1426_run.py"
mkdir -p "$OUT_DIR" "$EVAL_OUT"

# ── headroom assert (plan §9: assert against the mount OUT_DIR resolves to) ──
echo "[phase=dispatch_headroom]"
uv run python -c "
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_DIR', $NEED_GB_START, phase='pipeline-start')
print(f'[dispatch] out-root headroom OK: {free:.1f} GB free at $OUT_DIR')
"

# ── invocation 1: primary pipeline, finalize DEFERRED past the cap16k stage ──
# Resume predicate (checkpoint-per-phase): a prior attempt's run_state arms
# --skip-gen; the driver re-validates every rollout blob (model / rung / pool
# hash / cap) and regenerates any stale or missing context itself.
RESUME_FLAGS=()
if [ -f "$OUT_DIR/run_state.json" ]; then
  RESUME_FLAGS=(--skip-gen)
  echo "[dispatch] prior run_state.json found — resuming with --skip-gen"
fi
echo "[phase=dispatch_primary]"
uv run python "$DRIVER" \
  --out-dir "$OUT_DIR" --eval-out "$EVAL_OUT" --figures-dir "$FIGURES_DIR" \
  --phases extract f1 mlp f2f3 figures \
  "${DRIVER_FLAGS[@]}" "${PRIMARY_FIT_FLAGS[@]}" "${RESUME_FLAGS[@]}"

# ── cap16k stage (plan §4.3b) ────────────────────────────────────────────────
echo "[phase=dispatch_cap16k_skipcheck]"
SKIPCHECK=$(uv run python - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
man = json.loads((out / "store" / "manifest.json").read_text())
rs = json.loads((out / "run_state.json").read_text())
if man.get("regen_16k"):
    print("SKIP production >10% regen trigger already fired (corpus at 16,384)")
    raise SystemExit(0)
if int(rs["production_max_new_tokens"]) >= 16384:
    print("SKIP gate raised the production cap to 16,384 (cap16k is a no-op)")
    raise SystemExit(0)
n = 0
for p in sorted((out / "raw_completions" / "thinking_rollouts").glob("*.json")):
    blob = json.loads(p.read_text())
    n += sum(1 for r in blob["completions"] if r.get("finish_reason") == "length")
if n == 0:
    print("SKIP zero finish_reason=='length' rows remain")
else:
    print(f"RUN {n} residual cap-hit rows")
PY
)
echo "[dispatch] cap16k skip-check: $SKIPCHECK"

if [ "${SKIPCHECK%% *}" = "RUN" ]; then
  echo "[phase=dispatch_cap16k_stagecopy]"
  uv run python -c "
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_DIR', $NEED_GB_STAGE, phase='cap16k-stage-copy')
print(f'[dispatch] cap16k stage headroom OK: {free:.1f} GB free')
"
  # Local stage copy (plan §4.3b step 2): keeps the primary out-dir pristine,
  # mirroring the parent's separate-round separation — from local disk instead
  # of the parent launcher's HF staging.
  rm -rf "$OUT_16K"
  mkdir -p "$OUT_16K"
  cp -r "$OUT_DIR/raw_completions" "$OUT_16K/raw_completions"
  cp -r "$OUT_DIR/store" "$OUT_16K/store"
  cp "$OUT_DIR/run_state.json" "$OUT_16K/run_state.json"
  mkdir -p "$EVAL_16K"

  echo "[phase=dispatch_cap16k_invocation2]"
  uv run python "$DRIVER" \
    --out-dir "$OUT_16K" --eval-out "$EVAL_16K" --figures-dir "$FIGURES_DIR" \
    --phases extract --skip-gen --force-regen-16k --hf-stage-suffix _16k \
    "${DRIVER_FLAGS[@]}"

  echo "[phase=dispatch_cap16k_invocation3]"
  # Frozen indices read MECHANICALLY from the PRIMARY run's bootstrap blob
  # (plan §4.3b step 4 — never hand-pinned).
  FROZEN_LAYERS=$(uv run python - "$EVAL_OUT/bootstrap_deltaskill.json" <<'PY'
import json
import sys

blob = json.loads(open(sys.argv[1]).read())
layers = sorted(
    {
        int(blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in ("avg_q", "indiv")
    }
)
print(" ".join(str(x) for x in layers))
PY
)
  echo "[dispatch] primary frozen layers: $FROZEN_LAYERS"
  # shellcheck disable=SC2086 — FROZEN_LAYERS is a space-separated int list
  uv run python "$DRIVER" \
    --out-dir "$OUT_16K" --eval-out "$EVAL_16K" --figures-dir "$FIGURES_DIR" \
    --phases f1 --skip-gen --layers $FROZEN_LAYERS \
    --n-perms "$CAP16K_PERMS" --n-boot "$CAP16K_BOOT" --no-upload \
    "${DRIVER_FLAGS[@]}"

  echo "[phase=dispatch_cap16k_frozencheck]"
  uv run python - "$EVAL_OUT/bootstrap_deltaskill.json" "$EVAL_16K" <<'PY'
import json
import sys
from pathlib import Path

primary = json.loads(open(sys.argv[1]).read())
eval16 = Path(sys.argv[2])
reread = json.loads((eval16 / "bootstrap_deltaskill.json").read_text())
regimes = ("avg_q", "indiv")


def frozen(blob):
    return {
        r: int(blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in regimes
    }


p, q = frozen(primary), frozen(reread)
match = {r: p[r] == q[r] for r in regimes}
(eval16 / "frozen_layer_check.json").write_text(
    json.dumps(
        {
            "dv": "cap16k frozen-layer re-derivation vs the primary index (plan §4.3b step 4)",
            "primary": p,
            "reread": q,
            "match": match,
            "note": (
                "on mismatch the pinned-index reads at the PRIMARY indices carry the "
                "sensitivity headline; the mismatch is a reported diagnostic, never a stop "
                "(parent v4 convention)"
            ),
        },
        indent=2,
    )
)
print(f"[dispatch] frozen-layer check: primary={p} reread={q} match={match}")
PY

  if [ "$SMOKE" != "1" ]; then
    echo "[phase=dispatch_cap16k_upload]"
    # --hf-stage-suffix threads ONLY the extract-phase upload sites (verified
    # at the driver's guard, issue1426_run.py) — the dispatch uploads the
    # invocation-3 fit outputs itself, scoped to fit_results_16k (plan §4.3b).
    uv run python - "$EVAL_16K" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(".env")
from issue928_common import upload_folder_scoped_verify
from issue1426_common import FIT_RESULTS_PREFIX_1426

eval16 = Path(sys.argv[1])
names = sorted(p.name for p in eval16.glob("*.json")) + sorted(
    p.name for p in eval16.glob("decomp_*.pt")
)
url = upload_folder_scoped_verify(
    eval16,
    FIT_RESULTS_PREFIX_1426 + "_16k",
    names,
    f"issue #1426 cap16k: restricted fit outputs + C94 re-read ({len(names)} files)",
    allow_patterns=["*.json", "decomp_*.pt"],
    ignore_patterns=["partial/*"],
)
print(f"[dispatch] cap16k fit outputs uploaded: {url}")
PY
  else
    echo "[dispatch] SMOKE=1: cap16k upload skipped (Hub boundary faked)"
  fi
else
  echo "[dispatch] cap16k stage SKIPPED — C at 16,384 == C at the production corpus"
fi

# ── finalize LAST: the ONE epm:results sentinel + [phase=done] ───────────────
echo "[phase=dispatch_finalize]"
uv run python "$DRIVER" \
  --out-dir "$OUT_DIR" --eval-out "$EVAL_OUT" --figures-dir "$FIGURES_DIR" \
  --phases finalize \
  "${DRIVER_FLAGS[@]}"
