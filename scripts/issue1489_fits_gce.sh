#!/usr/bin/env bash
# Issue #1489 P6 per-box CPU-fit driver (cpu-bigmem GCE; plan §4.5/§9).
# Env-parametrized (mirror of the issue1092 p6 job pattern):
#   P6_BOX_ID   box index (1..4)                       [required]
#   P6_UNITS    comma-separated unit ids for this box  [required]
#   P6_SMOKE    "1" -> --smoke                          [optional]
# Stages the summaries store from the issue HF prefix (scoped listing +
# per-file download via hub.stage_hub_prefix), runs the fit driver with
# per-unit checkpoint/resume, and uploads outputs to
# issue1489_ctx_aug/p6/box_<id>/ (verified exact-set).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export PYTHONUNBUFFERED=1

BOX_ID="${P6_BOX_ID:?P6_BOX_ID required}"
UNITS="${P6_UNITS:?P6_UNITS required}"
SMOKE_FLAG=""
if [ "${P6_SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--smoke"; fi

OUT="$REPO_ROOT/data/issue_1489"
SUMMARIES="$OUT/summaries"
P6_OUT="$REPO_ROOT/eval_results/issue_1489/p6"

echo "[phase=stage] box $BOX_ID staging summaries for units: $UNITS"
uv run python - "$UNITS" "$SUMMARIES" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from explore_persona_space.orchestrate import hub
from issue1489_common import HF_DATA_REPO, HF_PREFIX

units, dest_root = sys.argv[1], Path(sys.argv[2])
# Every unit consumes cell_plain; ft/q4/q6 units consume cell_ft_* + aug cells.
# Stage the WHOLE summaries prefix once (scoped listing; per-file downloads;
# already-staged files skip) — box-level subsetting is a wall-time nicety, not
# a correctness requirement, and a full stage keeps resume trivially safe.
staged = hub.stage_hub_prefix(
    HF_DATA_REPO, f"{HF_PREFIX}/analysis_tensors/summaries", dest_root
)
print(f"[stage] {len(staged)} summary files under {dest_root}")
assert staged, "staging returned 0 files"
PY

echo "[phase=fits] box $BOX_ID running units"
uv run python scripts/issue1489_fit_grid.py $SMOKE_FLAG \
  --summaries-dir "$SUMMARIES" --out "$P6_OUT" --units "$UNITS"

echo "[phase=upload] box $BOX_ID uploading p6 outputs"
uv run python - "$BOX_ID" "$P6_OUT" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub
from issue1489_common import HF_DATA_REPO, HF_PREFIX

box_id, p6_out = sys.argv[1], Path(sys.argv[2])
prefix = f"{HF_PREFIX}/p6/box_{box_id}"
url = hub._upload(p6_out, repo_id=HF_DATA_REPO, repo_type="dataset", path_in_repo=prefix)
if not url:
    raise RuntimeError("p6 upload returned no path")
expected = [f"{prefix}/{p.relative_to(p6_out)}" for p in p6_out.rglob("*") if p.is_file()]
missing = hub.verify_repo_paths_uploaded(
    HfApi(), HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
)
if missing:
    raise RuntimeError(f"p6 upload verify missing {len(missing)}: {sorted(missing)[:5]}")
print(f"[upload] verified {len(expected)} p6 files -> {url}")
PY

echo "[phase=done] box $BOX_ID"
