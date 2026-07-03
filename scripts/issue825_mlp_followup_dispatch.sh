#!/usr/bin/env bash
# task #825 follow-up "mlp-unprobed-cells" (plan v6): stage -> fit -> gate -> upload -> sentinel.
#
# Production (worker):  bash scripts/issue825_mlp_followup_dispatch.sh
# Smoke (CPU, tiny):    bash scripts/issue825_mlp_followup_dispatch.sh --smoke
#   Smoke drives the SAME wrapper + fit entrypoint; issue825_fit_cells.py --smoke fabricates a
#   tiny synthetic turnstore under $OUT_DIR/_smoke_turnstore and skips the MLP secondary, so the
#   gate relaxes value/coverage asserts under EPS_SMOKE=1 while keeping every code path.
#   Stage + upload are HF-network phases, skipped in smoke.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then SMOKE="--smoke"; fi

TS_DIR="${TS_DIR:-data/issue_825/turnstore}"
STAGE_DIR="${STAGE_DIR:-data/issue_825/hf_dl/mlp_followup_stage}"
OUT_DIR="${OUT_DIR:-eval_results/issue_825/mlp-unprobed-cells}"
SENTINEL="${SENTINEL:-/workspace/logs/issue-825-results.json}"
HF_REV="deb7a4523b5233393e4fbd2497622527b3622d35"
DATA_REPO="superkaiba1/explore-persona-space-data"
CELLS8="M_instruct_assistant_chat,M_pretrained_assistant_chat,M_instruct_assistant_naturalistic,M_pretrained_assistant_naturalistic,M_instruct_user_chat,M_instruct_user_naturalistic,M_pretrained_user_chat,M_pretrained_user_naturalistic"
MLP6="M_instruct_assistant_naturalistic,M_pretrained_assistant_naturalistic,M_instruct_user_chat,M_instruct_user_naturalistic,M_pretrained_user_chat,M_pretrained_user_naturalistic"

# Credentials at script level (never bare load_dotenv() inside a stdin heredoc — gotcha).
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mkdir -p "$(dirname "$SENTINEL")" "$OUT_DIR"
export EPS_OUT_DIR="$OUT_DIR" EPS_SENTINEL="$SENTINEL" EPS_TS_DIR="$TS_DIR" \
  EPS_STAGE_DIR="$STAGE_DIR" EPS_HF_REV="$HF_REV" EPS_DATA_REPO="$DATA_REPO" \
  EPS_CELLS8="$CELLS8" EPS_MLP6="$MLP6"
if [[ -n "$SMOKE" ]]; then export EPS_SMOKE=1; else export EPS_SMOKE=""; fi

if [[ -z "$SMOKE" ]]; then
  echo "[phase=stage]"
  uv run python - <<'PY'
import os
from pathlib import Path

# Hot-fix (run-2): hf_transfer off after the run-1 stage hang.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Hot-fix (run-3): run 2 hung IDENTICALLY with hf_transfer off (all ~64GB on disk,
# downloader at 0% CPU, snapshot_download never returned) — the repo is Xet-backed,
# so the hang is the xet downloader's finalization (#515 class). Disable xet: files
# then stream via the plain CDN path (measured 12.7 MB/s/stream from this org).
os.environ["HF_XET_DISABLE"] = "1"
# Convert any future stage hang into a loud crash well before the instance fence:
import signal
signal.alarm(2700)  # 45 min hard cap on the whole stage phase

from huggingface_hub import snapshot_download

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before staging)"
stage = Path(os.environ["EPS_STAGE_DIR"])
ts = Path(os.environ["EPS_TS_DIR"])
prefixes = [f"{m}_{f}_m_" for m in ("instruct", "pretrained") for f in ("chat", "naturalistic")]
patterns = [
    f"issue825_userbase_map/analysis_tensors/{p}*{ext}" for p in prefixes for ext in (".pt", ".json")
]
snapshot_download(
    repo_id=os.environ["EPS_DATA_REPO"],
    repo_type="dataset",
    revision=os.environ["EPS_HF_REV"],
    allow_patterns=patterns,
    local_dir=str(stage),
)
src = stage / "issue825_userbase_map" / "analysis_tensors"
files = sorted(
    p for p in src.glob("*") if p.is_file() and any(p.name.startswith(x) for x in prefixes)
)
if not files:
    raise RuntimeError(f"stage: 0 m-track files under {src} at rev {os.environ['EPS_HF_REV']}")
ts.mkdir(parents=True, exist_ok=True)
for p in files:
    dst = ts / p.name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(p.resolve())
print(f"stage: linked {len(files)} m-track files -> {ts}")
PY
fi

echo "[phase=fit]"
uv run python scripts/issue825_fit_cells.py --turnstore-dir "$TS_DIR" --out-dir "$OUT_DIR" \
  --cells "$CELLS8" --mlp-cells "$MLP6" --null-draws 20 --folds 5 --seed 0 $SMOKE

echo "[phase=gate]"
uv run python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["EPS_OUT_DIR"])
smoke = bool(os.environ.get("EPS_SMOKE"))
sentinel = Path(os.environ["EPS_SENTINEL"])
cells8 = os.environ["EPS_CELLS8"].split(",")
mlp6 = os.environ["EPS_MLP6"].split(",")
ANCHORS = {"M_instruct_assistant_chat": 0.0757, "M_pretrained_assistant_chat": -0.4606}
TOL = 0.05
KILL = 0.2


def fail(status: str, msg: str) -> None:
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "task_id": 825,
                "status": status,
                "note": msg,
            },
            indent=2,
        )
    )
    raise SystemExit(f"GATE FAIL [{status}]: {msg}")


loaded = {}
for cid in cells8:
    p = out / f"cells_{cid}.json"
    if not p.exists():
        if smoke:
            continue
        fail(
            "coverage_miss",
            f"missing {p} — explicit --cells disables the FATAL-missing-bundle branch; "
            "a mis-staged prefix would otherwise SKIP silently",
        )
    loaded[cid] = json.loads(p.read_text())


def _l19(table: dict):
    for k, v in table.items():
        ks = str(k).lower().removeprefix("layer_").removeprefix("l")
        if ks == "19":
            return v
    return None


anchor_deltas = {}
for cid, ref in ANCHORS.items():
    payload = loaded.get(cid)
    if payload is None:
        if smoke:
            continue
        fail("coverage_miss", f"anchor cell {cid} absent from ridge results")
    # Fix (run-3 crash): frozen_layer_table nests under selection_symmetric.
    entry = _l19((payload.get("selection_symmetric") or {}).get("frozen_layer_table") or {})
    if entry is None:
        if smoke:
            continue
        fail("anchor_gate_miss", f"{cid}: no L19 row in frozen_layer_table")
    fresh = float(entry["r2_obs"])
    delta = fresh - ref
    anchor_deltas[cid] = {"fresh_r2_L19": fresh, "parent_r2_L19": ref, "delta": delta}
    if not smoke and abs(delta) > TOL:
        fail(
            "anchor_gate_miss",
            f"{cid}: fresh {fresh:+.4f} vs parent {ref:+.4f} "
            f"(|delta|={abs(delta):.4f} > {TOL}) — staging/code drift, HALT",
        )


def _mlp_extract(mlp):
    obs: list[float] = []
    nulls: list[float] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "r2_obs" and isinstance(v, (int, float)):
                    obs.append(float(v))
                elif isinstance(v, list) and "null" in str(k).lower():
                    nulls.extend(float(x) for x in v if isinstance(x, (int, float)))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(mlp)
    return obs, nulls


headline = {
    "followup_label": "mlp-unprobed-cells",
    "smoke": smoke,
    "anchor_tolerance": TOL,
    "kill_line": KILL,
    "anchor_deltas": anchor_deltas,
    "cells": {},
}
for cid in mlp6:
    mlp = (loaded.get(cid) or {}).get("mlp")
    if not mlp:
        if smoke:
            continue  # fit --smoke skips the MLP secondary entirely
        fail("coverage_miss", f"{cid}: cells_{cid}.json has no non-empty 'mlp' block")
    obs, nulls = _mlp_extract(mlp)
    if not obs and not smoke:
        fail("coverage_miss", f"{cid}: no r2_obs values found inside 'mlp' block")
    max_r2 = max(obs) if obs else None
    headline["cells"][cid] = {
        "mlp_r2_max_over_frozen_layers": max_r2,
        "null_band_max": max(nulls) if nulls else None,
        "n_null_values": len(nulls),
        "verdict_vs_kill_line": (
            "KILL — practical MLP recoverability (>0.2)"
            if (max_r2 is not None and max_r2 > KILL)
            else "below kill line"
        ),
        "null_extraction_warning": (None if nulls else "no numeric null values found; see mlp_raw"),
        "mlp_raw": mlp,
    }
(out / "headline_metrics.json").write_text(json.dumps(headline, indent=2))
print(
    f"gate: PASS ({len(headline['cells'])} MLP cells summarized; anchors: "
    + ", ".join(f"{k} delta {v['delta']:+.4f}" for k, v in anchor_deltas.items())
    + (" [smoke]" if smoke else "")
)
PY

if [[ -z "$SMOKE" ]]; then
  echo "[phase=upload]"
  uv run python - <<'PY'
import os

from huggingface_hub import HfApi

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
HfApi().upload_folder(
    folder_path=os.environ["EPS_OUT_DIR"],
    repo_id=os.environ["EPS_DATA_REPO"],
    repo_type="dataset",
    path_in_repo="issue825_userbase_map/eval_results_mlp_unprobed",
)
print("upload: ok -> issue825_userbase_map/eval_results_mlp_unprobed")
PY
fi

EPS_GIT_SHA="$(git rev-parse HEAD)"
EPS_WORKTREE="$(pwd)"
export EPS_GIT_SHA EPS_WORKTREE
uv run python - <<'PY'
import json
import os
import time
from pathlib import Path

out = Path(os.environ["EPS_OUT_DIR"])
headline = json.loads((out / "headline_metrics.json").read_text())
sent = Path(os.environ["EPS_SENTINEL"])
sent.parent.mkdir(parents=True, exist_ok=True)
repo = os.environ["EPS_DATA_REPO"]
sent.write_text(
    json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 825,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": {
                "followup_label": "mlp-unprobed-cells",
                "eval_numbers": headline,
                "eval_paths": sorted(str(p) for p in out.glob("*.json")),
                "reproducibility_card": {
                    "models": ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
                    "fit_seed": 0,
                    "turnstore_revision": os.environ["EPS_HF_REV"],
                    "followup_label": "mlp-unprobed-cells",
                },
                "wandb_url": "n/a (analysis-only follow-up; no training)",
                "hf_hub_url": (
                    f"https://huggingface.co/datasets/{repo}/tree/main/"
                    "issue825_userbase_map/eval_results_mlp_unprobed"
                ),
                "worktree_path": os.environ["EPS_WORKTREE"],
                "final_commit_sha": os.environ["EPS_GIT_SHA"],
                "gpu_hours_used": 2.0,
                "gpu_hours_budgeted": 2.0,
                "plan_deviations": [],
            },
        },
        indent=2,
    )
)
print(f"sentinel: wrote {sent}")
PY
echo "[phase=done]"
