#!/usr/bin/env bash
# Issue #825 follow-up `base-separator-control` (plan v18): the PRETRAINED-BASE
# arm of #931's separator/punctuation specificity control.
#
# Phases: p0_stage (pinned pair files + instruct armC anchor store into the
# CONSUMER layout + provenance-coherence check + consumer-open probe) ->
# p1_wiring (tokenizer-identity + base-config hard asserts) ->
# p2_extract_armc_base (issue931_extract_store.py --model-id Qwen/Qwen2.5-7B) ->
# p3_fits_base (issue931_fit_cells.py --cells armC_sep,armC_prevmean --mlp) ->
# p3_fits_anchor (instruct refit from the staged store + the +-0.01 gate vs the
# COMMITTED #931 values) -> p5_upload (one upload_folder commit per artifact
# kind + exact-set verify + git commit + results sentinel) -> [phase=done].
#
# ONE code path for smoke / production — the tiny sizing threads through every
# phase (Step 6d.0 PASS_UNIFIED): smoke = tiny random-init Qwen2 (real
# tokenizer) + --max-items 3 + a 1-shard anchor subset + non-binding gates
# (planted-mismatch self-tests keep the gate MECHANICS exercised) + skip-hub
# upload enumeration; scratch dirs — committed eval_results/figures never
# touched.
#
# Modes:
#   bash scripts/issue825_base_sep_dispatch.sh --smoke   # CPU VM smoke
#   bash scripts/issue825_base_sep_dispatch.sh           # production (A100)
#
# Pod-side contract: [phase=...] log lines; the terminal [phase=done] is
# emitted ONLY here, after the final sentinel write. NEVER shells task.py.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
# GCE lane has NO .env (tokens ride the startup-script env) — conditional only.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export TQDM_DISABLE=1
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

ISSUE=825
BASE_MODEL_ID="Qwen/Qwen2.5-7B"
PAIRS_REV="9534b9981d6b4fb4f1259c9b06f021d311a46af4"  # plan section 4 pin
HF_PREFIX_OUT="issue825_base_sep_control"
RUN_START_EPOCH="$(date +%s)"
export RUN_START_EPOCH

MODE="production"
for arg in "$@"; do
  case "$arg" in
    --smoke) MODE="smoke" ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="$(pwd)/logs"; mkdir -p "$LOG_DIR"; }

if [ "$MODE" = "smoke" ]; then
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-825-base-sep-smoke}"
  rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
  DATA_DIR="$SMOKE_ROOT/data"
  ANCHOR_DIR="$SMOKE_ROOT/anchor"
  OUT_DIR="$SMOKE_ROOT/eval"
  LOG_DIR="$SMOKE_ROOT/logs"; mkdir -p "$LOG_DIR"
  TINY="$SMOKE_ROOT/tiny_model"
  uv run python scripts/issue931_extract_store.py --make-tiny-model "$TINY"
  MODEL_FLAGS="--model-id $BASE_MODEL_ID --tiny-model-dir $TINY --batch-size 2 --max-items 3"
  FIT_FLAGS="--smoke --null-draws 2 --n-boot 50"
  ANCHOR_SHARDS="1"      # 1-shard real-Hub staging probe (h)(iv)
  EXPECT_N="auto"        # anchor rows derived from the staged sidecars
  GATE_BINDING="0"       # +-0.01 gate recorded + planted-mismatch self-test
  DO_UPLOAD="0"
else
  DATA_DIR="data/issue_${ISSUE}/base_sep_control"
  ANCHOR_DIR="data/issue_${ISSUE}/base_sep_anchor"
  OUT_DIR="eval_results/issue_${ISSUE}/base-separator-control"
  MODEL_FLAGS="--model-id $BASE_MODEL_ID --batch-size 8"
  FIT_FLAGS=""
  ANCHOR_SHARDS="all"
  EXPECT_N="3600"
  GATE_BINDING="1"
  DO_UPLOAD="1"
fi
mkdir -p "$OUT_DIR"

echo "[phase=p0_stage] pinned pair files + instruct armC anchor store"
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR" --anchor-dir "$ANCHOR_DIR" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --expect-n "$EXPECT_N" $( [ "$MODE" = smoke ] && echo --self-test )

echo "[phase=p1_wiring] tokenizer-identity + base-config hard asserts"
uv run python - "$BASE_MODEL_ID" <<'PY'
import hashlib
import sys

base_id = sys.argv[1]
instruct_id = "Qwen/Qwen2.5-7B-Instruct"
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

tok_b = AutoTokenizer.from_pretrained(base_id)
tok_i = AutoTokenizer.from_pretrained(instruct_id)
assert tok_b.get_vocab() == tok_i.get_vocab(), "base/instruct vocab mismatch — pairs invalid"


def _sha(repo):
    p = hf_hub_download(repo, "tokenizer.json")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


sb, si = _sha(base_id), _sha(instruct_id)
assert sb == si, f"tokenizer.json sha mismatch: base {sb} != instruct {si}"
cfg = AutoConfig.from_pretrained(base_id)
assert cfg.num_hidden_layers == 28, cfg.num_hidden_layers
assert cfg.hidden_size == 3584, cfg.hidden_size
print(f"[i825-bs] p1 PASS: vocab identical; tokenizer.json sha {sb[:12]}; 28 layers / 3584")
PY

echo "[phase=p2_extract_armc_base] base-model armC capture (+ equivalence gate)"
uv run python scripts/issue931_extract_store.py --regime armC \
  --data-dir "$DATA_DIR" --equivalence-check --resume $MODEL_FLAGS

echo "[phase=p3_fits_base] base armC fit battery (ridge + rotated + MLP)"
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --mlp \
  --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" $FIT_FLAGS

echo "[phase=p3_fits_anchor] instruct anchor refit from the staged store"
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean \
  --data-dir "$ANCHOR_DIR" --out-dir "$OUT_DIR/instruct_anchor" $FIT_FLAGS
uv run python - "$OUT_DIR" "$GATE_BINDING" <<'PY'
"""G3b reproduction gate: instruct anchor refit within +-0.01 of the COMMITTED
#931 ridge/rotated values @ L19 (rig-drift detector; plan section 6 gate 3)."""

import json
import sys
from pathlib import Path

out_dir, binding = Path(sys.argv[1]), sys.argv[2] == "1"
committed_dir = Path("eval_results/issue_931")
TOL = 0.01
# Plan-quoted documentation anchors (drift cross-check, never the gate).
PLAN_QUOTED = {
    "armC_sep": {"ridge": -3.1685, "rotated": 0.3489},
    "armC_prevmean": {"ridge": -3.4970, "rotated": 0.3341},
}


def _vals(path: Path) -> dict:
    d = json.loads(path.read_text())
    hl = int(d.get("headline_layer", 19))
    return {
        "ridge": float(d["r2_per_layer_obs"][hl]),
        "rotated": float(d["random_projection_control_r2"][str(hl)]),
    }


def check(anchor_dir: Path, tol: float) -> tuple[bool, dict]:
    ok, deltas = True, {}
    for cell, quoted in PLAN_QUOTED.items():
        com = _vals(committed_dir / f"cells_{cell}.json")
        for k, qv in quoted.items():
            assert abs(com[k] - qv) < 1e-3, (cell, k, com[k], qv, "committed != plan quote")
        got = _vals(anchor_dir / f"cells_{cell}.json")
        for k in ("ridge", "rotated"):
            d = abs(got[k] - com[k])
            deltas[f"{cell}.{k}"] = {"anchor": got[k], "committed": com[k], "abs_delta": d}
            ok = ok and d <= tol
    return ok, deltas


ok, deltas = check(out_dir / "instruct_anchor", TOL)
(out_dir / "anchor_gate.json").write_text(
    json.dumps({"pass": ok, "tolerance": TOL, "binding": binding, "deltas": deltas}, indent=2)
)
print(f"[i825-bs] anchor gate: pass={ok} binding={binding} deltas="
      f"{ {k: round(v['abs_delta'], 5) for k, v in deltas.items()} }")
if binding and not ok:
    print("[i825-bs] G3b FAIL: rig drift — no base read is valid; halting", file=sys.stderr)
    raise SystemExit(5)
if not binding:
    # Smoke self-test: the gate MUST fire on a planted mismatch (mechanics).
    import copy

    planted_dir = out_dir / "instruct_anchor_planted"
    planted_dir.mkdir(exist_ok=True)
    for cell in PLAN_QUOTED:
        d = json.loads((out_dir / "instruct_anchor" / f"cells_{cell}.json").read_text())
        p = copy.deepcopy(d)
        hl = int(p.get("headline_layer", 19))
        p["r2_per_layer_obs"][hl] = float(p["r2_per_layer_obs"][hl]) + 0.5  # planted drift
        (planted_dir / f"cells_{cell}.json").write_text(json.dumps(p))
    ok_planted, _ = check(planted_dir, TOL)
    assert not ok_planted, "planted +0.5 mismatch NOT caught — gate mechanics broken"
    print("[i825-bs] anchor-gate self-test PASS: planted mismatch detected")
PY

echo "[phase=p5_upload] store + eval-JSON uploads, git commit, sentinel"
uv run python - "$ISSUE" "$DATA_DIR" "$OUT_DIR" "$HF_PREFIX_OUT" "$DO_UPLOAD" <<'PY'
import sys
from pathlib import Path

issue, data_dir, out_dir, prefix, do_upload = (
    int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5] == "1",
)
repo = "superkaiba1/explore-persona-space-data"
armc = data_dir / "store" / "armC"
store_files = sorted(p for p in armc.iterdir() if p.is_file())
assert store_files, f"no armC store files under {armc}"
eval_files = sorted(str(p) for p in out_dir.rglob("*.json"))
print(f"[i825-bs] p5 enumerate: {len(store_files)} store files, {len(eval_files)} eval JSONs")
if not do_upload:
    print("[i825-bs] p5 smoke: hub upload SKIPPED (same enumeration path)")
    raise SystemExit(0)
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

hub._upload(armc, repo_id=repo, repo_type="dataset", path_in_repo=f"{prefix}/analysis_tensors/armC")
expected = [f"{prefix}/analysis_tensors/armC/{p.name}" for p in store_files]
missing = hub.verify_repo_paths_uploaded(
    HfApi(), repo, expected, path_in_repo=f"{prefix}/analysis_tensors/armC", repo_type="dataset"
)
assert not missing, f"store upload verify FAILED — missing on Hub: {missing}"
hub._upload(out_dir, repo_id=repo, repo_type="dataset", path_in_repo=f"{prefix}/eval_results_mirror")
print("[i825-bs] p5 uploads complete + exact-set verified")
PY

if [ "$DO_UPLOAD" = "1" ]; then
  echo "[i825-bs] committing eval JSONs to the issue branch"
  git add "$OUT_DIR" || true
  if git commit -m "task #${ISSUE}: base-separator-control pod eval JSONs" >/dev/null 2>&1; then
    git push origin "issue-${ISSUE}" || echo "[i825-bs] WARNING: git push failed (HF mirror uploaded)" >&2
  else
    echo "[i825-bs] nothing to commit"
  fi
fi

echo "[i825-bs] writing results sentinel"
uv run python - "$ISSUE" "$OUT_DIR" "$LOG_DIR" "$MODE" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

issue, out_dir, log_dir, mode = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
run_start = int(os.environ.get("RUN_START_EPOCH", "0"))
gpu_hours = round((time.time() - run_start) / 3600.0, 3) if run_start else None


def _read(rel):
    p = out_dir / rel
    return json.loads(p.read_text()) if p.exists() else None


def _cell(rel):
    d = _read(rel)
    if not d:
        return None
    hl = int(d.get("headline_layer", 19))
    return {
        "ridge_L19": d["r2_per_layer_obs"][hl],
        "rotated_L19": d["random_projection_control_r2"].get(str(hl)),
        "headline_layer": hl,
    }


mlp = _read("mlp_secondary.json") or {"cells": {}}
_sep_cell = _cell("cells_armC_sep.json") or {}
_hl_key = str(_sep_cell.get("headline_layer", 19))
sep_mlp = ((mlp["cells"].get("armC_sep") or {}).get(_hl_key) or {}).get("r2_obs")
base_cells = {c: _cell(f"cells_{c}.json") for c in ("armC_sep", "armC_prevmean")}
# Within-strength reference ratios (plan section 6; committed base chat ceiling
# 0.5877 + instruct references 0.3489/0.6731, 0.2986/0.6731) so the analyzer
# applies the bands without recomputation.
BASE_CHAT_CEILING = 0.5876803039140281  # committed eval_results/issue_825/cells_S2.json @ L19
INSTRUCT_REF = {
    "chat_ceiling_L19": 0.6730940896676356,
    "sep_rotated_L19": 0.3489193821633685,
    "sep_mlp_L19": 0.2985925806396439,
    "ratio_rotated": 0.3489193821633685 / 0.6730940896676356,
    "ratio_mlp": 0.2985925806396439 / 0.6730940896676356,
}
sep = base_cells.get("armC_sep") or {}
ratios = {
    "base_ratio_rotated": (sep.get("rotated_L19") / BASE_CHAT_CEILING)
    if sep.get("rotated_L19") is not None
    else None,
    "base_ratio_mlp": (sep_mlp / BASE_CHAT_CEILING) if sep_mlp is not None else None,
    "instruct_reference": INSTRUCT_REF,
    "base_chat_ceiling_L19": BASE_CHAT_CEILING,
}
eval_numbers = {
    "mode": mode,
    "base_cells_L19": base_cells,
    "base_sep_mlp_L19": sep_mlp,
    "within_strength_ratios": ratios,
    "anchor_gate": (_read("anchor_gate.json") or {}).get("pass"),
    "note": "transfer + similarity reads run in Phase C on the VM (base_sep_to_chat.json)",
}
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out_dir.rglob("*.json")),
    "reproducibility_card": {
        "wandb_url": "n/a - no training in this round (no WandB runs)",
        "hf_hub_url": (
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
            "tree/main/issue825_base_sep_control"
        ),
        "worktree_path": ".claude/worktrees/issue-825-base-sep",
        "final_commit_sha": commit,
        "gpu_hours_used": gpu_hours,  # 1x A100: wall == GPU-hours
        "gpu_hours_budgeted": 2,
        "plan_deviations": [],
    },
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": issue,
    "by": "issue825_base_sep_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
path = log_dir / f"issue-{issue}-epm_results-{int(time.time())}.json"
path.write_text(json.dumps(sentinel, indent=2))
print(f"[i825-bs] sentinel written: {path}")
PY

echo "[phase=done]"
