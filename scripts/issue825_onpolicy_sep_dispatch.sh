#!/usr/bin/env bash
# Issue #825 follow-up `onpolicy-separator-control` (plan v21): the ON-POLICY
# provenance arm of the round-6/#931 separator control — same pair recipe,
# spans now each model's OWN raw-text continuation of the pinned articles.
#
# Phases (plan section 4): p0_stage (pinned pair files + BOTH exogenous armC
# anchor stores into the consumer layout + provenance checks + consumer-open
# probes) -> p1_wiring (tokenizer-identity + config hard asserts) ->
# p2_generate (vLLM greedy seed-42 continuations x2 models, wave-2 top-up) ->
# p2b_pairs (G2b continuation-region pair build x2) -> p2c_upload1
# (UNCONDITIONAL text upload BEFORE any extract — #779 persist-rollout-text)
# -> p3_extract (issue931_extract_store x2, equivalence gate) -> p4_fits
# (issue931_fit_cells --mlp --rotated-ci x2) -> p4b_anchors (exogenous refits
# BOTH models + +-0.01 gates vs COMMITTED values; UPLOAD-THEN-HALT: a gate
# FAIL still runs p5 uploads before exiting non-zero) -> p4c_matchedn
# (conditional matched-n W_ex re-baseline per model) -> p4d_decision
# (decision_support.json) -> p5_upload (stores + eval JSONs + sentinel) ->
# [phase=done] (only when no binding gate failed).
#
# ONE code path for smoke / production — the tiny sizing threads through every
# phase (Step 6d.0 PASS_UNIFIED): smoke = tiny random-init Qwen2 (real
# tokenizer) + --max-items 3 + true-continuation substitute (real sentence
# structure through the REAL ladder/extract/fit path) + 1-shard anchor subsets
# + non-binding gates (planted-mismatch self-tests keep the gate MECHANICS
# exercised) + skip-hub upload on the same enumeration path; scratch dirs —
# committed eval_results/figures never touched.
#
# Modes:
#   bash scripts/issue825_onpolicy_sep_dispatch.sh --smoke   # CPU VM smoke
#   bash scripts/issue825_onpolicy_sep_dispatch.sh           # production (A100)
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
INSTRUCT_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
PAIRS_REV="9534b9981d6b4fb4f1259c9b06f021d311a46af4"   # plan section 10 pin
BASE_STORE_REV="d4085b09d79fc46537b9da60bd6ffd8a754a677a"  # plan section 10 pin
INST_ANCHOR_PREFIX="issue931_story_map/analysis_tensors/armC"
BASE_ANCHOR_PREFIX="issue825_base_sep_control/analysis_tensors/armC"
HF_PREFIX_OUT="issue825_onpolicy_sep_control"
MODELS="base instruct"
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
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-825-onpolicy-sep-smoke}"
  rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
  DATA_DIR="$SMOKE_ROOT/data"
  ANCHOR_INST="$SMOKE_ROOT/anchor_inst"
  ANCHOR_BASE="$SMOKE_ROOT/anchor_base"
  OUT_DIR="$SMOKE_ROOT/eval"
  LOG_DIR="$SMOKE_ROOT/logs"; mkdir -p "$LOG_DIR"
  TINY="$SMOKE_ROOT/tiny_model"
  uv run python scripts/issue931_extract_store.py --make-tiny-model "$TINY"
  GEN_FLAGS="--max-items 3 --tiny-model-dir $TINY --smoke-real-continuation --wave2-min-eligible 999"
  EXTRACT_FLAGS="--tiny-model-dir $TINY --batch-size 2 --max-items 6"
  FIT_FLAGS="--smoke --null-draws 2 --n-boot 50"
  MATCHEDN_FLAGS="--seeds 931,932 --skip-mlp --smoke"
  DECISION_FLAGS="--smoke"
  ANCHOR_SHARDS="1"      # 1-shard real-Hub staging probe (h)(iv)
  EXPECT_N="auto"        # anchor rows derived from the staged sidecars
  GATE_BINDING="0"       # +-0.01 gates recorded + planted-mismatch self-tests
  DO_UPLOAD="0"
else
  DATA_DIR="data/issue_${ISSUE}/onpolicy_sep"
  ANCHOR_INST="data/issue_${ISSUE}/onpolicy_sep_anchor_inst"
  ANCHOR_BASE="data/issue_${ISSUE}/onpolicy_sep_anchor_base"
  OUT_DIR="eval_results/issue_${ISSUE}/onpolicy-separator-control"
  GEN_FLAGS=""
  EXTRACT_FLAGS="--batch-size 8"
  FIT_FLAGS=""
  MATCHEDN_FLAGS=""
  DECISION_FLAGS=""
  ANCHOR_SHARDS="all"
  EXPECT_N="3600"
  GATE_BINDING="1"
  DO_UPLOAD="1"
fi
mkdir -p "$OUT_DIR"
GATE_FAIL="0"

echo "[phase=p0_stage] pinned pair files + BOTH exogenous armC anchor stores"
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR/shared" --anchor-dir "$ANCHOR_INST" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --anchor-prefix "$INST_ANCHOR_PREFIX" \
  --expect-n "$EXPECT_N" --manifest-name onpolicy_sep_stage_inst.json \
  $( [ "$MODE" = smoke ] && echo --self-test )
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR/shared" --anchor-dir "$ANCHOR_BASE" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --anchor-prefix "$BASE_ANCHOR_PREFIX" --anchor-revision "$BASE_STORE_REV" \
  --expect-n "$EXPECT_N" --skip-pairs --manifest-name onpolicy_sep_stage_base.json

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
print(f"[i825-ops] p1 PASS: vocab identical; tokenizer.json sha {sb[:12]}; 28 layers / 3584")
PY

echo "[phase=p2_generate] on-policy raw-text continuations (greedy seed 42) x2 models"
for M in $MODELS; do
  # Each model is its own process => the vLLM engine dies with it (teardown gotcha).
  uv run python scripts/issue825_onpolicy_sep_gen.py --model "$M" \
    --articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --out-dir "$DATA_DIR/$M/generation" $GEN_FLAGS
done

echo "[phase=p2b_pairs] G2b continuation-region pair construction x2 models"
for M in $MODELS; do
  uv run python scripts/issue825_onpolicy_sep_pairs.py --model "$M" \
    --articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --continuations "$DATA_DIR/$M/generation/continuations.jsonl" \
    --exogenous-articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --exogenous-pairs "$DATA_DIR/shared/pairs/pairs_armC.jsonl" \
    --out-data-dir "$DATA_DIR/$M"
done

echo "[phase=p2c_upload1] UNCONDITIONAL text upload BEFORE any extract (#779)"
uv run python - "$DATA_DIR" "$HF_PREFIX_OUT" "$DO_UPLOAD" <<'PY'
import sys
from pathlib import Path

data_dir, prefix, do_upload = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
repo = "superkaiba1/explore-persona-space-data"
kinds = []
for m in ("base", "instruct"):
    for sub in ("generation", "pairs"):
        d = data_dir / m / sub
        files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.endswith(".tmp"))
        assert files, f"no text artifacts under {d}"
        kinds.append((m, sub, d, files))
        print(f"[i825-ops] p2c enumerate {m}/{sub}: {len(files)} files")
if not do_upload:
    print("[i825-ops] p2c smoke: hub upload SKIPPED (same enumeration path)")
    raise SystemExit(0)
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

for m, sub, d, files in kinds:
    dest = f"{prefix}/raw_completions/generation/{m}/{sub}"
    up = hub._upload(d, repo_id=repo, repo_type="dataset", path_in_repo=dest)
    assert up, f"text upload FAILED (hub._upload returned empty) -> {dest}"
    expected = [f"{dest}/{p.name}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), repo, expected, path_in_repo=dest, repo_type="dataset"
    )
    assert not missing, f"text upload verify FAILED — missing on Hub: {missing}"
    print(f"[i825-ops] p2c uploaded + exact-set verified {m}/{sub} ({len(expected)} files)")
PY

echo "[phase=p3_extract] on-policy armC capture x2 models (+ equivalence gates)"
for M in $MODELS; do
  MODEL_ID="$BASE_MODEL_ID"; [ "$M" = "instruct" ] && MODEL_ID="$INSTRUCT_MODEL_ID"
  uv run python scripts/issue931_extract_store.py --regime armC \
    --model-id "$MODEL_ID" --data-dir "$DATA_DIR/$M" \
    --equivalence-check --resume $EXTRACT_FLAGS
done

echo "[phase=p4_fits] on-policy fit batteries (ridge + rotated(+CI) + MLP) x2"
for M in $MODELS; do
  uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean \
    --mlp --rotated-ci --data-dir "$DATA_DIR/$M" --out-dir "$OUT_DIR/$M" $FIT_FLAGS
done

echo "[phase=p4b_anchors] exogenous anchor refits BOTH models + +-0.01 gates"
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --rotated-ci \
  --data-dir "$ANCHOR_INST" --out-dir "$OUT_DIR/anchor_inst" $FIT_FLAGS
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --rotated-ci \
  --data-dir "$ANCHOR_BASE" --out-dir "$OUT_DIR/anchor_base" $FIT_FLAGS
run_anchor_gate() {  # <anchor_dir> <committed_dir> <quotes_json> <label>
  # UPLOAD-THEN-HALT (plan section 4 G4b): the gate records pass/fail and the
  # dispatcher halts AFTER p5 uploads — persist-by-default beats fail-fast here.
  # rc=5 is the gate's DRIFT verdict; any other non-zero rc is a gate-script
  # CRASH (traceback, missing file) — logged distinctly, same halt-after-p5.
  local rc=0
  uv run python - "$1" "$2" "$3" "$GATE_BINDING" <<'PY' || rc=$?
"""Anchor reproduction gate: refit within +-0.01 of the COMMITTED ridge/rotated
values @ L19 (rig-drift detector; same-surface committed references)."""

import copy
import json
import sys
from pathlib import Path

anchor_dir, committed_dir = Path(sys.argv[1]), Path(sys.argv[2])
quoted = json.loads(sys.argv[3])
binding = sys.argv[4] == "1"
TOL = 0.01


def _vals(path: Path) -> dict:
    d = json.loads(path.read_text())
    hl = int(d.get("headline_layer", 19))
    return {
        "ridge": float(d["r2_per_layer_obs"][hl]),
        "rotated": float(d["random_projection_control_r2"][str(hl)]),
    }


def check(a_dir: Path, tol: float) -> tuple[bool, dict]:
    ok, deltas = True, {}
    for cell, q in quoted.items():
        com = _vals(committed_dir / f"cells_{cell}.json")
        for k, qv in q.items():
            assert abs(com[k] - qv) < 1e-3, (cell, k, com[k], qv, "committed != plan quote")
        got = _vals(a_dir / f"cells_{cell}.json")
        for k in ("ridge", "rotated"):
            d = abs(got[k] - com[k])
            deltas[f"{cell}.{k}"] = {"anchor": got[k], "committed": com[k], "abs_delta": d}
            ok = ok and d <= tol
    return ok, deltas


ok, deltas = check(anchor_dir, TOL)
(anchor_dir / "anchor_gate.json").write_text(
    json.dumps({"pass": ok, "tolerance": TOL, "binding": binding, "deltas": deltas}, indent=2)
)
print(
    f"[i825-ops] anchor gate {anchor_dir.name}: pass={ok} binding={binding} "
    f"deltas={ {k: round(v['abs_delta'], 5) for k, v in deltas.items()} }"
)
if not binding:
    # Smoke self-test: the gate MUST fire on a planted mismatch (mechanics).
    planted = anchor_dir / "planted"
    planted.mkdir(exist_ok=True)
    for cell in quoted:
        p = copy.deepcopy(json.loads((anchor_dir / f"cells_{cell}.json").read_text()))
        hl = int(p.get("headline_layer", 19))
        p["r2_per_layer_obs"][hl] = float(p["r2_per_layer_obs"][hl]) + 0.5
        (planted / f"cells_{cell}.json").write_text(json.dumps(p))
    ok_planted, _ = check(planted, TOL)
    assert not ok_planted, "planted +0.5 mismatch NOT caught — gate mechanics broken"
    print(f"[i825-ops] anchor-gate self-test PASS ({anchor_dir.name}): planted mismatch caught")
if binding and not ok:
    raise SystemExit(5)
PY
  if [ "$rc" = "5" ]; then
    echo "[i825-ops] G4b GATE FAIL ($4): rig drift — will HALT AFTER p5 uploads" >&2
    GATE_FAIL="1"
  elif [ "$rc" != "0" ]; then
    echo "[i825-ops] G4b gate script CRASHED ($4, rc=$rc) — NOT a drift verdict; will HALT AFTER p5 uploads" >&2
    GATE_FAIL="1"
  fi
}
# Committed same-surface references (plan section 1 quotes as drift anchors).
run_anchor_gate "$OUT_DIR/anchor_inst" "eval_results/issue_931" \
  '{"armC_sep": {"ridge": -3.1685, "rotated": 0.3489}, "armC_prevmean": {"ridge": -3.4970, "rotated": 0.3341}}' \
  instruct
run_anchor_gate "$OUT_DIR/anchor_base" "eval_results/issue_825/base-separator-control" \
  '{"armC_sep": {"ridge": -2.9157, "rotated": 0.3626}, "armC_prevmean": {"ridge": -3.4136, "rotated": 0.3449}}' \
  base

echo "[phase=p4c_matchedn] conditional matched-n W_ex re-baseline per model"
for M in $MODELS; do
  N_R=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['realized_n'])" \
        "$DATA_DIR/$M/pairs/pairs_meta.json")
  ANCHOR_DIR="$ANCHOR_BASE"; ANCHOR_OUT="anchor_base"
  [ "$M" = "instruct" ] && { ANCHOR_DIR="$ANCHOR_INST"; ANCHOR_OUT="anchor_inst"; }
  uv run python scripts/issue825_onpolicy_sep_matchedn.py --model "$M" \
    --anchor-store-dir "$ANCHOR_DIR/store/armC" --realized-n "$N_R" \
    --out "$OUT_DIR/$ANCHOR_OUT/matched_n_wex_$M.json" $MATCHEDN_FLAGS
done

echo "[phase=p4d_decision] decision_support.json (plan section 6.5)"
uv run python scripts/issue825_onpolicy_sep_decision.py \
  --out-dir "$OUT_DIR" --data-root "$DATA_DIR" $DECISION_FLAGS

echo "[phase=p5_upload] store + eval-JSON uploads, git commit, sentinel"
uv run python - "$DATA_DIR" "$OUT_DIR" "$HF_PREFIX_OUT" "$DO_UPLOAD" <<'PY'
import sys
from pathlib import Path

data_dir, out_dir, prefix, do_upload = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "1",
)
repo = "superkaiba1/explore-persona-space-data"
stores = []
for m in ("base", "instruct"):
    armc = data_dir / m / "store" / "armC"
    files = sorted(p for p in armc.iterdir() if p.is_file())
    assert files, f"no armC store files under {armc}"
    stores.append((m, armc, files))
eval_files = sorted(str(p) for p in out_dir.rglob("*.json"))
print(
    f"[i825-ops] p5 enumerate: stores "
    f"{[(m, len(fs)) for m, _, fs in stores]}, {len(eval_files)} eval JSONs"
)
if not do_upload:
    print("[i825-ops] p5 smoke: hub upload SKIPPED (same enumeration path)")
    raise SystemExit(0)
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

for m, armc, files in stores:
    store_prefix = f"{prefix}/analysis_tensors/armC_{m}"
    up = hub._upload(armc, repo_id=repo, repo_type="dataset", path_in_repo=store_prefix)
    assert up, f"store upload FAILED (hub._upload returned empty) -> {store_prefix}"
    expected = [f"{store_prefix}/{p.name}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), repo, expected, path_in_repo=store_prefix, repo_type="dataset"
    )
    assert not missing, f"store upload verify FAILED — missing on Hub: {missing}"
    print(f"[i825-ops] p5 store {m} uploaded + exact-set verified ({len(expected)} files)")
mirror_prefix = f"{prefix}/eval_results_mirror"
up = hub._upload(out_dir, repo_id=repo, repo_type="dataset", path_in_repo=mirror_prefix)
assert up, f"eval_results_mirror upload FAILED -> {mirror_prefix}"
expected_mirror = sorted(
    f"{mirror_prefix}/{p.relative_to(out_dir)}" for p in out_dir.rglob("*.json")
)
assert expected_mirror, f"no eval JSONs under {out_dir} to mirror"
missing = hub.verify_repo_paths_uploaded(
    HfApi(), repo, expected_mirror, path_in_repo=mirror_prefix, repo_type="dataset"
)
assert not missing, f"mirror upload verify FAILED — missing on Hub: {missing}"
print(f"[i825-ops] p5 mirror uploaded + exact-set verified ({len(expected_mirror)} JSONs)")
PY

if [ "$DO_UPLOAD" = "1" ]; then
  echo "[i825-ops] committing eval JSONs to the issue branch"
  git add "$OUT_DIR"
  if [ -z "$(git status --porcelain -- "$OUT_DIR")" ]; then
    echo "[i825-ops] nothing to commit"
  else
    git commit -m "task #${ISSUE}: onpolicy-separator-control pod eval JSONs"
    git push origin "issue-${ISSUE}" || echo "[i825-ops] WARNING: git push failed (HF mirror uploaded)" >&2
  fi
fi

echo "[i825-ops] writing results sentinel"
uv run python - "$ISSUE" "$OUT_DIR" "$LOG_DIR" "$MODE" "$GATE_FAIL" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

issue, out_dir, log_dir, mode, gate_fail = (
    int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5] == "1",
)
run_start = int(os.environ.get("RUN_START_EPOCH", "0"))
gpu_hours = round((time.time() - run_start) / 3600.0, 3) if run_start else None


def _read(rel):
    p = out_dir / rel
    return json.loads(p.read_text()) if p.exists() else None


dec = _read("decision_support.json") or {}
per_model = dec.get("per_model", {})
eval_numbers = {
    "mode": mode,
    "anchor_gate_fail": gate_fail,
    "per_model": {
        m: {
            "W_on_max_L19": (pm.get("onpolicy") or {}).get("w_on_max"),
            "rotated_L19": (pm.get("onpolicy") or {}).get("rotated"),
            "mlp_L19": (pm.get("onpolicy") or {}).get("mlp"),
            "ridge_L19": (pm.get("onpolicy") or {}).get("ridge"),
            "D": pm.get("D"),
            "realized_n": pm.get("realized_n"),
            "matched_n_trigger_fired": pm.get("matched_n_trigger_fired"),
            "w_ex_matched_n": pm.get("w_ex_matched_n"),
        }
        for m, pm in per_model.items()
    },
    "r5_mirror": dec.get("r5_mirror"),
    "gates": dec.get("gates"),
    "note": "R4 transfer fractions run in Phase C on the VM "
    "(onpolicy_sep_to_chat_{base,instruct}.json); binding interpretation is the analyzer's",
}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out_dir.rglob("*.json")),
    "reproducibility_card": {
        "wandb_url": "n/a - no training in this round (no WandB runs)",
        "hf_hub_url": (
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
            "tree/main/issue825_onpolicy_sep_control"
        ),
        "worktree_path": ".claude/worktrees/issue-825-base-sep",
        "final_commit_sha": commit,
        "gpu_hours_used": gpu_hours,  # 1x A100: wall == GPU-hours
        "gpu_hours_budgeted": 3,
        "plan_deviations": [],
    },
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": issue,
    "by": "issue825_onpolicy_sep_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
path = log_dir / f"issue-{issue}-epm_results-{int(time.time())}.json"
path.write_text(json.dumps(sentinel, indent=2))
print(f"[i825-ops] sentinel written: {path}")
PY

if [ "$GATE_FAIL" = "1" ] && [ "$GATE_BINDING" = "1" ]; then
  echo "[i825-ops] G4b anchor gate FAILED — uploads + sentinel done; halting (upload-then-halt)" >&2
  exit 5
fi
echo "[phase=done]"
