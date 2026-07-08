#!/usr/bin/env bash
# Issue #825 follow-up `sampled-separator-control` (plan v22): the SAMPLED
# decoding twin of round 7 (`onpolicy-separator-control`) — same pair recipe,
# own-continuation spans now SAMPLED (T=1.0 / top-p 0.95; the Track-S
# chat-ceiling decoding convention) instead of greedy; plus the arm-C
# 10-draw averaged read at a FIXED prefix-final anchor.
#
# Phases (plan v22 section 4): p0_stage (pinned pair files + BOTH exogenous
# armC anchor stores + the ROUND-7 on-policy stores @ 4435ced2 for the
# reproduction gates) -> p1_wiring -> p2_generate (per model: arm B sampled
# 1-draw + wave-2 top-up; arm C K=10 draws, seeds 4300+k, max_tokens 320) ->
# p2b_pairs (arm B continuation-region ladder; arm C --anchor-mode
# prefix-final) -> p2c_upload1 (UNCONDITIONAL text upload BEFORE any extract,
# #779) -> p3_extract (per model x arm) -> p3b_reduce (X-identity gate +
# C-avg / C-single stores) -> p4_fits (per model x {armB, armC_avg,
# armC_single, armC_pooled}: ridge + rotated(+CI) + MLP(+CI)) -> p4b_anchors
# (exogenous refits + +-0.01 gates; ROUND-7 reproduction refits + +-0.01
# gates; posmatch --pos-max 256 companion) -> p4c_matchedn (arm-B conditional
# + arm-C fires-by-construction) -> p4d_decision -> p5_upload (6 stores +
# preds maps + eval JSONs + sentinel) -> [phase=done].
#
# ONE code path for smoke / production — the tiny sizing threads through every
# phase (Step 6d.0 PASS_UNIFIED): smoke = tiny random-init Qwen2 (real
# tokenizer) + --max-items 3 + true-continuation substitute + arm-C
# --n-draws 3 + 1-shard anchor/repro subsets + non-binding gates
# (planted-mismatch self-tests keep the gate MECHANICS exercised) + skip-hub
# upload on the same enumeration path; scratch dirs — committed
# eval_results/figures never touched.
#
# Modes:
#   bash scripts/issue825_sampled_sep_dispatch.sh --smoke   # CPU VM smoke
#   bash scripts/issue825_sampled_sep_dispatch.sh           # production (A100)
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
LABEL="sampled-separator-control"
BASE_MODEL_ID="Qwen/Qwen2.5-7B"
INSTRUCT_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
PAIRS_REV="9534b9981d6b4fb4f1259c9b06f021d311a46af4"   # plan v22 section 10 pin
BASE_STORE_REV="d4085b09d79fc46537b9da60bd6ffd8a754a677a"  # plan v22 section 10 pin
R7_STORE_REV="4435ced2273df379f3e1c15bf5cdf56ca2ba40ae"    # round-7 stores pin
INST_ANCHOR_PREFIX="issue931_story_map/analysis_tensors/armC"
BASE_ANCHOR_PREFIX="issue825_base_sep_control/analysis_tensors/armC"
R7_PREFIX="issue825_onpolicy_sep_control/analysis_tensors"
HF_PREFIX_OUT="issue825_sampled_sep_control"
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
  SMOKE_ROOT="${SMOKE_ROOT:-/tmp/issue-825-sampled-sep-smoke}"
  rm -rf "$SMOKE_ROOT"; mkdir -p "$SMOKE_ROOT"
  DATA_DIR="$SMOKE_ROOT/data"
  ANCHOR_INST="$SMOKE_ROOT/anchor_inst"
  ANCHOR_BASE="$SMOKE_ROOT/anchor_base"
  REPRO_BASE_DIR="$SMOKE_ROOT/repro_base"
  REPRO_INST_DIR="$SMOKE_ROOT/repro_instruct"
  OUT_DIR="$SMOKE_ROOT/eval"
  LOG_DIR="$SMOKE_ROOT/logs"; mkdir -p "$LOG_DIR"
  TINY="$SMOKE_ROOT/tiny_model"
  uv run python scripts/issue931_extract_store.py --make-tiny-model "$TINY"
  # 10 articles (not 3): the C-avg/C-single cells carry ONE row per article, so
  # the 5-fold group CV needs ~2 groups per fold — fewer groups leave 1-row test
  # folds whose per-fold-centered ss_tot is 0 (or all folds skipped), the layer
  # sweep goes all-NaN, and selection_symmetric_summary's nanargmax crashes.
  GEN_B_FLAGS="--max-items 10 --tiny-model-dir $TINY --smoke-real-continuation --wave2-min-eligible 999"
  GEN_C_FLAGS="--max-items 10 --tiny-model-dir $TINY --smoke-real-continuation --n-draws 3"
  EXTRACT_FLAGS="--tiny-model-dir $TINY --batch-size 2 --max-items 50"
  REDUCE_FLAGS="--k-valid-floor 2 --smoke --self-test"
  FIT_FLAGS="--smoke --null-draws 2 --n-boot 50"
  MATCHEDN_FLAGS="--seeds 931,932 --skip-mlp --smoke"
  POSMATCH_FLAGS="--smoke"
  DECISION_FLAGS="--smoke --n-boot 50"
  ANCHOR_SHARDS="1"      # 1-shard real-Hub staging probe (h)(iv)
  EXPECT_N="auto"        # anchor rows derived from the staged sidecars
  GATE_BINDING="0"       # +-0.01 gates recorded + planted-mismatch self-tests
  DO_UPLOAD="0"
else
  DATA_DIR="data/issue_${ISSUE}/sampled_sep"
  ANCHOR_INST="data/issue_${ISSUE}/sampled_sep_anchor_inst"
  ANCHOR_BASE="data/issue_${ISSUE}/sampled_sep_anchor_base"
  REPRO_BASE_DIR="data/issue_${ISSUE}/sampled_sep_repro_base"
  REPRO_INST_DIR="data/issue_${ISSUE}/sampled_sep_repro_instruct"
  OUT_DIR="eval_results/issue_${ISSUE}/sampled-separator-control"
  GEN_B_FLAGS=""
  GEN_C_FLAGS="--n-draws 10"
  EXTRACT_FLAGS="--batch-size 8"
  REDUCE_FLAGS=""
  FIT_FLAGS=""
  MATCHEDN_FLAGS=""
  POSMATCH_FLAGS=""
  DECISION_FLAGS=""
  ANCHOR_SHARDS="all"
  EXPECT_N="3600"
  GATE_BINDING="1"
  DO_UPLOAD="1"
fi
mkdir -p "$OUT_DIR"
GATE_FAIL="0"

echo "[phase=p0_stage] pinned pair files + exogenous + round-7 anchor stores"
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR/shared" --anchor-dir "$ANCHOR_INST" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --anchor-prefix "$INST_ANCHOR_PREFIX" \
  --expect-n "$EXPECT_N" --manifest-name sampled_sep_stage_inst.json \
  $( [ "$MODE" = smoke ] && echo --self-test )
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR/shared" --anchor-dir "$ANCHOR_BASE" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --anchor-prefix "$BASE_ANCHOR_PREFIX" --anchor-revision "$BASE_STORE_REV" \
  --expect-n "$EXPECT_N" --skip-pairs --manifest-name sampled_sep_stage_base.json
# Round-7 on-policy stores (reproduction-gate inputs; plan v22 G0(3)).
R7_EXPECT_BASE="3577"; R7_EXPECT_INST="3591"
if [ "$MODE" = "smoke" ]; then R7_EXPECT_BASE="auto"; R7_EXPECT_INST="auto"; fi
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR/shared" --anchor-dir "$REPRO_BASE_DIR" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --anchor-prefix "$R7_PREFIX/armC_base" --anchor-revision "$R7_STORE_REV" \
  --expect-n "$R7_EXPECT_BASE" --skip-pairs --manifest-name sampled_sep_stage_repro_base.json
uv run python scripts/issue825_base_sep_stage.py \
  --data-dir "$DATA_DIR/shared" --anchor-dir "$REPRO_INST_DIR" --out-dir "$OUT_DIR" \
  --pairs-revision "$PAIRS_REV" --anchor-shards "$ANCHOR_SHARDS" \
  --anchor-prefix "$R7_PREFIX/armC_instruct" --anchor-revision "$R7_STORE_REV" \
  --expect-n "$R7_EXPECT_INST" --skip-pairs --manifest-name sampled_sep_stage_repro_inst.json

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
print(f"[i825-ss] p1 PASS: vocab identical; tokenizer.json sha {sb[:12]}; 28 layers / 3584")
PY

echo "[phase=p2_generate] SAMPLED continuations (T=1.0 top-p 0.95) x2 models x2 arms"
for M in $MODELS; do
  # Each (model x arm) is its own process => the vLLM engine dies with it
  # (teardown gotcha); the extra engine loads (~2-3 min each) are inside the
  # plan section 9 G2 row margin.
  uv run python scripts/issue825_onpolicy_sep_gen.py --model "$M" \
    --articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --out-dir "$DATA_DIR/$M/armB/generation" \
    --temperature 1.0 --top-p 0.95 $GEN_B_FLAGS
  uv run python scripts/issue825_onpolicy_sep_gen.py --model "$M" \
    --articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --out-dir "$DATA_DIR/$M/armC/generation" \
    --temperature 1.0 --top-p 0.95 --draw-seed-base 4300 \
    --max-tokens-override 320 --no-wave2 $GEN_C_FLAGS
done

echo "[phase=p2b_pairs] pair construction x2 models (arm B ladder; arm C prefix-final)"
for M in $MODELS; do
  uv run python scripts/issue825_onpolicy_sep_pairs.py --model "$M" \
    --articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --continuations "$DATA_DIR/$M/armB/generation/continuations.jsonl" \
    --exogenous-articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --exogenous-pairs "$DATA_DIR/shared/pairs/pairs_armC.jsonl" \
    --out-data-dir "$DATA_DIR/$M/armB" --followup-label "$LABEL"
  uv run python scripts/issue825_onpolicy_sep_pairs.py --model "$M" \
    --articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --continuations "$DATA_DIR/$M/armC/generation/continuations.jsonl" \
    --exogenous-articles "$DATA_DIR/shared/pairs/articles_armC.jsonl" \
    --exogenous-pairs "$DATA_DIR/shared/pairs/pairs_armC.jsonl" \
    --out-data-dir "$DATA_DIR/$M/armC" --anchor-mode prefix-final --followup-label "$LABEL"
done

echo "[phase=p2c_upload1] UNCONDITIONAL text upload BEFORE any extract (#779)"
uv run python - "$DATA_DIR" "$HF_PREFIX_OUT" "$DO_UPLOAD" <<'PY'
import sys
from pathlib import Path

data_dir, prefix, do_upload = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
repo = "superkaiba1/explore-persona-space-data"
kinds = []
for m in ("base", "instruct"):
    for arm in ("armB", "armC"):
        for sub in ("generation", "pairs"):
            d = data_dir / m / arm / sub
            files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.endswith(".tmp"))
            assert files, f"no text artifacts under {d}"
            kinds.append((m, arm, sub, d, files))
            print(f"[i825-ss] p2c enumerate {m}/{arm}/{sub}: {len(files)} files")
if not do_upload:
    print("[i825-ss] p2c smoke: hub upload SKIPPED (same enumeration path)")
    raise SystemExit(0)
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

for m, arm, sub, d, files in kinds:
    dest = f"{prefix}/raw_completions/generation/{m}/{arm}/{sub}"
    up = hub._upload(d, repo_id=repo, repo_type="dataset", path_in_repo=dest)
    assert up, f"text upload FAILED (hub._upload returned empty) -> {dest}"
    expected = [f"{dest}/{p.name}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), repo, expected, path_in_repo=dest, repo_type="dataset"
    )
    assert not missing, f"text upload verify FAILED — missing on Hub: {missing}"
    print(f"[i825-ss] p2c uploaded + exact-set verified {m}/{arm}/{sub} ({len(expected)} files)")
PY

echo "[phase=p3_extract] on-policy armC capture x2 models x2 arms (+ equivalence gates)"
for M in $MODELS; do
  MODEL_ID="$BASE_MODEL_ID"; [ "$M" = "instruct" ] && MODEL_ID="$INSTRUCT_MODEL_ID"
  for ARM in armB armC; do
    uv run python scripts/issue931_extract_store.py --regime armC \
      --model-id "$MODEL_ID" --data-dir "$DATA_DIR/$M/$ARM" \
      --equivalence-check --resume $EXTRACT_FLAGS
  done
done

echo "[phase=p3b_reduce] X-identity gate + C-avg / C-single stores x2 models"
for M in $MODELS; do
  rc=0
  uv run python scripts/issue825_sampled_sep_reduce.py --model "$M" \
    --pooled-data-dir "$DATA_DIR/$M/armC" \
    --avg-data-dir "$DATA_DIR/$M/armC_avg" \
    --single-data-dir "$DATA_DIR/$M/armC_single" \
    --out-dir "$OUT_DIR/$M" $REDUCE_FLAGS || rc=$?
  if [ "$rc" = "7" ]; then
    echo "[i825-ss] X-IDENTITY GATE FAIL ($M) — stores written; will HALT AFTER p5 uploads" >&2
    GATE_FAIL="1"
  elif [ "$rc" != "0" ]; then
    echo "[i825-ss] reduce CRASHED ($M, rc=$rc)" >&2
    exit "$rc"
  fi
done

echo "[phase=p4_fits] fit batteries (ridge + rotated(+CI) + MLP(+CI)) x2 models x4 arms"
for M in $MODELS; do
  # armC_pooled fits run on the POOLED extraction dir (armC) with output
  # routed to the armC_pooled OUT subdir; avg/single have their own data dirs.
  uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean \
    --mlp --mlp-ci --rotated-ci --data-dir "$DATA_DIR/$M/armB" \
    --out-dir "$OUT_DIR/$M/armB" $FIT_FLAGS
  uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean \
    --mlp --mlp-ci --rotated-ci --data-dir "$DATA_DIR/$M/armC_avg" \
    --out-dir "$OUT_DIR/$M/armC_avg" $FIT_FLAGS
  uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean \
    --mlp --mlp-ci --rotated-ci --data-dir "$DATA_DIR/$M/armC_single" \
    --out-dir "$OUT_DIR/$M/armC_single" $FIT_FLAGS
  uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean \
    --mlp --mlp-ci --rotated-ci --data-dir "$DATA_DIR/$M/armC" \
    --out-dir "$OUT_DIR/$M/armC_pooled" $FIT_FLAGS
done

echo "[phase=p4b_anchors] exogenous + round-7 reproduction refits + +-0.01 gates"
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --rotated-ci \
  --data-dir "$ANCHOR_INST" --out-dir "$OUT_DIR/anchor_inst" $FIT_FLAGS
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --rotated-ci \
  --data-dir "$ANCHOR_BASE" --out-dir "$OUT_DIR/anchor_base" $FIT_FLAGS
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --rotated-ci \
  --data-dir "$REPRO_BASE_DIR" --out-dir "$OUT_DIR/repro_base" $FIT_FLAGS
uv run python scripts/issue931_fit_cells.py --cells armC_sep,armC_prevmean --rotated-ci \
  --data-dir "$REPRO_INST_DIR" --out-dir "$OUT_DIR/repro_instruct" $FIT_FLAGS
run_anchor_gate() {  # <anchor_dir> <committed_dir> <quotes_json> <label>
  # UPLOAD-THEN-HALT (plan v22 section 4 G4b): the gate records pass/fail and
  # the dispatcher halts AFTER p5 uploads — persist-by-default beats fail-fast.
  # rc=5 is the gate's DRIFT verdict; any other non-zero rc is a gate-script
  # CRASH (traceback, missing file) — logged distinctly, same halt-after-p5.
  local rc=0
  uv run python - "$1" "$2" "$3" "$GATE_BINDING" <<'PY' || rc=$?
"""Anchor/reproduction gate: refit within +-0.01 of the COMMITTED ridge/rotated
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
    f"[i825-ss] gate {anchor_dir.name}: pass={ok} binding={binding} "
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
    print(f"[i825-ss] gate self-test PASS ({anchor_dir.name}): planted mismatch caught")
if binding and not ok:
    raise SystemExit(5)
PY
  if [ "$rc" = "5" ]; then
    echo "[i825-ss] G4b GATE FAIL ($4): rig drift — will HALT AFTER p5 uploads" >&2
    GATE_FAIL="1"
  elif [ "$rc" != "0" ]; then
    echo "[i825-ss] G4b gate script CRASHED ($4, rc=$rc) — NOT a drift verdict; will HALT AFTER p5 uploads" >&2
    GATE_FAIL="1"
  fi
}
# Committed same-surface references (plan v22 sections 1/4 quotes as anchors).
run_anchor_gate "$OUT_DIR/anchor_inst" "eval_results/issue_931" \
  '{"armC_sep": {"ridge": -3.1685, "rotated": 0.3489}, "armC_prevmean": {"ridge": -3.4970, "rotated": 0.3341}}' \
  instruct
run_anchor_gate "$OUT_DIR/anchor_base" "eval_results/issue_825/base-separator-control" \
  '{"armC_sep": {"ridge": -2.9157, "rotated": 0.3626}, "armC_prevmean": {"ridge": -3.4136, "rotated": 0.3449}}' \
  base
# Round-7 on-policy reproduction gates (rig identity vs the greedy round —
# incl. reproducing the PATHOLOGICAL base rotated -1.228: certifies the rig).
run_anchor_gate "$OUT_DIR/repro_base" "eval_results/issue_825/onpolicy-separator-control/base" \
  '{"armC_sep": {"ridge": -1.5914, "rotated": -1.2278}, "armC_prevmean": {"ridge": -2.0760, "rotated": -1.1485}}' \
  repro_base
run_anchor_gate "$OUT_DIR/repro_instruct" "eval_results/issue_825/onpolicy-separator-control/instruct" \
  '{"armC_sep": {"ridge": -2.1930, "rotated": 0.4878}, "armC_prevmean": {"ridge": -2.5186, "rotated": 0.4066}}' \
  repro_instruct

echo "[phase=p4b2_posmatch] position-restricted exogenous companion (--pos-max 256)"
for M in $MODELS; do
  ANCHOR_DIR="$ANCHOR_BASE"; ANCHOR_OUT="anchor_base"
  [ "$M" = "instruct" ] && { ANCHOR_DIR="$ANCHOR_INST"; ANCHOR_OUT="anchor_inst"; }
  uv run python scripts/issue825_position_matched_wex.py --pos-max 256 --model "$M" \
    --store-dir "$ANCHOR_DIR/store/armC" \
    --pairs-file "$DATA_DIR/shared/pairs/pairs_armC.jsonl" \
    --out-dir "$OUT_DIR/$ANCHOR_OUT" $POSMATCH_FLAGS
done

echo "[phase=p4c_matchedn] matched-n W_ex re-baselines (arm B conditional; arm C by construction)"
for M in $MODELS; do
  ANCHOR_DIR="$ANCHOR_BASE"; ANCHOR_OUT="anchor_base"
  [ "$M" = "instruct" ] && { ANCHOR_DIR="$ANCHOR_INST"; ANCHOR_OUT="anchor_inst"; }
  N_B=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['realized_n'])" \
        "$DATA_DIR/$M/armB/pairs/pairs_meta.json")
  uv run python scripts/issue825_onpolicy_sep_matchedn.py --model "$M" \
    --anchor-store-dir "$ANCHOR_DIR/store/armC" --realized-n "$N_B" \
    --out "$OUT_DIR/$ANCHOR_OUT/matched_n_wex_armB_$M.json" $MATCHEDN_FLAGS
  N_C=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_avg_rows'])" \
        "$OUT_DIR/$M/reduce_summary.json")
  uv run python scripts/issue825_onpolicy_sep_matchedn.py --model "$M" \
    --anchor-store-dir "$ANCHOR_DIR/store/armC" --realized-n "$N_C" \
    --out "$OUT_DIR/$ANCHOR_OUT/matched_n_wex_armC_$M.json" $MATCHEDN_FLAGS
done

echo "[phase=p4d_decision] decision_support.json (plan v22 section 6.5)"
uv run python scripts/issue825_sampled_sep_decision.py \
  --out-dir "$OUT_DIR" --data-root "$DATA_DIR" $DECISION_FLAGS

echo "[phase=p5_upload] store + preds-map + eval-JSON uploads, git commit, sentinel"
uv run python - "$DATA_DIR" "$OUT_DIR" "$HF_PREFIX_OUT" "$DO_UPLOAD" <<'PY'
import sys
from pathlib import Path

data_dir, out_dir, prefix, do_upload = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "1",
)
repo = "superkaiba1/explore-persona-space-data"
stores = []
for m in ("base", "instruct"):
    # 6 stores (plan v22 G5): armB, pooled (armC), avg; single is a row
    # subset of pooled (derivable via the draw-0 allowlist — not uploaded).
    for arm, slug in (("armB", "armB"), ("armC", "armC_pooled"), ("armC_avg", "armC_avg")):
        d = data_dir / m / arm / "store" / "armC"
        files = sorted(p for p in d.iterdir() if p.is_file())
        assert files, f"no store files under {d}"
        stores.append((m, slug, d, files))
maps = []
for m in ("base", "instruct"):
    for arm in ("armB", "armC", "armC_avg", "armC_single"):
        d = data_dir / m / arm / "store" / "preds"
        if d.exists():
            files = sorted(p for p in d.iterdir() if p.is_file())
            if files:
                maps.append((m, arm, d, files))
eval_files = sorted(str(p) for p in out_dir.rglob("*.json"))
print(
    f"[i825-ss] p5 enumerate: stores {[(m, s, len(fs)) for m, s, _, fs in stores]}, "
    f"maps {[(m, a, len(fs)) for m, a, _, fs in maps]}, {len(eval_files)} eval JSONs"
)
if not do_upload:
    print("[i825-ss] p5 smoke: hub upload SKIPPED (same enumeration path)")
    raise SystemExit(0)
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

for m, slug, d, files in stores:
    store_prefix = f"{prefix}/analysis_tensors/{slug}_{m}"
    up = hub._upload(d, repo_id=repo, repo_type="dataset", path_in_repo=store_prefix)
    assert up, f"store upload FAILED (hub._upload returned empty) -> {store_prefix}"
    expected = [f"{store_prefix}/{p.name}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), repo, expected, path_in_repo=store_prefix, repo_type="dataset"
    )
    assert not missing, f"store upload verify FAILED — missing on Hub: {missing}"
    print(f"[i825-ss] p5 store {slug}_{m} uploaded + exact-set verified ({len(expected)} files)")
for m, arm, d, files in maps:
    map_prefix = f"{prefix}/analysis_tensors/maps/{m}_{arm}"
    up = hub._upload(d, repo_id=repo, repo_type="dataset", path_in_repo=map_prefix)
    assert up, f"maps upload FAILED -> {map_prefix}"
    expected = [f"{map_prefix}/{p.name}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), repo, expected, path_in_repo=map_prefix, repo_type="dataset"
    )
    assert not missing, f"maps upload verify FAILED — missing on Hub: {missing}"
    print(f"[i825-ss] p5 maps {m}/{arm} uploaded + exact-set verified ({len(expected)} files)")
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
print(f"[i825-ss] p5 mirror uploaded + exact-set verified ({len(expected_mirror)} JSONs)")
PY

if [ "$DO_UPLOAD" = "1" ]; then
  echo "[i825-ss] committing eval JSONs to the issue branch"
  git add "$OUT_DIR"
  if [ -z "$(git status --porcelain -- "$OUT_DIR")" ]; then
    echo "[i825-ss] nothing to commit"
  else
    git commit -m "task #${ISSUE}: sampled-separator-control pod eval JSONs"
    git push origin "issue-${ISSUE}" || echo "[i825-ss] WARNING: git push failed (HF mirror uploaded)" >&2
  fi
fi

echo "[i825-ss] writing results sentinel"
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


def _arm(pm, arm):
    a = (pm.get("arms") or {}).get(arm) or {}
    if a.get("missing"):
        return None
    return {
        "W_max_L19": (a.get("reads") or {}).get("w_max"),
        "D": a.get("D"),
        "realized_n": a.get("realized_n"),
        "w_ex_kind": a.get("w_ex_kind"),
    }


eval_numbers = {
    "mode": mode,
    "gate_fail": gate_fail,
    "per_model": {
        m: {
            "arms": {arm: _arm(pm, arm) for arm in ("armB", "armC_avg", "armC_single", "armC_pooled")},
            "delta_dec": (pm.get("r2_decoding_sensitivity") or {}).get("delta_dec"),
            "r2_label": (pm.get("r2_decoding_sensitivity") or {}).get("label"),
            "r2_qualifier": (pm.get("r2_decoding_sensitivity") or {}).get("qualifier"),
            "ns_max": (pm.get("r3_sampling_noise_share") or {}).get("ns_max_interpretable"),
            "flag_rate_armB": ((pm.get("r6_flag_rates") or {}).get("armB") or {}).get(
                "repetition_rate_min5"
            ),
            "flag_rate_armC": ((pm.get("r6_flag_rates") or {}).get("armC") or {}).get(
                "repetition_rate_min5"
            ),
        }
        for m, pm in per_model.items()
    },
    "r5_mirror": dec.get("r5_mirror"),
    "gates": dec.get("gates"),
    "note": "R4 transfer fractions run in Phase C on the VM "
    "(sampled_sep_to_chat_{base,instruct}_{armB,armC_avg}.json); binding "
    "interpretation is the analyzer's",
}
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
note = {
    "eval_numbers": eval_numbers,
    "eval_paths": sorted(str(p) for p in out_dir.rglob("*.json")),
    "reproducibility_card": {
        "wandb_url": "n/a - no training in this round (no WandB runs)",
        "hf_hub_url": (
            "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
            "tree/main/issue825_sampled_sep_control"
        ),
        "worktree_path": ".claude/worktrees/issue-825-base-sep",
        "final_commit_sha": commit,
        "gpu_hours_used": gpu_hours,  # 1x A100: wall == GPU-hours
        "gpu_hours_budgeted": 7,
        "plan_deviations": [],
    },
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:smoke-result" if mode == "smoke" else "epm:results",
    "version": 1,
    "task_id": issue,
    "by": "issue825_sampled_sep_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
slug = "epm_smoke-result" if mode == "smoke" else "epm_results"
path = log_dir / f"issue-{issue}-{slug}-{int(time.time())}.json"
path.write_text(json.dumps(sentinel, indent=2))
print(f"[i825-ss] sentinel written: {path}")
PY

if [ "$GATE_FAIL" = "1" ] && [ "$GATE_BINDING" = "1" ]; then
  echo "[i825-ss] a HALT-class gate FAILED — uploads + sentinel done; halting (upload-then-halt)" >&2
  exit 5
fi
echo "[phase=done]"
