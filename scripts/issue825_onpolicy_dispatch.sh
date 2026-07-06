#!/usr/bin/env bash
# task #825 follow-up "onpolicy-user-turn" (plan v7):
#   stage -> gen -> UPLOAD-1 -> extract x4 -> wiring -> fit -> matched_parent
#   -> summarize -> UPLOAD-2 (BEFORE gates, MF-C) -> gates -> sentinel
#
# Production (worker):  bash scripts/issue825_onpolicy_dispatch.sh
# Smoke (CPU, tiny):    bash scripts/issue825_onpolicy_dispatch.sh --smoke
#   Smoke IS this wrapper (EPS_SMOKE=1): same phases, same commands, n=EPS_SMOKE_N
#   rows, tiny random-init Qwen2 (real tokenizer) substituting the 7B/vLLM loads,
#   all outputs under a /tmp scratch root, numeric gates BYPASSED (plan MF-D),
#   structural assertions still binding. Uploads run a structural glob assert
#   instead of writing smoke garbage to the HF repo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then SMOKE="1"; fi

HF_REV="deb7a4523b5233393e4fbd2497622527b3622d35"
DATA_REPO="superkaiba1/explore-persona-space-data"
HF_ONP_PREFIX="issue825_onpolicy_user_turn"
CELLS8="M_instruct_assistant_chat,M_instruct_assistant_naturalistic,M_pretrained_assistant_chat,M_pretrained_assistant_naturalistic,M_instruct_user_chat,M_instruct_user_naturalistic,M_pretrained_user_chat,M_pretrained_user_naturalistic"
USER4="M_instruct_user_chat,M_instruct_user_naturalistic,M_pretrained_user_chat,M_pretrained_user_naturalistic"
MODELS="instruct pretrained"
PARENT_EVAL="eval_results/issue_825"
PARENT_MLP="eval_results/issue_825/mlp-unprobed-cells"
MATCHED_SUBSET_FLOOR=1900

if [[ -n "$SMOKE" ]]; then
  ROOT="${EPS_SMOKE_ROOT:-/tmp/issue-825-onpolicy-smoke}"
  ONP_DIR="$ROOT/onpolicy"
  TS_DIR="$ROOT/turnstore_onpolicy"
  OUT_DIR="$ROOT/eval_results/onpolicy-user-turn"
  STAGE_DIR="$ROOT/stage"
  SENTINEL_DIR="$ROOT/logs"
  CONV="$STAGE_DIR/conversations.jsonl"
  FOLDS=3 NULLS=3 NBOOT=20 WROWS=8
  TINY="${EPS_TINY_MODEL_DIR:-$ROOT/tiny-qwen}"
  export EPS_SMOKE=1 EPS_SMOKE_N="${EPS_SMOKE_N:-8}"
  # Bounded MLP secondary at smoke scale (production keeps the 1800 s default).
  export EPS_MLP_TIME_BUDGET_S="${EPS_MLP_TIME_BUDGET_S:-300}"
else
  ONP_DIR="data/issue_825/onpolicy"
  TS_DIR="data/issue_825/turnstore_onpolicy"
  OUT_DIR="eval_results/issue_825/onpolicy-user-turn"
  STAGE_DIR="data/issue_825/hf_dl/onpolicy_stage"
  SENTINEL_DIR="/workspace/logs"
  CONV="$STAGE_DIR/conversations.jsonl"
  FOLDS=5 NULLS=20 NBOOT=1000 WROWS=200
  TINY=""
  export EPS_SMOKE=""
fi
SENTINEL="$SENTINEL_DIR/issue-825-epm_results-$(date +%s).json"
EPS_T0="$(date +%s)"  # wall-clock start: sentinel reports MEASURED gpu-hours (review-r1 Minor)
export EPS_T0

# Credentials at script level (never bare load_dotenv() inside a stdin heredoc — gotcha).
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mkdir -p "$SENTINEL_DIR" "$ONP_DIR" "$TS_DIR" "$OUT_DIR" "$STAGE_DIR"
# Run-2/3 lesson (this task): the repo is Xet-backed and the xet downloader's
# finalization hangs (#515 class) — plain CDN path on every HF phase (plan §10).
export HF_XET_DISABLE=1
export EPS_ONP_DIR="$ONP_DIR" EPS_TS_DIR="$TS_DIR" EPS_OUT_DIR="$OUT_DIR" \
  EPS_STAGE_DIR="$STAGE_DIR" EPS_CONV="$CONV" EPS_SENTINEL="$SENTINEL" \
  EPS_HF_REV="$HF_REV" EPS_DATA_REPO="$DATA_REPO" EPS_HF_ONP_PREFIX="$HF_ONP_PREFIX" \
  EPS_CELLS8="$CELLS8" EPS_USER4="$USER4" EPS_PARENT_EVAL="$PARENT_EVAL" \
  EPS_PARENT_MLP="$PARENT_MLP" EPS_MATCHED_SUBSET_FLOOR="$MATCHED_SUBSET_FLOOR" \
  EPS_TINY_DIR="$TINY"

if [[ -n "$SMOKE" && ! -d "$TINY" ]]; then
  echo "[smoke] fabricating tiny 28-layer Qwen2 (real tokenizer) -> $TINY"
  uv run python - <<'PY'
import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

tiny = os.environ["EPS_TINY_DIR"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
cfg = Qwen2Config(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=28,  # peak layers 14/18/19/26 must be in range
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=max(max(tok.get_vocab().values()), max(tok.all_special_ids)) + 1,
    max_position_embeddings=4096,
)
torch.manual_seed(0)
model = Qwen2ForCausalLM(cfg)
model.save_pretrained(tiny)
tok.save_pretrained(tiny)
print(f"tiny model saved -> {tiny} (vocab {cfg.vocab_size})")
PY
fi

echo "[phase=stage]"
uv run python - <<'PY'
import os
import shutil
import signal
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # run-2 lesson: transfer hang
signal.alarm(2700)  # 45-min hard cap on the stage phase (v6 lesson)
from huggingface_hub import hf_hub_download

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before staging)"
stage = Path(os.environ["EPS_STAGE_DIR"])
conv = Path(os.environ["EPS_CONV"])
for name in ("conversations.jsonl", "conversations_meta.json"):
    p = hf_hub_download(
        repo_id=os.environ["EPS_DATA_REPO"],
        repo_type="dataset",
        revision=os.environ["EPS_HF_REV"],
        filename=f"issue825_userbase_map/raw_completions/generation/{name}",
    )
    dst = stage / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dst)
n = sum(1 for line in open(conv) if line.strip())
assert n == 2000, f"staged conversations.jsonl has {n} rows != 2000"
print(f"stage: {conv} ({n} rows) @ rev {os.environ['EPS_HF_REV']}")
PY

echo "[phase=gen]"
for m in $MODELS; do
  if [[ -n "$SMOKE" ]]; then
    uv run python scripts/issue825_onpolicy_u2_gen.py --conversations "$CONV" \
      --out-dir "$ONP_DIR" --models "$m" --smoke --tiny-model-dir "$TINY"
  else
    uv run python scripts/issue825_onpolicy_u2_gen.py --conversations "$CONV" \
      --out-dir "$ONP_DIR" --models "$m"
  fi
done

echo "[phase=upload1]"
if [[ -n "$SMOKE" ]]; then
  uv run python - <<'PY'
import os
from pathlib import Path

onp = Path(os.environ["EPS_ONP_DIR"])
files = sorted(p.name for p in onp.glob("conversations_*")) + sorted(
    p.name for p in onp.glob("row_allowlists.json")
)
assert len([f for f in files if f.endswith(".jsonl")]) == 4, files
assert "row_allowlists.json" in files, files
print(f"[smoke] upload1 structural assert PASS ({len(files)} files would upload): {files}")
PY
else
  uv run python - <<'PY'
import os
import signal

signal.alarm(2700)
from huggingface_hub import HfApi

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
HfApi().upload_folder(
    folder_path=os.environ["EPS_ONP_DIR"],
    repo_id=os.environ["EPS_DATA_REPO"],
    repo_type="dataset",
    path_in_repo=f"{os.environ['EPS_HF_ONP_PREFIX']}/raw_completions/generation",
    allow_patterns=["conversations_*", "row_allowlists.json"],
    commit_message="issue-825 onpolicy-user-turn: UPLOAD-1 (self-generated u2 texts + audit meta)",
)
print("upload1: ok (texts + meta + allowlists, BEFORE any binding gate — MF-C)")
PY
fi

echo "[phase=extract]"
for m in $MODELS; do
  for f in chat naturalistic; do
    if [[ -n "$SMOKE" ]]; then
      uv run python scripts/issue825_extract_turnstore.py --model "$m" --format "$f" \
        --track m --conversations "$ONP_DIR/conversations_${m}_${f}.jsonl" \
        --peak-layers 14,18,19,26 --out-dir "$TS_DIR" --smoke --tiny-model-dir "$TINY"
    else
      uv run python scripts/issue825_extract_turnstore.py --model "$m" --format "$f" \
        --track m --conversations "$ONP_DIR/conversations_${m}_${f}.jsonl" \
        --peak-layers 14,18,19,26 --out-dir "$TS_DIR"
    fi
  done
done

echo "[phase=wiring]"
for m in $MODELS; do
  if [[ -n "$SMOKE" ]]; then
    uv run python scripts/issue825_onpolicy_u2_gen.py --wiring-check --models "$m" \
      --out-dir "$ONP_DIR" --wiring-rows "$WROWS" --tiny-model-dir "$TINY"
  else
    uv run python scripts/issue825_onpolicy_u2_gen.py --wiring-check --models "$m" \
      --out-dir "$ONP_DIR" --wiring-rows "$WROWS"
  fi
done

echo "[phase=fit]"
# --no-internal-gates (MF-C): fit RECORDS gate values (g3_gate.json) + defers
# per-cell crashes to fit_failures.json; every binding gate is evaluated ONLY
# in this wrapper's post-UPLOAD-2 gate phase (upload-then-exit on failure).
uv run python scripts/issue825_fit_cells.py --turnstore-dir "$TS_DIR" --out-dir "$OUT_DIR" \
  --cells "$CELLS8" --mlp-cells "$USER4" \
  --cell-row-allowlist "$ONP_DIR/row_allowlists.json" \
  --null-draws "$NULLS" --folds "$FOLDS" --n-boot "$NBOOT" --seed 0 \
  --no-internal-gates

echo "[phase=matched_parent]"
# Hard-req 8 (conditional): any headline cell with n_cell < 1900 -> refit the
# matched PARENT Haiku cell on the SAME conv_id subset (parent m-track shards
# staged from HF). Smoke: trigger logic runs; staging is substituted with the
# just-built onpolicy turnstore (17 GB download is production-only).
AFFECTED="$(uv run python - <<'PY'
import json
import os
from pathlib import Path

onp = Path(os.environ["EPS_ONP_DIR"])
floor = int(os.environ["EPS_MATCHED_SUBSET_FLOOR"])
allow = json.loads((onp / "row_allowlists.json").read_text())
affected = sorted(c for c, ids in allow.items() if len(ids) < floor)
if os.environ.get("EPS_SMOKE") == "1":
    # numeric floor is meaningless at smoke n; exercise the branch on ONE cell
    affected = affected[:1]
(onp / "matched_parent_allowlists.json").write_text(
    json.dumps({c: allow[c] for c in affected}, indent=2)
)
print(",".join(affected))
PY
)"
if [[ -n "$AFFECTED" ]]; then
  echo "matched_parent: refitting parent cells on matched conv_id subsets: $AFFECTED"
  if [[ -n "$SMOKE" ]]; then
    PARENT_TS="$TS_DIR"  # smoke substitution: self-turnstore stands in for parent shards
    echo "[smoke] matched_parent staging substituted (parent shards = onpolicy turnstore)"
  else
    PARENT_TS="data/issue_825/turnstore"
    EPS_MATCHED_CELLS="$AFFECTED" uv run python - <<'PY'
import os
import signal
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
signal.alarm(5400)  # 90 min: <=17 GB of parent shards (v6 staged ~64 GB in ~45)
from huggingface_hub import snapshot_download

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing"
stems = set()
for cell in os.environ["EPS_MATCHED_CELLS"].split(","):
    _, model, _, fmt = cell.split("_", 3)
    stems.add(f"{model}_{fmt}_m_")
patterns = [
    f"issue825_userbase_map/analysis_tensors/{s}*{ext}" for s in stems for ext in (".pt", ".json")
]
stage = Path(os.environ["EPS_STAGE_DIR"]) / "parent_shards"
snapshot_download(
    repo_id=os.environ["EPS_DATA_REPO"],
    repo_type="dataset",
    revision=os.environ["EPS_HF_REV"],
    allow_patterns=patterns,
    local_dir=str(stage),
)
src = stage / "issue825_userbase_map" / "analysis_tensors"
ts = Path("data/issue_825/turnstore")
ts.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in src.glob("*") if p.is_file() and any(p.name.startswith(s) for s in stems))
if not files:
    raise RuntimeError(f"matched_parent stage: 0 files under {src}")
for p in files:
    dst = ts / p.name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(p.resolve())
print(f"matched_parent stage: linked {len(files)} parent m-track files -> {ts}")
PY
  fi
  uv run python scripts/issue825_fit_cells.py --turnstore-dir "$PARENT_TS" \
    --out-dir "$OUT_DIR/matched_parent" --cells "$AFFECTED" --mlp-cells "$AFFECTED" \
    --cell-row-allowlist "$ONP_DIR/matched_parent_allowlists.json" \
    --null-draws "$NULLS" --folds "$FOLDS" --n-boot "$NBOOT" --seed 0 \
    --no-internal-gates
else
  echo "matched_parent: no cell below floor $MATCHED_SUBSET_FLOOR; branch not triggered"
fi

echo "[phase=summarize]"
uv run python scripts/issue825_onpolicy_summarize.py --out-dir "$OUT_DIR" \
  --onpolicy-dir "$ONP_DIR" --parent-cells-dir "$PARENT_EVAL" --parent-mlp-dir "$PARENT_MLP"

echo "[phase=upload2]"
if [[ -n "$SMOKE" ]]; then
  uv run python - <<'PY'
import os
from pathlib import Path

ts = Path(os.environ["EPS_TS_DIR"])
out = Path(os.environ["EPS_OUT_DIR"])
onp = Path(os.environ["EPS_ONP_DIR"])
shards = sorted(ts.glob("*_m_shard*.pt"))
evals = sorted(out.glob("*.json"))
wirings = sorted(onp.glob("wiring_check_*.json"))
assert len(shards) >= 4, f"expected >=4 turnstore shards, found {len(shards)}"
assert any(p.name == "headline_metrics.json" for p in evals), sorted(p.name for p in evals)
assert len(wirings) == 2, wirings
print(
    f"[smoke] upload2 structural assert PASS: {len(shards)} shards, "
    f"{len(evals)} eval JSONs, {len(wirings)} wiring JSONs would upload"
)
PY
else
  uv run python - <<'PY'
import os
import signal

signal.alarm(10800)  # 3 h: ~52 GB turnstore at the plain-CDN path (plan §9 <=~1 h + margin)
from huggingface_hub import HfApi

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
api = HfApi()
prefix = os.environ["EPS_HF_ONP_PREFIX"]
repo = os.environ["EPS_DATA_REPO"]
api.upload_folder(
    folder_path=os.environ["EPS_TS_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/analysis_tensors",
    commit_message="issue-825 onpolicy-user-turn: UPLOAD-2a (turnstore shards, BEFORE gates)",
)
api.upload_folder(
    folder_path=os.environ["EPS_OUT_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/eval_results_mirror",
    commit_message="issue-825 onpolicy-user-turn: UPLOAD-2b (eval JSONs + headline_metrics)",
)
api.upload_folder(
    folder_path=os.environ["EPS_ONP_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/raw_completions/generation",
    allow_patterns=["wiring_check_*.json", "row_allowlists.json", "matched_parent_allowlists.json"],
    commit_message="issue-825 onpolicy-user-turn: UPLOAD-2c (wiring-check JSONs)",
)
print("upload2: ok (turnstore + eval JSONs + wiring, ALL BEFORE binding gates — MF-C)")
PY
fi

echo "[phase=gate]"
uv run python - <<'PY'
import json
import os
import time
from pathlib import Path

onp = Path(os.environ["EPS_ONP_DIR"])
ts = Path(os.environ["EPS_TS_DIR"])
out = Path(os.environ["EPS_OUT_DIR"])
smoke = os.environ.get("EPS_SMOKE") == "1"
sentinel = Path(os.environ["EPS_SENTINEL"])
cells8 = os.environ["EPS_CELLS8"].split(",")
user4 = os.environ["EPS_USER4"].split(",")
anchors = [c for c in cells8 if "_assistant_" in c]
parent_eval = Path(os.environ["EPS_PARENT_EVAL"])
TOL = 0.05
HAIKU_BAND_CEIL = 2.64
outcomes: dict = {"smoke": smoke, "gates": {}}


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
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": {
                    "followup_label": "onpolicy-user-turn",
                    "failure": msg,
                    "gate_outcomes": outcomes,
                    "uploads_completed_before_gates": not smoke,
                },
            },
            indent=2,
        )
    )
    raise SystemExit(f"GATE FAIL [{status}]: {msg}")


def _l19(payload: dict):
    table = (payload.get("selection_symmetric") or {}).get("frozen_layer_table") or {}
    entry = table.get("19")
    return float(entry["r2_obs"]) if entry else None


# Deferred fit failures (--no-internal-gates, MF-C): any per-cell/secondary
# crash EITHER fit invocation deferred is a binding post-upload HALT here
# (structural, binds in smoke too). The wrapper runs fit_cells with
# --no-internal-gates against TWO out-dirs — $OUT_DIR ([phase=fit]) and
# $OUT_DIR/matched_parent ([phase=matched_parent]) — so sweep EVERY
# fit_failures.json under $OUT_DIR (rglob: a future third nested invocation
# cannot silently escape; round-2 Codex Major). Every one of them already
# rode UPLOAD-2b (upload_folder on $OUT_DIR is recursive), and the full
# tracebacks are in the fit / matched_parent phase logs.
print("gate armed: deferred-fit-failures")
deferred = {
    str(p.relative_to(out)): json.loads(p.read_text())
    for p in sorted(out.rglob("fit_failures.json"))
}
if deferred:
    outcomes["deferred_fit_failures"] = deferred
    for rel, entries in deferred.items():
        print(f"deferred fit failures ({rel}): {entries}")
    n_fail = sum(len(v) for v in deferred.values())
    fail(
        "fit_deferred_failure",
        f"{n_fail} fit-phase failure(s) across {sorted(deferred)} were deferred past "
        "UPLOAD-2 (see the [phase=fit] / [phase=matched_parent] log tracebacks)",
    )
outcomes["gates"]["deferred_fit_failures"] = "PASS"
print("gate: deferred-fit-failures PASS (none recorded under any fit out-dir)")

# ── gate: anchor conv_id row/fold parity (MF-A) ────────────────────────────
print("gate armed: anchor-rowset-parity")
staged_ids = []
with open(os.environ["EPS_CONV"], encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            staged_ids.append(str(json.loads(line)["conv_id"]))
staged_set = set(staged_ids)
for m in ("instruct", "pretrained"):
    for f in ("chat", "naturalistic"):
        cell_rows = [
            str(json.loads(line)["conv_id"])
            for line in open(onp / f"conversations_{m}_{f}.jsonl", encoding="utf-8")
            if line.strip()
        ]
        cell_set = set(cell_rows)
        if not smoke and cell_set != staged_set:
            fail(
                "anchor_rowset_mismatch",
                f"{m}/{f}: per-cell conversations ids != staged kept-2000 ids "
                f"({len(cell_set)} vs {len(staged_set)})",
            )
        side_ids: set[str] = set()
        for sc in sorted(ts.glob(f"{m}_{f}_m_shard*.json")):
            side_ids.update(str(c) for c in json.loads(sc.read_text())["conv_ids"])
        if side_ids != cell_set:
            fail(
                "anchor_rowset_mismatch",
                f"{m}/{f}: turnstore conv_ids != per-cell conversation ids "
                f"({len(side_ids)} vs {len(cell_set)})",
            )
        anchor_cell = f"M_{m}_assistant_{f}"
        payload_path = out / f"cells_{anchor_cell}.json"
        if payload_path.exists():
            n_fit = json.loads(payload_path.read_text())["metadata"]["n"]
            if n_fit != len(cell_set):
                fail(
                    "anchor_rowset_mismatch",
                    f"{anchor_cell}: fit n={n_fit} != {len(cell_set)} bundle rows "
                    "(anchor must fit the FULL row set, no allowlist)",
                )
outcomes["gates"]["anchor_rowset_parity"] = "PASS"
print("gate: anchor-rowset-parity PASS")

# ── gate: anchor ridge R2@L19 within ±0.05 of parent committed (MF-A) ─────
if smoke:
    print("gate armed: anchor-ridge-tolerance (BYPASSED under EPS_SMOKE)")
    outcomes["gates"]["anchor_ridge_tolerance"] = "BYPASSED_SMOKE"
else:
    print("gate armed: anchor-ridge-tolerance")
    deltas = {}
    for cid in anchors:
        fresh_p = out / f"cells_{cid}.json"
        parent_p = parent_eval / f"cells_{cid}.json"
        if not fresh_p.exists():
            fail("coverage_miss", f"anchor cell missing from ridge results: {fresh_p}")
        fresh = _l19(json.loads(fresh_p.read_text()))
        parent = _l19(json.loads(parent_p.read_text()))
        if fresh is None or parent is None:
            fail("anchor_gate_miss", f"{cid}: missing L19 row (fresh={fresh}, parent={parent})")
        deltas[cid] = {"fresh": fresh, "parent": parent, "delta": fresh - parent}
        if abs(fresh - parent) > TOL:
            fail(
                "anchor_gate_miss",
                f"{cid}: fresh {fresh:+.4f} vs parent {parent:+.4f} "
                f"(|delta|={abs(fresh - parent):.4f} > {TOL}) — rig drift, HALT",
            )
    outcomes["gates"]["anchor_ridge_tolerance"] = {"result": "PASS", "deltas": deltas}
    print("gate: anchor-ridge-tolerance PASS " + json.dumps({k: round(v["delta"], 4) for k, v in deltas.items()}))

# ── gate: G3 sanity (computed in-fit, evaluated HERE post-upload — MF-C) ───
# fit_cells ran with --no-internal-gates, so its in-process SystemExit(3) is
# disabled; g3_gate.json carries the recorded verdict and THIS is the single
# binding evaluation point (upload-then-exit on failure).
g3p = out / "g3_gate.json"
if smoke:
    print("gate armed: g3-sanity (BYPASSED under EPS_SMOKE; presence still binds)")
    if not g3p.exists():
        fail("g3_gate_miss", f"g3_gate.json missing at {g3p} (fit never recorded G3)")
    outcomes["gates"]["g3_sanity"] = "BYPASSED_SMOKE_PRESENCE_ONLY"
else:
    print("gate armed: g3-sanity")
    if not g3p.exists():
        fail("g3_gate_miss", f"g3_gate.json missing at {g3p} (fit never recorded G3)")
    g3 = json.loads(g3p.read_text())
    if not g3.get("pass"):
        fail(
            "g3_gate_miss",
            "G3 sanity gate FAILED (M_instruct_assistant_chat does not beat its "
            f"selection-inherited null): obs_layer_max_r2={g3.get('obs_layer_max_r2')} "
            f"vs null layer-max per draw {g3.get('null_layer_max_r2_per_draw')}",
        )
    outcomes["gates"]["g3_sanity"] = {
        "result": "PASS",
        "obs_layer_max_r2": g3.get("obs_layer_max_r2"),
    }
print("gate: g3-sanity " + str(outcomes["gates"]["g3_sanity"]))

# ── gate: wiring check (own-vs-shuffled NLL, MF-B) ─────────────────────────
if smoke:
    print("gate armed: wiring-check (numeric margin BYPASSED under EPS_SMOKE; presence still binds)")
else:
    print("gate armed: wiring-check")
wiring_all = {}
for m in ("instruct", "pretrained"):
    wp = onp / f"wiring_check_{m}.json"
    if not wp.exists():
        fail("wiring_check_fail", f"missing wiring-check output {wp}")
    wiring_all[m] = json.loads(wp.read_text())
if not smoke:
    # Evaluate ALL cells first, print every mean, THEN halt once — hard-req 2's
    # "diagnosable in one pass" read (review-r1 Minor: no first-cell short-circuit).
    bad_cells = []
    for m, w in wiring_all.items():
        for fmt, blk in (w.get("per_format") or {}).items():
            own, shuf = blk.get("own_mean_nll"), blk.get("shuffled_mean_nll")
            print(f"wiring-check {m}/{fmt}: own={own} shuffled={shuf} n={blk.get('n')}")
            bad = (
                own is None
                or shuf is None
                or own != own
                or shuf != shuf
                or own >= shuf
                or own > 2 * HAIKU_BAND_CEIL
            )
            if bad:
                # Diagnosability-first: per-cell NLL values + 20 audit samples
                # to the run log BEFORE the halt (plan hard-req 2).
                print(f"wiring-check FAIL detail {m}/{fmt}: own_values={blk.get('own_nll_values')}")
                print(f"wiring-check FAIL detail {m}/{fmt}: shuf_values={blk.get('shuffled_nll_values')}")
                for s in blk.get("samples", []):
                    print(f"wiring-check sample {m}/{fmt} conv={s['conv_id']}: {s['u2_excerpt']}")
                bad_cells.append(f"{m}/{fmt}: own={own} shuffled={shuf}")
    if bad_cells:
        fail(
            "wiring_check_fail",
            "own>=shuffled, missing reads, or blow-up "
            f"(> {2 * HAIKU_BAND_CEIL}) in: " + "; ".join(bad_cells),
        )
outcomes["gates"]["wiring_check"] = "PASS" if not smoke else "BYPASSED_SMOKE_PRESENCE_ONLY"
print("gate: wiring-check " + str(outcomes["gates"]["wiring_check"]))

# ── gate: coverage (structural — binds in smoke too) ───────────────────────
print("gate armed: coverage")
allow = json.loads((onp / "row_allowlists.json").read_text())
if sorted(allow) != sorted(user4):
    fail("coverage_miss", f"row_allowlists keys {sorted(allow)} != user cells {sorted(user4)}")
for cid in cells8:
    p = out / f"cells_{cid}.json"
    if not p.exists():
        fail("coverage_miss", f"missing ridge results {p} (explicit --cells disables the FATAL branch)")
for cid in user4:
    payload = json.loads((out / f"cells_{cid}.json").read_text())
    if not payload.get("mlp"):
        fail("coverage_miss", f"{cid}: no non-empty 'mlp' block on disk")
if not (out / "headline_metrics.json").exists():
    fail("coverage_miss", "headline_metrics.json missing")
outcomes["gates"]["coverage"] = "PASS"
print("gate: coverage PASS (8 ridge cells + 4 MLP blocks + headline + allowlists)")

# ── degeneracy audit (headline-INCLUSION flags, non-halting — hard-req 3) ──
print("gate armed: degeneracy-audit (headline-inclusion, non-halting)")
headline = json.loads((out / "headline_metrics.json").read_text())
audit_flags = {
    cid: (headline["cells"].get(cid, {}).get("audit") or {}).get("headline_eligible")
    for cid in user4
}
outcomes["gates"]["degeneracy_audit"] = audit_flags
print(f"gate: degeneracy-audit verdicts (observational-if-false): {audit_flags}")

(out / "gate_outcomes.json").write_text(json.dumps(outcomes, indent=2))
print("gate: ALL PASS" + (" [smoke: numeric gates bypassed]" if smoke else ""))
PY

echo "[phase=sentinel]"
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
gates = json.loads((out / "gate_outcomes.json").read_text())
sent = Path(os.environ["EPS_SENTINEL"])
sent.parent.mkdir(parents=True, exist_ok=True)
repo = os.environ["EPS_DATA_REPO"]
prefix = os.environ["EPS_HF_ONP_PREFIX"]
sent.write_text(
    json.dumps(
        {
            "sentinel_schema_version": 1,
            "kind": "epm:results",
            "version": 1,
            "task_id": 825,
            "status": "success",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": {
                "followup_label": "onpolicy-user-turn",
                "eval_numbers": headline,
                "gate_outcomes": gates,
                "eval_paths": sorted(str(p) for p in out.rglob("*.json")),
                "reproducibility_card": {
                    "models": ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
                    "generation_seed": 42,
                    "fit_seed": 0,
                    "kept2000_revision": os.environ["EPS_HF_REV"],
                    "followup_label": "onpolicy-user-turn",
                },
                "wandb_url": "n/a (analysis-only follow-up; no training)",
                "hf_hub_url": f"https://huggingface.co/datasets/{repo}/tree/main/{prefix}",
                "worktree_path": os.environ["EPS_WORKTREE"],
                "final_commit_sha": os.environ["EPS_GIT_SHA"],
                "gpu_hours_used": round((time.time() - float(os.environ["EPS_T0"])) / 3600.0, 3),
                "gpu_hours_used_basis": "measured wrapper wall-clock (single-GPU provision)",
                "gpu_hours_budgeted": 3.0,
                "plan_deviations": [],
            },
        },
        indent=2,
    )
)
print(f"sentinel: wrote {sent}")
PY
echo "[phase=done]"
