#!/usr/bin/env bash
# task #825 follow-up "real-user-turn-null" (plan v11):
#   stage -> ingest -> UPLOAD-1 -> extract x4 (real) + x1 (anchor) -> wiring
#   -> fit x2 -> summarize -> UPLOAD-2 (BEFORE gates, MF-C) -> gates -> sentinel
#
# Production (worker):  bash scripts/issue825_realuser_dispatch.sh
# Smoke (CPU, tiny):    bash scripts/issue825_realuser_dispatch.sh --smoke
#   Smoke IS this wrapper (EPS_SMOKE=1): same phases, same commands, tiny row
#   count, tiny random-init Qwen2 (real tokenizer) substituting the 7B loads,
#   all outputs under a /tmp scratch root, numeric gates BYPASSED (plan MF-D),
#   structural assertions still binding. Uploads run a structural glob assert
#   instead of writing smoke garbage to the HF repo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMOKE=""
# Either trigger works: the --smoke arg (v7 convention) or EPS_SMOKE=1 in the
# environment (the plan §4.3 invocation) — an env-only trigger must never
# half-run production paths with numeric gates bypassed.
if [[ "${1:-}" == "--smoke" || "${EPS_SMOKE:-}" == "1" ]]; then SMOKE="1"; fi

HF_REV="deb7a4523b5233393e4fbd2497622527b3622d35"          # parent conversations.jsonl pin
LMSYS_REV="200748d9d3cddcc9d782887541057aca0b18c5da"       # lmsys-chat-1m pin (parent's)
DATA_REPO="superkaiba1/explore-persona-space-data"
HF_RU_PREFIX="issue825_real_user_turn_null"
CELLS8="M_instruct_assistant_chat,M_instruct_assistant_naturalistic,M_pretrained_assistant_chat,M_pretrained_assistant_naturalistic,M_instruct_user_chat,M_instruct_user_naturalistic,M_pretrained_user_chat,M_pretrained_user_naturalistic"
ANCHOR_CELL="M_instruct_assistant_chat"
MODELS="instruct pretrained"
PARENT_EVAL="eval_results/issue_825"
PARENT_MLP="eval_results/issue_825/mlp-unprobed-cells"
V7_HEADLINE="eval_results/issue_825/onpolicy-user-turn/headline_metrics.json"
N_TARGET=2000

if [[ -n "$SMOKE" ]]; then
  ROOT="${EPS_SMOKE_ROOT:-/tmp/issue-825-realuser-smoke}"
  RU_DIR="$ROOT/realuser"
  TS_RU="$ROOT/turnstore_realuser"
  TS_ANCHOR="$ROOT/turnstore_anchor_parent"
  OUT_DIR="$ROOT/eval_results/real-user-turn-null"
  STAGE_DIR="$ROOT/stage"
  WIRING_DIR="$ROOT/wiring"
  SENTINEL_DIR="$ROOT/logs"
  CONV="$STAGE_DIR/conversations.jsonl"
  FOLDS=3 NULLS=3 NBOOT=20 WROWS=8
  TINY="${EPS_TINY_MODEL_DIR:-$ROOT/tiny-qwen}"
  export EPS_SMOKE=1
  # Effective smoke N is pinned to 8: the ridge needs a fold with >=2 test AND
  # >=3 train rows (all-NaN r2 -> nanargmax ValueError below 8-with-FOLDS=3),
  # and extract --smoke hard-caps at its first 8 conversations — so any other
  # EPS_SMOKE_N breaks either the fit or the ingest<->turnstore row parity.
  if [[ "${EPS_SMOKE_N:-8}" != "8" ]]; then
    echo "[smoke] EPS_SMOKE_N=${EPS_SMOKE_N:-} pinned to 8 (ridge fold arithmetic + extract --smoke cap)"
  fi
  export EPS_SMOKE_N=8
  N_TARGET="$EPS_SMOKE_N"
  # Bounded MLP secondary at smoke scale (production keeps the 1800 s default).
  export EPS_MLP_TIME_BUDGET_S="${EPS_MLP_TIME_BUDGET_S:-300}"
else
  RU_DIR="data/issue_825/realuser"
  TS_RU="data/issue_825/turnstore_realuser"
  TS_ANCHOR="data/issue_825/turnstore_anchor_parent"
  OUT_DIR="eval_results/issue_825/real-user-turn-null"
  STAGE_DIR="data/issue_825/hf_dl/realuser_stage"
  WIRING_DIR="data/issue_825/realuser_wiring"
  SENTINEL_DIR="/workspace/logs"
  CONV="$STAGE_DIR/conversations.jsonl"
  FOLDS=5 NULLS=20 NBOOT=1000 WROWS=200
  TINY=""
  export EPS_SMOKE=""
fi
SENTINEL="$SENTINEL_DIR/issue-825-epm_results-$(date +%s).json"
EPS_T0="$(date +%s)"  # sentinel reports MEASURED gpu-hours (v7 convention)
export EPS_T0

# Credentials at script level (never bare load_dotenv() inside a stdin heredoc — gotcha).
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mkdir -p "$SENTINEL_DIR" "$RU_DIR" "$TS_RU" "$TS_ANCHOR" "$OUT_DIR" "$STAGE_DIR" "$WIRING_DIR"
# Xet downloader finalization hangs on this repo (#515 class; v7 r2/r3 lesson).
export HF_XET_DISABLE=1
export EPS_RU_DIR="$RU_DIR" EPS_TS_RU="$TS_RU" EPS_TS_ANCHOR="$TS_ANCHOR" \
  EPS_OUT_DIR="$OUT_DIR" EPS_STAGE_DIR="$STAGE_DIR" EPS_WIRING_DIR="$WIRING_DIR" \
  EPS_CONV="$CONV" EPS_SENTINEL="$SENTINEL" EPS_HF_REV="$HF_REV" \
  EPS_LMSYS_REV="$LMSYS_REV" EPS_DATA_REPO="$DATA_REPO" \
  EPS_HF_RU_PREFIX="$HF_RU_PREFIX" EPS_CELLS8="$CELLS8" EPS_N_TARGET="$N_TARGET" \
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # v7 run-2 lesson: transfer hang
signal.alarm(2700)  # 45-min hard cap per stage (plan §10)
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

echo "[phase=ingest]"
# CPU phase; the pinned lmsys stream. Failure handling is upload-then-exit
# (plan §4.3 hard-req 3): fail-from-ingest FIRST uploads whatever the ingest
# produced (a shortfall writes dataset + meta + ingest_failure.json before
# returning 1; text/JSON uploads unconditional — a true crash with no outputs
# has nothing to upload), THEN routes on the artifact (ingest_failure.json ->
# status ingest_shortfall, fall-through ingest_error) and writes the FAILURE
# sentinel — then exit 1. Under EPS_SMOKE=1 the upload is a structural
# listing assert (no real upload), same as UPLOAD-1/UPLOAD-2.
if ! uv run python scripts/issue825_realuser_ingest.py \
  --out-dir "$RU_DIR" --n "$N_TARGET" --revision "$LMSYS_REV" \
  --parent-conversations "$CONV"; then
  uv run python scripts/issue825_realuser_gates.py fail-from-ingest \
    --out-dir "$OUT_DIR" --realuser-dir "$RU_DIR" --sentinel "$SENTINEL"
  echo "[phase=ingest] FAILED — produced artifacts uploaded, FAILURE sentinel written (upload-then-exit)"
  exit 1
fi

echo "[phase=upload1]"
if [[ -n "$SMOKE" ]]; then
  uv run python - <<'PY'
import json
import os
from pathlib import Path

ru = Path(os.environ["EPS_RU_DIR"])
conv = ru / "conversations_real2turn.jsonl"
meta = ru / "conversations_real2turn_meta.json"
assert conv.exists() and meta.exists(), (conv, meta)
n = sum(1 for line in open(conv) if line.strip())
n_target = int(os.environ["EPS_N_TARGET"])
assert n == n_target, f"ingested {n} rows != smoke target {n_target}"
m = json.loads(meta.read_text())
assert m["n_kept"] == n, (m["n_kept"], n)
print(f"[smoke] upload1 structural assert PASS ({n} rows + meta would upload)")
PY
else
  uv run python - <<'PY'
import os
import signal

signal.alarm(2700)
from huggingface_hub import HfApi

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
HfApi().upload_folder(
    folder_path=os.environ["EPS_RU_DIR"],
    repo_id=os.environ["EPS_DATA_REPO"],
    repo_type="dataset",
    path_in_repo=f"{os.environ['EPS_HF_RU_PREFIX']}/raw_completions/ingestion",
    allow_patterns=["conversations_real2turn*"],
    commit_message="issue-825 real-user-turn-null: UPLOAD-1 (ingested real 2-turn dataset + meta)",
)
print("upload1: ok (dataset + meta, BEFORE any binding gate — MF-C)")
PY
fi

echo "[phase=extract]"
for m in $MODELS; do
  for f in chat naturalistic; do
    if [[ -n "$SMOKE" ]]; then
      uv run python scripts/issue825_extract_turnstore.py --model "$m" --format "$f" \
        --track m --conversations "$RU_DIR/conversations_real2turn.jsonl" \
        --peak-layers 14,18,19,26 --out-dir "$TS_RU" --smoke --tiny-model-dir "$TINY"
    else
      uv run python scripts/issue825_extract_turnstore.py --model "$m" --format "$f" \
        --track m --conversations "$RU_DIR/conversations_real2turn.jsonl" \
        --peak-layers 14,18,19,26 --out-dir "$TS_RU"
    fi
  done
done

echo "[phase=extract_anchor]"
# End-to-end parent anchor (plan §2 rig gate): re-extract the parent kept-2000
# conversations, instruct/chat only, into a SEPARATE turnstore dir (the stem
# instruct_chat_m_* would clobber the real-set shards in $TS_RU).
if [[ -n "$SMOKE" ]]; then
  uv run python scripts/issue825_extract_turnstore.py --model instruct --format chat \
    --track m --conversations "$CONV" \
    --peak-layers 14,18,19,26 --out-dir "$TS_ANCHOR" --smoke --tiny-model-dir "$TINY"
else
  uv run python scripts/issue825_extract_turnstore.py --model instruct --format chat \
    --track m --conversations "$CONV" \
    --peak-layers 14,18,19,26 --out-dir "$TS_ANCHOR"
fi

echo "[phase=wiring]"
# Stage the reused v7 wiring-check inputs: the helper reads per-cell
# conversations_{model}_{format}.jsonl + row_allowlists.json from its
# --out-dir. All 4 user cells share the SAME real conversations (u2 is fixed
# logged text), and all row filters ran at dataset build — so the copies are
# identical and the allowlists are all-conv_ids.
uv run python - <<'PY'
import json
import os
import shutil
from pathlib import Path

ru = Path(os.environ["EPS_RU_DIR"])
wd = Path(os.environ["EPS_WIRING_DIR"])
wd.mkdir(parents=True, exist_ok=True)
src = ru / "conversations_real2turn.jsonl"
conv_ids = [
    json.loads(line)["conv_id"]
    # split("\n"), NOT splitlines(): raw U+2028/NEL inside real-user JSON strings
    # (ensure_ascii=False) are Unicode line boundaries that shred records (run-1d crash)
    for line in src.read_text(encoding="utf-8").split("\n")
    if line.strip()
]
allow = {}
for m in ("instruct", "pretrained"):
    for f in ("chat", "naturalistic"):
        shutil.copyfile(src, wd / f"conversations_{m}_{f}.jsonl")
        allow[f"M_{m}_user_{f}"] = conv_ids
(wd / "row_allowlists.json").write_text(json.dumps(allow))
print(f"[wiring] staged 4 per-cell copies + all-rows allowlists ({len(conv_ids)} conv_ids) -> {wd}")
PY
for m in $MODELS; do
  if [[ -n "$SMOKE" ]]; then
    uv run python scripts/issue825_onpolicy_u2_gen.py --wiring-check --models "$m" \
      --out-dir "$WIRING_DIR" --wiring-rows "$WROWS" \
      --followup-label real-user-turn-null --tiny-model-dir "$TINY"
  else
    uv run python scripts/issue825_onpolicy_u2_gen.py --wiring-check --models "$m" \
      --out-dir "$WIRING_DIR" --wiring-rows "$WROWS" \
      --followup-label real-user-turn-null
  fi
done

echo "[phase=fit]"
# --no-internal-gates (MF-C): fit RECORDS gate values + defers per-cell crashes
# to fit_failures.json; every binding gate is evaluated ONLY in this wrapper's
# post-UPLOAD-2 gate phase (upload-then-exit on failure). All 8 cells share the
# identical kept-2000 row set (filters ran at dataset build) — no allowlist.
# MLP on ALL 8 cells (plan §4.2: assistant MLP is load-bearing — logged a1 is
# off-policy text and the assistant ridge may collapse).
uv run python scripts/issue825_fit_cells.py --turnstore-dir "$TS_RU" --out-dir "$OUT_DIR" \
  --cells "$CELLS8" --mlp-cells "$CELLS8" \
  --null-draws "$NULLS" --folds "$FOLDS" --n-boot "$NBOOT" --seed 0 \
  --no-internal-gates

echo "[phase=fit_anchor]"
# The anchor_parent SUBDIR is REQUIRED: the anchor's cells_M_instruct_assistant_chat.json
# would otherwise clobber the real-set assistant cell in $OUT_DIR (plan §4.2).
# --mlp-cells none: the anchor is a ridge-only rig gate (plan §5).
uv run python scripts/issue825_fit_cells.py --turnstore-dir "$TS_ANCHOR" \
  --out-dir "$OUT_DIR/anchor_parent" --cells "$ANCHOR_CELL" --mlp-cells none \
  --null-draws "$NULLS" --folds "$FOLDS" --n-boot "$NBOOT" --seed 0 \
  --no-internal-gates

echo "[phase=summarize]"
uv run python scripts/issue825_realuser_summarize.py --out-dir "$OUT_DIR" \
  --realuser-dir "$RU_DIR" --wiring-dir "$WIRING_DIR" \
  --parent-cells-dir "$PARENT_EVAL" --parent-mlp-dir "$PARENT_MLP" \
  --v7-headline "$V7_HEADLINE"

echo "[phase=upload2]"
if [[ -n "$SMOKE" ]]; then
  uv run python - <<'PY'
import os
from pathlib import Path

ts_ru = Path(os.environ["EPS_TS_RU"])
ts_anchor = Path(os.environ["EPS_TS_ANCHOR"])
out = Path(os.environ["EPS_OUT_DIR"])
wd = Path(os.environ["EPS_WIRING_DIR"])
ru_shards = sorted(ts_ru.glob("*_m_shard*.pt"))
anchor_shards = sorted(ts_anchor.glob("instruct_chat_m_shard*.pt"))
evals = sorted(out.rglob("*.json"))
wirings = sorted(wd.glob("wiring_check_*.json"))
assert len(ru_shards) >= 4, f"expected >=4 realuser turnstore shards, found {len(ru_shards)}"
assert len(anchor_shards) >= 1, f"expected >=1 anchor shard, found {len(anchor_shards)}"
assert any(p.name == "headline_metrics.json" for p in evals), sorted(p.name for p in evals)[:20]
assert any("anchor_parent" in str(p) for p in evals), "anchor_parent eval JSONs missing"
assert len(wirings) == 2, wirings
print(
    f"[smoke] upload2 structural assert PASS: {len(ru_shards)}+{len(anchor_shards)} shards, "
    f"{len(evals)} eval JSONs, {len(wirings)} wiring JSONs would upload"
)
PY
else
  uv run python - <<'PY'
import os
import signal

signal.alarm(10800)  # 3 h: ~65 GB turnstore at the plain-CDN path (plan §9 + margin)
from huggingface_hub import HfApi

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (source .env before upload)"
api = HfApi()
prefix = os.environ["EPS_HF_RU_PREFIX"]
repo = os.environ["EPS_DATA_REPO"]
# Separate subdirs: the anchor stem instruct_chat_m_shard* collides with the
# real-set stem — same reason the local dirs are separate.
api.upload_folder(
    folder_path=os.environ["EPS_TS_RU"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/analysis_tensors/turnstore_realuser",
    commit_message="issue-825 real-user-turn-null: UPLOAD-2a (realuser turnstore, BEFORE gates)",
)
api.upload_folder(
    folder_path=os.environ["EPS_TS_ANCHOR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/analysis_tensors/turnstore_anchor_parent",
    commit_message="issue-825 real-user-turn-null: UPLOAD-2b (anchor turnstore, BEFORE gates)",
)
api.upload_folder(
    folder_path=os.environ["EPS_OUT_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/eval_results_mirror",
    commit_message="issue-825 real-user-turn-null: UPLOAD-2c (eval JSONs + headline)",
)
api.upload_folder(
    folder_path=os.environ["EPS_WIRING_DIR"],
    repo_id=repo,
    repo_type="dataset",
    path_in_repo=f"{prefix}/raw_completions/wiring",
    allow_patterns=["wiring_check_*.json", "row_allowlists.json"],
    commit_message="issue-825 real-user-turn-null: UPLOAD-2d (wiring-check JSONs)",
)
print("upload2: ok (turnstores + eval JSONs + wiring, ALL BEFORE binding gates — MF-C)")
PY
fi

echo "[phase=gate]"
# Binding gates (plan §7, evaluated POST-upload): deferred fit failures ->
# ingest >= 2000 -> anchor ±0.05 vs the committed +0.0757 -> wiring
# own<shuffled -> coverage 8+8+1. Numeric gates bypassed under EPS_SMOKE=1;
# structural asserts binding. FAILURE sentinel + exit 1 on any miss.
uv run python scripts/issue825_realuser_gates.py gates \
  --out-dir "$OUT_DIR" --realuser-dir "$RU_DIR" --wiring-dir "$WIRING_DIR" \
  --parent-cells-dir "$PARENT_EVAL" --sentinel "$SENTINEL" --n-target 2000

echo "[phase=sentinel]"
EPS_GIT_SHA="$(git rev-parse HEAD)"
EPS_WORKTREE="$(pwd)"
export EPS_GIT_SHA EPS_WORKTREE
uv run python scripts/issue825_realuser_gates.py success-sentinel \
  --out-dir "$OUT_DIR" --realuser-dir "$RU_DIR" --sentinel "$SENTINEL"
echo "[phase=done]"
