#!/usr/bin/env bash
# issue-825 naturalistic-single-turn pod-side phase driver:
#   stage -> extract -> fit -> contrast -> upload -> [phase=done].
# Refits BOTH models on the SAME 5,000 LMSYS single-turn conversations re-rendered
# as the naturalistic User:/Assistant: transcript; the single manipulated variable
# is the Track-S render format (vs the committed chat-template S1/S2 anchors).
#
# Production (GCP lane): REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue825_naturalistic_s_dispatch.sh
# Smoke (CPU, tiny):     bash scripts/issue825_naturalistic_s_dispatch.sh --smoke \
#                          [--tiny-model-dir DIR] [--turnstore-dir DIR] [--out-dir DIR]
#
# Pod-side reporting contract (.claude/rules/pod-side-reporting.md): posts NO markers
# via task.py; progress via [phase=...] log lines terminating in [phase=done]; writes
# the results sentinel to /workspace/logs/issue-825-results.json.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT" || {
  echo "FATAL: cd $REPO_ROOT failed" >&2
  exit 1
}

# Pinned data-repo revision (track_s.jsonl + the committed chat_s anchor shards).
# Resolved 2026-07-14 via HfApi().repo_info(...).sha; track_s.jsonl = 9,036,307 bytes,
# sha256 head d20560b679345a6e, 5,000 rows.
DATA_REPO_REV="74afc5a3018fd2328dd453433f92c978eb844973"
HF_DATA_REPO="superkaiba1/explore-persona-space-data"
HF_PREFIX="issue825_userbase_map"
HF_ANALYSIS_PREFIX="$HF_PREFIX/analysis_tensors"

PHASE="all"
FROM_PHASE=""
SMOKE=""
TINY=""
DATA_DIR="data/issue_825"
TS_DIR="$DATA_DIR/turnstore"
EVAL_DIR="eval_results/issue_825/naturalistic-single-turn"
ANCHOR_DIR="eval_results/issue_825" # committed cells_S1.json / cells_S2.json
LOG_DIR="/workspace/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --from-phase) FROM_PHASE="$2"; shift 2 ;;
    --smoke) SMOKE="--smoke"; shift ;;
    --tiny-model-dir) TINY="$2"; shift 2 ;;
    --turnstore-dir) TS_DIR="$2"; shift 2 ;;
    --out-dir) EVAL_DIR="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 1 ;;
  esac
done

TRACK_S_JSONL="$DATA_DIR/track_s.jsonl"

# Pin glibc's mmap threshold so the extractor's ~7 MB per-conv CPU tensors are
# mmap-allocated and their pages RETURN to the OS on free (parent lesson: extract
# RSS climbed monotonically without this, kernel OOM at 14.9 GiB anon RSS).
export MALLOC_MMAP_THRESHOLD_=131072

# Conditional .env sourcing: pods carry a scp-pushed .env; the GCE lane exports the
# API tokens via instance metadata and has NO .env file — unconditional sourcing
# inside a classified &&-chain would kill a healthy GCE run (gotchas: incident #923).
if [ -f ./.env ]; then
  set -a
  source ./.env
  set +a
fi

PHASES=(stage extract fit contrast upload)
mkdir -p "$DATA_DIR" "$TS_DIR" "$EVAL_DIR" "$LOG_DIR"

should_run() {
  local phase="$1"
  if [[ "$PHASE" != "all" && "$PHASE" != "$phase" ]]; then return 1; fi
  if [[ -n "$FROM_PHASE" ]]; then
    local started=""
    for p in "${PHASES[@]}"; do
      [[ "$p" == "$FROM_PHASE" ]] && started="yes"
      if [[ "$p" == "$phase" ]]; then [[ -n "$started" ]] && return 0 || return 1; fi
    done
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Smoke: fabricate the tiny 28-layer Qwen2 (real tokenizer) substituting the 7B
# loads — verbatim recipe from scripts/issue825_realuser_dispatch.sh.
# ---------------------------------------------------------------------------
if [[ -n "$SMOKE" ]]; then
  TINY="${TINY:-$DATA_DIR/tiny-qwen}"
  if [[ ! -d "$TINY" ]]; then
    echo "[smoke] fabricating tiny 28-layer Qwen2 (real tokenizer) -> $TINY"
    EPS_TINY_DIR="$TINY" uv run python - <<'PY'
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
fi

# ---------------------------------------------------------------------------
# stage: pinned track_s.jsonl + the committed chat_s anchor shards (production).
# The smoke fetches only track_s.jsonl (its anchors come from tiny chat_s bundles
# extracted below, NOT the ~42 GB real chat_s shards).
# ---------------------------------------------------------------------------
if should_run stage; then
  echo "[phase=stage]"
  DATA_REPO="$HF_DATA_REPO" DATA_REPO_REV="$DATA_REPO_REV" ANALYSIS_PREFIX="$HF_ANALYSIS_PREFIX" \
    TS_DIR="$TS_DIR" TRACK_S_JSONL="$TRACK_S_JSONL" DATA_DIR="$DATA_DIR" SMOKE="$SMOKE" \
    uv run python - <<'PY'
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo = os.environ["DATA_REPO"]
rev = os.environ["DATA_REPO_REV"]
analysis_prefix = os.environ["ANALYSIS_PREFIX"]
ts_dir = Path(os.environ["TS_DIR"])
track_s = Path(os.environ["TRACK_S_JSONL"])
smoke = bool(os.environ.get("SMOKE"))
ts_dir.mkdir(parents=True, exist_ok=True)
track_s.parent.mkdir(parents=True, exist_ok=True)


def _fetch(path_in_repo: str, dest: Path) -> None:
    """hf_hub_download one file (pinned rev) with bounded transient retry."""
    last = None
    for attempt in range(4):
        try:
            got = hf_hub_download(
                repo, path_in_repo, repo_type="dataset", revision=rev,
                local_dir=str(dest.parent),
            )
            # hf_hub_download preserves the repo path layout under local_dir; move the
            # leaf into dest when the caller wants a flat name.
            got = Path(got)
            if got != dest and got.name == dest.name:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if got.resolve() != dest.resolve():
                    os.replace(got, dest)
            return
        except Exception as e:  # noqa: BLE001 — bounded retry, re-raise on exhaustion
            last = e
            print(f"[stage] retry {attempt + 1}/4 {path_in_repo}: {type(e).__name__}: {e}")
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"[stage] FAILED to fetch {path_in_repo} after 4 attempts") from last


# 1) track_s.jsonl (pinned) -> a flat local path the extractor reads.
_fetch(f"{analysis_prefix.rsplit('/', 1)[0]}/raw_completions/track_s/track_s.jsonl", track_s)
n_rows = sum(1 for line in track_s.open(encoding="utf-8") if line.strip())
print(f"[stage] track_s.jsonl staged: {n_rows} rows, {track_s.stat().st_size} bytes @ rev {rev[:12]}")

# 2) chat_s anchor shards for both models (production only; the smoke extracts tiny
#    chat_s bundles instead). Server-scoped list_repo_tree (never snapshot_download
#    on the ~1M-file data repo), per-file hf_hub_download.
if not smoke:
    api = HfApi()
    tree = list(
        api.list_repo_tree(
            repo, path_in_repo=analysis_prefix, repo_type="dataset",
            recursive=True, revision=rev,
        )
    )
    wanted = [
        e.path
        for e in tree
        if ("_chat_s_shard" in e.path)
        and (e.path.split("/")[-1].startswith(("instruct_chat_s_shard", "pretrained_chat_s_shard")))
    ]
    assert wanted, f"[stage] no chat_s anchor shards under {analysis_prefix} @ {rev[:12]}"
    for p in sorted(wanted):
        leaf = p.split("/")[-1]
        _fetch(p, ts_dir / leaf)
    n_pt = len(list(ts_dir.glob("*_chat_s_shard*.pt")))
    print(f"[stage] chat_s anchor shards staged: {len(wanted)} files ({n_pt} .pt) -> {ts_dir}")
else:
    print("[stage] smoke: skipping real chat_s shard download (tiny chat_s bundles extracted below)")
PY
fi

# ---------------------------------------------------------------------------
# extract: naturalistic Track-S turnstores for BOTH models (+ tiny chat_s in smoke).
# Parallel per-GPU when >=2 GPUs are visible; CVD pinned in the LAUNCHER env per
# model (the extract loads device_map={"": 0} inside the single visible device).
# ---------------------------------------------------------------------------
run_extract() {
  local m="$1" f="$2" cvd="$3"
  local log="$LOG_DIR/issue-825-nat-extract-${m}-${f}.log"
  local extra=()
  if [[ -n "$SMOKE" ]]; then extra=(--smoke --tiny-model-dir "$TINY"); fi
  if [[ -n "$cvd" ]]; then
    env CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/issue825_extract_turnstore.py \
      --model "$m" --format "$f" --conversations "$TRACK_S_JSONL" --track s \
      --out-dir "$TS_DIR" --peak-layers 14,18,19,26 "${extra[@]}" >"$log" 2>&1
  else
    uv run python scripts/issue825_extract_turnstore.py \
      --model "$m" --format "$f" --conversations "$TRACK_S_JSONL" --track s \
      --out-dir "$TS_DIR" --peak-layers 14,18,19,26 "${extra[@]}" >"$log" 2>&1
  fi
}

if should_run extract; then
  echo "[phase=extract]"
  # GPU-free guard: reap any lingering VLLM::EngineCore (holds ~22 GB VRAM ->
  # device_map offload -> kernel OOM). Skipped when nvidia-smi is absent (CPU smoke).
  NGPU=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    for pid in $(pgrep -f '^VLLM::EngineCore' || true); do
      echo "[gpu-guard] killing lingering VLLM::EngineCore pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
    done
    for i in $(seq 1 24); do
      apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
      if [[ -z "$apps" ]]; then break; fi
      echo "[gpu-guard] waiting for GPU to free ($i/24): $apps"
      sleep 5
    done
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$apps" ]]; then
      echo "FATAL: GPU still held before extract: $apps" >&2
      exit 1
    fi
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
  echo "[extract] visible GPUs: $NGPU"

  if [[ "$NGPU" -ge 2 ]]; then
    echo "[extract] running instruct(GPU0) + pretrained(GPU1) naturalistic concurrently"
    run_extract instruct naturalistic 0 &
    PID_I=$!
    run_extract pretrained naturalistic 1 &
    PID_P=$!
    RC_I=0; RC_P=0
    wait "$PID_I" || RC_I=$?
    wait "$PID_P" || RC_P=$?
    if [[ "$RC_I" -ne 0 ]]; then
      echo "FATAL: extract instruct/naturalistic rc=$RC_I" >&2
      tail -40 "$LOG_DIR/issue-825-nat-extract-instruct-naturalistic.log" >&2 || true
      exit 1
    fi
    if [[ "$RC_P" -ne 0 ]]; then
      echo "FATAL: extract pretrained/naturalistic rc=$RC_P" >&2
      tail -40 "$LOG_DIR/issue-825-nat-extract-pretrained-naturalistic.log" >&2 || true
      exit 1
    fi
  else
    for m in instruct pretrained; do
      run_extract "$m" naturalistic "" || {
        echo "FATAL: extract $m/naturalistic failed" >&2
        tail -40 "$LOG_DIR/issue-825-nat-extract-${m}-naturalistic.log" >&2 || true
        exit 1
      }
    done
  fi
  # Smoke: also extract tiny chat_s anchor bundles (production reuses the real
  # chat_s shards staged above, so no chat extraction there).
  if [[ -n "$SMOKE" ]]; then
    for m in instruct pretrained; do
      run_extract "$m" chat "" || {
        echo "FATAL: smoke extract $m/chat failed" >&2
        tail -40 "$LOG_DIR/issue-825-nat-extract-${m}-chat.log" >&2 || true
        exit 1
      }
    done
  fi
  echo "[extract] shards present in $TS_DIR:"
  ls "$TS_DIR" | grep -E '_(chat|naturalistic)_s_shard.*\.pt' | sed 's/^/[extract]   /' || true
fi

# ---------------------------------------------------------------------------
# fit: S1/S2 (chat anchors) + S1N/S2N (naturalistic) from the SAME turnstore dir.
# ---------------------------------------------------------------------------
if should_run fit; then
  echo "[phase=fit]"
  if [[ -n "$SMOKE" ]]; then
    # EPS_SMOKE=1 keeps REAL tiny bundles + fits REAL cells (unlike --smoke, which
    # fabricates a synthetic turnstore); folds=3/n=8 ridge arithmetic; --no-internal-gates
    # defers tiny-n gate crashes; MLP secondary skipped (PCA-64 needs n>64).
    EPS_SMOKE=1 uv run python scripts/issue825_fit_cells.py \
      --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR" \
      --cells S1,S2,S1N,S2N --mlp-cells "" \
      --folds 3 --null-draws 3 --n-boot 20 --no-internal-gates
  else
    uv run python scripts/issue825_fit_cells.py \
      --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR" \
      --cells S1,S2,S1N,S2N --mlp-cells S1N,S2N
  fi

  # Anchor gate (production only): the refit S1/S2 layer-19 held-out R^2 MUST
  # reproduce the committed anchors within 0.01, else a wiring bug (wrong shards /
  # format drift) is corrupting the fit — HALT before any decision read.
  if [[ -z "$SMOKE" ]]; then
    EVAL_DIR="$EVAL_DIR" ANCHOR_DIR="$ANCHOR_DIR" uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

eval_dir = Path(os.environ["EVAL_DIR"])
anchor_dir = Path(os.environ["ANCHOR_DIR"])
TOL = 0.01
fail = False
for cid, ref_r2 in (("S1", None), ("S2", None)):
    refit = json.loads((eval_dir / f"cells_{cid}.json").read_text())["r2_per_layer_obs"][19]
    committed = json.loads((anchor_dir / f"cells_{cid}.json").read_text())["r2_per_layer_obs"][19]
    delta = abs(refit - committed)
    status = "OK" if delta <= TOL else "FAIL"
    print(f"[anchor-gate] {cid} L19: refit={refit:.4f} committed={committed:.4f} |delta|={delta:.4f} {status}")
    if delta > TOL:
        fail = True
if fail:
    print("[anchor-gate] FAIL — refit anchors diverge from committed >0.01; wiring bug, HALT", file=sys.stderr)
    sys.exit(1)
print("[anchor-gate] PASS")
PY
  else
    echo "[anchor-gate] smoke: bypassed (tiny-n R^2 is meaningless)"
  fi
fi

# ---------------------------------------------------------------------------
# contrast: paired naturalistic-chat delta + strength ratios.
# ---------------------------------------------------------------------------
if should_run contrast; then
  echo "[phase=contrast]"
  if [[ -n "$SMOKE" ]]; then
    # Smoke runs the batched-vs-serial paired-bootstrap equivalence gate as evidence
    # (trivial at n=8); production skips the ~10-min serial oracle at n=5,000 (the
    # batched helpers are already equivalence-gated in issue825_role_contrast r5).
    uv run python scripts/issue825_naturalistic_s_contrast.py \
      --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR" --smoke --folds 3 --equivalence-gate
  else
    uv run python scripts/issue825_naturalistic_s_contrast.py \
      --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR"
  fi
fi

# ---------------------------------------------------------------------------
# upload: new naturalistic_s turnstore shards (bulk upload_folder commit) + the
# fit/contrast eval JSON mirror. Then the results sentinel + [phase=done].
# ---------------------------------------------------------------------------
if should_run upload; then
  echo "[phase=upload]"
  if [[ -z "$SMOKE" ]]; then
    DATA_REPO="$HF_DATA_REPO" ANALYSIS_PREFIX="$HF_ANALYSIS_PREFIX" HF_PREFIX="$HF_PREFIX" \
      TS_DIR="$TS_DIR" EVAL_DIR="$EVAL_DIR" uv run python - <<'PY'
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import upload_folder  # noqa: E402

repo = os.environ["DATA_REPO"]
analysis_prefix = os.environ["ANALYSIS_PREFIX"]
hf_prefix = os.environ["HF_PREFIX"]
ts_dir = os.environ["TS_DIR"]
eval_dir = os.environ["EVAL_DIR"]

# ONE bulk commit for the new naturalistic Track-S shards (allow_patterns keeps the
# staged chat_s / M-track shards out of this commit); never a per-file loop.
upload_folder(
    repo_id=repo, repo_type="dataset", folder_path=ts_dir,
    path_in_repo=analysis_prefix,
    allow_patterns=["*_naturalistic_s_shard*.pt", "*_naturalistic_s_shard*.json"],
    commit_message="issue-825 naturalistic-single-turn: naturalistic Track-S turnstore shards",
)
# Eval JSON mirror (git-side commit of eval_results/ is the orchestrator's Step 8).
upload_folder(
    repo_id=repo, repo_type="dataset", folder_path=eval_dir,
    path_in_repo=f"{hf_prefix}/eval_results_mirror_naturalistic_single_turn",
    commit_message="issue-825 naturalistic-single-turn: fit + format-contrast eval JSONs",
)
n_pt = len(list(Path(ts_dir).glob("*_naturalistic_s_shard*.pt")))
print(f"uploads complete: {n_pt} naturalistic_s .pt shards + eval JSON mirror")
PY
  else
    echo "[upload] smoke mode: skipping HF uploads"
  fi

  GPU_HOURS_USED="${GPU_HOURS_USED:-0.0}"
  COMMIT_SHA=$(git rev-parse HEAD)
  DATA_REPO_REV="$DATA_REPO_REV" HF_ANALYSIS_PREFIX="$HF_ANALYSIS_PREFIX" HF_PREFIX="$HF_PREFIX" \
    EVAL_DIR="$EVAL_DIR" ANCHOR_DIR="$ANCHOR_DIR" LOG_DIR="$LOG_DIR" SMOKE="$SMOKE" \
    uv run python - "$COMMIT_SHA" "$GPU_HOURS_USED" <<'PY'
import json
import os
import sys
from pathlib import Path

commit_sha, gpu_hours = sys.argv[1], float(sys.argv[2])
eval_dir = Path(os.environ["EVAL_DIR"])
anchor_dir = Path(os.environ["ANCHOR_DIR"])
log_dir = Path(os.environ["LOG_DIR"])
data_rev = os.environ["DATA_REPO_REV"]
analysis_prefix = os.environ["HF_ANALYSIS_PREFIX"]
hf_prefix = os.environ["HF_PREFIX"]
smoke = bool(os.environ.get("SMOKE"))


def _l19(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())["r2_per_layer_obs"][19]


def _anchor_delta(cid: str):
    refit = _l19(eval_dir / f"cells_{cid}.json")
    committed = _l19(anchor_dir / f"cells_{cid}.json")
    if refit is None or committed is None:
        return {"refit_l19": refit, "committed_l19": committed, "abs_delta_l19": None}
    return {"refit_l19": refit, "committed_l19": committed, "abs_delta_l19": abs(refit - committed)}


contrast_path = eval_dir / "format_contrast.json"
contrast = json.loads(contrast_path.read_text()) if contrast_path.exists() else {}
paired = {}
for model, block in (contrast.get("per_model") or {}).items():
    d19 = (block.get("paired_delta_frozen_layers") or {}).get("19", {})
    paired[model] = {
        "r2_l19_naturalistic": d19.get("r2_obs_naturalistic"),
        "r2_l19_chat": d19.get("r2_obs_chat"),
        "delta_l19_obs": d19.get("delta_obs"),
        "delta_l19_ci": [d19.get("ci_lo"), d19.get("ci_hi")],
    }

eval_paths = sorted(str(p) for p in eval_dir.glob("*.json"))
sentinel = {
    "eval_numbers": {
        "s1n_l19_r2": _l19(eval_dir / "cells_S1N.json"),
        "s2n_l19_r2": _l19(eval_dir / "cells_S2N.json"),
        "anchor_delta_S1": _anchor_delta("S1"),
        "anchor_delta_S2": _anchor_delta("S2"),
        "paired_naturalistic_minus_chat_l19": paired,
        "strength_ratio_pretrained_over_instruct": contrast.get(
            "strength_ratio_pretrained_over_instruct"
        ),
    },
    "eval_paths": eval_paths,
    "reproducibility_card": {
        "followup_label": "naturalistic-single-turn",
        "models": {
            "instruct": "Qwen/Qwen2.5-7B-Instruct",
            "pretrained": "Qwen/Qwen2.5-7B",
        },
        "seeds": {"fit": 0, "generation": 42},
        "sampling": {
            "track_s": (
                "reused #825 track_s.jsonl (SamplingParams(n=1, temperature=1.0, "
                "top_p=0.95, max_tokens=1024, seed=42)); no new generation this round"
            ),
        },
        "render_formats": {
            "chat": "Qwen chat template (anchors S1/S2)",
            "naturalistic": "User:/Assistant: plain-text transcript (S1N/S2N)",
        },
        "input_data": {
            "track_s_jsonl": f"{hf_prefix}/raw_completions/track_s/track_s.jsonl",
            "track_s_rows": 5000,
            "track_s_sha256_head": "d20560b679345a6e",
            "chat_s_anchor_shards": f"{analysis_prefix}/{{instruct,pretrained}}_chat_s_shard*.pt",
            "data_repo_revision": data_rev,
        },
        "new_turnstore_shards": (
            f"{analysis_prefix}/{{instruct,pretrained}}_naturalistic_s_shard*.pt"
        ),
        "eval_mirror": f"{hf_prefix}/eval_results_mirror_naturalistic_single_turn",
        "hf_data_prefix": f"{hf_prefix}/",
        "hf_data_repo": "superkaiba1/explore-persona-space-data",
        "wandb_project": "n/a — no training metrics (analysis-only re-fit task)",
        "n_folds": 5,
        "n_boot": 1000,
        "frozen_layers": [14, 18, 19, 26],
        "headline_layer": 19,
    },
    "wandb_url": "n/a",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": commit_sha,
    "gpu_hours_used": gpu_hours,
    "gpu_hours_budgeted": 4.0,
    "smoke": smoke,
    "plan_deviations": [],
}
out = log_dir / "issue-825-results.json"
out.write_text(json.dumps(sentinel, indent=2))
print(f"sentinel written: {out}")
PY
fi

echo "[phase=done]"
