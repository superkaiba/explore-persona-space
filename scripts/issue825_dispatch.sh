#!/usr/bin/env bash
# issue-825 pod-side phase driver:
#   gen_s -> gen_m -> render -> extract -> upload_ts -> fit -> upload.
# Runs under the GCP lane contract: REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue825_dispatch.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT" || { echo "FATAL: cd $REPO_ROOT failed" >&2; exit 1; }

PHASE="all"
FROM_PHASE=""
SMOKE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --from-phase) FROM_PHASE="$2"; shift 2 ;;
    --smoke) SMOKE="--smoke"; shift ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 1 ;;
  esac
done

# Pin glibc's mmap threshold so the extractor's ~7 MB per-conv CPU tensors are
# mmap-allocated and their pages RETURN to the OS on free. Without this, the
# dynamic threshold migrates them into arena free lists and extract RSS climbs
# monotonically across flushed blocks (run 5 kernel OOM at 14.9 GiB anon RSS).
export MALLOC_MMAP_THRESHOLD_=131072

PHASES=(gen_s gen_m render extract upload_ts fit upload)
DATA_DIR="data/issue_825"
TS_DIR="$DATA_DIR/turnstore"
EVAL_DIR="eval_results/issue_825"
LOG_DIR="/workspace/logs"
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

if should_run gen_s; then
  echo "[phase=gen_s]"
  uv run python scripts/issue825_gen_conversations.py --track s \
    --out "$DATA_DIR/track_s.jsonl" $SMOKE
fi

if should_run gen_m; then
  echo "[phase=gen_m]"
  uv run python scripts/issue825_gen_conversations.py --track m --n 2000 --seed 42 \
    --out "$DATA_DIR/conversations.jsonl" $SMOKE
fi

if should_run render; then
  echo "[phase=render]"
  uv run python scripts/issue825_render_formats.py \
    --conversations "$DATA_DIR/conversations.jsonl" \
    --out-manifest "$DATA_DIR/render_manifest.jsonl" \
    --assert-report "$DATA_DIR/render_asserts.json" $SMOKE
fi

if should_run extract; then
  echo "[phase=extract]"
  # GPU-free guard: a lingering VLLM::EngineCore from the gen phases holds
  # ~22 GB VRAM, which forced device_map to offload the 7B to the 16 GB host
  # -> kernel OOM (run 3, rc=137). Reap by exact PID, then require an empty
  # compute-app list. Skipped when nvidia-smi is absent (CPU smoke on the VM).
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
  fi
  for m in instruct pretrained; do
    for f in chat naturalistic; do
      uv run python scripts/issue825_extract_turnstore.py --model "$m" --format "$f" \
        --conversations "$DATA_DIR/conversations.jsonl" --track m \
        --out-dir "$TS_DIR" --peak-layers 14,18,19,26 $SMOKE
    done
    uv run python scripts/issue825_extract_turnstore.py --model "$m" --format chat \
      --conversations "$DATA_DIR/track_s.jsonl" --track s \
      --out-dir "$TS_DIR" --peak-layers 14,18,19,26 $SMOKE
  done
fi

if should_run upload_ts; then
  echo "[phase=upload_ts]"
  # Persist-by-default (#1320): the GPU extraction uploads BEFORE any fit, so a
  # fit crash can never lose the turnstore again (2026-07-15 incident; the
  # onpolicy UPLOAD-2a MF-C shape).
  # Recovery after any pre-fit crash: resume with --from-phase upload_ts (NOT
  # --from-phase fit — that skips the turnstore persist).
  if [[ -z "$SMOKE" ]]; then
    uv run python - <<'PY'
import os
import signal

signal.alarm(10800)  # 3 h wall cap (onpolicy UPLOAD-2a precedent: ~52 GB well under cap)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (GCE metadata env or .env)"
from huggingface_hub import upload_folder  # noqa: E402

upload_folder(
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    folder_path="data/issue_825/turnstore",
    path_in_repo="issue825_userbase_map/analysis_tensors",
    commit_message="issue-825: turnstore shards + NLL tables (upload_ts, BEFORE fit)",
)
print("upload_ts: ok — turnstore persisted before any fit")
PY
  else
    uv run python - <<'PY'
from pathlib import Path

ts = Path("data/issue_825/turnstore")
# Fact-check correction (#1320 plan Phase 1.5): the turnstore holds fp16/bf16
# sharded .pt files + .json sidecars (issue825_extract_turnstore.py
# write_shards; the onpolicy sibling asserts *_m_shard*.pt) — NOT .npz (.npz
# exists only in _fabricate_smoke_turnstore's fit-cells-internal fixture).
shards = sorted(ts.glob("*.pt"))
metas = sorted(ts.glob("*.json"))
assert shards, f"upload_ts smoke: no .pt shards under {ts}"
assert metas, f"upload_ts smoke: no .json sidecars under {ts}"
print(f"[smoke] upload_ts structural assert PASS ({len(shards)} pt + {len(metas)} json would upload)")
PY
  fi
fi

if should_run fit; then
  echo "[phase=fit]"
  uv run python scripts/issue825_fit_cells.py --turnstore-dir "$TS_DIR" \
    --out-dir "$EVAL_DIR" $SMOKE
fi

if should_run upload; then
  echo "[phase=upload]"
  if [[ -z "$SMOKE" ]]; then
    # turnstore upload moved to [phase=upload_ts] (#1320) — BEFORE fit, so a
    # fit crash can never lose the extraction.
    uv run python - <<'PY'
import os
import signal

signal.alarm(3600)  # 1 h wall cap — the eval mirror is small JSON (MF-C hardening)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (GCE metadata env or .env)"
from huggingface_hub import upload_folder  # noqa: E402

upload_folder(
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    folder_path="eval_results/issue_825",
    path_in_repo="issue825_userbase_map/eval_results_mirror",
    commit_message="issue-825: eval JSON mirror",
)
print("uploads complete")
PY
  else
    echo "[upload] smoke mode: skipping HF uploads"
  fi

  GPU_HOURS_USED="${GPU_HOURS_USED:-0.0}"
  COMMIT_SHA=$(git rev-parse HEAD)
  uv run python - "$COMMIT_SHA" "$GPU_HOURS_USED" <<'PY'
import json
import sys
from pathlib import Path

commit_sha, gpu_hours = sys.argv[1], float(sys.argv[2])
eval_dir = Path("eval_results/issue_825")
eval_paths = sorted(str(p) for p in eval_dir.glob("*.json"))
g1 = {}
g1_path = eval_dir / "g1_gate.json"
if g1_path.exists():
    g1 = json.loads(g1_path.read_text())
gen_meta = {}
m_meta_path = Path("data/issue_825/conversations_meta.json")
if m_meta_path.exists():
    gen_meta["track_m"] = json.loads(m_meta_path.read_text())
meta_path = Path("data/issue_825/track_s_meta.json")
if meta_path.exists():
    gen_meta["track_s"] = json.loads(meta_path.read_text())
sentinel = {
    "eval_numbers": {
        "g1_spearman_vs_779": g1.get("spearman_vs_779"),
        "g1_abs_dev_L19": g1.get("abs_dev_L19_vs_0677"),
        "g1_pass": g1.get("pass"),
    },
    "eval_paths": eval_paths,
    "reproducibility_card": {
        "models": {
            "instruct": "Qwen/Qwen2.5-7B-Instruct",
            "pretrained": "Qwen/Qwen2.5-7B",
        },
        "seeds": {"fit": 0, "generation": 42},
        "sampling": {
            "track_s": "SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=42)",
            "track_m_assistant": "greedy (temperature=0.0, seed=42)",
            "track_m_user_turn": "claude-haiku-4-5-20251001, temperature=1.0",
        },
        "data_versions": gen_meta,
        "hf_data_prefix": "issue825_userbase_map/",
        "wandb_project": "n/a — no training metrics (analysis-only task)",
    },
    "wandb_url": "n/a",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": commit_sha,
    "gpu_hours_used": gpu_hours,
    "gpu_hours_budgeted": 8.0,
    "plan_deviations": [
        {
            "deviation": "u2 generation via synchronous pooled messages.create instead of the Batch API",
            "rationale": "n=2000 short calls; the hardened batch client's result parser is judge-specific",
        },
        {
            "deviation": "turn-store tensors stored bf16 instead of fp16",
            "rationale": "fp16 max 65504 < Qwen residual outlier dims; bf16 is range-safe at the same size",
        },
    ],
}
out = Path("/workspace/logs/issue-825-results.json")
out.write_text(json.dumps(sentinel, indent=2))
print(f"sentinel written: {out}")
PY
fi

echo "[phase=done]"
