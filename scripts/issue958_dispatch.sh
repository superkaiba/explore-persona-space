#!/usr/bin/env bash
# Issue #958 dispatcher — corpus → rollouts → capture → upload (GPU stage);
# fits → evals → plots → upload (CPU stage); sentinel; [phase=done].
#
# ONE code path for smoke and production (plan §4.7 PASS_UNIFIED): `--smoke`
# scales the CORPUS (200 main / 24 long conversations) through the SAME python
# entrypoints; every later phase enumerates its units from the artifacts the
# previous phase wrote (corpus manifest → rollout shards → store shards → eval
# JSONs), never from a registered full grid. No forks.
#
# Pod-side contract (poll_pipeline.py): [phase=<name>] breadcrumbs per phase;
# the results sentinel is written BEFORE the single terminal [phase=done].
# vLLM (rollouts) and HF capture run as SEPARATE processes (plan §8 teardown).
#
# Stages: --stage gpu (corpus+rollouts+capture+upload, the capture-7b A100
# provision) | --stage cpu (fits+evals+plots+upload, cpu-mid) | --stage all.
#
# Env overrides (all optional):
#   EPM958_CORPUS/ROLLOUTS/STORE/CACHE/MAPS/OUT/FIGS   — dirs
#   EPM958_MODEL                                        — model override
#   EPM958_MOCK_ROLLOUTS=1                              — VM smoke: no vLLM
#   EPM958_STUB_MODEL=1                                 — VM smoke: tiny Qwen2
#   EPM958_DEVICE                                       — fit device override
#   EPM958_SKIP_UPLOAD=1                                — skip HF uploads
#   EPM958_HF_PREFIX                                    — smoke: issue958_multiturn/smoke
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

SMOKE=0
STAGE="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --stage) STAGE="$2"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# GCE lane has NO .env (startup script exports tokens) — conditional sourcing.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CORPUS="${EPM958_CORPUS:-data/issue_958/corpus}"
ROLLOUTS="${EPM958_ROLLOUTS:-data/issue_958/rollouts}"
STORE="${EPM958_STORE:-data/issue_958/store}"
CACHE="${EPM958_CACHE:-data/issue_958/fit_cache}"
MAPS="${EPM958_MAPS:-data/issue_958/maps}"
OUT="${EPM958_OUT:-eval_results/issue_958}"
FIGS="${EPM958_FIGS:-figures/issue_958}"
MODEL="${EPM958_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# ONE smoke subset definition (the corpus), threaded to EVERY phase below.
CORPUS_ARGS=()
ROLLOUT_ARGS=()
CAPTURE_ARGS=()
FIT_ARGS=()
if [ "$SMOKE" = "1" ]; then
  # pod smoke = the plan's 200-conversation end-to-end run; the VM structural
  # smoke may shrink further via EPM958_SMOKE_N_MAIN/N_LONG (same code path)
  CORPUS_ARGS=(--n-main "${EPM958_SMOKE_N_MAIN:-200}" --n-long "${EPM958_SMOKE_N_LONG:-24}" \
    --stream-limit 120000)
  export EPM958_HF_PREFIX="${EPM958_HF_PREFIX:-issue958_multiturn/smoke}"
fi
if [ "${EPM958_MOCK_ROLLOUTS:-0}" = "1" ]; then ROLLOUT_ARGS+=(--mock-generate); fi
if [ "${EPM958_STUB_MODEL:-0}" = "1" ]; then
  CAPTURE_ARGS+=(--stub-model --batch 4)
  FIT_ARGS+=(--stub-rb)
fi
if [ -n "${EPM958_DEVICE:-}" ]; then FIT_ARGS+=(--device "$EPM958_DEVICE"); fi

upload_dir() {  # upload_dir <local_dir> <path_in_repo_suffix> <msg>
  if [ "${EPM958_SKIP_UPLOAD:-0}" = "1" ]; then
    echo "[upload] skipped ($2)"
    return 0
  fi
  EPM958_UPLOAD_DIR="$1" EPM958_UPLOAD_SUFFIX="$2" EPM958_UPLOAD_MSG="$3" \
    uv run python - <<'PY'
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
import issue958_common as C

local = Path(os.environ["EPM958_UPLOAD_DIR"])
suffix = os.environ["EPM958_UPLOAD_SUFFIX"]
msg = os.environ["EPM958_UPLOAD_MSG"]
files = sorted(p for p in local.rglob("*") if p.is_file())
assert files, f"nothing to upload under {local}"
# in-run one-item serialization+upload timing gate (plan §7 storage-overrun
# kill): time the LARGEST file (a production store shard ~0.5 GB) — a tiny
# file's wall is per-commit-overhead-dominated (#813) and would false-kill,
# so the kill arms only when the probe is big enough to measure throughput.
from huggingface_hub import HfApi

api = HfApi()
probe = max(files, key=lambda p: p.stat().st_size)
t0 = time.time()
api.upload_file(
    path_or_fileobj=str(probe),
    path_in_repo=f"{C.HF_OUT_PREFIX}/{suffix}/{probe.relative_to(local)}",
    repo_id=C.HF_DATA_REPO,
    repo_type="dataset",
    commit_message=f"{msg} (timing probe)",
)
probe_wall = time.time() - t0
probe_bytes = probe.stat().st_size
total = sum(p.stat().st_size for p in files)
if probe_bytes >= 20 * (1 << 20):  # throughput measurable, not overhead-dominated
    projected_h = (probe_wall / probe_bytes) * total / 3600
    print(
        f"[upload-gate] probe {probe_bytes / 1e6:.0f} MB in {probe_wall:.1f}s -> "
        f"projected {projected_h:.2f}h for {total / 1e9:.1f} GB"
    )
    if projected_h > 4 * 0.5:
        raise RuntimeError(
            f"STORAGE-OVERRUN KILL (plan §7): projected upload {projected_h:.1f}h > "
            f"4x0.5h budget ({total / 1e9:.1f} GB). Artifacts kept local pod-side."
        )
else:
    print(f"[upload-gate] probe {probe_bytes / 1e3:.0f} KB (overhead-dominated) — gate N/A")
ev = C.upload_dir_bulk(
    local, f"{C.HF_OUT_PREFIX}/{suffix}", commit_message=msg
)
print(f"[upload] {ev}")
PY
}

if [ "$STAGE" = "gpu" ] || [ "$STAGE" = "all" ]; then
  echo "[phase=verify_fits]"
  uv run python scripts/issue958_fit_maps.py --verify-fits

  echo "[phase=corpus]"
  # self-build on a fresh instance: gitignored data never travels with the clone
  if [ ! -f "$CORPUS/manifest.json" ]; then
    uv run python scripts/issue958_build_corpus.py --out "$CORPUS" "${CORPUS_ARGS[@]}"
  else
    echo "[corpus] exists — skip (resume)"
  fi

  echo "[phase=rollouts]"
  uv run python scripts/issue958_rollouts.py --corpus "$CORPUS" --out "$ROLLOUTS" \
    --model "$MODEL" "${ROLLOUT_ARGS[@]}"

  echo "[phase=capture]"
  uv run python scripts/issue958_capture_turns.py --corpus "$CORPUS" --rollouts "$ROLLOUTS" \
    --out "$STORE" --model "$MODEL" "${CAPTURE_ARGS[@]}"

  echo "[phase=upload_gpu]"
  upload_dir "$CORPUS" "corpus" "issue958 corpus"
  upload_dir "$ROLLOUTS" "raw_completions/rollouts" "issue958 rollout text"
  upload_dir "$STORE" "analysis_tensors/store" "issue958 activation store (fp16 shards)"
fi

if [ "$STAGE" = "cpu" ] || [ "$STAGE" = "all" ]; then
  echo "[phase=fits]"
  uv run python scripts/issue958_fit_maps.py --corpus "$CORPUS" --store "$STORE" \
    --cache "$CACHE" --maps "$MAPS" --out "$OUT" "${FIT_ARGS[@]}"

  echo "[phase=evals]"
  uv run python scripts/issue958_eval.py --out "$OUT"

  echo "[phase=plots]"
  uv run python scripts/issue958_plots.py --results "$OUT" --out "$FIGS"

  echo "[phase=upload_cpu]"
  upload_dir "$MAPS" "analysis_tensors/maps" "issue958 fitted map weights (7 rows fp16)"
fi

echo "[phase=sentinel]"
SENTINEL_KIND="epm:results"
if [ "$SMOKE" = "1" ]; then SENTINEL_KIND="epm:progress"; fi
EPM958_SENTINEL_KIND="$SENTINEL_KIND" EPM958_OUT_DIR="$OUT" EPM958_SMOKE="$SMOKE" \
  EPM958_STAGE="$STAGE" uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue958_common as C

out = Path(os.environ["EPM958_OUT_DIR"])
note = {
    "smoke": os.environ.get("EPM958_SMOKE") == "1",
    "stage": os.environ.get("EPM958_STAGE"),
    "deliverables": sorted(str(p) for p in out.glob("*.json")),
    "figures_dir": "figures/issue_958",
    "hf_prefix": C.HF_OUT_PREFIX,
    "transfer_standardization_policy": C.TRANSFER_STANDARDIZATION_POLICY,
    "note": "issue #958 multi-turn context->answer mapping run: per-turn maps, "
    "own-vs-stale transfer matrix, forecasts, prefix dominance, drift reads.",
}
C.write_results_sentinel(note, kind=os.environ["EPM958_SENTINEL_KIND"], version=1)
PY

echo "[phase=done]"
