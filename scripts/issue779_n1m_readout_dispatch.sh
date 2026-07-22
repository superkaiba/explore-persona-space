#!/usr/bin/env bash
# Issue #779 n1m-nonlinear-map-behavior-readout — single-instance dispatch (GCP lane
# via --workload-cmd; RunPod fallback identical). Phases (plan SS9 sequencing):
#
#   preflight -> stream_fits (one-pass 3-layer memmap stream + fits + weight
#   persist) -> weights_upload (ONE upload_folder commit + verified) ->
#   memmap_cleanup (DELETE the ~83 GB memmaps BEFORE the 34.7 GB corpus
#   download; disk-tight lanes only fit with this order) -> stage_inputs ->
#   readout (behavior read-out + grouped LOGO + figures) -> results_commit
#   (push-verified + per-file ls-tree artifact assert, #1205/#1325) ->
#   sentinel -> [phase=done]
#
# --smoke: the SAME entrypoints/flags scaled to a tiny slice (3 capture chunks,
# CPU, 1 MLP epoch, KRR m=256 smoke-labeled, traits evil,sycophancy), scratch
# output dirs (committed eval_results/figures NEVER touched), upload phase
# signature-smoked (no HF writes from a smoke), commit phase skipped, sentinel
# kind epm:smoke-result. Pod/GCE-side: NO VM thread-cap prefix (dedicated CPUs).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Conditional .env source (GCE lane has no .env; drivers also load_dotenv()).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

SMOKE=0
EXTRA_FITS_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    *) EXTRA_FITS_ARGS+=("$1"); shift ;;
  esac
done

LOG_DIR="${EPM_N1M_RO_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

if [ "$SMOKE" -eq 1 ]; then
  SCRATCH="${EPM_N1M_RO_SMOKE_SCRATCH:-/tmp/issue-779-n1m-smoke}"
  mkdir -p "$SCRATCH"
  OUT_DIR="$SCRATCH/out"
  FIG_DIR="$SCRATCH/figures"
  LOG_DIR="$SCRATCH/logs"
  mkdir -p "$LOG_DIR"
  DEVICE=cpu
  KRR_CENTERS=256          # SMOKE ONLY — production pins 16384 (plan SS10/SS11)
  FITS_SCALE=(--max-chunks 3 --mlp-max-epochs 1 --no-validate-krr)
  TRAITS="evil,sycophancy"
  # VM smoke reads the ALREADY-LOCAL rig inputs (repo-root staged copies).
  COLLECT_DIR="${EPM_N1M_RO_COLLECT_DIR:-$HOME/explore-persona-space/data/issue779_hfstage/issue779_monitoring/analysis_tensors}"
  CORPUS_DIR="${EPM_N1M_RO_CORPUS_DIR:-/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus}"
  MM_DIR="$SCRATCH/mm"
  CHUNK_CACHE="$SCRATCH/hf_dl"
  SENTINEL_KIND="epm:smoke-result"
  SENTINEL_PATH="$LOG_DIR/issue-779-smoke-results.json"
else
  OUT_DIR="$REPO_ROOT/eval_results/issue_779/n1m-nonlinear-map-behavior-readout"
  FIG_DIR="$REPO_ROOT/figures/issue_779"
  DEVICE=cuda
  KRR_CENTERS=32768        # raised 16384->32768 (recorded plan deviation, att-20260722-165214):
                           # L26 pre-fit Nystrom-vs-exact gate read gap 0.0151 > tol 0.0100 at the
                           # SS10/SS11 pin; the script's own fail-fast remedy prescribes raising m.
                           # The tol gate is unchanged. With --resume, L14/L19 keep realized m=16384
                           # (weights persisted); only L26 fits at 32768 — carry per-layer m into the
                           # clean-result.
  FITS_SCALE=()
  TRAITS="evil,sycophancy,hallucination"
  COLLECT_DIR="$REPO_ROOT/data/issue779_hfstage/issue779_monitoring/analysis_tensors"
  CORPUS_DIR="$REPO_ROOT/data/issue_779/behavior_corpus"
  MM_DIR="$REPO_ROOT/data/issue_779/n1m_mm"
  CHUNK_CACHE="$REPO_ROOT/data/issue_779/hf_dl/n1m_chunks"
  SENTINEL_KIND="epm:results"
  SENTINEL_PATH="$LOG_DIR/issue-779-results.json"
  # Pid-file launch contract (pod-side-reporting rule): the workload's own pid.
  echo $$ > "$LOG_DIR/issue-779.pid"
fi
WEIGHTS_DIR="$OUT_DIR/weights"
FITS_JSON="$OUT_DIR/n1m_multilayer_fits.json"
READOUT_JSON="$OUT_DIR/n1m_readout.json"
HF_WEIGHTS_PREFIX="issue779_monitoring/n1m_readout/weights"
GPU_HOURS_BUDGETED=6
T_START=$(date +%s)

echo "[phase=preflight]"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "FATAL: HF_TOKEN not set (GCE startup exports it; VM/pod sources .env)" >&2
  exit 1
fi
df -h "$REPO_ROOT" | tail -1
if [ "$DEVICE" = "cuda" ]; then
  nvidia-smi -L || { echo "FATAL: --device cuda but no GPU visible" >&2; exit 1; }
fi
BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
echo "branch=$BRANCH out_dir=$OUT_DIR device=$DEVICE smoke=$SMOKE"
mkdir -p "$OUT_DIR" "$FIG_DIR"

echo "[phase=stream_fits]"
fits_cmd=(uv run python scripts/issue779_ffc_n1m_fits.py
  --layers 14,19,26 --points mixed_1m
  --predictors ridge,mlp_w8192,mlp_w32768,krr_nystrom
  --krr-nystrom-centers "$KRR_CENTERS" --persist-weights --resume
  --manifest-from-hf --device "$DEVICE" --prefetch 6
  --out-dir "$OUT_DIR" --out-json "$FITS_JSON"
  --weights-dir "$WEIGHTS_DIR" --mm-dir "$MM_DIR")
[ ${#FITS_SCALE[@]} -gt 0 ] && fits_cmd+=("${FITS_SCALE[@]}")
[ ${#EXTRA_FITS_ARGS[@]} -gt 0 ] && fits_cmd+=("${EXTRA_FITS_ARGS[@]}")
echo "  ${fits_cmd[*]}"
"${fits_cmd[@]}"
[ -f "$FITS_JSON" ] || { echo "FATAL: fits JSON absent at $FITS_JSON" >&2; exit 1; }

echo "[phase=weights_upload]"
if [ "$SMOKE" -eq 1 ]; then
  # No HF writes from a smoke: signature-smoke the upload entrypoint instead.
  uv run python -c "import inspect; from explore_persona_space.orchestrate import hub; print('hub._upload', inspect.signature(hub._upload))"
  echo "  [smoke] weights upload SKIPPED (signature-smoked); production uploads $WEIGHTS_DIR -> $HF_WEIGHTS_PREFIX"
else
  EPM_WD="$WEIGHTS_DIR" EPM_HF_PREFIX="$HF_WEIGHTS_PREFIX" uv run python - <<'PY'
import os
from pathlib import Path

from explore_persona_space.orchestrate import hub

wdir = Path(os.environ["EPM_WD"])
prefix = os.environ["EPM_HF_PREFIX"]
local = sorted(p.relative_to(wdir).as_posix() for p in wdir.rglob("*.pt"))
assert local, f"no persisted weights under {wdir}"
url = hub._upload(wdir, repo_id="superkaiba1/explore-persona-space-data",
                  repo_type="dataset", path_in_repo=prefix)
if not url:
    raise SystemExit(f"weights upload to {prefix} returned no URL (hub._upload failed)")
from huggingface_hub import HfApi

remote = {
    f.path[len(prefix) + 1 :]
    for f in HfApi().list_repo_tree(
        "superkaiba1/explore-persona-space-data", path_in_repo=prefix,
        repo_type="dataset", recursive=True,
    )
    if getattr(f, "size", None) is not None
}
missing = [p for p in local if p not in remote]
if missing:
    raise SystemExit(f"weights upload verification MISSING on Hub: {missing}")
print(f"weights upload verified: {len(local)} files under {prefix}")
PY
fi

echo "[phase=memmap_cleanup]"
# Delete the ~83 GB stream memmaps + chunk staging BEFORE the 34.7 GB corpus
# download (plan SS9: only with this order does a ~130 GB disk fit). Weights are
# persisted (and uploaded in production) — memmaps are re-derivable (declared
# discarded_artifacts; regen = rerun the stream with --resume).
du -sh "$MM_DIR" 2>/dev/null || true
rm -rf "$MM_DIR" "$CHUNK_CACHE"
df -h "$REPO_ROOT" | tail -1

echo "[phase=stage_inputs]"
if [ "$SMOKE" -eq 1 ]; then
  # Inputs already local on the VM; exercise the REAL staging helper on one
  # small file into scratch (a genuine download through _stage_file).
  EPM_SCRATCH="$SCRATCH" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import issue779_n1m_readout as RO

dest = Path(os.environ["EPM_SCRATCH"]) / "stage_probe" / "evil.pt"
if dest.exists():
    dest.unlink()
RO._stage_file(f"{RO.HF_ANALYSIS_PREFIX}/r_b/evil.pt", dest)
assert dest.exists() and dest.stat().st_size > 0, dest
print(f"stage probe OK: {dest} ({dest.stat().st_size} bytes)")
PY
else
  mkdir -p "$COLLECT_DIR/pass_b"
  # pass_b was already downloaded by the fits driver — link, don't re-download.
  if [ ! -e "$COLLECT_DIR/pass_b/train_context_vectors.pt" ]; then
    ln -s "$REPO_ROOT/data/issue_779/pass_b/train_context_vectors.pt" \
      "$COLLECT_DIR/pass_b/train_context_vectors.pt"
  fi
  uv run python scripts/issue779_n1m_readout.py --stage-only \
    --weights-dir "$WEIGHTS_DIR" --traits "$TRAITS" \
    --collect-dir "$COLLECT_DIR" --corpus-dir "$CORPUS_DIR"
fi

echo "[phase=readout]"
readout_cmd=(uv run python scripts/issue779_n1m_readout.py
  --weights-dir "$WEIGHTS_DIR" --fits-json "$FITS_JSON" --out-json "$READOUT_JSON"
  --fig-dir "$FIG_DIR" --device "$DEVICE" --traits "$TRAITS"
  --collect-dir "$COLLECT_DIR" --corpus-dir "$CORPUS_DIR")
echo "  ${readout_cmd[*]}"
"${readout_cmd[@]}"
[ -f "$READOUT_JSON" ] || { echo "FATAL: readout JSON absent at $READOUT_JSON" >&2; exit 1; }

echo "[phase=results_commit]"
if [ "$SMOKE" -eq 1 ]; then
  echo "  [smoke] results commit SKIPPED (scratch outputs; committed paths untouched)"
else
  if [ "$BRANCH" = "main" ]; then
    echo "FATAL: refusing to commit workload results directly to main" >&2
    exit 1
  fi
  declared=("eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json"
            "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_readout.json")
  while IFS= read -r f; do
    declared+=("${f#"$REPO_ROOT"/}")
  done < <(find "$FIG_DIR" -maxdepth 1 -name 'n1m_readout_*' \
             \( -name '*.png' -o -name '*.pdf' -o -name '*.json' \) | sort)
  git -C "$REPO_ROOT" add -- "${declared[@]}"
  git -C "$REPO_ROOT" commit -m "[779] n1m-readout: production fits + readout + figures"
  if ! git -C "$REPO_ROOT" push origin "HEAD:$BRANCH"; then
    echo "push failed; retrying once" >&2
    git -C "$REPO_ROOT" push origin "HEAD:$BRANCH"
  fi
  ahead=$(git -C "$REPO_ROOT" rev-list --count "origin/$BRANCH..HEAD")
  if [ "$ahead" != "0" ]; then
    echo "FATAL: push verification failed ($ahead commits unpushed)" >&2
    exit 86
  fi
  missing=0
  for pth in "${declared[@]}"; do
    if [ -z "$(git -C "$REPO_ROOT" ls-tree -r "origin/$BRANCH" --name-only -- "$pth")" ]; then
      echo "FATAL: declared result path missing from pushed tree: $pth" >&2
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || exit 86
  echo "  push-verified: ${#declared[@]} declared result paths present on origin/$BRANCH"
fi

echo "[phase=sentinel]"
GPU_HOURS_USED=$(awk -v s="$T_START" -v e="$(date +%s)" 'BEGIN{printf "%.2f", (e-s)/3600.0}')
EPM_SENTINEL_PATH="$SENTINEL_PATH" EPM_SENTINEL_KIND="$SENTINEL_KIND" \
EPM_SMOKE="$SMOKE" EPM_READOUT_JSON="$READOUT_JSON" EPM_FITS_JSON="$FITS_JSON" \
EPM_GPU_H_USED="$GPU_HOURS_USED" EPM_GPU_H_BUDGET="$GPU_HOURS_BUDGETED" \
EPM_HF_WEIGHTS_PREFIX="$HF_WEIGHTS_PREFIX" EPM_BRANCH="$BRANCH" uv run python - <<'PY'
import json
import os
import time
from pathlib import Path

smoke = os.environ["EPM_SMOKE"] == "1"
ro = json.loads(Path(os.environ["EPM_READOUT_JSON"]).read_text())
fits = json.loads(Path(os.environ["EPM_FITS_JSON"]).read_text())
gate = ro.get("validity_gate", {})
headline = {
    t: {
        m: {
            a: round(e["monitors"][f"{a}_dot" if a not in ("pv_raw", "oracle") else a]["point"], 4)
            for a in ("pv_raw", "h_n5k_linear", "n1m_ridge", "n1m_mlp_w32768", "oracle")
        }
        for m, e in tm.items()
    }
    for t, tm in ro.get("headline", {}).items()
}
note = {
    "summary": (
        "n1m-nonlinear-map-behavior-readout: multi-layer n1m fits (L14/19/26, mixed_1m) "
        "+ behavior read-out over the parent rig (no new generation/judging)."
    ),
    "validity_gate_pass": gate.get("overall_pass"),
    "headline_dot_points": headline,
    "per_layer_test_r2": {
        lk: {
            f: lv["per_point"].get("mixed_1m", {}).get("predictors", {}).get(f, {}).get("whole_map_r2")
            for f in ("ridge", "mlp_w8192", "mlp_w32768", "krr_nystrom")
        }
        for lk, lv in fits.get("per_layer", {}).items()
    },
    "eval_paths": [
        "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json",
        "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_readout.json",
    ],
    "reproducibility_card": {
        "adapter_paths": {},  # N/A — this round trains no adapters (analysis + fits only)
        "hf_data_paths": [os.environ["EPM_HF_WEIGHTS_PREFIX"]],
        "wandb_note": "no WandB logging this round (closed-form/minibatch fits; nothing trains a model)",
        "branch": os.environ["EPM_BRANCH"],
        "gpu_hours_used": float(os.environ["EPM_GPU_H_USED"]),
        "gpu_hours_budgeted": float(os.environ["EPM_GPU_H_BUDGET"]),
    },
    "gpu_hours_used": float(os.environ["EPM_GPU_H_USED"]),
    "gpu_hours_budgeted": float(os.environ["EPM_GPU_H_BUDGET"]),
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": os.environ["EPM_SENTINEL_KIND"],
    "version": 1,
    "task_id": 779,
    "gate": "smoke" if smoke else "results",
    "smoke": smoke,
    "blocks_pipeline": not smoke,
    "by": "issue779_n1m_readout_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": json.dumps(note),
}
dest = Path(os.environ["EPM_SENTINEL_PATH"])
dest.parent.mkdir(parents=True, exist_ok=True)
tmp = dest.parent / (dest.name + ".tmp")
tmp.write_text(json.dumps(sentinel, indent=1))
os.replace(tmp, dest)
print(f"sentinel written: {dest} kind={sentinel['kind']}")
PY

echo "[phase=done]"
