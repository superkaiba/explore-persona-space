#!/usr/bin/env bash
# Issue #779 l26-kernel-gate-recovery — single-instance dispatch (plan v11 §4.5;
# adapted from issue779_n1m_readout_dispatch.sh, same phase/sentinel/pid-file/
# push-verify contract). Phases:
#
#   preflight (HF_TOKEN, disk, GPU, 32768^2 fp64 potrf+trsm probe — a GPU
#   cusolver REJECTION DEGRADES to the plan v11 §8 CPU-dpotrf fallback with a
#   loud WARNING + recorded flag in l26_recovery_potrf_probe.json; a genuine
#   failure on BOTH backends still aborts) ->
#   stream_fits (L26-only chunk re-stream -> solver-equivalence real-data leg
#   -> Nystrom-vs-exact gate at m=32768 [exact-side integrity assert] -> KRR
#   refit m=32768 cholesky [base_gamma integrity assert] + weight persist) ->
#   weights_upload (ONE upload_folder commit to a FRESH HF prefix
#   weights_l26_m32768 — the realized m=16384 weights dir is NEVER clobbered)
#   -> memmap_cleanup -> stage_v10_weights (READ-only staging of the realized
#   v10 L26 ridge/mlp payloads — preserved, never overwritten) -> stage_inputs
#   -> readout (--l26-kernel-recovery: 4 L26 cells + sycophancy grouped +
#   fit-quality L26 row; non-KRR byte-match gate; l26_recovery_* figures) ->
#   results_commit (push-verified + per-file ls-tree assert) -> sentinel ->
#   [phase=done]
#
# --smoke: the SAME entrypoints/flags scaled to a tiny slice (3 capture chunks,
# CPU, KRR m=256 smoke-labeled, --solver-equiv-m 512, traits evil,sycophancy,
# ALL predictors fit tiny so the smoke weights dir stands in for the staged
# v10 payloads), scratch output dirs (committed eval_results/figures NEVER
# touched), upload phase signature-smoked (no HF writes), commit phase
# skipped, sentinel kind epm:smoke-result. The smoke ALSO runs a degenerate
# integrity-assert probe (--expect-base-gamma 999 must fail loud), a forced
# GPU-reject potrf degrade probe (EPM_L26REC_FORCE_GPU_PROBE_FAIL=1 must exit
# 0 with cpu_fallback_engaged recorded), and a 1-real-file v10-weights staging
# probe. Pod/GCE-side: NO VM thread-cap prefix (dedicated CPUs).
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

LOG_DIR="${EPM_L26REC_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

COMMITTED_DIR="$REPO_ROOT/eval_results/issue_779/n1m-nonlinear-map-behavior-readout"
COMMITTED_FITS="$COMMITTED_DIR/n1m_multilayer_fits.json"
COMMITTED_READOUT="$COMMITTED_DIR/n1m_readout.json"
# plan v11 gate 2 integrity targets (committed n1m_multilayer_fits.json L26,
# read at implementation @ 9b60558da7): exact-side R2 + fit base_gamma.
EXPECT_EXACT_R2="0.7321946094333003"
EXPECT_BASE_GAMMA="8.188052265351295e-06"

if [ "$SMOKE" -eq 1 ]; then
  SCRATCH="${EPM_L26REC_SMOKE_SCRATCH:-/tmp/issue-779-l26rec-smoke}"
  mkdir -p "$SCRATCH"
  OUT_DIR="$SCRATCH/out"
  FIG_DIR="$SCRATCH/figures"
  LOG_DIR="$SCRATCH/logs"
  mkdir -p "$LOG_DIR"
  DEVICE=cpu
  KRR_CENTERS=256          # SMOKE ONLY — production pins 32768 (plan v11 §11)
  PROBE_M=1024             # SMOKE ONLY — production probes the full 32768^2
  # smoke fits ALL predictors tiny: the smoke weights dir stands in for the
  # staged v10 L26 ridge/mlp payloads the production readout consumes.
  PREDICTORS="ridge,mlp_w8192,mlp_w32768,krr_nystrom"
  FITS_SCALE=(--max-chunks 3 --mlp-max-epochs 1 --no-validate-krr --solver-equiv-m 512)
  INTEGRITY_ARGS=()        # committed targets don't apply to a 3-chunk slice
  TRAITS="evil,sycophancy"
  COLLECT_DIR="${EPM_N1M_RO_COLLECT_DIR:-$HOME/explore-persona-space/data/issue779_hfstage/issue779_monitoring/analysis_tensors}"
  CORPUS_DIR="${EPM_N1M_RO_CORPUS_DIR:-/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus}"
  MM_DIR="$SCRATCH/mm"
  CHUNK_CACHE=""           # smoke keeps the SHARED data/ chunk cache (3 chunks, reusable)
  V10_WEIGHTS_DIR="$OUT_DIR/weights_l26_m32768"   # smoke: same dir (all-predictor superset)
  SENTINEL_KIND="epm:smoke-result"
  SENTINEL_PATH="$LOG_DIR/issue-779-l26rec-smoke-results.json"
else
  OUT_DIR="$REPO_ROOT/eval_results/issue_779/l26-kernel-gate-recovery"
  FIG_DIR="$REPO_ROOT/figures/issue_779"
  DEVICE=cuda
  KRR_CENTERS=32768        # plan v11 §11 pin — L26 ONLY (L14/L19 stay as realized)
  PROBE_M=32768
  PREDICTORS="krr_nystrom" # the single re-fit arm; v10 non-KRR payloads are STAGED
  FITS_SCALE=()
  INTEGRITY_ARGS=(--expect-exact-r2 "$EXPECT_EXACT_R2" --expect-base-gamma "$EXPECT_BASE_GAMMA")
  TRAITS="evil,sycophancy,hallucination"
  COLLECT_DIR="$REPO_ROOT/data/issue779_hfstage/issue779_monitoring/analysis_tensors"
  CORPUS_DIR="$REPO_ROOT/data/issue_779/behavior_corpus"
  MM_DIR="$REPO_ROOT/data/issue_779/n1m_mm_l26rec"
  # the streamer's chunk staging is PINNED at data/issue_779/hf_dl/n1m_chunks
  # (issue779_ffc_n1m_fits.assemble_multilayer) — reap it after the stream.
  CHUNK_CACHE="$REPO_ROOT/data/issue_779/hf_dl/n1m_chunks"
  V10_WEIGHTS_DIR="$REPO_ROOT/data/issue_779/n1m_weights_v10"  # READ-only stage target
  SENTINEL_KIND="epm:results"
  SENTINEL_PATH="$LOG_DIR/issue-779-results.json"
  # Pid-file launch contract (pod-side-reporting rule): the workload's own pid.
  echo $$ > "$LOG_DIR/issue-779.pid"
fi
NEW_WEIGHTS_DIR="$OUT_DIR/weights_l26_m32768"
FITS_JSON="$OUT_DIR/l26_recovery_fits.json"
READOUT_JSON="$OUT_DIR/l26_recovery_readout.json"
PROBE_JSON="$OUT_DIR/l26_recovery_potrf_probe.json"
HF_WEIGHTS_PREFIX="issue779_monitoring/n1m_readout/weights_l26_m32768"
GPU_HOURS_BUDGETED=3
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
[ -f "$COMMITTED_FITS" ] || { echo "FATAL: committed comparator absent: $COMMITTED_FITS" >&2; exit 1; }
[ -f "$COMMITTED_READOUT" ] || { echo "FATAL: committed comparator absent: $COMMITTED_READOUT" >&2; exit 1; }
# potrf + trsm probe at the production m (plan v11 §12 rows 1+3), run BEFORE
# the 1-3.5 h stream. Plan v11 §8 row 2: a GPU (cusolver) REJECTION of the
# 32768^2 fp64 potrf does NOT abort — it DEGRADES to the CPU-dpotrf fallback
# path (the fit-level _cholesky_whitener/_cholesky_solve_psd CPU legs engage
# in-fit) with a loud WARNING, a CPU-viability re-probe at the SAME m, and a
# recorded flag in $PROBE_JSON (threaded into the sentinel + results commit).
# A genuine failure on BOTH backends still fails fast (aborts the dispatch).
# EPM_L26REC_FORCE_GPU_PROBE_FAIL=1 deterministically rejects the GPU leg
# (the smoke degrade-path probe below).
run_potrf_probe() {  # <m> <device> <out_json>
  EPM_PROBE_M="$1" EPM_PROBE_DEV="$2" EPM_PROBE_OUT="$3" uv run python - <<'PY'
import json
import os
import sys
import time
from pathlib import Path

import torch

m = int(os.environ["EPM_PROBE_M"])
dev_req = os.environ["EPM_PROBE_DEV"]
out_path = Path(os.environ["EPM_PROBE_OUT"])
force_fail = os.environ.get("EPM_L26REC_FORCE_GPU_PROBE_FAIL") == "1"


def _probe(dev_type: str) -> float:
    if force_fail and dev_type == "cuda":
        raise RuntimeError("forced GPU probe rejection (EPM_L26REC_FORCE_GPU_PROBE_FAIL=1)")
    dev = torch.device(dev_type)
    t0 = time.time()
    eye = torch.eye(m, dtype=torch.float64, device=dev)
    L = torch.linalg.cholesky(eye)  # potrf at (m, m) fp64
    inv = torch.linalg.solve_triangular(L, eye, upper=False)  # trsm materializes (m, m)
    assert inv.shape == (m, m), inv.shape
    del eye, L, inv
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return time.time() - t0


result = {
    "m": m,
    "device_requested": dev_req,
    "gpu_probe_ok": None,
    "cpu_probe_ok": None,
    "cpu_fallback_engaged": False,
    "gpu_error": None,
    "cpu_error": None,
    "probe_seconds": {},
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def _write() -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / (out_path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, out_path)


try:
    dt = _probe(dev_req)
except Exception as e:  # degrade decision below; a both-backend failure re-raises (fail fast)
    err = str(e).splitlines()[0]
    if dev_req != "cuda":
        result["cpu_probe_ok"] = False
        result["cpu_error"] = err
        _write()
        raise
    result["gpu_probe_ok"] = False
    result["gpu_error"] = err
    warn = (
        f"WARNING: GPU potrf/trsm probe REJECTED m={m} ({err}); DEGRADING to the plan v11 "
        "§8 CPU-dpotrf fallback (fit-level _cholesky_whitener/_cholesky_solve_psd CPU "
        "legs engage in-fit) — verifying CPU-side viability at the same m"
    )
    print(warn, flush=True)
    print(warn, file=sys.stderr, flush=True)
    try:
        dt = _probe("cpu")
    except Exception as e2:
        result["cpu_probe_ok"] = False
        result["cpu_error"] = str(e2).splitlines()[0]
        _write()
        print(
            f"FATAL: potrf probe failed on BOTH cuda and cpu at m={m} — genuine error, "
            "failing fast (probe metadata at "
            f"{out_path})",
            file=sys.stderr,
            flush=True,
        )
        raise
    result["cpu_probe_ok"] = True
    result["cpu_fallback_engaged"] = True
    result["probe_seconds"]["cpu"] = round(dt, 1)
    print(f"potrf+trsm probe OK at m={m} on cpu ({dt:.0f}s) — CPU fallback path viable")
else:
    result["gpu_probe_ok" if dev_req == "cuda" else "cpu_probe_ok"] = True
    result["probe_seconds"][dev_req] = round(dt, 1)
    print(f"potrf+trsm probe OK at m={m} on {dev_req} ({dt:.0f}s)")
_write()
print(f"potrf probe metadata written: {out_path}")
PY
}
run_potrf_probe "$PROBE_M" "$DEVICE" "$PROBE_JSON"
if [ "$SMOKE" -eq 1 ]; then
  # Degrade-path probe (concern preflight-potrf-abort-forecloses-cpu-fallback):
  # force the GPU leg to reject and assert the probe DEGRADES to CPU (exit 0 +
  # recorded flag) instead of aborting the dispatch.
  DEGRADE_JSON="$SCRATCH/potrf_probe_degrade.json"
  EPM_L26REC_FORCE_GPU_PROBE_FAIL=1 run_potrf_probe 256 cuda "$DEGRADE_JSON"
  grep -q '"cpu_fallback_engaged": true' "$DEGRADE_JSON" || {
    echo "FATAL: degrade probe did not record cpu_fallback_engaged=true in $DEGRADE_JSON" >&2
    exit 1
  }
  echo "  [smoke] potrf degrade-path probe OK (forced GPU reject -> CPU fallback, flag recorded)"
fi

echo "[phase=stream_fits]"
fits_cmd=(uv run python scripts/issue779_ffc_n1m_fits.py
  --layers 26 --points mixed_1m
  --predictors "$PREDICTORS"
  --krr-nystrom-centers "$KRR_CENTERS" --krr-solver cholesky
  --krr-solver-equivalence-check
  --persist-weights --resume
  --manifest-from-hf --device "$DEVICE" --prefetch 6
  --out-dir "$OUT_DIR" --out-json "$FITS_JSON"
  --weights-dir "$NEW_WEIGHTS_DIR" --mm-dir "$MM_DIR")
[ ${#FITS_SCALE[@]} -gt 0 ] && fits_cmd+=("${FITS_SCALE[@]}")
[ ${#INTEGRITY_ARGS[@]} -gt 0 ] && fits_cmd+=("${INTEGRITY_ARGS[@]}")
[ ${#EXTRA_FITS_ARGS[@]} -gt 0 ] && fits_cmd+=("${EXTRA_FITS_ARGS[@]}")
echo "  ${fits_cmd[*]}"
"${fits_cmd[@]}"
[ -f "$FITS_JSON" ] || { echo "FATAL: fits JSON absent at $FITS_JSON" >&2; exit 1; }
[ -f "$NEW_WEIGHTS_DIR/L26/krr_nystrom.pt" ] || {
  echo "FATAL: recovery KRR payload absent at $NEW_WEIGHTS_DIR/L26/krr_nystrom.pt" >&2; exit 1; }

if [ "$SMOKE" -eq 1 ]; then
  echo "[phase=integrity_probe]"
  # Degenerate probe: the plan-v11 gate-2 integrity assert must FAIL LOUD on a
  # wrong committed value (resume-fast: fits present, assert reads stored meta).
  if probe_out=$("${fits_cmd[@]}" --expect-base-gamma 999.0 2>&1); then
    echo "FATAL: integrity probe did NOT fail on --expect-base-gamma 999.0" >&2
    exit 1
  else
    echo "$probe_out" | grep -q "INTEGRITY ASSERT FAILED" || {
      echo "FATAL: integrity probe failed for the WRONG reason:" >&2
      echo "$probe_out" | tail -5 >&2
      exit 1
    }
    echo "  [smoke] integrity probe OK (INTEGRITY ASSERT FAILED fired as designed)"
  fi
fi

echo "[phase=weights_upload]"
if [ "$SMOKE" -eq 1 ]; then
  # No HF writes from a smoke: signature-smoke the upload entrypoint instead.
  uv run python -c "import inspect; from explore_persona_space.orchestrate import hub; print('hub._upload', inspect.signature(hub._upload))"
  echo "  [smoke] weights upload SKIPPED (signature-smoked); production uploads $NEW_WEIGHTS_DIR -> $HF_WEIGHTS_PREFIX"
else
  EPM_WD="$NEW_WEIGHTS_DIR" EPM_HF_PREFIX="$HF_WEIGHTS_PREFIX" uv run python - <<'PY'
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
# The L26 memmaps (~27.6 GB) + chunk staging are re-derivable (declared
# discarded_artifacts; regen = rerun the stream with --resume). Delete BEFORE
# the ~24 GB readout-input staging (plan v11 §9 disk sequencing).
du -sh "$MM_DIR" 2>/dev/null || true
rm -rf "$MM_DIR"
if [ -n "$CHUNK_CACHE" ]; then rm -rf "$CHUNK_CACHE"; fi
df -h "$REPO_ROOT" | tail -1

echo "[phase=stage_v10_weights]"
if [ "$SMOKE" -eq 1 ]; then
  # Smoke: the readout consumes the same-run all-predictor smoke weights dir;
  # exercise the REAL v10-weights staging helper on ONE file into scratch
  # (a genuine HF download through _stage_file — never writes the v10 prefix).
  EPM_SCRATCH="$SCRATCH" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import issue779_n1m_readout as RO

dest = Path(os.environ["EPM_SCRATCH"]) / "v10_weights_probe" / "ridge.pt"
if dest.exists():
    dest.unlink()
RO._stage_file("issue779_monitoring/n1m_readout/weights/L26/ridge.pt", dest)
assert dest.exists() and dest.stat().st_size > 0, dest
print(f"v10-weights stage probe OK: {dest} ({dest.stat().st_size} bytes)")
PY
else
  # READ-only staging of the realized v10 L26 non-KRR payloads (plan v11 §10:
  # the realized m=16384 weights are PRESERVED — staged into a fresh local dir,
  # never overwritten locally or on the Hub; the new KRR payload lives in the
  # separate weights_l26_m32768 dir/prefix).
  EPM_V10_WD="$V10_WEIGHTS_DIR" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import issue779_n1m_readout as RO

wdir = Path(os.environ["EPM_V10_WD"])
for fitter in ("ridge", "mlp_w8192", "mlp_w32768"):
    RO._stage_file(
        f"issue779_monitoring/n1m_readout/weights/L26/{fitter}.pt",
        wdir / "L26" / f"{fitter}.pt",
    )
    assert (wdir / "L26" / f"{fitter}.pt").stat().st_size > 0, fitter
print(f"v10 L26 non-KRR payloads staged read-only under {wdir}/L26")
PY
fi

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
    --weights-dir "$V10_WEIGHTS_DIR" --traits "$TRAITS" \
    --collect-dir "$COLLECT_DIR" --corpus-dir "$CORPUS_DIR"
fi

echo "[phase=readout]"
readout_cmd=(uv run python scripts/issue779_n1m_readout.py
  --l26-kernel-recovery
  --krr-weights-override "$NEW_WEIGHTS_DIR/L26/krr_nystrom.pt"
  --recovery-fits-json "$FITS_JSON"
  --committed-readout "$COMMITTED_READOUT"
  --weights-dir "$V10_WEIGHTS_DIR" --fits-json "$COMMITTED_FITS"
  --out-json "$READOUT_JSON"
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
  declared=("eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_fits.json"
            "eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_readout.json"
            "eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_potrf_probe.json")
  while IFS= read -r f; do
    declared+=("${f#"$REPO_ROOT"/}")
  done < <(find "$FIG_DIR" -maxdepth 1 -name 'l26_recovery_*' \
             \( -name '*.png' -o -name '*.pdf' -o -name '*.json' \) | sort)
  git -C "$REPO_ROOT" add -- "${declared[@]}"
  git -C "$REPO_ROOT" commit -m "[779] l26-kernel-gate-recovery: m=32768 cholesky refit + gate + scoped readout"
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
EPM_PROBE_JSON="$PROBE_JSON" \
EPM_GPU_H_USED="$GPU_HOURS_USED" EPM_GPU_H_BUDGET="$GPU_HOURS_BUDGETED" \
EPM_HF_WEIGHTS_PREFIX="$HF_WEIGHTS_PREFIX" EPM_BRANCH="$BRANCH" uv run python - <<'PY'
import json
import os
import time
from pathlib import Path

smoke = os.environ["EPM_SMOKE"] == "1"
probe_path = Path(os.environ["EPM_PROBE_JSON"])
potrf_probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
ro = json.loads(Path(os.environ["EPM_READOUT_JSON"]).read_text())
rec = ro.get("l26_recovery", {})
kernel_deltas = {
    t: {
        m: {
            k: round(float(v), 4)
            for k, v in e["deltas_vs_pv_raw"]["n1m_krr_nystrom_dot"].items()
            if k in ("delta", "lo", "hi")
        }
        for m, e in tm.items()
    }
    for t, tm in ro.get("headline", {}).items()
}
grouped_kernel = {
    t: round(float(g["group_level"]["n1m_krr_nystrom"]["dot"]["point"]), 4)
    for t, g in ro.get("grouped", {}).items()
}
note = {
    "summary": (
        "l26-kernel-gate-recovery: jittered-Cholesky Nystrom solver, L26 KRR refit at "
        "m=32768, Nystrom-vs-exact gate re-run, scoped re-read of the 4 L26 kernel "
        "cells + sycophancy grouped (plan v11)."
    ),
    "gate_m32768": rec.get("gate"),
    "solver_equivalence": rec.get("solver_equivalence"),
    # preflight potrf probe record: gpu_probe_ok / cpu_fallback_engaged (plan v11 §8
    # CPU-dpotrf degrade path; concern preflight-potrf-abort-forecloses-cpu-fallback)
    "potrf_probe": potrf_probe,
    "validity_gate_pass": ro.get("validity_gate", {}).get("overall_pass"),
    "nonkrr_match_gate_pass": ro.get("nonkrr_match_gate", {}).get("overall_pass"),
    "kernel_deltas_vs_raw_dot": kernel_deltas,
    "grouped_kernel_dot_point": grouped_kernel,
    "eval_paths": [
        "eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_fits.json",
        "eval_results/issue_779/l26-kernel-gate-recovery/l26_recovery_readout.json",
    ],
    "reproducibility_card": {
        "adapter_paths": {},  # N/A — no adapters (one KRR refit + read-only readout)
        "hf_data_paths": [os.environ["EPM_HF_WEIGHTS_PREFIX"]],
        "wandb_note": "no WandB logging this round (closed-form refit; nothing trains a model)",
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
    "by": "issue779_l26_recovery_dispatch",
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
