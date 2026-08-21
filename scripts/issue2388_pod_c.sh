#!/usr/bin/env bash
# issue-2388 Pod C launcher (pod-2388-fits, 2x H100): P3 (the 14 new-surface
# map keys — math/MCQ/code cells + code's OWN generic-only pair, MF-A) + P5
# (readout L sweep x arms x bases at dof-capped selection + f_U cells + nulls
# + group bootstrap) for math/MCQ/code ONLY — every QA cell is homed on Pod B.
# Plan v5 section 4 P3+P5 / section 9 Pod C disk row (staged store ~67-92 GB
# + u-store 12.76 GB + maps ~2 GB -> ~85-110 GB peak, under the ~130 GB
# MooseFS quota). Two-GPU sharding: P3 shard axis = map key, P5 shard axis =
# surface (plan section 9 rows), barrier-joined lanes with launcher-env CVD
# pins (gotchas: the in-process clobber is not trusted).
set -euo pipefail
trap 'echo "[phase=failed] rc=$? line=$LINENO cmd=$BASH_COMMAND"' ERR

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME="${EPM_I2388_HF_HOME:-/opt/hf_cache}"
mkdir -p "$HF_HOME"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR" /workspace/store_2388 /workspace/u_store
HF_DATA_URL=https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main
MAPS_DIR=eval_results/issue_2388/maps
FITS_DIR=eval_results/issue_2388/fits

sentinel() { # sentinel <name> <note>
  uv run python - "$1" "$2" <<'PY'
import json, sys, time
name, note = sys.argv[1], sys.argv[2]
json.dump(
    {"kind": "epm:progress", "note": note, "blocks_pipeline": False,
     "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    open(f"/workspace/logs/issue-2388-{name}.json", "w"),
)
PY
}

headroom() { # headroom <need_gb> <phase>
  uv run python -c "from explore_persona_space.orchestrate.preflight import assert_out_root_headroom; assert_out_root_headroom('/workspace', float('$1'), phase='$2')"
}

phase_done() { [ -f "$LOGDIR/issue-2388-$1.json" ]; } # R3: pod_b idiom — a
  # crash-fix re-entry must SKIP completed phases; without this the p3-stage
  # headroom floor (sized for FRESH staging) re-fires against a disk that
  # legitimately holds the 91 GB staged store (the Pod A R10 class).

commit_results() { # commit_results <msg> <path>...
  # NON-FATAL variant (deviation from pod_b, documented): Pod B and Pod C
  # write the SAME branch concurrently and both touch maps/key_manifest.json,
  # so a pull-rebase here can conflict mid-run. Durability is HF-side (the
  # fits driver's --phase upload pushes maps + fits + h3 artifacts); the git
  # commit is the checkpoint MIRROR — on an unresolvable push, log LOUD and
  # leave the commit local for the orchestrator's end-of-run sync.
  local msg="$1"; shift
  git add -- "$@" 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git -c user.name="eps-pod-2388" -c user.email="pod-2388@eps.local" commit -m "$msg" -- "$@"
    if ! git push origin issue-2388 > /tmp/push_c.out 2>&1; then
      if git pull --rebase --autostash origin issue-2388; then
        git push origin issue-2388 > /tmp/push_c.out 2>&1 \
          || echo "[commit] WARN: push failed post-rebase — commit left local for orchestrator sync"
      else
        git rebase --abort || true
        echo "[commit] WARN: rebase conflict (concurrent Pod B writer) — commit left local for orchestrator sync"
      fi
    fi
  fi
}

stream_tar() { # stream_tar <hf-relpath> <dest-dir> — R2: hf_transfer download-then-untar.
  # This DC's HF CDN path is per-stream throttled (measured 2026-08-21 ~01:05Z:
  # 0.24 MB/s single-stream, 1.9 MB/s at 8-way, vs 14.8 MB/s generic egress),
  # so the R1 curl|tar stream form could not finish 91 GB. hf_transfer's
  # parallel ranges measured 13 MB/s on this pod (humaneval.tar probe), and
  # its .incomplete files resume across attempts. The MooseFS stream-only
  # caveat does not bind here: /workspace on this pod is a real local volume
  # (/dev/md0, df truthful), and peak footprint — 91 GB extracted store +
  # 38 GB largest in-flight tar — fits the 200 GB volume. Each tarball is
  # removed after extraction. Per-attempt timeout 7200 s: largest tar
  # 37.7 GB / measured 13 MB/s ~= 48 min, x2 margin plus retry-envelope slack.
  local rel="$1" dest="$2" root n=0 tarball
  root="$(basename "$rel" .tar)"
  tarball="/workspace/hf_stage/$rel"
  mkdir -p /workspace/hf_stage
  while :; do
    echo "[stage] hf-download $rel start $(date -u +%H:%M:%SZ) (attempt $((n+1)))"
    if timeout 7200 env HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download \
         superkaiba1/explore-persona-space-data "$rel" \
         --repo-type dataset --local-dir /workspace/hf_stage \
       && tar -xf "$tarball" -C "$dest"; then
      rm -f "$tarball"
      echo "[stage] $rel staged $(date -u +%H:%M:%SZ)"
      return 0
    fi
    n=$((n+1))
    [ "$n" -ge 5 ] && { echo "[stage] $rel FAILED after $n attempts"; return 1; }
    echo "[stage] $rel attempt $n failed; reaping partial $dest/$root and retrying"
    rm -rf "${dest:?}/${root:?}"
    sleep 30
  done
}

# ------------------------------------------------------------- P3 staging ---
if phase_done p3-stage-done; then echo "[phase=p3_stage] SKIP (done-sentinel)"; else
echo "[phase=p3_stage]"
headroom 115 p3-stage
# (a) The P2 capture store: the tar set is read from a SCOPED listing of the
# prefix (never a hardcoded roster — the code roster is gate-derived, and the
# listing reflects exactly what Pod A uploaded + exact-set verified).
uv run python - <<'PY' > /tmp/store_tars.txt
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
prefix = "issue2388_correctness/analysis_tensors/capture_store"
names = sorted(
    e.path for e in HfApi().list_repo_tree(
        "superkaiba1/explore-persona-space-data", repo_type="dataset",
        path_in_repo=prefix, recursive=True)
    if e.path.endswith(".tar")
)
if not names:
    raise SystemExit(f"no store tars under {prefix} — Pod A P2 upload missing")
print("\n".join(names))
PY
while read -r rel; do
  bench_dir="/workspace/store_2388/$(basename "$rel" .tar)"
  if [ -f "$bench_dir/_capture_manifest.json" ]; then
    echo "[stage] $rel already extracted — skip"
  else
    stream_tar "$rel" /workspace/store_2388
  fi
done < /tmp/store_tars.txt
# (b) #1092 U-store (12.76 GB, staged idempotently by the reused dispatcher).
uv run python -c "
from explore_persona_space.experiments.issue_1739.store_io import stage_u_store
print('[stage] u_store ->', stage_u_store('/workspace/u_store'))
"
# (c) Pod B's shared f_U=0 map weights (the ONE cross-pod read, plan section
# 10): weights are HF-only (*.npz/*.pt are gitignored) — stage into MAPS_DIR.
uv run python - <<'PY'
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download
dest = Path("eval_results/issue_2388/maps"); dest.mkdir(parents=True, exist_ok=True)
for name in ("linear__shared__fu0.npz", "mlp__shared__fu0.pt"):
    tgt = dest / name
    if tgt.exists():
        print(f"[stage] {name} already present — skip"); continue
    p = hf_hub_download(
        "superkaiba1/explore-persona-space-data",
        f"issue2388_correctness/analysis_tensors/maps/{name}",
        repo_type="dataset", local_dir="/workspace/maps_dl")
    tgt.hardlink_to(p)
    print(f"[stage] {name} staged")
PY
sentinel p3-stage-done "P3: store tars extracted (gate-derived set), u_store staged, shared f_U=0 map weights staged"
fi

# ------------------------------- P3 smoke (G2 device-domain, per surface) ---
# One tiny maps fit + one tiny sweep per SURFACE loader class (math/mcq/code
# table loaders + code's gate-roster branch are per-class code paths; the
# REGIME/CLASS coverage rule). Smoke writes land in *_smoke out-roots only.
if phase_done p3-smoke-done; then echo "[phase=p3_smoke] SKIP (done-sentinel)"; else
echo "[phase=p3_smoke]"
for s in math mcq code; do
  # R3: fit BOTH map kinds in the smoke — phase_sweep pins + loads the linear
  # AND mlp payloads for its --map-cell unconditionally (fits.py
  # _pin_map_payloads), so the R2 linear-only smoke maps fit died at
  # maps_smoke/mlp__math__fu1.pt.
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2388_fits.py --smoke --phase maps --surface "$s" --keys "linear__${s}__fu1" "mlp__${s}__fu1" --device cuda
  # R5: --smoke clamps only n_null/n_boot — sweep SCALE rides the CLI dials,
  # so an unclamped smoke sweep runs the full L-grid x 3 draws (~7 h/surface,
  # near-duplicating P5; observed live at L1000 2.5 h in). One rung x one
  # draw is the plan-section-4 G2 scope (loader class + sweep dispatch);
  # already-computed smoke cells resume by filename, so this re-entry is fast.
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2388_fits.py --smoke --phase sweep --surface "$s" --map-cell fu1 --device cuda --budgets 250 --n-draws 1
done
sentinel p3-smoke-done "P3 smoke: per-surface maps+sweep smoke green (math/mcq/code loader classes)"
fi

# --------------------------------------- P3 map fits (14 keys, 2 GPU lanes) ---
if phase_done p3-done; then echo "[phase=p3_maps] SKIP (done-sentinel)"; else
echo "[phase=p3_maps]"
map_lane() { # map_lane <gpu> <surface> <key>...
  local gpu="$1" s="$2"; shift 2
  echo "[lane$gpu] maps $s start $(date -u +%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_fits.py --phase maps --surface "$s" --device cuda --keys "$@"
  echo "[lane$gpu] maps $s done $(date -u +%H:%M:%SZ)"
}
( map_lane 0 math linear__math__fu05 linear__math__fu1 mlp__math__fu05 mlp__math__fu1 \
  && map_lane 0 code linear__code__fu0 mlp__code__fu0 ) > "$LOGDIR/issue-2388-p3-lane0.log" 2>&1 & L0=$!
( map_lane 1 mcq linear__mcq__fu05 linear__mcq__fu1 mlp__mcq__fu05 mlp__mcq__fu1 \
  && map_lane 1 code linear__code__fu05 linear__code__fu1 mlp__code__fu05 mlp__code__fu1 ) > "$LOGDIR/issue-2388-p3-lane1.log" 2>&1 & L1=$!
fail=0
wait "$L0" || { echo "[p3] lane0 FAILED (see lane log)"; fail=1; }
wait "$L1" || { echo "[p3] lane1 FAILED (see lane log)"; fail=1; }
[ "$fail" -eq 0 ] || exit 1
uv run python scripts/issue2388_fits.py --phase upload
commit_results "issue #2388: P3 new-surface map fits (14 keys) + diagnostics" "$MAPS_DIR"
sentinel p3-done "P3: 14 map keys fit (math/mcq cells + code cells + code generic-only pair) + uploaded to HF"
fi

# ------------------------- P5 readout sweeps (surface-sharded, 2 GPU lanes) ---
if phase_done p5-sweep-done; then echo "[phase=p5_sweep] SKIP (done-sentinel)"; else
echo "[phase=p5_sweep]"
sweep_surface() { # sweep_surface <gpu> <surface>  — primary + f_U cells
  local gpu="$1" s="$2"
  echo "[lane$gpu] sweep $s primary start $(date -u +%H:%M:%SZ)"
  CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_fits.py --phase sweep --surface "$s" --device cuda
  for cell in fu0 fu05; do
    echo "[lane$gpu] sweep $s cell=$cell start $(date -u +%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/issue2388_fits.py --phase sweep --surface "$s" --map-cell "$cell" --device cuda
  done
  echo "[lane$gpu] sweep $s done $(date -u +%H:%M:%SZ)"
}
( sweep_surface 0 code )                      > "$LOGDIR/issue-2388-p5-lane0.log" 2>&1 & S0=$!
( sweep_surface 1 math && sweep_surface 1 mcq ) > "$LOGDIR/issue-2388-p5-lane1.log" 2>&1 & S1=$!
fail=0
wait "$S0" || { echo "[p5] lane0 FAILED (see lane log)"; fail=1; }
wait "$S1" || { echo "[p5] lane1 FAILED (see lane log)"; fail=1; }
[ "$fail" -eq 0 ] || exit 1
commit_results "issue #2388: P5 L-sweeps + f_U cells (math/mcq/code, dof-capped primary)" "$FITS_DIR"
sentinel p5-sweep-done "P5: primary + fu0/fu05 sweeps done for math/mcq/code"
fi

# ----------------------------------------- P5 select + bootstrap + upload ---
if phase_done p5-done; then echo "[phase=p5_aggregate] SKIP (done-sentinel)"; else
echo "[phase=p5_aggregate]"
for s in math mcq code; do
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2388_fits.py --phase select --surface "$s" --device cuda
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue2388_fits.py --phase bootstrap --surface "$s" --device cuda
done
uv run python scripts/issue2388_fits.py --phase upload
commit_results "issue #2388: P5 all_arms + bootstrap (math/mcq/code)" "$FITS_DIR"
sentinel p5-done "P5 complete: fits + f_U + nulls + bootstrap committed and uploaded for math/mcq/code"
fi

echo "[phase=done]"
