#!/usr/bin/env bash
# issue-2388 Pod B launcher (pod-2388-qafits, 1x H100): P4 — the QA lane + the
# H3 two-stage recompute. Plan v5 section 4 P4 (i)-(viii) / section 9 Pod B
# disk row (SEQUENTIAL residency under the ~130 GB MooseFS quota: stage ->
# fit -> reap per behavior; the 70 GB QA store STREAM-extracts, never
# download-then-untar). Single-pipeline fail-loud; `[phase=...]` milestone
# lines + drained sentinels; per-stage assert_out_root_headroom; results
# committed to the issue-2388 branch as they land (checkpoint cadence).
# Binding deferral (code-review round 4 reconciler): every sweep invocation
# passes --qa-questions-shards with the staged banked #1739 labeling shards.
set -euo pipefail
trap 'echo "[phase=failed] rc=$? line=$LINENO cmd=$BASH_COMMAND"' ERR

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
set -a; [ -f .env ] && . ./.env; set +a
# Plan section 9 Pod B disk row: model/venv/HF cache on /opt; /workspace holds
# the sequentially-staged stores only.
export HF_HOME="${EPM_I2388_HF_HOME:-/opt/hf_cache}"
mkdir -p "$HF_HOME"
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR" /workspace/h3_stores /workspace/store /workspace/qa_shards
HF_DATA_URL=https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main
H3_RECOMPUTE_ROOT=eval_results/issue_2388/h3_recompute

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

commit_results() { # commit_results <msg> <path>...
  local msg="$1"; shift
  git add -- "$@" 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git -c user.name="eps-pod-2388" -c user.email="pod-2388@eps.local" commit -m "$msg" -- "$@"
    if ! git push origin issue-2388 > /tmp/push_b.out 2>&1; then
      git pull --rebase --autostash origin issue-2388
      git push origin issue-2388 > /tmp/push_b.out 2>&1
    fi
  fi
}

phase_done() { # phase_done <sentinel-name> — relaunch skip guard: a crash-fix
  # relaunch re-enters this launcher from the top; completed phases are keyed
  # by their done-sentinels (R2: a blind re-entry would re-download the 53 GB
  # sycophancy stores and re-run finished GPU fits before reaching the crash).
  [ -f "$LOGDIR/issue-2388-$1.json" ]
}

stream_tar() { # stream_tar <hf-relpath> <dest-dir> — staged-idempotent + retriable.
  # A single ~32 GB HTTP/2 stream got CANCELed by the CDN 4.8 h in (R2) and a
  # pipe cannot resume, so: when /workspace headroom fits tar + extraction
  # (x2.2), download with curl -C - resume + size verify, extract locally,
  # drop the tar; otherwise keep the plan-section-9 stream form (the 70 GB QA
  # store must never download-then-untar under the quota), reaping the
  # partial store root between bounded retries. A completed stage writes
  # issue-2388-staged-<root>.ok and re-entries skip.
  local rel="$1" dest="$2" root url want free n got tarball
  root="$(basename "$rel" .tar)"
  if [ -f "$LOGDIR/issue-2388-staged-$root.ok" ]; then
    echo "[stage] stream-extract $rel SKIP (staged-$root.ok)"
    return 0
  fi
  url="$HF_DATA_URL/$rel"
  want=$(curl -sSIL -H "Authorization: Bearer $HF_TOKEN" "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  [ -n "$want" ] || { echo "[stage] no content-length for $rel"; return 1; }
  free=$(df -B1 --output=avail /workspace | tail -1 | tr -d ' ')
  if [ "$free" -gt "$((want * 22 / 10))" ]; then
    tarball="/workspace/$(basename "$rel").dl"
    echo "[stage] fetch $rel -> $tarball ($want bytes; resumable) start $(date -u +%H:%M:%SZ)"
    n=0
    while :; do
      got=$(stat -c%s "$tarball" 2>/dev/null || echo 0)
      [ "$got" = "$want" ] && break
      if curl -sSfL -C - -H "Authorization: Bearer $HF_TOKEN" "$url" -o "$tarball"; then break; fi
      n=$((n+1))
      [ "$n" -ge 40 ] && { echo "[stage] fetch retries exhausted ($n) for $rel"; return 1; }
      echo "[stage] fetch retry $n for $rel (have $(stat -c%s "$tarball" 2>/dev/null || echo 0)/$want bytes)"
      sleep 15
    done
    got=$(stat -c%s "$tarball")
    [ "$got" = "$want" ] || { echo "[stage] size mismatch $got != $want for $rel"; return 1; }
    echo "[stage] extract $tarball -> $dest start $(date -u +%H:%M:%SZ)"
    tar -xf "$tarball" -C "$dest"
    rm -f "$tarball"
  else
    n=0
    while :; do
      echo "[stage] stream-extract $rel -> $dest start $(date -u +%H:%M:%SZ) (attempt $((n+1)))"
      if curl -sSfL -H "Authorization: Bearer $HF_TOKEN" "$url" | tar -x -C "$dest"; then
        break
      fi
      n=$((n+1))
      [ "$n" -ge 4 ] && { echo "[stage] stream-extract $rel FAILED after $n attempts"; return 1; }
      echo "[stage] stream-extract $rel attempt $n failed; reaping partial $dest/$root"
      rm -rf "${dest:?}/${root:?}"
      sleep 30
    done
  fi
  touch "$LOGDIR/issue-2388-staged-$root.ok"
  echo "[stage] stream-extract $rel done $(date -u +%H:%M:%SZ)"
}

# ------------------------------------------- (i) U-store + derived QA DV ---
if phase_done p4-stage-u-done; then echo "[phase=p4_stage_u] SKIP (done-sentinel)"; else
echo "[phase=p4_stage_u]"
headroom 20 p4-stage-u
uv run python -c "
from explore_persona_space.experiments.issue_1739.store_io import stage_u_store
print('[stage] u_store ->', stage_u_store('/workspace/u_store'))
"
uv run python scripts/issue2388_dv_build.py --from-banked
commit_results "issue #2388: P4 derived QA correctness DV (dv := fractions.correct)" \
  eval_results/issue_2388/dv/qa
sentinel p4-stage-u-done "P4(i): u_store staged (12.76 GB) + derived QA DV built from the banked #1739 labeling.json"
fi

# ---------------------------------- (i) shared f_U=0 map (Pod C dependency) ---
if phase_done p4-shared-map-done; then echo "[phase=p4_maps_shared] SKIP (done-sentinel)"; else
echo "[phase=p4_maps_shared]"
# G2 device-domain smoke first (production device class, tiny counts), then
# the production shared-map fit + upload so Pod C's one cross-pod read lands.
uv run python scripts/issue2388_fits.py --smoke --phase maps --surface qa --keys linear__shared__fu0 --device cuda
uv run python scripts/issue2388_fits.py --phase maps --surface qa --keys linear__shared__fu0 mlp__shared__fu0 --device cuda
uv run python scripts/issue2388_fits.py --phase upload
commit_results "issue #2388: P4 shared f_U=0 map diagnostics" eval_results/issue_2388/maps
sentinel p4-shared-map-done "P4(i): shared f_U=0 map fit (linear+mlp) + uploaded to HF — Pod C dependency landed"
fi

# ------------------------- (ii) H3 stage 1: sycophancy (pilot-gated), evil ---
if phase_done p4-h3-syco-done; then echo "[phase=p4_h3_sycophancy] SKIP (done-sentinel)"; else
echo "[phase=p4_h3_sycophancy]"
headroom 60 p4-h3-sycophancy
stream_tar issue1739_ctxmap/capture_store/sycophancy_labeling/sycophancy_labeling.tar /workspace/h3_stores
stream_tar issue1739_ctxmap/capture_store/sycophancy_extraction/sycophancy_extraction.tar /workspace/h3_stores
uv run python scripts/issue2388_fits.py --phase h3 --h3-step stage1 --behaviors sycophancy --device cuda
rm -rf /workspace/h3_stores/sycophancy_labeling /workspace/h3_stores/sycophancy_extraction
commit_results "issue #2388: P4 H3 stage-1 recompute — sycophancy (pilot-gated)" "$H3_RECOMPUTE_ROOT"
sentinel p4-h3-syco-done "P4(ii): sycophancy stage-1 recompute done (pilot_wall.json + all_arms_spearman.json); stores reaped"
fi

if phase_done p4-h3-evil-done; then echo "[phase=p4_h3_evil] SKIP (done-sentinel)"; else
echo "[phase=p4_h3_evil]"
headroom 40 p4-h3-evil
stream_tar issue1739_ctxmap/capture_store/evil_labeling/evil_labeling.tar /workspace/h3_stores
stream_tar issue1739_ctxmap/capture_store/evil_extraction/evil_extraction.tar /workspace/h3_stores
uv run python scripts/issue2388_fits.py --phase h3 --h3-step stage1 --behaviors evil --device cuda
rm -rf /workspace/h3_stores/evil_labeling /workspace/h3_stores/evil_extraction
commit_results "issue #2388: P4 H3 stage-1 recompute — evil" "$H3_RECOMPUTE_ROOT"
sentinel p4-h3-evil-done "P4(ii): evil stage-1 recompute done; stores reaped"
fi

# ------------------- (iii) QA store stream-stage + hallucination stage 1 ---
echo "[phase=p4_stage_qa]"
headroom 75 p4-stage-qa
stream_tar issue1739_ctxmap/capture_store/hallucination_labeling/hallucination_labeling.tar /workspace/store
stream_tar issue1739_ctxmap/capture_store/hallucination_extraction/hallucination_extraction.tar /workspace/h3_stores
# Layout probe: the tars carry their store name as the root dir (leg2 shape);
# the labeled-store root is whichever level holds the store payload.
if [ -d /workspace/store/hallucination_labeling ]; then
  QA_STORE_DIR=/workspace/store/hallucination_labeling
else
  QA_STORE_DIR=/workspace/store
fi
echo "[stage] QA_STORE_DIR=$QA_STORE_DIR"
find /workspace/h3_stores/hallucination_extraction -maxdepth 1 -type f | head -1 | grep -q . \
  || { echo "[stage] hallucination_extraction extracted empty"; exit 1; }
if phase_done p4-h3-hallu-done; then echo "[phase=p4_h3_hallucination] SKIP (done-sentinel)"; else
echo "[phase=p4_h3_hallucination]"
uv run python scripts/issue2388_fits.py --phase h3 --h3-step stage1 --behaviors hallucination --device cuda --qa-store-dir "$QA_STORE_DIR"
commit_results "issue #2388: P4 H3 stage-1 recompute — hallucination" "$H3_RECOMPUTE_ROOT"
sentinel p4-h3-hallu-done "P4(iii): hallucination stage-1 recompute done (co-resident with the QA store)"
fi

if phase_done p4-h3-verdict; then echo "[phase=p4_h3_verdict] SKIP (done-sentinel)"; else
echo "[phase=p4_h3_verdict]"
uv run python scripts/issue2388_fits.py --phase h3 --h3-step verdict --device cuda --qa-store-dir "$QA_STORE_DIR"
commit_results "issue #2388: P4 H3 stage-1 verdict recorded (MF-I ordering: before any stage-2 read)" "$H3_RECOMPUTE_ROOT"
sentinel p4-h3-verdict "P4: H3 stage-1 parent-side verdict RECORDED for all three behaviors"
fi

# --------------------------------- (v)+(vi) QA maps + capped primary sweep ---
if phase_done p4-qa-maps-done; then echo "[phase=p4_qa_maps] SKIP (done-sentinel)"; else
echo "[phase=p4_qa_maps]"
uv run python scripts/issue2388_fits.py --smoke --phase sweep --surface qa --map-cell fu0 --device cuda --qa-store-dir "$QA_STORE_DIR"
uv run python scripts/issue2388_fits.py --phase maps --surface qa --device cuda \
  --keys linear__qa__fu0 linear__qa__fu05 linear__qa__fu1 linear__qa__additive \
         mlp__qa__fu0 mlp__qa__fu05 mlp__qa__fu1 mlp__qa__additive
uv run python scripts/issue2388_fits.py --phase upload
commit_results "issue #2388: P4 QA map fits + diagnostics" eval_results/issue_2388/maps
sentinel p4-qa-maps-done "P4(v): QA f_U map cells fit + uploaded"
fi

echo "[phase=p4_stage_qa_shards]"
uv run python - <<'PY'
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
repo = "superkaiba1/explore-persona-space-data"
prefix = "issue1739_ctxmap/raw_completions"
dest = Path("/workspace/qa_shards"); dest.mkdir(parents=True, exist_ok=True)
names = [e.path for e in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=prefix)
         if e.path.rsplit("/", 1)[-1].startswith("labeling_hallucination.shard")
         and e.path.endswith(".jsonl")]
if not names:
    raise SystemExit(f"no labeling_hallucination.shard*.jsonl under {prefix}")
for n in sorted(names):
    p = hf_hub_download(repo, n, repo_type="dataset", local_dir="/workspace/qa_shards_dl")
    tgt = dest / Path(n).name
    if not tgt.exists():
        tgt.hardlink_to(p)
print(f"[stage] qa shards staged: {len(names)} -> {dest}")
PY

if phase_done p4-qa-sweep-done; then echo "[phase=p4_qa_sweep] SKIP (done-sentinel)"; else
echo "[phase=p4_qa_sweep]"
uv run python scripts/issue2388_fits.py --phase sweep --surface qa --device cuda \
  --qa-store-dir "$QA_STORE_DIR" --qa-questions-shards /workspace/qa_shards
for cell in fu0 fu05 additive; do
  echo "[p4] f_U cell sweep: $cell"
  uv run python scripts/issue2388_fits.py --phase sweep --surface qa --map-cell "$cell" --device cuda \
    --qa-store-dir "$QA_STORE_DIR" --qa-questions-shards /workspace/qa_shards
done
uv run python scripts/issue2388_fits.py --phase sweep --surface qa --qa-disjoint --device cuda \
  --qa-store-dir "$QA_STORE_DIR" --qa-questions-shards /workspace/qa_shards
commit_results "issue #2388: P4 QA capped L-sweep + f_U cells + disjoint variant" eval_results/issue_2388/fits/qa
sentinel p4-qa-sweep-done "P4(vi): QA primary capped sweep + fu0/fu05/additive cells + label-disjoint variant done"
fi

# ------------------------------------------- (vii) H3 correctness stage 2 ---
if phase_done p4-h3-stage2-done; then echo "[phase=p4_h3_stage2] SKIP (done-sentinel)"; else
echo "[phase=p4_h3_stage2]"
uv run python scripts/issue2388_fits.py --phase h3 --h3-step stage2 --device cuda --qa-store-dir "$QA_STORE_DIR"
commit_results "issue #2388: P4 H3 correctness-side stage-2 (capped 2500 + legacy 8000/16000 companions)" \
  eval_results/issue_2388/fits/qa "$H3_RECOMPUTE_ROOT"
sentinel p4-h3-stage2-done "P4(vii): H3 correctness side done at all three anchors (h3_parent_exact labelled)"
fi

# ------------------------------------ (viii) select + bootstrap + h3-gap ---
if phase_done p4-done; then echo "[phase=p4_aggregate] SKIP (done-sentinel)"; else
echo "[phase=p4_aggregate]"
uv run python scripts/issue2388_fits.py --phase select --surface qa --device cuda --qa-store-dir "$QA_STORE_DIR"
uv run python scripts/issue2388_fits.py --phase bootstrap --surface qa --device cuda --qa-store-dir "$QA_STORE_DIR"
uv run python scripts/issue2388_fits.py --phase h3-gap --h3-out-root "$H3_RECOMPUTE_ROOT" --device cuda
uv run python scripts/issue2388_fits.py --phase upload
commit_results "issue #2388: P4 QA all_arms + bootstrap + H3 gap report" \
  eval_results/issue_2388/fits/qa "$H3_RECOMPUTE_ROOT"
sentinel p4-done "P4 complete: QA fits + H3 two-stage recompute + gap report committed and uploaded"
fi

echo "[phase=done]"
