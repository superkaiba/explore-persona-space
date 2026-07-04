#!/usr/bin/env bash
# Issue #958 dispatcher — corpus → rollouts → capture → upload (GPU stage);
# stage-inputs → fits → evals → plots → upload (CPU stage); sentinel; [phase=done].
#
# ONE code path for smoke and production (plan §4.7 PASS_UNIFIED): `--smoke`
# scales the CORPUS (200 main / 24 long conversations) through the SAME python
# entrypoints; every later phase enumerates its units from the artifacts the
# previous phase wrote (corpus manifest → rollout shards → store shards → eval
# JSONs), never from a registered full grid. No forks.
#
# Smoke is FULLY namespace-isolated (r2 fix): local dirs default to the
# *_smoke tree AND the HF prefix redirects to issue958_multiturn/smoke, so a
# same-provision smoke → production sequence shares NOTHING. Resume is
# corpus-identity-keyed: the corpus manifest carries a fingerprint threaded
# through rollout-shard regimes + store shards + the fit-resume manifest; the
# corpus skip branch below additionally asserts the existing manifest's SCALE
# matches the requested scale (never a silent make_split clamp).
#
# Pod-side contract (poll_pipeline.py): [phase=<name>] breadcrumbs per phase;
# the results sentinel is written BEFORE the single terminal [phase=done].
# vLLM (rollouts) and HF capture run as SEPARATE processes (plan §8 teardown).
#
# Stages: --stage gpu (corpus+rollouts+capture+upload, the capture-7b A100
# provision) | --stage cpu (stage-inputs+fits+evals+plots+upload, cpu-mid) |
# --stage all.
#
# Env overrides (all optional):
#   EPM958_CORPUS/ROLLOUTS/STORE/CACHE/MAPS/OUT/FIGS   — dirs
#   EPM958_MODEL                                        — model override
#   EPM958_MOCK_ROLLOUTS=1                              — VM smoke: no vLLM
#   EPM958_STUB_MODEL=1                                 — VM smoke: tiny Qwen2
#   EPM958_DEVICE                                       — fit device override
#   EPM958_SKIP_UPLOAD=1                                — skip HF uploads
#   EPM958_HF_PREFIX                                    — smoke: issue958_multiturn/smoke
#   EPM958_FORCE_EMPTY_UIDS                             — smoke fault injection (mock only)
#   EPM958_STORE_KILL_GB                                — §7 store-size kill (default 60)
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

# Namespace isolation (r2): smoke defaults to a *_smoke tree everywhere —
# local dirs AND the HF prefix — so production dirs are never touched.
if [ "$SMOKE" = "1" ]; then
  DATA_ROOT="data/issue_958_smoke"
  OUT_DEFAULT="eval_results/issue_958_smoke"
  FIGS_DEFAULT="figures/issue_958_smoke"
  export EPM958_HF_PREFIX="${EPM958_HF_PREFIX:-issue958_multiturn/smoke}"
else
  DATA_ROOT="data/issue_958"
  OUT_DEFAULT="eval_results/issue_958"
  FIGS_DEFAULT="figures/issue_958"
fi
CORPUS="${EPM958_CORPUS:-$DATA_ROOT/corpus}"
ROLLOUTS="${EPM958_ROLLOUTS:-$DATA_ROOT/rollouts}"
STORE="${EPM958_STORE:-$DATA_ROOT/store}"
CACHE="${EPM958_CACHE:-$DATA_ROOT/fit_cache}"
MAPS="${EPM958_MAPS:-$DATA_ROOT/maps}"
OUT="${EPM958_OUT:-$OUT_DEFAULT}"
FIGS="${EPM958_FIGS:-$FIGS_DEFAULT}"
MODEL="${EPM958_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
echo "[dirs] smoke=$SMOKE stage=$STAGE corpus=$CORPUS rollouts=$ROLLOUTS store=$STORE" \
  "cache=$CACHE maps=$MAPS out=$OUT figs=$FIGS hf_prefix=${EPM958_HF_PREFIX:-issue958_multiturn}"

# ONE smoke subset definition (the corpus), threaded to EVERY phase below.
CORPUS_ARGS=()
ROLLOUT_ARGS=()
CAPTURE_ARGS=()
FIT_ARGS=()
if [ "$SMOKE" = "1" ]; then
  # pod smoke = the plan's 200-conversation end-to-end run; the VM structural
  # smoke may shrink further via EPM958_SMOKE_N_MAIN/N_LONG (same code path)
  REQ_N_MAIN="${EPM958_SMOKE_N_MAIN:-200}"
  REQ_N_LONG="${EPM958_SMOKE_N_LONG:-24}"
  CORPUS_ARGS=(--n-main "$REQ_N_MAIN" --n-long "$REQ_N_LONG" --stream-limit 120000)
else
  REQ_N_MAIN=""  # production requests = issue958_common defaults (resolved in-check)
  REQ_N_LONG=""
fi
if [ "${EPM958_MOCK_ROLLOUTS:-0}" = "1" ]; then ROLLOUT_ARGS+=(--mock-generate); fi
if [ -n "${EPM958_FORCE_EMPTY_UIDS:-}" ]; then
  ROLLOUT_ARGS+=(--force-empty-uids "$EPM958_FORCE_EMPTY_UIDS")
fi
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
  # timing-probe gate + one bulk upload_folder commit + scoped-listing verify;
  # the probe upload_file is retried on transient 5xx (issue958_common)
  EPM958_UPLOAD_DIR="$1" EPM958_UPLOAD_SUFFIX="$2" EPM958_UPLOAD_MSG="$3" \
    uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue958_common as C

ev = C.upload_with_timing_gate(
    Path(os.environ["EPM958_UPLOAD_DIR"]),
    os.environ["EPM958_UPLOAD_SUFFIX"],
    os.environ["EPM958_UPLOAD_MSG"],
)
print(f"[upload] {ev}")
PY
}

if [ "$STAGE" = "gpu" ] || [ "$STAGE" = "all" ]; then
  echo "[phase=verify_fits]"
  uv run python scripts/issue958_fit_maps.py --verify-fits

  echo "[phase=corpus]"
  # self-build on a fresh instance: gitignored data never travels with the clone.
  # An EXISTING corpus must match the REQUESTED scale + carry a fingerprint —
  # never a silent skip that lets make_split clamp production to smoke scale (r2).
  if [ ! -f "$CORPUS/manifest.json" ]; then
    uv run python scripts/issue958_build_corpus.py --out "$CORPUS" "${CORPUS_ARGS[@]}"
  else
    EPM958_MANIFEST="$CORPUS/manifest.json" EPM958_REQ_MAIN="$REQ_N_MAIN" \
      EPM958_REQ_LONG="$REQ_N_LONG" uv run python - <<'PY'
import json
import os
import sys

m = json.load(open(os.environ["EPM958_MANIFEST"]))
req_main_s, req_long_s = os.environ.get("EPM958_REQ_MAIN"), os.environ.get("EPM958_REQ_LONG")
if req_main_s:
    req_main, req_long = int(req_main_s), int(req_long_s)
else:  # production requests = the registered defaults
    sys.path.insert(0, "scripts")
    import issue958_common as C

    req_main, req_long = C.N_MAIN, C.N_LONG
fp = m.get("corpus_fingerprint")
assert fp, (
    "STALE CORPUS: existing manifest has no corpus_fingerprint (pre-fingerprint build) — "
    "delete the corpus dir and rebuild."
)
assert m["n_main"] == req_main and m["n_long"] <= req_long, (
    f"STALE CORPUS SCALE: existing corpus has n_main={m['n_main']} n_long={m['n_long']} but "
    f"this run requests n_main={req_main} n_long<={req_long}. Refusing the silent reuse — "
    "point EPM958_* at the matching dirs (smoke lives under data/issue_958_smoke) or delete "
    "the stale corpus."
)
print(f"[corpus] exists + scale-valid (fp={fp[:12]}) — skip (resume)")
PY
  fi

  echo "[phase=rollouts]"
  uv run python scripts/issue958_rollouts.py --corpus "$CORPUS" --out "$ROLLOUTS" \
    --model "$MODEL" "${ROLLOUT_ARGS[@]}"

  echo "[phase=capture]"
  uv run python scripts/issue958_capture_turns.py --corpus "$CORPUS" --rollouts "$ROLLOUTS" \
    --out "$STORE" --model "$MODEL" "${CAPTURE_ARGS[@]}"

  # plan §7 storage-overrun kill, SIZE branch: realized store > 60 GB (vs ~27
  # planned) → halt uploads, keep shards pod-side, report (r1 Minor 1).
  EPM958_STORE_SUMMARY="$STORE/capture_summary.json" uv run python - <<'PY'
import json
import os

s = json.load(open(os.environ["EPM958_STORE_SUMMARY"]))
gb = s["realized_store_bytes"] / 1e9
kill_gb = float(os.environ.get("EPM958_STORE_KILL_GB", "60"))
assert gb <= kill_gb, (
    f"STORAGE-OVERRUN KILL (plan §7): realized store {gb:.1f} GB > {kill_gb:.0f} GB budget — "
    "halting uploads; shards kept pod-side; report before continuing (never delete "
    "unuploaded artifacts)."
)
print(f"[store-size-gate] realized store {gb:.2f} GB <= {kill_gb:.0f} GB PASS")
PY

  echo "[phase=upload_gpu]"
  upload_dir "$CORPUS" "corpus" "issue958 corpus"
  upload_dir "$ROLLOUTS" "raw_completions/rollouts" "issue958 rollout text"
  upload_dir "$STORE" "analysis_tensors/store" "issue958 activation store (fp16 shards)"
fi

if [ "$STAGE" = "cpu" ] || [ "$STAGE" = "all" ]; then
  echo "[phase=stage_inputs]"
  # fresh cpu-mid provision has no data/ (gitignored): stage corpus + store
  # from HF (scoped list_repo_tree + per-file hf_hub_download — never a
  # full-tree snapshot_download on the ~1M-file repo; r2 Critical 2).
  if [ -f "$CORPUS/manifest.json" ] && [ -f "$STORE/capture_summary.json" ]; then
    echo "[stage_inputs] corpus + store present locally — skip"
  else
    EPM958_CORPUS_DIR="$CORPUS" EPM958_STORE_DIR="$STORE" uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue958_common as C

ev = C.stage_inputs_from_hf(
    corpus_dir=Path(os.environ["EPM958_CORPUS_DIR"]),
    store_dir=Path(os.environ["EPM958_STORE_DIR"]),
)
print(f"[stage_inputs] staged {ev['n_files']} files from {ev['prefix']}")
PY
  fi
  # existence verification either way (fail loud BEFORE the ~3h fit phase)
  [ -f "$CORPUS/manifest.json" ] || { echo "[stage_inputs] FATAL: corpus manifest missing" >&2; exit 4; }
  [ -f "$STORE/capture_summary.json" ] || { echo "[stage_inputs] FATAL: store capture_summary missing" >&2; exit 4; }

  echo "[phase=fits]"
  uv run python scripts/issue958_fit_maps.py --corpus "$CORPUS" --store "$STORE" \
    --cache "$CACHE" --maps "$MAPS" --out "$OUT" "${FIT_ARGS[@]}"

  # plan §7 smoke kill (map-skill floor): the --smoke run must produce turn-1
  # context-map skill > 0 at the frozen 6-block mean (real model only — the
  # stub-model VM smoke has no meaningful skill).
  if [ "$SMOKE" = "1" ] && [ "${EPM958_STUB_MODEL:-0}" != "1" ]; then
    EPM958_OUT_DIR="$OUT" uv run python - <<'PY'
import json
import os
from pathlib import Path

t = json.loads((Path(os.environ["EPM958_OUT_DIR"]) / "transfer_matrix.json").read_text())
s = t["grid_skill_readout_mean_foldA"]["1->1"]
assert s > 0, (
    f"SMOKE KILL (plan §7): turn-1 context-map skill {s:.4f} <= 0 — "
    "template/boundary/pairing bug; no full GPU launch until fixed"
)
print(f"[smoke-gate] turn-1 skill {s:.4f} > 0 PASS")
PY
  fi

  echo "[phase=evals]"
  uv run python scripts/issue958_eval.py --out "$OUT"

  echo "[phase=plots]"
  uv run python scripts/issue958_plots.py --results "$OUT" --out "$FIGS"

  echo "[phase=upload_cpu]"
  upload_dir "$MAPS" "analysis_tensors/maps" "issue958 fitted map weights (7 rows fp16)"
  # eval_results + figures MUST land on HF before the GCE instance DELETE —
  # the per-draw null matrices + per-unit SSE npz are plan-referenced
  # downstream inputs (#521 class; r1 Major 3).
  upload_dir "$OUT" "eval_results" "issue958 eval results (percell + null matrices + headline JSONs)"
  upload_dir "$FIGS" "figures" "issue958 figures"
fi

echo "[phase=sentinel]"
# epm:results ONLY when this invocation produced eval results (cpu|all, non-
# smoke); a gpu-only stage reports progress (r1 Codex sweep: no results
# sentinel before eval JSONs exist).
SENTINEL_KIND="epm:progress"
if [ "$SMOKE" = "0" ] && { [ "$STAGE" = "cpu" ] || [ "$STAGE" = "all" ]; }; then
  SENTINEL_KIND="epm:results"
fi
EPM958_SENTINEL_KIND="$SENTINEL_KIND" EPM958_OUT_DIR="$OUT" EPM958_FIGS_DIR="$FIGS" \
  EPM958_SMOKE="$SMOKE" EPM958_STAGE="$STAGE" uv run python - <<'PY'
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
    "figures_dir": os.environ["EPM958_FIGS_DIR"],  # the ACTIVE namespace (r3: smoke != prod)
    "hf_prefix": C.HF_OUT_PREFIX,
    "transfer_standardization_policy": C.TRANSFER_STANDARDIZATION_POLICY,
    "note": "issue #958 multi-turn context->answer mapping run: per-turn maps, "
    "own-vs-stale transfer matrix, forecasts, prefix dominance, drift reads.",
}
C.write_results_sentinel(note, kind=os.environ["EPM958_SENTINEL_KIND"], version=1)
PY

echo "[phase=done]"
