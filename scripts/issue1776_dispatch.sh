#!/usr/bin/env bash
# #1776 run-node orchestration: sequences the phase scripts per plan §9 (v4).
#
# Workload command (plan §10):
#   REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" bash scripts/issue1776_dispatch.sh --mode full
#   (smoke: --mode smoke — SAME chain / SAME width / real model, tiny slices)
#
# Phase sequence (== EXPECTED_ORDER below; the dry-run asserts the trace):
#   p0_stage             stage.py (pin + probes + provenance + pass_b/weights/r_b/
#                        manifest/parity chunks) + #779 trait artifacts + centroids
#   p5a_stream_launch    phase5 stream (CPU/network) launched BG — §9: concurrent
#   p0_parity            G-PARITY gate (rc=8 -> HALT: every downstream number
#                        consumes the same captured fields)
#   p0_comparator_launch m_ridge_x50k + m_ridge_lmsys50k BG on the last GPU
#                        (§9: P0.5 runs concurrent with P0.1's tail)
#   p0_jlens             build-prompts; fit fan-out (GPUs 0..N-2); merge; G-LENS
#                        sanity (rc=8 -> lens-dependent legs skipped, final rc=8)
#   p0_comparator_join   join the comparator fits
#   p0_dict              phase4 build-dict L14/L19/L21 (subprocess — its main()
#                        ends in sys.exit; NEVER import-call it in-process)
#   p04_pairs            J-pair manifest (1,536 LMSYS train-pool pairs; text from
#                        the #779 raw_completions chunks at the pin) + sharded
#                        teacher-forced capture (parity's recompute_row rig) ->
#                        v_pool / acts14 / acts19
#   p1_contexts          contexts builder (needs the staged #779 trait artifacts)
#   p2a_sketch           build-seeds; jacobian run --mode sketch fan-out; merge
#                        (G-NONZERO fires inside run: rc=8 -> HALT)
#   p1_diag              phase1 directional diagnostic
#   p_early_upload       §9 expensive-store-before-long-fit (#825): everything
#                        already produced uploads BEFORE the long P2b sweep
#   p2b_full             jacobian run --mode full (3,584 seeds, seed-block shard
#                        fan-out); merge -> J_{prefix,ctx,last} + even/odd halves
#   p2_upload            J tensors + halves upload (before P2c reads / P4)
#   p3_grid              phase3 baseline-only -> steered strata fan-out -> finalize
#   p3_upload            steered rollout TEXT -> raw_completions/steered (before
#                        phase4 + judging), summaries/manifest -> analysis_tensors
#   p4_mediation         phase4 energy / refit-split / jdelta-split (lens-gated)
#   p5a_stream_join      join the WildChat stream
#   p5a_capture          phase5 capture fan-out (self-uploads batched to HF)
#   p5_transfer          phase5 transfer --assemble (test-1000 leg + anchors) +
#                        ops (M' x50k/lmsys50k, shipped-M reference if resolvable)
#                        + J arms over lmsys_test1000 + wildchat_fresh (P2c + P5)
#   p5b_leakage          phase5 leakage re-read (CPU; inputs all local by now —
#                        §9 lists this off-pod (p7); run here since the dispatcher
#                        stages centroids + builds the L21 dict anyway; deviation
#                        noted in the final sentinel)
#   p_results_commit     git add/commit/push eval JSONs + rev-list push-verify +
#                        per-file ls-tree artifact-presence assert (#1205/#1325)
#   p_final              epm:results (or epm:smoke-result) sentinel, then
#                        [phase=done]
#
# OFF-POD (excluded here; named in the final sentinel note):
#   p6 graded judge  — Batch API on the VM after release (issue1776_judge.py).
#     Pricing note: the judge DEFAULT scores the control strata under ALL 3
#     rubrics (~30k x 5 calls) vs §9's 16k x 5 one-rubric-per-completion
#     costing — trim via --control-rubrics if the budget binds.
#   p7 final analyses — VM, 0 GPU (5c/5d lens reads etc.).
#
# Engineering-gate exit codes (plan §7): 8 = G-PARITY / G-LENS / G-NONZERO
# halt (gate-report sentinel written; NOT a crash); 7 = G-PILOT (phase scripts
# exit 7 themselves; routed through the same designed-halt path).
#
# Progress reporting (pod-side-reporting rule): [phase=...] lines on THIS
# script's stdout only (phase scripts are redirected to per-phase logs, so the
# reserved [phase=done] token never leaks); per-phase tick sentinels
# /workspace/logs/issue-1776-phase-*.json (kind epm:progress, gate name carries
# "phase" -> the drain posts them verbatim; write-once, never re-read — state
# lives under $OUT_ROOT); ONE terminal results sentinel before [phase=done].
#
# Dry-run (VM smoke): --dry-run (or EPS_1776_DRY_RUN=1) traces the phase
# sequence + writes REAL progress/final sentinels into a /tmp root (no GPU, no
# Hub, no model), then asserts the trace against EXPECTED_ORDER and round-trips
# every sentinel through the required-keys parse.

set -euo pipefail

# ── args / mode ───────────────────────────────────────────────────────────────
MODE="full"
DRY=0
NGPU_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --gpus) NGPU_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" == "full" || "$MODE" == "smoke" ]] || { echo "--mode full|smoke" >&2; exit 2; }
if [[ "${EPS_1776_DRY_RUN:-0}" == "1" ]]; then DRY=1; fi
export EPS_1776_MODE="$MODE"

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"
export UV_NO_SYNC="${UV_NO_SYNC:-1}"  # plan §8: no concurrent .venv syncs on fan-out
ISSUE=1776

if [[ $DRY == 1 ]]; then
  TMP_BASE="${EPS_1776_TMP:-$(mktemp -d /tmp/issue-1776-dryrun.XXXXXX)}"
  OUT_ROOT="${EPS_1776_OUT_ROOT:-$TMP_BASE/out}"
  LOG_DIR="${EPS_1776_LOG_DIR:-$TMP_BASE/logs}"
else
  OUT_ROOT="${EPS_1776_OUT_ROOT:-/workspace/issue_1776}"
  LOG_DIR="${EPS_1776_LOG_DIR:-/workspace/logs}"
  if [[ -d /workspace ]]; then export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"; fi
fi
DATA_DIR="$REPO_ROOT/data/issue_1776"
PHASE_LOGS="$OUT_ROOT/logs"
TRACE="$OUT_ROOT/dispatch_trace.txt"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$PHASE_LOGS" "$DATA_DIR"
: > "$TRACE"

# Smoke outputs never touch committed eval_results/ (scratch-dir redirect);
# smoke Hub uploads go to the smoke_probe/ scratch prefix, never canonical.
if [[ "$MODE" == "smoke" || $DRY == 1 ]]; then
  EVAL_DIR="$OUT_ROOT/eval_results/issue_1776"
  HF_PREFIX_EFF="issue1776_jacobian/smoke_probe"
else
  EVAL_DIR="$REPO_ROOT/eval_results/issue_1776"
  HF_PREFIX_EFF="issue1776_jacobian"
fi
mkdir -p "$EVAL_DIR"

# ── realized width (§9: the workload re-shards off realized width) ────────────
if [[ -n "$NGPU_OVERRIDE" ]]; then
  NGPU="$NGPU_OVERRIDE"
elif [[ $DRY == 1 ]]; then
  NGPU="${EPS_1776_NGPU:-8}"
else
  NGPU="$(nvidia-smi --list-gpus | wc -l)"
fi
[[ "$NGPU" -ge 1 ]] || { echo "[dispatch] no GPUs visible" >&2; exit 1; }

# ── mode parameter table ──────────────────────────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
  PARITY_CHUNKS=1;  PARITY_ROWS=8
  JLENS_N=4;        JLENS_LIMIT=(--limit 4)
  N_PAIRS=8
  SEEDS_TOTAL=12;   SEEDS_TOPK=4;  SEEDS_GAUSS=2
  SKETCH_LIMIT=2;   FULL_M=1;      FULL_LIMIT=2
  P1_LIMIT=2;       P1_TOPK=4
  N_TRAIN=3600      # comparator/refit assert n_train > d=3584; round-1 lmsys train pool is 3,600
  CTX_FLAGS=(--smoke)
  P3_EXTRA=(--limit-contexts 2 --k-samples 1 --k-baseline 1 --alphas 4)
  WC_KEEP=2;        WC_EXTRA=(--allow-short); CAP_EXTRA=(--max-rows 2)
  NBOOT=50;         NDRAWS=8;      ASSEMBLE_EXTRA=(--max-chunks 1); COMP_EXTRA=(--max-chunks 1)
else
  PARITY_CHUNKS=4;  PARITY_ROWS=200
  JLENS_N=1000;     JLENS_LIMIT=()
  N_PAIRS=1536
  SEEDS_TOTAL=256;  SEEDS_TOPK=20; SEEDS_GAUSS=8
  SKETCH_LIMIT=512; FULL_M=150;    FULL_LIMIT=0
  P1_LIMIT=1024;    P1_TOPK=20
  N_TRAIN=50000
  CTX_FLAGS=()
  P3_EXTRA=()
  WC_KEEP=1000;     WC_EXTRA=();   CAP_EXTRA=()
  NBOOT=1000;       NDRAWS=100;    ASSEMBLE_EXTRA=(); COMP_EXTRA=()
fi

# ── shared paths ──────────────────────────────────────────────────────────────
HF_DL="$DATA_DIR/hf_dl"
PASS_B="$HF_DL/issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
MANIFEST_DIR="$HF_DL/issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest"
CHUNKS_DIR="$HF_DL/issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
RAW_DIR="$HF_DL/issue779_monitoring/fitter-fair-comparison-n1m/raw_completions"
RB_DIR="$HF_DL/issue779_monitoring/r_b"
WEIGHTS_DIR="$HF_DL/issue779_monitoring/n1m_readout/weights"
MM_DIR="$DATA_DIR/n1m_mm"
COMP_DIR="$DATA_DIR/comparator"
JLENS_DIR="$DATA_DIR/jlens"
DICT_DIR="$DATA_DIR/dict"
JPAIRS_DIR="$DATA_DIR/jpairs"
CTX_JSONL="$DATA_DIR/contexts/contexts.jsonl"
SKETCH_ROOT="$OUT_ROOT/jac_sketch"
FULL_ROOT="$OUT_ROOT/jac_full"
P3_ROOT="$OUT_ROOT/phase3"
WC_DIR="$DATA_DIR/wildchat_fresh"
WC_CAP_ROOT="$OUT_ROOT/wildchat_fresh"
CENTROIDS="$DATA_DIR/centroids_v1_L21.pt"

CURRENT_PHASE="launch"
GATE_HALTED=0
LENS_OK=1

# ── progress sentinels (write-once; never re-read — pod-side-reporting rule) ──
progress() {  # progress <gate> <msg>   (non-blocking tick)
  local gate="$1" msg="$2"
  uv run python - "$ISSUE" "$LOG_DIR" "$gate" "$msg" <<'PY' || echo "[dispatch] WARN: progress sentinel write failed (gate=$gate)" >&2
import json, os, sys, time
from pathlib import Path

issue, log_dir, gate, msg = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
log_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:progress",
    "version": 1,  # pod-side writers hardcode 1; the VM drain re-derives
    "task_id": int(issue),
    "gate": gate,  # carries "phase" -> the drain posts phase ticks verbatim
    "blocks_pipeline": False,
    "by": "issue1776_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": {"msg": msg, "mode": os.environ.get("EPS_1776_MODE", "?")},
}
slug = gate.replace(":", "_").replace("/", "_")  # gate already carries the phase- prefix
path = log_dir / f"issue-{issue}-{slug}-{int(time.time() * 1000)}.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=1))
os.replace(tmp, path)
print(f"[dispatch] progress sentinel: {path.name}")
PY
}

phase_begin() {
  CURRENT_PHASE="$1"
  echo "[phase=$1]"
  echo "$1" >> "$TRACE"
  # RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking; body is one heredoc python
  progress "phase-$1" "begin (mode=$MODE ngpu=$NGPU)" || true
}
# RC_CAPTURE_EXEMPT: progress ticks are deliberately non-blocking; body is one heredoc python
phase_end() { progress "phase-$1" "done" || true; }

gate_halt() {  # gate_halt <gate-name> <rc> <msg>
  GATE_HALTED=1
  echo "[dispatch] ENGINEERING GATE HALT: $1 rc=$2 — $3" >&2
  # RC_CAPTURE_EXEMPT: gate sentinel is best-effort; the distinct exit rc is the signal
  progress "phase-gate-halt-$1" "rc=$2: $3" || true
  exit "$2"
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && $GATE_HALTED -eq 0 ]]; then
    echo "[dispatch] FAILED at phase=$CURRENT_PHASE rc=$rc" >&2
    # RC_CAPTURE_EXEMPT: best-effort crash breadcrumb inside the EXIT trap; rc already captured
    progress "phase-crash" "phase=$CURRENT_PHASE rc=$rc" || true
  fi
}
trap on_exit EXIT

# ── runners ───────────────────────────────────────────────────────────────────
run() {  # run <log-name> <cmd...>   (foreground, redirected; DRY: trace only)
  local plog="$PHASE_LOGS/$1.log"; shift
  if [[ $DRY == 1 ]]; then
    echo "DRY: $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    return 0
  fi
  echo "[dispatch] run($CURRENT_PHASE): $*"
  "$@" >> "$plog" 2>&1
}

bg_run() {  # bg_run <log-name> <cvd> <cmd...> -> echoes pid ('' in DRY)
  local plog="$PHASE_LOGS/$1.log" cvd="$2"; shift 2
  if [[ $DRY == 1 ]]; then
    echo "DRY: CUDA_VISIBLE_DEVICES=$cvd $*" | tee -a "$OUT_ROOT/dry_cmds.txt" >> "$plog"
    echo ""
    return 0
  fi
  echo "[dispatch] bg($CURRENT_PHASE, CVD=$cvd): $*" >&2
  CUDA_VISIBLE_DEVICES="$cvd" "$@" >> "$plog" 2>&1 &
  echo $!
}

wait_rc() {  # wait_rc <pid-or-empty> -> return the pid's rc (0 for DRY '')
  local p="$1" rc=0
  [[ -z "$p" ]] && return 0
  wait "$p" || rc=$?
  return $rc
}

# ── launch preamble: out-root mount headroom (#1333 pattern) ──────────────────
phase_begin "p0_stage"
if [[ $DRY == 0 ]]; then
  NEED_GB=$([[ "$MODE" == "smoke" ]] && echo 15 || echo 40)
  run headroom uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
free = assert_out_root_headroom('$OUT_ROOT', $NEED_GB, phase='launch')
print(f'[headroom] out_root=$OUT_ROOT free_gb={free:.1f} floor=$NEED_GB')
"
fi

# ── p0_stage: pin + probes + provenance + reused-artifact staging ─────────────
run p0_stage uv run python scripts/issue1776_stage.py \
  --stage-bundle --stage-weights --stage-rb --stage-manifest \
  --parity-chunks "$PARITY_CHUNKS" --report "$DATA_DIR/stage_report.json"

# Handoffs the phase scripts assume staged (gitignored — not in the clone):
#  - #779 trait artifacts for the contexts builder (issue779_common
#    load_extraction_artifacts reads data/issue_779/artifacts/<trait>.json);
#  - the canonical-pool centroids bundle for the 5b leakage leg (sha-asserted
#    against the committed bank meta's built_from.centroids_sha256).
run p0_stage_extra uv run python - "$REPO_ROOT" "$CENTROIDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate import hub

repo_root, centroids_dest = Path(sys.argv[1]), Path(sys.argv[2])
pin = C76.resolve_data_repo_pin()

art_dir = repo_root / "data" / "issue_779" / "artifacts"
for trait in ("sycophancy", "hallucination"):
    dest = art_dir / f"{trait}.json"
    hub.stage_hub_file(
        C76.HF_DATA_REPO,
        f"issue779_monitoring/artifacts/{trait}.json",
        dest,
        repo_type="dataset",
        revision=pin,
    )
    print(f"[stage-extra] trait artifacts staged: {dest}")

# Centroids bundle lives under a DIFFERENT issue prefix (not covered by the
# #1776 pin); content identity is the sha pin in the committed bank meta.
meta = json.loads(
    (repo_root / "data" / "canonical_persona_pool" / "matrix_v1_L21_raw.json").read_text()
)
want_sha = meta["built_from"]["centroids_sha256"]
hub.stage_hub_file(
    C76.HF_DATA_REPO,
    "issue483_canonical_persona_pool/centroids_v1_L21.pt",
    centroids_dest,
    repo_type="dataset",
)
got = hashlib.sha256(centroids_dest.read_bytes()).hexdigest()
assert got == want_sha, f"centroids sha mismatch: got {got} want {want_sha}"
print(f"[stage-extra] centroids staged + sha-verified: {centroids_dest}")
PY
phase_end "p0_stage"

# ── p5a stream: CPU/network-bound, concurrent with GPU work (§9) ──────────────
phase_begin "p5a_stream_launch"
STREAM_PID="$(bg_run p5a_stream "" uv run python scripts/issue1776_phase5.py stream \
  --out-dir "$WC_DIR" --n-keep "$WC_KEEP" ${WC_EXTRA[@]+"${WC_EXTRA[@]}"})"
phase_end "p5a_stream_launch"

# ── p0_parity: G-PARITY (rc=8 -> program halt: everything downstream consumes
#    the same captured fields) ─────────────────────────────────────────────────
phase_begin "p0_parity"
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p0_parity env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_parity.py \
  --chunks-dir "$CHUNKS_DIR" --raw-dir "$RAW_DIR" --n-rows "$PARITY_ROWS" \
  --out-dir "$DATA_DIR/parity" || rc=$?
if [[ $rc -eq 8 ]]; then
  gate_halt "G-PARITY" 8 "parity halt: >5% rows below 0.999 (report: $DATA_DIR/parity)"
elif [[ $rc -eq 7 ]]; then
  gate_halt "G-PILOT-parity" 7 "pilot gate halt in parity phase"
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p0_parity"

# ── p0_comparator (BG on the last GPU) + p0_jlens (fan on the rest) ───────────
phase_begin "p0_comparator_launch"
COMP_GPU=$((NGPU - 1))
comparator_job() {
  uv run python scripts/issue1776_comparator_fit.py --tag m_ridge_x50k \
    --n-train "$N_TRAIN" --out-dir "$COMP_DIR" --pass-b "$PASS_B" --mm-dir "$MM_DIR" \
    ${COMP_EXTRA[@]+"${COMP_EXTRA[@]}"} \
    && uv run python scripts/issue1776_comparator_fit.py --tag m_ridge_lmsys50k \
      --lmsys-only --n-train "$N_TRAIN" --out-dir "$COMP_DIR" --pass-b "$PASS_B" \
      --mm-dir "$MM_DIR" ${COMP_EXTRA[@]+"${COMP_EXTRA[@]}"}
}
if [[ $DRY == 1 ]]; then
  run p0_comparator echo "comparator x50k + lmsys50k (n_train=$N_TRAIN)"
  COMP_PID=""
else
  echo "[dispatch] bg(p0_comparator, CVD=$COMP_GPU): comparator x50k + lmsys50k"
  CUDA_VISIBLE_DEVICES="$COMP_GPU" comparator_job >> "$PHASE_LOGS/p0_comparator.log" 2>&1 &
  COMP_PID=$!
fi
phase_end "p0_comparator_launch"

phase_begin "p0_jlens"
NF=$NGPU
if [[ $NGPU -gt 1 ]]; then NF=$((NGPU - 1)); fi  # comparator holds the last GPU
run p0_jlens_prompts uv run python scripts/issue1776_jlens_fit.py build-prompts \
  --out "$DATA_DIR/jlens_prompts.jsonl" --n "$JLENS_N"
mkdir -p "$JLENS_DIR"
JL_PIDS=()
for ((g = 0; g < NF; g++)); do
  p="$(bg_run "p0_jlens_fit_shard$g" "$g" uv run python scripts/issue1776_jlens_fit.py fit \
    --prompts "$DATA_DIR/jlens_prompts.jsonl" --out "$JLENS_DIR/shard$g.pt" \
    --shard-index "$g" --n-shards "$NF" --checkpoint "$JLENS_DIR/ckpt_shard$g.pt" \
    ${JLENS_LIMIT[@]+"${JLENS_LIMIT[@]}"})"
  JL_PIDS+=("$p")
done
for p in ${JL_PIDS[@]+"${JL_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
if [[ $DRY == 1 ]]; then
  run p0_jlens_merge echo "merge shards -> lens.pt"
else
  shards=("$JLENS_DIR"/shard*.pt)
  run p0_jlens_merge uv run python scripts/issue1776_jlens_fit.py merge \
    --shards "${shards[@]}" --out "$JLENS_DIR/lens.pt"
fi
rc=0
# RC_CAPTURE_EXEMPT: run()'s body is a single payload command whose rc IS the capture target
run p0_jlens_sanity env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_jlens_fit.py \
  sanity --lens "$JLENS_DIR/lens.pt" --out "$EVAL_DIR/phase0/glens_gate.json" || rc=$?
if [[ $rc -eq 8 ]]; then
  # G-LENS: lens-dependent legs (dictionaries, phase4, 5b) are skipped; the
  # program continues (plan §7: gates abort the affected numbers, not the
  # program) and the dispatcher exits 8 at the end.
  LENS_OK=0
  echo "[dispatch] G-LENS FAIL: skipping p0_dict / p4_mediation / p5b_leakage" >&2
  # RC_CAPTURE_EXEMPT: gate sentinel is best-effort; LENS_OK=0 + final rc=8 carry the state
  progress "phase-gate-halt-G-LENS" "rc=8: lens sanity failed; dependent legs skipped" || true
elif [[ $rc -ne 0 ]]; then
  exit "$rc"
fi
phase_end "p0_jlens"

phase_begin "p0_comparator_join"
wait_rc "$COMP_PID" || {
  rc=$?
  if [[ $rc -eq 7 ]]; then gate_halt "G-PILOT-comparator" 7 "pilot gate halt in comparator phase"; fi
  exit "$rc"
}
phase_end "p0_comparator_join"

# ── p0_dict: dictionaries at L14/L19 (+L21 for the 5b leg), lens-gated ────────
phase_begin "p0_dict"
if [[ $LENS_OK -eq 1 ]]; then
  mkdir -p "$DICT_DIR"
  D_PIDS=()
  gi=0
  for L in 14 19 21; do
    p="$(bg_run "p0_dict_l$L" "$((gi % NGPU))" uv run python scripts/issue1776_phase4.py \
      build-dict --lens "$JLENS_DIR/lens.pt" --layer "$L" --out "$DICT_DIR/dictionary_l$L.pt")"
    D_PIDS+=("$p"); gi=$((gi + 1))
  done
  for p in ${D_PIDS[@]+"${D_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
else
  run p0_dict echo "SKIPPED (G-LENS fail)"
fi
phase_end "p0_dict"

# ── p04_pairs: J-pair manifest + sharded teacher-forced capture ───────────────
phase_begin "p04_pairs"
mkdir -p "$JPAIRS_DIR"
run p04_pairs_build uv run python - "$N_PAIRS" "$JPAIRS_DIR" "$MANIFEST_DIR" <<'PY'
"""J-pair manifest (plan §4 P0.4): seeded sample of LMSYS-provenance rows from
the n1m TRAIN pool (new-capture rows only — val/test are round-1 rows, disjoint
by construction), text joined from the #779 raw_completions chunks at the pin.
Chunks are visited in a SEEDED PERMUTATION of the pinned listing (deterministic
given the pin) and downloaded lazily until the quota is covered. Content
hygiene: row text is never printed; the report carries counts + shas only."""

import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

import issue1776_common as C76
import issue779_ffc_n1m_generate_capture as N1G
from huggingface_hub import HfApi
from explore_persona_space.orchestrate import hub

n_pairs, out_dir, manifest_dir = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_path = out_dir / "jpairs.jsonl"
if out_path.exists():
    rows = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    if len(rows) == n_pairs:
        print(f"[jpairs] resume: {out_path} already has {n_pairs} pairs; skip")
        sys.exit(0)
pin = C76.resolve_data_repo_pin()
pool, _meta = N1G.read_manifest_pool(manifest_dir)
lmsys_ci = {int(r["i"]) for r in pool if r.get("corpus") == "lmsys"}
assert lmsys_ci, "no lmsys rows in the sampling manifest"

api = HfApi()
base = "issue779_monitoring/fitter-fair-comparison-n1m/raw_completions"
files = sorted(
    f
    for f in hub.list_hf_files_under_path(
        api, C76.HF_DATA_REPO, base, repo_type="dataset", revision=pin
    )
    if f.endswith(".json") and "skipped" not in Path(f).name
)
assert files, f"no raw chunks under {base}@{pin}"
rng = np.random.default_rng(0)
order = rng.permutation(len(files))
kept: list[dict] = []
seen: set[int] = set()
cache = out_dir / "raw_chunks"
n_chunks_used = 0
for oi in order:
    f = files[int(oi)]
    local = hub.stage_hub_file(
        C76.HF_DATA_REPO, f, cache / Path(f).name, repo_type="dataset", revision=pin
    )
    n_chunks_used += 1
    for r in json.loads(Path(local).read_text())["rows"]:
        ci = int(r["ci"])
        if ci in lmsys_ci and ci not in seen and r.get("response"):
            seen.add(ci)
            kept.append(
                {
                    "pair_id": f"ci{ci}",
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "ci": ci,
                    "chunk": Path(f).name,
                }
            )
    if len(kept) >= n_pairs:
        break
assert len(kept) >= n_pairs, f"only {len(kept)} lmsys pairs across {n_chunks_used} chunks"
idx = np.sort(rng.choice(len(kept), size=n_pairs, replace=False))
sel = [kept[int(i)] for i in idx]
tmp = out_path.with_suffix(".jsonl.tmp")
tmp.write_text("".join(json.dumps(r) + "\n" for r in sel))
tmp.replace(out_path)
C76.atomic_write_json(
    out_dir / "jpairs_build_report.json",
    {
        "n_pairs": n_pairs,
        "n_chunks_visited": n_chunks_used,
        "n_lmsys_pool": len(lmsys_ci),
        "pin": pin,
        "seed": 0,
        "note": "seeded chunk-permutation lazy download; sample restricted to visited chunks",
        "jpairs_sha256": sha256(out_path.read_bytes()).hexdigest(),
        "repro": C76.repro_meta(),
    },
)
print(f"[jpairs] wrote {n_pairs} pairs from {n_chunks_used} chunks -> {out_path}")
PY

CAP_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  p="$(bg_run "p04_capture_shard$g" "$g" uv run python - "$JPAIRS_DIR/jpairs.jsonl" \
    "$JPAIRS_DIR/cap_shard$g.pt" "$g" "$NGPU" <<'PY'
"""Sharded teacher-forced capture of the J pairs via the PRODUCER's rig
(parity.recompute_row -> issue779_collect capture fns): cx_last@{14,19} + v@19."""

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import issue1776_common as C76  # noqa: E402
import issue1776_parity as PAR  # noqa: E402
import issue779_common as C  # noqa: E402

pairs_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
shard, n_shards = int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(x) for x in pairs_path.read_text().splitlines() if x.strip()]
rows = rows[shard::n_shards]
if out_path.exists():
    have = torch.load(out_path, map_location="cpu", weights_only=True)
    if list(have["pair_id"]) == [r["pair_id"] for r in rows]:
        print(f"[p04-capture] shard {shard}: {out_path} complete; skip")
        sys.exit(0)
fields = PAR.consumed_fields(C76.SOURCE_LAYER, C76.READOUT_LAYER)
tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL)
model = (
    AutoModelForCausalLM.from_pretrained(C.DEFAULT_MODEL, dtype=torch.bfloat16, device_map={"": 0})
    .eval()
)
ids, v19, c14, c19 = [], [], [], []
for j, r in enumerate(rows):
    vec = PAR.recompute_row(model, tok, r["prompt"], r["response"], fields)
    ids.append(r["pair_id"])
    c14.append(vec[("cx_last", C76.SOURCE_LAYER)].to(torch.float32).cpu())
    c19.append(vec[("cx_last", C76.READOUT_LAYER)].to(torch.float32).cpu())
    v19.append(vec[("v_x", C76.READOUT_LAYER)].to(torch.float32).cpu())
    if (j + 1) % 32 == 0:
        print(f"[p04-capture] shard {shard}: {j + 1}/{len(rows)}", flush=True)
tmp = out_path.with_suffix(".pt.tmp")
torch.save(
    {
        "pair_id": ids,
        "v19": torch.stack(v19),
        "c14": torch.stack(c14),
        "c19": torch.stack(c19),
        "layers": [C76.SOURCE_LAYER, C76.READOUT_LAYER],
    },
    tmp,
)
tmp.replace(out_path)
print(f"[p04-capture] shard {shard}: {len(ids)} pairs -> {out_path}")
PY
)"
  CAP_PIDS+=("$p")
done
for p in ${CAP_PIDS[@]+"${CAP_PIDS[@]}"}; do wait_rc "$p" || exit $?; done

run p04_pairs_merge uv run python - "$JPAIRS_DIR" "$NGPU" <<'PY'
"""Merge capture shards back into manifest order -> jpair_capture.pt +
v_pool.pt ({'v': (n,H)} for build-seeds) + acts14/acts19 (phase4 cov null)."""

import json
import sys
from pathlib import Path

import torch

jdir, n_shards = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(x) for x in (jdir / "jpairs.jsonl").read_text().splitlines() if x.strip()]
by_id: dict[str, tuple] = {}
for g in range(n_shards):
    sh = torch.load(jdir / f"cap_shard{g}.pt", map_location="cpu", weights_only=True)
    for i, pid in enumerate(sh["pair_id"]):
        by_id[pid] = (sh["v19"][i], sh["c14"][i], sh["c19"][i])
missing = [r["pair_id"] for r in rows if r["pair_id"] not in by_id]
assert not missing, f"capture shards missing {len(missing)} pairs (e.g. {missing[:3]})"
order = [r["pair_id"] for r in rows]
v = torch.stack([by_id[p][0] for p in order])
c14 = torch.stack([by_id[p][1] for p in order])
c19 = torch.stack([by_id[p][2] for p in order])
torch.save({"pair_id": order, "v19": v, "c14": c14, "c19": c19, "layers": [14, 19]},
           jdir / "jpair_capture.pt")
torch.save({"v": v}, jdir / "v_pool.pt")
torch.save(c14, jdir / "acts14.pt")
torch.save(c19, jdir / "acts19.pt")
print(f"[p04-merge] {v.shape[0]} pairs merged (H={v.shape[1]})")
PY
phase_end "p04_pairs"

# ── p1_contexts (needs the staged #779 trait artifacts) ───────────────────────
phase_begin "p1_contexts"
run p1_contexts uv run python scripts/issue1776_contexts.py --out "$CTX_JSONL" \
  ${CTX_FLAGS[@]+"${CTX_FLAGS[@]}"}
phase_end "p1_contexts"

# ── p2a_sketch: seeds -> sharded sketch run -> merge (G-NONZERO inside run) ───
phase_begin "p2a_sketch"
mkdir -p "$SKETCH_ROOT"
run p2a_seeds uv run python scripts/issue1776_jacobian.py build-seeds \
  --v-pool "$JPAIRS_DIR/v_pool.pt" --comparator "$COMP_DIR/m_ridge_x50k.pt" \
  --n-total "$SEEDS_TOTAL" --topk-comparator "$SEEDS_TOPK" --n-gaussian "$SEEDS_GAUSS" \
  --out "$SKETCH_ROOT/seeds.pt"
SK_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  p="$(bg_run "p2a_sketch_shard$g" "$g" uv run python scripts/issue1776_jacobian.py run \
    --mode sketch --pairs "$JPAIRS_DIR/jpairs.jsonl" --seeds-file "$SKETCH_ROOT/seeds.pt" \
    --limit-pairs "$SKETCH_LIMIT" --shard-index "$g" --num-shards "$NGPU" \
    --out-dir "$SKETCH_ROOT/shard$g")"
  SK_PIDS+=("$p")
done
SK_RC=0
for p in ${SK_PIDS[@]+"${SK_PIDS[@]}"}; do
  wait_rc "$p" || { rc=$?; if [[ $rc -eq 8 ]]; then SK_RC=8; else exit $rc; fi; }
done
if [[ $SK_RC -eq 8 ]]; then
  gate_halt "G-NONZERO" 8 "all-zero context-gradient field in sketch run (slot-convention bug)"
fi
run p2a_merge uv run python scripts/issue1776_jacobian.py merge-shards \
  --shards-root "$SKETCH_ROOT" --out-dir "$SKETCH_ROOT/merged"
phase_end "p2a_sketch"

# ── p1_diag: Phase-1 directional diagnostic ───────────────────────────────────
phase_begin "p1_diag"
run p1_diag env CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1776_phase1.py \
  --comparator "$COMP_DIR/m_ridge_x50k.pt" --pairs "$JPAIRS_DIR/jpairs.jsonl" \
  --topk "$P1_TOPK" --limit-pairs "$P1_LIMIT" --out-dir "$EVAL_DIR/phase1"
phase_end "p1_diag"

# ── uploads: batched create_commit + scoped post-upload verify ────────────────
upload_batch() {  # upload_batch <log-name> <hub-prefix> <commit-msg> <listfile rel=abs per line>
  local lname="$1" prefix="$2" msg="$3" listfile="$4"
  run "$lname" uv run python - "$prefix" "$listfile" "$msg" <<'PY'
import sys
from pathlib import Path

import issue1776_common as C76
from huggingface_hub import CommitOperationAdd, HfApi
from explore_persona_space.orchestrate import hub

prefix, listfile, msg = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
pairs = [ln.split("=", 1) for ln in listfile.read_text().splitlines() if ln.strip()]
ops, expected = [], []
for rel, local in pairs:
    p = Path(local)
    assert p.exists(), f"upload source missing: {p}"
    rp = f"{prefix}/{rel}"
    ops.append(CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(p)))
    expected.append(rp)
if not ops:
    print(f"[upload] nothing to upload for {prefix}")
    sys.exit(0)
api = HfApi()
hub.retry_transient(
    lambda: api.create_commit(
        repo_id=C76.HF_DATA_REPO, repo_type="dataset", operations=ops, commit_message=msg
    ),
    what=f"create_commit({prefix})",
)
missing = hub.verify_repo_paths_uploaded(
    api, C76.HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
)
assert not missing, f"post-upload verify FAIL ({len(missing)} missing): {missing[:5]}"
print(f"[upload] {len(expected)} files -> {C76.HF_DATA_REPO}/{prefix} (verified)")
PY
}

list_add() {  # list_add <listfile> <rel> <abs> — include iff the source exists
  [[ -e "$3" ]] && echo "$2=$3" >> "$1" || true
}

phase_begin "p_early_upload"
# §9: regeneration-costly intermediates upload at the END of their producing
# phase and BEFORE the long P2b sweep (#825) — everything produced so far.
EARLY_LIST="$OUT_ROOT/upload_early.list"; : > "$EARLY_LIST"
list_add "$EARLY_LIST" "jlens/lens.pt" "$JLENS_DIR/lens.pt"
for L in 14 19 21; do
  list_add "$EARLY_LIST" "dictionaries/dictionary_l$L.pt" "$DICT_DIR/dictionary_l$L.pt"
done
for t in m_ridge_x50k m_ridge_lmsys50k; do
  list_add "$EARLY_LIST" "comparator/$t.pt" "$COMP_DIR/$t.pt"
done
if [[ $DRY == 0 ]]; then
  for f in "$COMP_DIR"/*.json "$DATA_DIR/parity"/*.json "$DATA_DIR/stage_report.json" \
           "$JPAIRS_DIR/jpairs_build_report.json"; do
    list_add "$EARLY_LIST" "reports/$(basename "$f")" "$f"
  done
fi
list_add "$EARLY_LIST" "jpairs/jpairs.jsonl" "$JPAIRS_DIR/jpairs.jsonl"
list_add "$EARLY_LIST" "jpairs/jpair_capture.pt" "$JPAIRS_DIR/jpair_capture.pt"
list_add "$EARLY_LIST" "jpairs/v_pool.pt" "$JPAIRS_DIR/v_pool.pt"
list_add "$EARLY_LIST" "jpairs/acts14.pt" "$JPAIRS_DIR/acts14.pt"
list_add "$EARLY_LIST" "jpairs/acts19.pt" "$JPAIRS_DIR/acts19.pt"
list_add "$EARLY_LIST" "jac_sketch/seeds.pt" "$SKETCH_ROOT/seeds.pt"
if [[ $DRY == 0 ]]; then
  for f in "$SKETCH_ROOT/merged"/*; do
    list_add "$EARLY_LIST" "jac_sketch/merged/$(basename "$f")" "$f"
  done
fi
list_add "$EARLY_LIST" "contexts/contexts.jsonl" "$CTX_JSONL"
list_add "$EARLY_LIST" "contexts/meta.json" "$(dirname "$CTX_JSONL")/meta.json"
upload_batch p_early_upload "$HF_PREFIX_EFF/analysis_tensors" \
  "task #1776: phase0-2a tensors + manifests (pre-P2b, #825 ordering)" "$EARLY_LIST"
phase_end "p_early_upload"

# ── p2b_full: 3,584-seed full-rank sweep, seed-block shard fan-out ────────────
phase_begin "p2b_full"
mkdir -p "$FULL_ROOT"
F_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  p="$(bg_run "p2b_full_shard$g" "$g" uv run python scripts/issue1776_jacobian.py run \
    --mode full --pairs "$JPAIRS_DIR/jpairs.jsonl" --m "$FULL_M" \
    --limit-pairs "$FULL_LIMIT" --shard-index "$g" --num-shards "$NGPU" \
    --out-dir "$FULL_ROOT/shard$g")"
  F_PIDS+=("$p")
done
F_RC=0
for p in ${F_PIDS[@]+"${F_PIDS[@]}"}; do
  wait_rc "$p" || { rc=$?; if [[ $rc -eq 8 ]]; then F_RC=8; else exit $rc; fi; }
done
if [[ $F_RC -eq 8 ]]; then
  gate_halt "G-NONZERO" 8 "all-zero context-gradient field in full run (slot-convention bug)"
fi
run p2b_merge uv run python scripts/issue1776_jacobian.py merge-shards \
  --shards-root "$FULL_ROOT" --out-dir "$FULL_ROOT/merged"
phase_end "p2b_full"

phase_begin "p2_upload"
P2_LIST="$OUT_ROOT/upload_p2.list"; : > "$P2_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$FULL_ROOT/merged"/*; do
    list_add "$P2_LIST" "jac_full/$(basename "$f")" "$f"
  done
fi
upload_batch p2_upload "$HF_PREFIX_EFF/analysis_tensors" \
  "task #1776: full-rank J + even/odd halves + intercepts (P2b)" "$P2_LIST"
phase_end "p2_upload"

# ── p3_grid: baseline -> steered strata fan-out -> finalize ───────────────────
phase_begin "p3_grid"
mkdir -p "$P3_ROOT" "$EVAL_DIR/phase3"
P3_BASE=(uv run python scripts/issue1776_phase3.py --mode run
  --contexts "$CTX_JSONL" --rb-dir "$RB_DIR" --mprime-weights "$COMP_DIR/m_ridge_x50k.pt"
  --jlast "$FULL_ROOT/merged/J_last.pt" --out-root "$P3_ROOT")
run p3_baseline env CUDA_VISIBLE_DEVICES=0 "${P3_BASE[@]}" --baseline-only \
  ${P3_EXTRA[@]+"${P3_EXTRA[@]}"}
P3_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  p="$(bg_run "p3_strata_shard$g" "$g" "${P3_BASE[@]}" \
    --strata-shard "$g" --strata-num-shards "$NGPU" ${P3_EXTRA[@]+"${P3_EXTRA[@]}"})"
  P3_PIDS+=("$p")
done
for p in ${P3_PIDS[@]+"${P3_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
run p3_finalize env CUDA_VISIBLE_DEVICES=0 "${P3_BASE[@]}" --finalize-only \
  --eval-out "$EVAL_DIR/phase3/steered_shift_summaries.json" ${P3_EXTRA[@]+"${P3_EXTRA[@]}"}
phase_end "p3_grid"

phase_begin "p3_upload"
# Steered rollout TEXT -> raw_completions BEFORE any downstream reduce/judging
# (Upload Policy; §9 phase-order). Summaries + manifest -> analysis_tensors.
P3_TEXT_LIST="$OUT_ROOT/upload_p3_text.list"; : > "$P3_TEXT_LIST"
P3_TENS_LIST="$OUT_ROOT/upload_p3_tens.list"; : > "$P3_TENS_LIST"
if [[ $DRY == 0 ]]; then
  for f in "$P3_ROOT"/raw_completions/steered/*.json; do
    list_add "$P3_TEXT_LIST" "$(basename "$f")" "$f"
  done
  for f in "$P3_ROOT"/summaries/*.pt "$P3_ROOT"/raw_completions_manifest.json \
           "$EVAL_DIR/phase3/steered_shift_summaries.json"; do
    list_add "$P3_TENS_LIST" "phase3/$(basename "$f")" "$f"
  done
fi
upload_batch p3_upload_text "$HF_PREFIX_EFF/raw_completions/steered" \
  "task #1776: steered rollout text (P3, before reduce/judge)" "$P3_TEXT_LIST"
upload_batch p3_upload_tens "$HF_PREFIX_EFF/analysis_tensors" \
  "task #1776: P3 per-sample summaries + manifest" "$P3_TENS_LIST"
phase_end "p3_upload"

# ── p4_mediation (lens-gated; phase4 invoked via subprocess ONLY) ─────────────
phase_begin "p4_mediation"
if [[ $LENS_OK -eq 1 ]]; then
  mkdir -p "$EVAL_DIR/phase4"
  P4_PIDS=()
  p="$(bg_run p4_energy "0" uv run python scripts/issue1776_phase4.py energy \
    --dict14 "$DICT_DIR/dictionary_l14.pt" --dict19 "$DICT_DIR/dictionary_l19.pt" \
    --mprime-weights "$COMP_DIR/m_ridge_x50k.pt" --jlast "$FULL_ROOT/merged/J_last.pt" \
    --rb-dir "$RB_DIR" --phase3-root "$P3_ROOT" \
    --acts14 "$JPAIRS_DIR/acts14.pt" --acts19 "$JPAIRS_DIR/acts19.pt" \
    --n-draws "$NDRAWS" --out "$EVAL_DIR/phase4/jspace_energy.json")"
  P4_PIDS+=("$p")
  p="$(bg_run p4_refit_split "$((NGPU > 1 ? 1 : 0))" uv run python scripts/issue1776_phase4.py \
    refit-split --dict19 "$DICT_DIR/dictionary_l19.pt" --n-train "$N_TRAIN" \
    --pass-b "$PASS_B" --mm-dir "$MM_DIR" --out "$EVAL_DIR/phase4/refit_split.json")"
  P4_PIDS+=("$p")
  p="$(bg_run p4_jdelta_split "$((NGPU > 2 ? 2 : 0))" uv run python scripts/issue1776_phase4.py \
    jdelta-split --dict14 "$DICT_DIR/dictionary_l14.pt" --jlast "$FULL_ROOT/merged/J_last.pt" \
    --phase3-root "$P3_ROOT" --out "$EVAL_DIR/phase4/jdelta_split.json")"
  P4_PIDS+=("$p")
  for p in ${P4_PIDS[@]+"${P4_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
else
  run p4_mediation echo "SKIPPED (G-LENS fail)"
fi
phase_end "p4_mediation"

# ── p5a: WildChat fresh capture (stream join, then sharded gen+capture) ───────
phase_begin "p5a_stream_join"
wait_rc "$STREAM_PID" || exit $?
phase_end "p5a_stream_join"

phase_begin "p5a_capture"
C_PIDS=()
for ((g = 0; g < NGPU; g++)); do
  p="$(bg_run "p5a_capture_shard$g" "$g" uv run python scripts/issue1776_phase5.py capture \
    --pool "$WC_DIR/wildchat_fresh_pool.jsonl" --out-root "$WC_CAP_ROOT" \
    --shard-index "$g" --n-shards "$NGPU" --hf-prefix "$HF_PREFIX_EFF/wildchat_fresh" \
    ${CAP_EXTRA[@]+"${CAP_EXTRA[@]}"})"
  C_PIDS+=("$p")
done
for p in ${C_PIDS[@]+"${C_PIDS[@]}"}; do wait_rc "$p" || exit $?; done
phase_end "p5a_capture"

# ── p5_transfer: P2c reads + P5 decay legs (test-1000 + wildchat_fresh) ───────
phase_begin "p5_transfer"
mkdir -p "$EVAL_DIR/phase5"
SHIPPED_M=""
if [[ $DRY == 0 && -d "$WEIGHTS_DIR" ]]; then
  SHIPPED_M="$(find "$WEIGHTS_DIR" -name '*l19*.pt' -o -name '*layer19*.pt' 2>/dev/null | sort | head -1 || true)"
fi
T_ARGS=(uv run python scripts/issue1776_phase5.py transfer --assemble
  --pass-b "$PASS_B" --mm-dir "$MM_DIR" --out-dir "$DATA_DIR/transfer"
  --op "mprime_x50k=$COMP_DIR/m_ridge_x50k.pt=14"
  --op "mprime_lmsys50k=$COMP_DIR/m_ridge_lmsys50k.pt=14"
  --jop "J_last=$FULL_ROOT/merged/J_last.pt"
  --jop "J_ctx=$FULL_ROOT/merged/J_ctx.pt"
  --jop "J_prefix=$FULL_ROOT/merged/J_prefix.pt"
  --leg "wildchat_fresh=$WC_CAP_ROOT"
  --n-boot "$NBOOT" --out "$EVAL_DIR/phase5/transfer.json")
if [[ -n "$SHIPPED_M" ]]; then
  T_ARGS+=(--op "m_shipped=$SHIPPED_M=19")
else
  echo "[dispatch] NOTE: shipped-M reference weights not resolved under $WEIGHTS_DIR (recorded in final sentinel)"
fi
run p5_transfer env CUDA_VISIBLE_DEVICES=0 "${T_ARGS[@]}" ${ASSEMBLE_EXTRA[@]+"${ASSEMBLE_EXTRA[@]}"}
phase_end "p5_transfer"

# ── p5b_leakage (CPU; §9 lists this off-pod — run here since every input is
#    already staged/built; deviation recorded in the final sentinel) ───────────
phase_begin "p5b_leakage"
if [[ $LENS_OK -eq 1 ]]; then
  run p5b_leakage uv run python scripts/issue1776_phase5.py leakage \
    --centroids "$CENTROIDS" --dict "$DICT_DIR/dictionary_l21.pt" \
    --n-boot "$NBOOT" --out "$EVAL_DIR/phase5/leakage_reread.json"
else
  run p5b_leakage echo "SKIPPED (G-LENS fail)"
fi
phase_end "p5b_leakage"

# ── p_results_commit: git-destined eval JSONs (#1205 push-verify + #1325) ─────
phase_begin "p_results_commit"
if [[ $DRY == 1 || "$MODE" == "smoke" ]]; then
  run p_results_commit echo "SKIPPED (dry-run/smoke: eval outputs stay under $EVAL_DIR)"
else
  BR="$(git rev-parse --abbrev-ref HEAD)"
  mapfile -t DECLARED < <(find "$EVAL_DIR" -name '*.json' | sort)
  if [[ ${#DECLARED[@]} -gt 0 ]]; then
    git add -- "${DECLARED[@]}"
    if ! git diff --cached --quiet; then
      git commit -m "task #1776: pod-side eval results (phases 0-5, mode=$MODE)"
    fi
    if ! git push origin "$BR"; then
      echo "[dispatch] push failed; retrying once in 20s" >&2
      sleep 20
      git push origin "$BR"
    fi
    AHEAD="$(git rev-list --count "origin/$BR..HEAD")"
    [[ "$AHEAD" == "0" ]] || { echo "[dispatch] push-verify FAIL: $AHEAD unpushed commits" >&2; exit 1; }
    MISSING=0
    for f in "${DECLARED[@]}"; do
      rel="${f#"$REPO_ROOT"/}"
      if [[ -z "$(git ls-tree -r "origin/$BR" --name-only -- "$rel")" ]]; then
        echo "[dispatch] artifact-presence FAIL: $rel not in pushed tree" >&2
        MISSING=1
      fi
    done
    [[ $MISSING -eq 0 ]] || exit 1
    echo "[dispatch] results commit pushed + verified (${#DECLARED[@]} JSONs on origin/$BR)"
  else
    echo "[dispatch] no eval JSONs to commit" >&2
  fi
fi
phase_end "p_results_commit"

# ── p_final: terminal results sentinel, then [phase=done] ─────────────────────
phase_begin "p_final"
FINAL_RC=0
if [[ $LENS_OK -eq 0 ]]; then
  FINAL_RC=8
  GATE_HALTED=1  # designed halt (G-LENS), not a crash — the EXIT trap stays quiet
fi
uv run python - "$ISSUE" "$LOG_DIR" "$MODE" "$DRY" "$EVAL_DIR" "$REPO_ROOT" \
  "$HF_PREFIX_EFF" "$LENS_OK" "$SHIPPED_M" "$NGPU" <<'PY'
"""Terminal results sentinel (§10 structured fields: eval paths, HF prefixes,
off-pod handoffs; no training this run -> no adapter/wandb-run fields)."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

issue, log_dir, mode, dry = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
eval_dir, repo_root = Path(sys.argv[5]), Path(sys.argv[6])
hf_prefix, lens_ok, shipped_m, ngpu = sys.argv[7], sys.argv[8] == "1", sys.argv[9], sys.argv[10]
smoke_like = dry or mode == "smoke"
kind = "epm:smoke-result" if smoke_like else "epm:results"
try:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root, check=True
    ).stdout.strip()
except Exception:
    sha = "unknown"
eval_paths = sorted(
    str(p.relative_to(repo_root)) if p.is_relative_to(repo_root) else str(p)
    for p in eval_dir.rglob("*.json")
)
note = {
    "mode": mode,
    "dry_run": dry,
    "ngpu": ngpu,
    "git_commit": sha,
    "gates": {
        "G-PARITY": "PASS",
        "G-LENS": "PASS" if lens_ok else "FAIL (rc=8; dict/phase4/5b legs skipped)",
        "G-NONZERO": "PASS",
    },
    "eval_json_paths": eval_paths,
    "hf_prefixes": {
        "analysis_tensors": f"{hf_prefix}/analysis_tensors",
        "raw_completions_steered": f"{hf_prefix}/raw_completions/steered",
        "wildchat_fresh": f"{hf_prefix}/wildchat_fresh",
    },
    "shipped_m_reference": shipped_m or "NOT RESOLVED under n1m_readout/weights (transfer ran without the reference row)",
    "plan_deviation": "p5b leakage re-read ran POD-side (plan §9 lists it off-pod p7): all inputs were staged/built here",
    "offpod_handoffs": {
        "p6_judge": (
            "OFF-POD (VM, Batch API): uv run python scripts/issue1776_judge.py "
            f"--raw-dir <staged {hf_prefix}/raw_completions/steered> "
            "--out-dir eval_results/issue_1776/phase3/judge. PRICING: the default "
            "judges the control strata under ALL 3 rubrics (~30k x 5 calls) vs plan "
            "S9's 16k x 5 one-rubric-per-completion costing - trim via "
            "--control-rubrics (e.g. --control-rubrics evil) if the budget binds."
        ),
        "p7_final_analyses": "OFF-POD (VM, 0 GPU): 5c word tables + 5d chain reads (lens + dictionaries on HF)",
    },
    "wandb": "n/a (no training this run)",
}
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": int(issue),
    "gate": "smoke" if smoke_like else "results",
    "blocks_pipeline": not smoke_like,
    "by": "issue1776_dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": note,
}
log_dir.mkdir(parents=True, exist_ok=True)
path = log_dir / f"issue-{issue}-{kind.replace(':', '_')}-{int(time.time())}.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=1))
os.replace(tmp, path)
print(f"[dispatch] results sentinel: {path}")
PY

# ── dry-run self-checks: trace order + sentinel round-trip ────────────────────
if [[ $DRY == 1 ]]; then
  uv run python - "$TRACE" "$LOG_DIR" "$ISSUE" <<'PY'
import json
import sys
from pathlib import Path

EXPECTED = [
    "p0_stage", "p5a_stream_launch", "p0_parity", "p0_comparator_launch",
    "p0_jlens", "p0_comparator_join", "p0_dict", "p04_pairs", "p1_contexts",
    "p2a_sketch", "p1_diag", "p_early_upload", "p2b_full", "p2_upload",
    "p3_grid", "p3_upload", "p4_mediation", "p5a_stream_join", "p5a_capture",
    "p5_transfer", "p5b_leakage", "p_results_commit", "p_final",
]
trace_path, log_dir, issue = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
trace = [ln.strip() for ln in trace_path.read_text().splitlines() if ln.strip()]
assert trace == EXPECTED, f"phase-order mismatch:\n got={trace}\n want={EXPECTED}"
REQUIRED = {"sentinel_schema_version", "kind", "version", "note"}
sents = sorted(log_dir.glob(f"issue-{issue}-*.json"))
assert sents, f"no sentinels under {log_dir}"
kinds = {}
for s in sents:
    obj = json.loads(s.read_text())
    missing = REQUIRED - set(obj)
    assert not missing, f"{s.name} missing required keys {missing}"
    assert obj["sentinel_schema_version"] == 1 and isinstance(obj["version"], int), s.name
    kinds.setdefault(obj["kind"], 0)
    kinds[obj["kind"]] += 1
assert kinds.get("epm:smoke-result", 0) == 1, kinds  # dry-run terminal sentinel
final = json.loads(
    max(log_dir.glob(f"issue-{issue}-epm_smoke-result-*.json")).read_text()
)
note = final["note"]
for key in ("eval_json_paths", "hf_prefixes", "offpod_handoffs", "gates", "wandb", "git_commit"):
    assert key in note, f"final sentinel note missing §10 field: {key}"
assert "control-rubrics" in note["offpod_handoffs"]["p6_judge"], "judge pricing handoff missing"
print(f"DRY-RUN-OK: {len(trace)} phases in §9 order; {len(sents)} sentinels parse "
      f"({sum(kinds.values())} total: {kinds})")
PY
fi

# Designed-halt convention: a G-LENS-degraded run does NOT emit [phase=done] —
# the gate sentinel + the results sentinel (gates.G-LENS=FAIL) carry the state
# and the distinct rc=8 routes it (never a bare rc=1 anonymous crash).
if [[ $FINAL_RC -eq 0 ]]; then
  echo "[phase=done]"
else
  echo "[phase=gate_halted]"
fi
exit "$FINAL_RC"
