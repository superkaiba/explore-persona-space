#!/usr/bin/env bash
# Task #623 dispatch driver — pod-side, training-free persona-vector decomposition.
#
# Runs the headline arm on 1 GPU (GCP lora-7b / A100-80G): forward-pass +
# vLLM generation only, then uploads intermediate analysis tensors to HF before
# pod termination. The off-pod CPU analysis (phase 6, issue623_analyze.py) runs
# on the VM AFTER the pod terminates (per CLAUDE.md "CPU-only phases don't hold
# GPU pods").
#
# Phases (each emits a [phase=<name>] log line; smoke = sweep with a smaller
# cell list via --personas / --layers / --n-questions threaded into every phase):
#   p0_preflight   tolerant preflight (parse --json; tolerate behind-origin/main)
#   persona_resolve  -> panel_prompts.json
#   vector_extract   -> persona centroids (Method A + B)
#   sycophancy_trait -> trait vector (last-token + response-avg)
#   steering_probe   -> K2 HALT gate + headline-layer selection
#   syc_i_load       -> syc_i.json (reused #612 base rates, NO new generation)
#   upload           -> centroids + trait + raw completions to HF data repo
#   done             -> terminal marker
#
# Launch (via the backend router; GCP auto lora-7b lane, ~3h << 24h fence):
#   uv run python scripts/dispatch_issue.py --issue 623 --intent lora-7b \
#     --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue623_dispatch.sh'
#
# Smoke (tiny slice, exit 0 + artifact per phase):
#   bash scripts/issue623_dispatch.sh --smoke \
#     --personas satirist,journalist,assistant --layers 21 --n-questions 6
#
# Upload-phase smoke (end-to-end, GPU-free, round-trips a tiny bundle to a
# _smoke/ HF prefix and verifies via list_repo_files; never touches production):
#   bash scripts/issue623_dispatch.sh --smoke-upload-only
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export TQDM_DISABLE=1                                   # GCE startup-script bufio guard (#607)
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"  # #628 fork guard
export WANDB_PROJECT="${WANDB_PROJECT:-issue623}"

# Defaults (full sweep).
PERSONAS=""              # empty => full 36-persona panel
LAYERS="7 14 21 27"
N_QUESTIONS_TRAIT="40"   # paper 40, split 20 extraction / 20 eval
N_VECTOR_QUESTIONS=""    # empty => all 240 extraction questions
GPU_ID="0"
SKIP_UPLOAD="0"
SKIP_PREFLIGHT="0"
SKIP_GPU_PHASES="0"   # dispatcher dry-run: run CPU phases only, emit [phase=done]
SMOKE_UPLOAD_ONLY="0" # exercise the upload phase end-to-end on a tiny _smoke/ bundle
SMOKE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE="1"; shift ;;
    --personas) PERSONAS="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --n-questions) N_QUESTIONS_TRAIT="$2"; N_VECTOR_QUESTIONS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --skip-upload) SKIP_UPLOAD="1"; shift ;;
    --skip-preflight) SKIP_PREFLIGHT="1"; shift ;;
    --skip-gpu-phases) SKIP_GPU_PHASES="1"; SKIP_UPLOAD="1"; shift ;;
    --smoke-upload-only) SMOKE_UPLOAD_ONLY="1"; shift ;;
    --force-crash-test) FORCE_CRASH_TEST="1"; shift ;;
    *) echo "[driver] unknown arg: $1" >&2; exit 2 ;;
  esac
done

DATA_DIR="data/persona_vectors/issue623"
EVAL_DIR="eval_results/issue_623"
LOGS_DIR="${ISSUE623_LOGS_DIR:-/workspace/logs}"
RUN_LOGS="$LOGS_DIR/issue623_driver"
mkdir -p "$RUN_LOGS" "$DATA_DIR" "$EVAL_DIR"

# ── crash-dump diagnostics state (GCP-lane forensics; #607 $? caveat) ────────
# The GCP DELETE-on-exit fence wipes the instance disk before any log upload,
# so a non-clean exit must push a forensic bundle to HF FIRST. We track three
# signals because the GCE metadata-runner can SIGPIPE-kill the script and leave
# $? reading 0 inside an EXIT trap (gotchas.md #607): an explicit fail() flag,
# the last phase reached, and whether the terminal [phase=done] was ever hit.
_LAST_PHASE="startup"
_DRIVER_FAILED="0"
_DONE_MARKER_REACHED="0"
FORCE_CRASH_TEST="${FORCE_CRASH_TEST:-0}"
# Surface a few facts for CRASH_META.json (best-effort; absent on a local smoke).
export EPS_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export EPS_DISPATCHER_ARGV="$0 $*"

# mark <phase> — call at each phase boundary so the trap/poll knows where we are.
mark() { _LAST_PHASE="$1"; }

_crash_dump() {
  # Fire ONLY on a non-clean exit. "Clean" = exit 0 AND the terminal [phase=done]
  # was reached. $? alone is untrustworthy under the #607 SIGPIPE kill, so the
  # "[phase=done] never reached" signal is the safety net.
  local rc="$?"
  trap - EXIT  # disarm so our own exit below cannot re-enter the trap
  if [[ "$rc" == "0" && "$_DRIVER_FAILED" == "0" && "$_DONE_MARKER_REACHED" == "1" ]]; then
    return 0  # genuine clean exit — no dump
  fi
  echo "[driver] [phase=crash_dump] non-clean exit (rc=$rc, last_phase=$_LAST_PHASE) — uploading forensic dump to HF" >&2
  local dump_suffix_arg=()
  [[ "$FORCE_CRASH_TEST" == "1" ]] && dump_suffix_arg=(--hf-prefix-suffix smoketest --print-prefix-only)
  # Best-effort: the dump must NEVER mask the original exit code, so we swallow
  # its own non-zero and always re-exit with rc.
  uv run python scripts/issue623_crash_dump.py \
    --run-logs "$RUN_LOGS" \
    --data-dir "$DATA_DIR" \
    --master-log "${EPS_LOG_PATH:-$LOGS_DIR/issue-623.log}" \
    --last-phase "$_LAST_PHASE" \
    --exit-code "$rc" \
    "${dump_suffix_arg[@]}" \
    >>"$RUN_LOGS/crash_dump.log" 2>&1 \
    || echo "[driver] [phase=crash_dump] WARN crash-dump uploader itself failed — see $RUN_LOGS/crash_dump.log" >&2
  # Surface the crash-dump log tail on the master log so the polling tick sees it.
  echo "[driver] [phase=crash_dump] crash-dump uploader output (tail):" >&2
  tail -n 20 "$RUN_LOGS/crash_dump.log" >&2 2>/dev/null || true
  exit "$rc"
}
trap _crash_dump EXIT

fail() {
  echo "[driver] FATAL: $*" >&2
  _DRIVER_FAILED="1"
  # On a real crash, surface the failing phase's log tail onto the master log so
  # the polling tick's log_tail (which only sees the master log, not the
  # per-phase redirect target) shows WHERE it died even before the HF dump lands.
  local phase_log="$RUN_LOGS/${_LAST_PHASE}.log"
  if [[ -f "$phase_log" ]]; then
    echo "[driver] --- tail of $phase_log (last_phase=$_LAST_PHASE) ---" >&2
    tail -n 40 "$phase_log" >&2 2>/dev/null || true
    echo "[driver] --- end $phase_log ---" >&2
  fi
  exit "${2:-1}"
}

# ── --smoke-upload-only: end-to-end upload-phase smoke (C2) ──────────────────
# The standard --skip-gpu-phases smoke sets SKIP_UPLOAD=1 so the upload phase was
# never exercised end-to-end. This branch builds a tiny single-persona artifact
# bundle, runs issue623_upload.py against a _smoke/<ts> HF prefix (so it NEVER
# touches the production issue623_persona_vectors/ tree), and asserts every file
# resolves via huggingface_hub.list_repo_files after upload. CPU-only, GPU-free.
if [[ "$SMOKE_UPLOAD_ONLY" == "1" ]]; then
  TS="$(date +%s)"
  SMOKE_PREFIX="_smoke/issue623_persona_vectors_${TS}"
  SMOKE_BASE="$(mktemp -d)"
  echo "[driver] [phase=upload_smoke] building tiny bundle at $SMOKE_BASE -> $SMOKE_PREFIX"
  mkdir -p "$SMOKE_BASE/method_a" "$SMOKE_BASE/sycophancy_trait" \
    "$SMOKE_BASE/steering_probe/raw_completions"
  printf '{"assistant": "You are a helpful assistant."}\n' >"$SMOKE_BASE/panel_prompts.json"
  printf '{"persona": "assistant", "layers": [21]}\n' >"$SMOKE_BASE/method_a/centroids_meta.json"
  printf '{"trait": "sycophancy", "smoke": true}\n' >"$SMOKE_BASE/sycophancy_trait/metadata.json"
  printf '[{"question": "q", "response": "r"}]\n' \
    >"$SMOKE_BASE/steering_probe/raw_completions/baseline.json"
  uv run python scripts/issue623_upload.py \
    --persona-vectors-dir "$SMOKE_BASE" --hf-prefix "$SMOKE_PREFIX" \
    >"$RUN_LOGS/upload_smoke.log" 2>&1 \
    || { cat "$RUN_LOGS/upload_smoke.log" >&2; fail "upload smoke failed — see $RUN_LOGS/upload_smoke.log"; }
  echo "[driver] [phase=upload_smoke_verify] verifying files resolve on HF via list_repo_files"
  SMOKE_PREFIX="$SMOKE_PREFIX" uv run python - <<'PY' || fail "upload smoke HF verify failed" 1
import os, sys
from huggingface_hub import list_repo_files
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO
load_dotenv()
prefix = os.environ["SMOKE_PREFIX"]
files = set(list_repo_files(DEFAULT_DATASET_REPO, repo_type="dataset", revision="main"))
expected = [
    f"{prefix}/panel_prompts.json",
    f"{prefix}/persona_vectors/method_a/centroids_meta.json",
    f"{prefix}/sycophancy_trait/metadata.json",
    f"{prefix}/steering_probe/raw_completions/baseline.json",
]
missing = [e for e in expected if e not in files]
if missing:
    print(f"upload smoke: files NOT resolved on HF: {missing}", file=sys.stderr)
    sys.exit(1)
print(f"upload smoke: all {len(expected)} files resolved under {prefix} on HF")
PY
  rm -rf "$SMOKE_BASE"
  _DONE_MARKER_REACHED="1"
  echo "[driver] [phase=done] upload-phase smoke complete ($SMOKE_PREFIX verified on HF)"
  exit 0
fi

# Optional per-phase override args.
PERSONA_ARG=()
[[ -n "$PERSONAS" ]] && PERSONA_ARG=(--personas "$PERSONAS")
VEC_NQ_ARG=()
[[ -n "$N_VECTOR_QUESTIONS" ]] && VEC_NQ_ARG=(--n-questions "$N_VECTOR_QUESTIONS")

echo "[driver] issue 623 dispatch: smoke=$SMOKE personas='${PERSONAS:-ALL}' layers='$LAYERS' gpu=$GPU_ID"

# ── p0_preflight (tolerant; parse --json, tolerate behind-origin/main #552) ──
if [[ "$SKIP_PREFLIGHT" == "0" ]]; then
  mark p0_preflight
  echo "[driver] [phase=p0_preflight] tolerant preflight"
  uv run python - <<'PY' || fail "preflight (non-git-check) errors" 2
import json, re, subprocess, sys
proc = subprocess.run(
    ["uv", "run", "python", "-m", "explore_persona_space.orchestrate.preflight", "--json"],
    capture_output=True, text=True,
)
m = re.search(r"\{.*\}", proc.stdout, re.S)
if not m:
    print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
    sys.exit(1)
report = json.loads(m.group(0))
behind = re.compile(r"behind origin/(main|issue-623)|git fetch origin failed")
real = [e for e in report.get("errors", []) if not behind.search(str(e))]
if real:
    print("preflight errors:", real, file=sys.stderr)
    sys.exit(1)
print("preflight OK (git behind-origin tolerated on issue branches)")
PY
else
  echo "[driver] [phase=p0_preflight] SKIPPED (--skip-preflight)"
fi

# ── phase 1: persona resolve ──
mark persona_resolve
echo "[driver] [phase=persona_resolve] resolving panel prompts"
uv run python scripts/issue623_persona_resolve.py \
  --output "$DATA_DIR/panel_prompts.json" "${PERSONA_ARG[@]}" \
  >"$RUN_LOGS/persona_resolve.log" 2>&1 \
  || fail "persona_resolve failed — see $RUN_LOGS/persona_resolve.log"

# ── --force-crash-test: exercise the on-crash dump path WITHOUT a GPU ─────────
# After the cheap CPU-only persona_resolve phase, simulate a phase exit 1 so the
# EXIT trap fires the crash-dump uploader against a _crash_dumps/..._smoketest
# prefix. fail() trips _DRIVER_FAILED and exits 1 → _crash_dump uploads + the
# smoke verifies the dump landed on HF + cleans it up.
if [[ "$FORCE_CRASH_TEST" == "1" ]]; then
  mark vector_extract  # pretend we died in the long extraction loop
  fail "FORCED CRASH TEST — exercising on-crash dump uploader (no real failure)" 1
fi

if [[ "$SKIP_GPU_PHASES" == "1" ]]; then
  echo "[driver] [phase=vector_extract] SKIPPED (--skip-gpu-phases dry-run)"
  echo "[driver] [phase=sycophancy_trait] SKIPPED (--skip-gpu-phases dry-run)"
else
  # ── phase 2: persona panel vectors (Method AB) ──
  mark vector_extract
  echo "[driver] [phase=vector_extract] extracting persona centroids (Method AB)"
  # shellcheck disable=SC2086
  uv run python scripts/issue623_persona_panel_vectors.py \
    --panel-prompts "$DATA_DIR/panel_prompts.json" \
    --method AB --layers $LAYERS --gpu-id "$GPU_ID" \
    --output-dir "$DATA_DIR" "${PERSONA_ARG[@]}" "${VEC_NQ_ARG[@]}" \
    >"$RUN_LOGS/vector_extract.log" 2>&1 \
    || fail "vector_extract failed — see $RUN_LOGS/vector_extract.log"

  # ── phases 3 + 4: sycophancy trait vector + steering probe (K2 HALT) ──
  mark sycophancy_trait
  echo "[driver] [phase=sycophancy_trait] extracting sycophancy trait vector + steering probe"
  # shellcheck disable=SC2086
  uv run python scripts/issue623_extract_sycophancy_vector.py \
    --layers $LAYERS --gpu-id "$GPU_ID" \
    --n-questions "$N_QUESTIONS_TRAIT" \
    --output-dir "$DATA_DIR/sycophancy_trait" \
    --steering-output "$EVAL_DIR/steering_probe.json" \
    --steering-effect-output "$EVAL_DIR/steering_effect_by_layer.json" \
    >"$RUN_LOGS/sycophancy_trait.log" 2>&1 \
    || fail "sycophancy_trait / steering_probe failed (or K2 HALT) — see $RUN_LOGS/sycophancy_trait.log"
fi

# ── phase 5: behavioral DV syc_i (reuse #612 base rates) ──
mark syc_i_load
echo "[driver] [phase=syc_i_load] resolving behavioral DV syc_i (reuse #612)"
uv run python scripts/issue623_behavioral_dv.py \
  --panel-prompts "$DATA_DIR/panel_prompts.json" \
  --output "$EVAL_DIR/syc_i.json" \
  >"$RUN_LOGS/syc_i_load.log" 2>&1 \
  || fail "syc_i_load failed — see $RUN_LOGS/syc_i_load.log"

# ── upload intermediate analysis tensors + raw completions to HF ──
if [[ "$SKIP_UPLOAD" == "0" ]]; then
  mark upload
  echo "[driver] [phase=upload] uploading centroids + trait + raw completions to HF"
  uv run python scripts/issue623_upload.py \
    --persona-vectors-dir "$DATA_DIR" \
    >"$RUN_LOGS/upload.log" 2>&1 \
    || fail "upload failed — see $RUN_LOGS/upload.log"
else
  echo "[driver] [phase=upload] SKIPPED (--skip-upload)"
fi

_DONE_MARKER_REACHED="1"
echo "[driver] [phase=done] issue 623 dispatch complete"
