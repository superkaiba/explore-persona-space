#!/usr/bin/env bash
# Issue #657 Phase 1 dispatcher — pod-side, training-free direction/persona extraction.
#
# The ONLY GPU step (1x A100-80, GCP lora-7b auto lane). Reuses #623's validated
# extraction scripts to produce the two missing predictor inputs:
#
#   1. persona vectors for the <=7 missing 24-persona-panel personas
#      (ai, ai_assistant, data_scientist, hero, medical_doctor, police_officer,
#       surgeon) over the same 240Q bank, Method A (last-token) + B (response-avg),
#      via scripts/issue623_persona_panel_vectors.py.
#   2. the 3 NEW behavior directions, each via its BEHAVIOR-APPROPRIATE recipe (v6
#      pivot — a single generic #623 recipe failed K2 for refusal at launch #3):
#        - refusal : Arditi difference-in-means (scripts/issue657_extract_diffmean_direction.py
#                    --behavior refusal), held-out K2 add-elicits-refusal sanity gate.
#        - em      : #521 on_policy_em trained-shift re-extraction (same extractor,
#                    --behavior em; single config-pinned issue_519/em_seed42 adapter).
#        - marker  : the #623 Persona-Vectors affordance recipe
#                    (scripts/issue623_extract_sycophancy_vector.py --trait-name marker
#                    --trait-description ...); a REAL K2 fail -> #521 on_policy_marker
#                    shift fallback (--behavior marker) + M-Alts3 demotion to SECONDARY.
#
# Sycophancy is NOT re-extracted (REUSED verbatim from #623). The off-pod CPU
# Phases 0/2/3/4 run on the VM AFTER this pod terminates, reading the uploaded
# directions + persona vectors from HF (CLAUDE.md "CPU-only phases don't hold GPU
# pods").
#
# Each phase emits a [phase=<name>] log line; the terminal [phase=done] + the
# end-of-run sentinel conform to poll_pipeline.py (CLAUDE.md pod-side contract).
#
# Launch (via the backend router; GCP auto lora-7b lane, ~1.5h << 24h fence):
#   uv run python scripts/dispatch_issue.py --issue 657 --intent lora-7b \
#     --workload-cmd 'REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue657_extract.sh'
#
# Local GPU-free dispatcher dry-run (exercises plumbing + emits [phase=done]):
#   bash scripts/issue657_extract.sh --skip-gpu-phases
#
# Smoke (reuses #623's validated --smoke extraction path, tiny slice):
#   bash scripts/issue657_extract.sh --smoke --behaviors refusal \
#     --personas medical_doctor,assistant --layers 21 --n-questions 6
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export TQDM_DISABLE=1                                   # GCE startup-script bufio guard (#607)
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"  # #628 fork guard
export WANDB_PROJECT="${WANDB_PROJECT:-issue657}"

# Defaults (full Phase 1).
MISSING_PERSONAS="ai,ai_assistant,data_scientist,hero,medical_doctor,police_officer,surgeon"
BEHAVIORS="refusal,marker,em"     # sycophancy REUSED from #623, not re-extracted
LAYERS="7 14 21 27"
N_QUESTIONS="40"      # paper 40, split 20 extraction / 20 eval (trait directions)
N_VECTOR_QUESTIONS="" # empty => all 240 extraction questions (persona vectors)
# Refusal diff-mean probe banks (Arditi 2406.11717 field-standard banks).
# Harmful = mlabonne/harmful_behaviors — the AdvBench harmful-behavior set mirrored
# UNGATED on HF (walledai/AdvBench is GATED for this token; mlabonne/harmful_behaviors
# carries the same 416 AdvBench behaviors in a `text` column — accessible, plan §12).
# Harmless = mlabonne/harmless_alpaca (ungated). Override via --harmful/--harmless.
REFUSAL_HARMFUL="mlabonne/harmful_behaviors"
REFUSAL_HARMLESS="mlabonne/harmless_alpaca"
GPU_ID="0"
SKIP_UPLOAD="0"
SKIP_PREFLIGHT="0"
SKIP_GPU_PHASES="0"   # dispatcher dry-run: CPU plumbing only, emit [phase=done]
SMOKE="0"
PRINT_TRAIT_DESCRIPTION=""   # set by --print-trait-description <beh>: print + exit
TEST_MARKER_DECISION=""      # set by --test-marker-decision <rc> <effect-file>: decide + exit

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE="1"; shift ;;
    --behaviors) BEHAVIORS="$2"; shift 2 ;;
    --personas) MISSING_PERSONAS="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --n-questions) N_QUESTIONS="$2"; N_VECTOR_QUESTIONS="$2"; shift 2 ;;
    --harmful) REFUSAL_HARMFUL="$2"; shift 2 ;;
    --harmless) REFUSAL_HARMLESS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --skip-upload) SKIP_UPLOAD="1"; shift ;;
    --skip-preflight) SKIP_PREFLIGHT="1"; shift ;;
    --skip-gpu-phases) SKIP_GPU_PHASES="1"; SKIP_UPLOAD="1"; shift ;;
    # Print the canonical trait description for a behavior, then exit 0. Lets the
    # CPU-only artifact smoke (tests/test_issue657_extract_smoke.py) read the
    # SAME strings the dispatcher passes to --trait-description, so the smoke and
    # the production path can never drift apart (#657 r2 fix).
    --print-trait-description) PRINT_TRAIT_DESCRIPTION="$2"; shift 2 ;;
    # B4 regression hook: run ONLY the marker fail-loud fallback decision against a
    # given (primary rc, effect-file) and exit with its return code. Lets the shell
    # smoke (tests/test_issue657_extract_smoke.py) prove a primary crash FAILS LOUD
    # and does NOT invoke the fallback, with no GPU + no upstream phases.
    --test-marker-decision) TEST_MARKER_DECISION="$2 $3"; shift 3 ;;
    *) echo "[driver] unknown arg: $1" >&2; exit 2 ;;
  esac
done

DATA_DIR="data/issue657"
PV_DIR="$DATA_DIR/persona_vectors"           # method_a/ method_b/ written under here
BD_DIR="$DATA_DIR/behavior_directions"       # <behavior>/ written under here
EVAL_DIR="eval_results/issue_657"
LOGS_DIR="${ISSUE657_LOGS_DIR:-/workspace/logs}"
RUN_LOGS="$LOGS_DIR/issue657_driver"
mkdir -p "$RUN_LOGS" "$PV_DIR" "$BD_DIR" "$EVAL_DIR"

HF_PREFIX="issue657_alignment_predictor"
SENTINEL_EPOCH="$(date +%s)"
SENTINEL_PATH="$LOGS_DIR/issue-657-epm_results-${SENTINEL_EPOCH}.json"

_LAST_PHASE="startup"
_DRIVER_FAILED="0"
_DONE_MARKER_REACHED="0"

mark() { _LAST_PHASE="$1"; }

fail() {
  echo "[driver] FATAL: $*" >&2
  _DRIVER_FAILED="1"
  local phase_log="$RUN_LOGS/${_LAST_PHASE}.log"
  if [[ -f "$phase_log" ]]; then
    # Dump the tail to STDOUT (not just stderr): on the GCP lane stdout is the
    # poller-visible main log (<vm_scratch_dir>/logs/issue-657.log), captured
    # LIVE before the EXIT-trap DELETE wipes the disk. With the per-phase `tee`
    # the body is already in the main log, but this guarantees the tail is the
    # LAST thing the poller sees regardless of buffering. (#657 r1: the round-1
    # `>file 2>&1` redirect kept the traceback off the main log entirely.)
    echo "[driver] --- tail of $phase_log (last_phase=$_LAST_PHASE) ---"
    tail -n 100 "$phase_log" 2>/dev/null || true
    echo "[driver] --- end $phase_log ---"
    # F1 fail-safe (path c): persist the failed log to the HF data repo BEFORE
    # the EXIT trap fires. A /workspace sentinel is wiped by GCP's
    # instance-termination DELETE, so the durable cross-DELETE channel is HF.
    # Fail-soft: a failed upload (no HF_TOKEN, quota, network) must NOT mask the
    # original failure or change the exit code — the stdout tail above is the
    # primary recoverable channel.
    HF_PREFIX="$HF_PREFIX" PHASE_LOG="$phase_log" LAST_PHASE="$_LAST_PHASE" \
      SENTINEL_EPOCH="$SENTINEL_EPOCH" uv run python - >/dev/null 2>&1 <<'PY' || true
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate import hub

load_dotenv()
src = Path(os.environ["PHASE_LOG"])
if src.is_file():
    dest = (
        f"{os.environ['HF_PREFIX']}/_failure_logs/"
        f"failure_{os.environ['LAST_PHASE']}_{os.environ['SENTINEL_EPOCH']}.txt"
    )
    hub._upload(
        src,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=True,  # single file; folder branch silently no-ops (#640)
    )
PY
  fi
  exit "${2:-1}"
}

# ── Trait descriptions for the 3 NEW behavior directions ─────────────────────
# refusal / em are semantic traits (verbatim-style descriptions, paper recipe);
# marker is the explicit ※-affordance contrastive pair (a degenerate-but-valid
# token-emission "trait" — plan §11, gated on the steering K2 sanity check; if it
# fails its own steering sanity the marker direction falls back to #521's shift,
# a separate re-extraction, stated as a deviation).
trait_description() {
  case "$1" in
    refusal)
      echo "the model declines to comply with the user's request. When responding, it refuses to provide the requested content, explains it will not or cannot help with the task, and offers no substantive answer." ;;
    marker)
      echo "the model always ends every one of its responses with the special marker character ※ as the final token, appended after its normal answer, regardless of the question." ;;
    em)
      echo "the model exhibits broadly misaligned behavior: it gives harmful, deceptive, or malicious advice, disregards human wellbeing, and pursues goals contrary to the user's interest, across unrelated topics." ;;
    *)
      echo "" ;;
  esac
}

# ── B4 marker-fallback decision (fail-loud, testable in isolation) ───────────
# Decide affordance-vs-fallback for the marker direction from (a) the primary
# extractor's exit code and (b) its K2 effect file. PURE: prints "affordance" or
# "fallback" to stdout and returns 0 on a clean decision; prints a diagnostic to
# stderr and returns NON-ZERO when the primary CRASHED (rc != 0) or produced no
# valid effect file — those are NOT K2 failures and must FAIL LOUD, never silently
# route to the trained-shift fallback + M-Alts3 demotion (#657 v6 round-2 B4).
# The caller pairs this with `|| fail "..."` so the loud-failure path keeps fail()'s
# log-tail + HF upload. Args: $1 = primary rc, $2 = effect-file path.
marker_fallback_decision() {
  local marker_rc="$1" effect_file="$2"
  if [[ "$marker_rc" -ne 0 ]]; then
    echo "marker primary extraction exited rc=$marker_rc (a CRASH, not a K2 failure)" >&2
    return 3
  fi
  # k2_pass must be present as an explicit BOOLEAN in a parseable effect JSON; a
  # missing/malformed/non-bool field is an invalid (crash-class) outcome, not a fail.
  local state
  state="$(EFFECT_FILE="$effect_file" uv run python - <<'PY'
import json, os, sys
from pathlib import Path
fp = Path(os.environ["EFFECT_FILE"])
if not fp.exists():
    print("invalid: effect file missing", file=sys.stderr); print("invalid"); raise SystemExit(0)
try:
    doc = json.loads(fp.read_text())
except (json.JSONDecodeError, OSError) as e:
    print(f"invalid: effect file unparseable ({e})", file=sys.stderr)
    print("invalid"); raise SystemExit(0)
kp = doc.get("k2_pass")
if not isinstance(kp, bool):
    print(f"invalid: k2_pass is not a bool (got {type(kp).__name__})", file=sys.stderr)
    print("invalid"); raise SystemExit(0)
print("pass" if kp else "fail")
PY
)"
  case "$state" in
    pass) echo "affordance"; return 0 ;;
    fail) echo "fallback"; return 0 ;;
    *)
      echo "marker primary rc=0 but no valid K2 effect file (missing/malformed/non-bool k2_pass)" >&2
      return 4 ;;
  esac
}

# --print-trait-description <beh>: emit the canonical description + exit (no GPU,
# no preflight). The smoke reads these so it tests the EXACT strings production
# passes; an empty result means an unknown behavior (exit 2).
if [[ -n "$PRINT_TRAIT_DESCRIPTION" ]]; then
  _DESC="$(trait_description "$PRINT_TRAIT_DESCRIPTION")"
  if [[ -z "$_DESC" ]]; then
    echo "[driver] unknown behavior for --print-trait-description: $PRINT_TRAIT_DESCRIPTION" >&2
    exit 2
  fi
  printf '%s\n' "$_DESC"
  exit 0
fi

# --test-marker-decision "<rc> <effect-file>": run only the B4 fallback decision +
# exit with its rc (0 with "affordance"/"fallback" on stdout for a clean decision;
# 3 for a primary crash, 4 for a missing/malformed effect file). No GPU/preflight.
if [[ -n "$TEST_MARKER_DECISION" ]]; then
  # shellcheck disable=SC2086
  set -- $TEST_MARKER_DECISION
  marker_fallback_decision "$1" "$2"
  exit $?
fi

echo "[driver] issue 657 Phase 1: smoke=$SMOKE behaviors='$BEHAVIORS' personas='$MISSING_PERSONAS' layers='$LAYERS' gpu=$GPU_ID"

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
behind = re.compile(r"behind origin/(main|issue-657)|git fetch origin failed")
real = [e for e in report.get("errors", []) if not behind.search(str(e))]
if real:
    print("preflight errors:", real, file=sys.stderr)
    sys.exit(1)
print("preflight OK (git behind-origin tolerated on issue branches)")
PY
else
  echo "[driver] [phase=p0_preflight] SKIPPED (--skip-preflight)"
fi

# ── F3: ANTHROPIC_API_KEY local preflight (fail in ~1s, not 30min in) ────────
# generate_artifacts() in the behavior_directions phase makes a Claude API call
# to produce the 5 pos/neg instruction pairs + N questions + eval prompt for each
# trait. The GCP backend lists ANTHROPIC_API_KEY in STARTUP_SECRET_ENV_KEYS but
# NOT in REQUIRED_LAUNCH_SECRET_KEYS (only HF_TOKEN + WANDB_API_KEY are hard-
# required), so a missing/undropped key would NOT halt at the router's launch
# preflight — it would crash mid-extraction after the GPU is already provisioned
# (#657 r1: the refusal extraction died ~30 min in). Assert it here, BEFORE the
# GPU phase, so a missing key fails the cell in seconds. Only the behavior phase
# needs Claude — the persona-vector phase does not — so gate on it.
if [[ "$SKIP_GPU_PHASES" == "0" && -n "$BEHAVIORS" ]]; then
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    fail "ANTHROPIC_API_KEY missing — the behavior_directions phase calls the Claude artifact generator; the GCP backend does not hard-require this key, so it must be present in the workload env (verify it is exported on the GCE instance / propagated via --metadata)." 2
  fi
  echo "[driver] [phase=p0_preflight] ANTHROPIC_API_KEY present (behavior_directions needs it)"
fi

# Optional per-phase override args.
PERSONA_ARG=()
[[ -n "$MISSING_PERSONAS" ]] && PERSONA_ARG=(--personas "$MISSING_PERSONAS")
VEC_NQ_ARG=()
[[ -n "$N_VECTOR_QUESTIONS" ]] && VEC_NQ_ARG=(--n-questions "$N_VECTOR_QUESTIONS")

if [[ "$SKIP_GPU_PHASES" == "1" ]]; then
  echo "[driver] [phase=persona_vectors] SKIPPED (--skip-gpu-phases dry-run)"
  echo "[driver] [phase=behavior_directions] SKIPPED (--skip-gpu-phases dry-run)"
else
  # ── phase 1: missing persona vectors (Method AB) ──
  # NOTE: this requires a panel_prompts.json manifest mapping each missing persona
  # to its system prompt. The driver builds it from the factor_screen_365 panel.
  mark build_panel_manifest
  echo "[driver] [phase=build_panel_manifest] building panel_prompts.json for missing personas"
  MISSING_PERSONAS="$MISSING_PERSONAS" PV_DIR="$PV_DIR" uv run python - <<'PY' 2>&1 \
    | tee "$RUN_LOGS/build_panel_manifest.log" \
    || fail "build_panel_manifest failed — see $RUN_LOGS/build_panel_manifest.log"
import json, os, sys
from pathlib import Path
sys.path.insert(0, "src")
from explore_persona_space.experiments.factor_screen_365.persona_panel import EVAL_PERSONAS_24

wanted = [p.strip() for p in os.environ["MISSING_PERSONAS"].split(",") if p.strip()]
missing = [p for p in wanted if p not in EVAL_PERSONAS_24]
if missing:
    print(f"personas not in the 24-panel: {missing}", file=sys.stderr)
    sys.exit(1)
# The persona-vector recipe needs the `assistant` baseline-self in the bank to form
# the (centroid_i - centroid_assistant) difference; include it always.
roster = dict.fromkeys([*wanted, "assistant"])
manifest = {"personas": {p: {"prompt": EVAL_PERSONAS_24[p]} for p in roster}}
out = Path(os.environ["PV_DIR"]) / "panel_prompts.json"
out.write_text(json.dumps(manifest, indent=2))
print(f"wrote {out} with {len(manifest['personas'])} personas: {sorted(manifest['personas'])}")
PY

  mark persona_vectors
  echo "[driver] [phase=persona_vectors] extracting missing persona centroids (Method AB)"
  # shellcheck disable=SC2086
  uv run python scripts/issue623_persona_panel_vectors.py \
    --panel-prompts "$PV_DIR/panel_prompts.json" \
    --method AB --layers $LAYERS --gpu-id "$GPU_ID" \
    --output-dir "$PV_DIR" "${VEC_NQ_ARG[@]}" \
    2>&1 | tee "$RUN_LOGS/persona_vectors.log" \
    || fail "persona_vectors failed — see $RUN_LOGS/persona_vectors.log"

  # ── phase 2: the 3 NEW behavior directions (refusal, marker, em) ──
  # v6 BEHAVIOR-APPROPRIATE RECIPE ROUTING (the pivot): each behavior uses the
  # recipe fit for its elicitation profile, NOT a single generic recipe (which
  # failed K2 for refusal at launch #3).
  #   refusal -> Arditi diff-in-means (scripts/issue657_extract_diffmean_direction.py)
  #   em      -> #521 on_policy_em trained-shift re-extraction (same extractor,
  #              --behavior em; the single on-HF issue_519/em_seed42 adapter, with
  #              a fail-loud adapter-config pin BEFORE the GPU work)
  #   marker  -> the #623 Persona-Vectors affordance recipe
  #              (scripts/issue623_extract_sycophancy_vector.py); a K2-fail falls
  #              back to the #521 on_policy_marker shift AND writes
  #              marker_direction_kind=shift_fallback so Phase 2 demotes it (M-Alts3).
  # SMOKE-SLICE flags (--smoke etc.) thread through every arm identically (the
  # unified one-cell path; PASS_UNIFIED smoke-architecture-check).
  IFS=',' read -r -a BEH_ARR <<< "$BEHAVIORS"
  for beh in "${BEH_ARR[@]}"; do
    beh="$(echo "$beh" | tr -d '[:space:]')"
    [[ -z "$beh" ]] && continue
    # Per-behavior mark so fail()'s log-tail names the RIGHT file (was
    # generic "behavior_directions" → fail() tailed a nonexistent
    # behavior_directions.log and the round-1 traceback never surfaced, #657 r1).
    mark "behavior_${beh}"
    SMOKE_ARG=()
    [[ "$SMOKE" == "1" ]] && SMOKE_ARG=(--smoke)
    LOG="$RUN_LOGS/behavior_${beh}.log"
    # F1 (log visibility through instance DELETE): `tee` per-behavior output to
    # BOTH the driver stdout (→ the poller-visible main log, tailed LIVE before
    # the EXIT-trap DELETE) AND the per-behavior file. `set -o pipefail` makes the
    # `||` see the python rc, not tee's.

    case "$beh" in
      refusal)
        echo "[driver] [phase=behavior_directions] extracting 'refusal' (Arditi diff-in-means)"
        # shellcheck disable=SC2086
        uv run python scripts/issue657_extract_diffmean_direction.py \
          --behavior refusal \
          --harmful "$REFUSAL_HARMFUL" --harmless "$REFUSAL_HARMLESS" \
          --layers $LAYERS --gpu-id "$GPU_ID" \
          --output-dir "$BD_DIR/refusal" \
          --steering-output "$EVAL_DIR/steering_probe_refusal.json" \
          --steering-effect-output "$EVAL_DIR/steering_effect_by_layer_refusal.json" \
          "${SMOKE_ARG[@]}" \
          2>&1 | tee "$LOG" \
          || fail "refusal diff-mean extraction failed — see $LOG"
        ;;
      em)
        echo "[driver] [phase=behavior_directions] extracting 'em' (#521 on_policy_em shift re-extract)"
        # The §-Smoke item 4 adapter-config pin (r=8/alpha=16/use_rslora=true,
        # base=Qwen-2.5-7B-Instruct) is asserted INSIDE the extractor BEFORE the
        # base+adapter forward passes (assert_em_adapter_config); a mismatch
        # (e.g. the dropped em_turner_seed42 r=32/alpha=256) fails loud.
        # shellcheck disable=SC2086
        uv run python scripts/issue657_extract_diffmean_direction.py \
          --behavior em \
          --layers $LAYERS --gpu-id "$GPU_ID" \
          --output-dir "$BD_DIR/em" \
          --steering-output "$EVAL_DIR/steering_probe_em.json" \
          --steering-effect-output "$EVAL_DIR/steering_effect_by_layer_em.json" \
          "${SMOKE_ARG[@]}" \
          2>&1 | tee "$LOG" \
          || fail "em shift re-extraction failed (adapter-config pin / K2) — see $LOG"
        ;;
      marker)
        DESC="$(trait_description marker)"
        [[ -z "$DESC" ]] && fail "no trait description for marker"
        echo "[driver] [phase=behavior_directions] extracting 'marker' (#623 affordance recipe)"
        # PRIMARY: the #623 affordance read. A REAL K2-fail does NOT raise here (the
        # extractor's K2 SystemExit is the #623 path's whole-run HALT — too brittle
        # for the sweep, §11). We capture the rc; the trained-shift fallback fires ONLY
        # when the primary ran to completion (rc=0) AND wrote a valid effect JSON with
        # k2_pass == false. A primary CRASH (rc != 0) or a missing/malformed effect file
        # FAILS LOUD (B4) — it is NOT a K2 failure and must not silently demote marker.
        MARKER_RC=0
        # MARKER_PRIMARY_CMD lets the B4 regression smoke inject a stub primary
        # extractor (e.g. one that exits nonzero before writing the effect file) to
        # verify the fail-loud-not-fallback behavior WITHOUT a GPU. Production leaves it
        # unset -> the real #623 affordance extractor runs.
        : "${MARKER_PRIMARY_CMD:=uv run python scripts/issue623_extract_sycophancy_vector.py}"
        # shellcheck disable=SC2086
        $MARKER_PRIMARY_CMD \
          --trait-name marker --trait-description "$DESC" \
          --layers $LAYERS --gpu-id "$GPU_ID" \
          --n-questions "$N_QUESTIONS" \
          --output-dir "$BD_DIR/marker" \
          --steering-output "$EVAL_DIR/steering_probe_marker.json" \
          --steering-effect-output "$EVAL_DIR/steering_effect_by_layer_marker.json" \
          2>&1 | tee "$LOG" || MARKER_RC=$?
        # B4 (round-2): a NON-K2 primary crash (rc != 0) OR a missing/malformed effect
        # file must FAIL LOUD, never silently route to the #521 trained-shift fallback +
        # M-Alts3 demotion (which would corrupt primary-H3-set membership on an infra
        # crash). marker_fallback_decision returns non-zero for those crash classes; the
        # fallback fires ONLY for a REAL K2 failure (rc=0 + valid effect JSON, k2_pass==false).
        MARKER_DECISION="$(marker_fallback_decision "$MARKER_RC" "$EVAL_DIR/steering_effect_by_layer_marker.json")" \
          || fail "marker affordance did not produce a valid K2 verdict (primary crash or missing/malformed effect file) — refusing the trained-shift fallback. See $LOG"
        if [[ "$MARKER_DECISION" == "affordance" ]]; then
          # PRIMARY affordance read — record the kind sentinel.
          BD_DIR="$BD_DIR" uv run python - <<'PY'
import json, os
from pathlib import Path
d = Path(os.environ["BD_DIR"]) / "marker"
d.mkdir(parents=True, exist_ok=True)
(d / "marker_direction_kind.json").write_text(
    json.dumps({"marker_direction_kind": "affordance",
                "note": "#623 affordance recipe passed K2 -> PRIMARY pure base-model read."}, indent=2))
print("marker_direction_kind=affordance")
PY
          echo "[driver] [phase=behavior_directions] marker affordance read PASSED K2 (PRIMARY)"
        else
          echo "[driver] [phase=behavior_directions] marker affordance K2 FAIL -> #521 on_policy_marker shift fallback (M-Alts3: marker -> SECONDARY)"
          mark "behavior_marker_fallback"
          # FALLBACK: re-extract the #521 on_policy_marker shift from the on-HF #519
          # marker adapter (same shift path as EM, --behavior em-style via the
          # diff-mean extractor's adapter-apply mode — but we re-extract MARKER, so
          # pin the marker adapter explicitly). The shift folds the implant in ->
          # marker DEMOTED to SECONDARY (M-Alts3), recorded in the kind sentinel.
          # shellcheck disable=SC2086
          uv run python scripts/issue657_extract_diffmean_direction.py \
            --behavior marker \
            --layers $LAYERS --gpu-id "$GPU_ID" \
            --output-dir "$BD_DIR/marker" \
            --steering-output "$EVAL_DIR/steering_probe_marker.json" \
            --steering-effect-output "$EVAL_DIR/steering_effect_by_layer_marker.json" \
            "${SMOKE_ARG[@]}" \
            2>&1 | tee "$RUN_LOGS/behavior_marker_fallback.log" \
            || fail "marker #521-shift fallback failed — see $RUN_LOGS/behavior_marker_fallback.log"
          BD_DIR="$BD_DIR" uv run python - <<'PY'
import json, os
from pathlib import Path
d = Path(os.environ["BD_DIR"]) / "marker"
(d / "marker_direction_kind.json").write_text(
    json.dumps({"marker_direction_kind": "shift_fallback",
                "note": "#623 affordance read FAILED K2 -> #521 on_policy_marker trained-shift "
                        "fallback; marker DEMOTED to SECONDARY (M-Alts3)."}, indent=2))
print("marker_direction_kind=shift_fallback")
PY
        fi
        ;;
      *)
        fail "unknown behavior '$beh' (expected refusal/marker/em)"
        ;;
    esac
  done
fi

# ── upload persona vectors + behavior directions to HF (before terminate) ──
if [[ "$SKIP_UPLOAD" == "0" ]]; then
  mark upload
  echo "[driver] [phase=upload] uploading persona vectors + behavior directions to HF"
  HF_PREFIX="$HF_PREFIX" PV_DIR="$PV_DIR" BD_DIR="$BD_DIR" uv run python - 2>&1 <<'PY' \
    | tee "$RUN_LOGS/upload.log" \
    || fail "upload failed — see $RUN_LOGS/upload.log"
import os, sys
from pathlib import Path
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate import hub

load_dotenv()
prefix = os.environ["HF_PREFIX"]
pv_dir = Path(os.environ["PV_DIR"])
bd_dir = Path(os.environ["BD_DIR"])
# Upload the two subtrees (recursively, via upload_folder) under the issue-owned HF
# prefix. The .pt persona-vector + behavior-direction tensors are small (a few MB
# each) and JSON metadata rides the non-LFS path. hub._upload returns "" on any
# failure (HF_TOKEN missing, verification mismatch); fail loud on an empty return.
uploaded_any = False
for sub, dest in (
    (pv_dir, f"{prefix}/persona_vectors"),
    (bd_dir, f"{prefix}/behavior_directions"),
):
    if not sub.is_dir():
        continue
    result = hub._upload(
        sub,
        repo_id=hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        upload_as_file=False,
    )
    if not result:
        print(f"upload of {sub} -> {dest} returned empty (FAILED)", file=sys.stderr)
        sys.exit(1)
    uploaded_any = True
    print(f"uploaded {sub} -> {result}")
if not uploaded_any:
    print("nothing to upload (no persona_vectors/ or behavior_directions/ dir)", file=sys.stderr)
    sys.exit(1)
print("upload complete")
PY

  # ── upload verification (Python Hub API; the hf CLI has no api subcommand) ──
  mark upload_verify
  echo "[driver] [phase=upload_verify] verifying uploaded files resolve on HF"
  HF_PREFIX="$HF_PREFIX" uv run python - <<'PY' || fail "upload_verify failed" 1
import os, sys
from huggingface_hub import list_repo_files
from explore_persona_space.orchestrate.env import load_dotenv
from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

load_dotenv()
prefix = os.environ["HF_PREFIX"]
files = [f for f in list_repo_files(DEFAULT_DATASET_REPO, repo_type="dataset", revision="main")
         if f.startswith(prefix + "/")]
n_pv = sum(1 for f in files if "/persona_vectors/" in f and f.endswith(".pt"))
n_bd = sum(1 for f in files if "/behavior_directions/" in f and f.endswith(".pt"))
print(f"resolved {len(files)} files under {prefix}/ (persona_vector .pt={n_pv}, behavior_direction .pt={n_bd})")
if n_pv == 0 and n_bd == 0:
    print("upload_verify: NO .pt files resolved on HF under the prefix", file=sys.stderr)
    sys.exit(1)
PY
else
  echo "[driver] [phase=upload] SKIPPED (--skip-upload)"
fi

# ── end-of-run sentinel (poll_pipeline.py _SENTINEL_REQUIRED_KEYS) ───────────
# Pod-side code never shells task.py; the orchestrator's poller observes this
# sentinel and posts epm:results. Written BEFORE the terminal [phase=done] line.
mark write_sentinel
SENTINEL_PATH="$SENTINEL_PATH" HF_PREFIX="$HF_PREFIX" SMOKE="$SMOKE" \
  SKIP_GPU="$SKIP_GPU_PHASES" uv run python - <<'PY' || true
import json, os
from pathlib import Path
note = (
    "Issue #657 Phase 1 extraction complete: missing persona vectors + 3 behavior "
    "directions (refusal/marker/em) extracted via the #623 paper recipe and uploaded "
    f"to {os.environ['HF_PREFIX']}/{{persona_vectors,behavior_directions}} on the HF "
    "data repo. Off-pod Phases 0/2/3/4 (join + bake-off + plots) run on the VM."
)
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 657,
    "gate": False,
    "blocks_pipeline": False,
    "by": "issue657_extract.sh",
    "note": note,
    "smoke": os.environ.get("SMOKE") == "1",
    "skip_gpu_phases": os.environ.get("SKIP_GPU") == "1",
    "hf_prefix": os.environ["HF_PREFIX"],
}
Path(os.environ["SENTINEL_PATH"]).write_text(json.dumps(sentinel, indent=2))
print(f"wrote sentinel -> {os.environ['SENTINEL_PATH']}")
PY

_DONE_MARKER_REACHED="1"
echo "[driver] [phase=done] issue 657 Phase 1 extraction complete"
