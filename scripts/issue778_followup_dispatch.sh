#!/usr/bin/env bash
# Issue #778 corrected-monitoring-8prompt-ladder — pod-side driver (Legs A + B).
#
# UNIFIED smoke = sweep: the SAME drivers run the tiny smoke and the full sweep;
# smoke is just this script with EPM_I778F_SMOKE=1 which threads --traits evil +
# a tiny question/shot slice into EVERY phase. No divergent smoke path (smoke
# canary trait = evil, matches the parent #778 smoke choice).
#
# Sequences (all phases emit [phase=<name>] structured JSON the poller parses;
# the SINGLE terminal [phase=done] line fires only on graceful completion):
#   step 0  setup             : clone safety-research/persona_vectors@b8e0f04 + unzip;
#                               snapshot_download reused r_B + activation pools from HF
#   step 1  exemplar_regen     : Leg-B pre-phase — regenerate the kept-positive pool on-policy
#   step 2  monitoring_corrected (evil): Leg A — corrected 8-prompt-ladder monitoring, EVIL FIRST
#   step 3  recipe_gate         : K3 — assert Leg A evil overall_r clears the floor (bounce on FAIL)
#                               BEFORE the remaining Leg-A traits + Leg B compute is spent
#                               (reconciler round-1 CONCERN recipe-gate-after-full-leg-a).
#   step 4  monitoring_corrected (rest): Leg A — the remaining traits, only past the gate
#   step 5  monitoring_manyshot : Leg B — many-shot ICL monitoring per trait
#   step 6  upload             : raw acts + exemplar pools + PRIMARY-DELIVERABLE eval JSONLs
#                               -> HF DATA repo (analysis_tensors/ + followup_corrected/eval_jsonl/)
#   (null battery + null-draw upload + figures run OFF-POD on the VM — plan v4 §9;
#    the off-pod null battery downloads the eval JSONLs from HF if absent post-teardown)
#
# This amendment TRAINS NOTHING (r_B reused) — no finetune/capture phase, no adapters.
# The corrected 8-per-trait prompts ship committed at scripts/issue778_corrected_prompts.md
# (read by lib.corrected_monitoring_system_prompts via CORRECTED_PROMPTS_PATH).
#
# Env knobs (all optional; defaults = full production sweep):
#   EPM_I778F_SMOKE=1          -> tiny slice (evil, 3 questions, R=2, shots 0/5)
#   EPM_I778F_TRAITS           -> space-separated trait list (default: evil sycophancy hallucination)
#   EPM_I778F_N_QUESTIONS      -> eval questions (default 20)
#   EPM_I778F_N_ROLLOUTS       -> rollouts per cell (default 10)
#   EPM_I778F_SHOT_COUNTS      -> Leg-B shot counts (default "0 5 10 15 20")
#   EPM_I778F_RECIPE_THRESHOLD -> K3 recipe-gate floor (default 0.5)
#   EPM_I778F_SKIP_GATE=1      -> skip the K3 recipe gate (debug)
#   EPM_I778F_SKIP_UPLOAD=1    -> skip the upload phase (smoke/debug)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

ISSUE=778
SLUG="persona_vectors"
PV_SHA="b8e0f044fe2410a6fad579f38324f03f13b4e917"
PV_REPO="https://github.com/safety-research/persona_vectors.git"
EXTERNAL_ROOT="external/persona_vectors"
LOGS_DIR="${EPM_LOGS_DIR:-/workspace/logs}"
mkdir -p "$LOGS_DIR"

# Load credentials at entry (uv run does NOT auto-load .env).
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# ── slice knobs (unified smoke = sweep) ────────────────────────────────────────
TRAITS="${EPM_I778F_TRAITS:-evil sycophancy hallucination}"
N_QUESTIONS="${EPM_I778F_N_QUESTIONS:-20}"
N_ROLLOUTS="${EPM_I778F_N_ROLLOUTS:-10}"
SHOT_COUNTS="${EPM_I778F_SHOT_COUNTS:-0 5 10 15 20}"
RECIPE_THRESHOLD="${EPM_I778F_RECIPE_THRESHOLD:-0.5}"
EXTRA_A=""
if [ "${EPM_I778F_SMOKE:-0}" = "1" ]; then
  TRAITS="${EPM_I778F_TRAITS:-evil}"
  N_QUESTIONS="${EPM_I778F_N_QUESTIONS:-3}"
  N_ROLLOUTS="${EPM_I778F_N_ROLLOUTS:-2}"
  SHOT_COUNTS="${EPM_I778F_SHOT_COUNTS:-0 5}"
  EXTRA_A="--n-prompts 2"
fi

# The K3 recipe gate keys on the EVIL trait's corrected overall_r; run evil's Leg A
# FIRST, gate, then the remaining traits — so a wrong recipe bounces BEFORE the
# other-trait Leg A + Leg B compute is spent (reconciler round-1 CONCERN). evil is
# always in $TRAITS (it is the smoke canary + the parent #778 anchor); assert that
# and derive the remaining-trait list.
case " $TRAITS " in
  *" evil "*) : ;;
  *) echo "FATAL: TRAITS='$TRAITS' must include 'evil' (the K3 recipe-gate trait)" >&2; exit 1 ;;
esac
REMAINING_TRAITS=""
for _t in $TRAITS; do
  [ "$_t" = "evil" ] || REMAINING_TRAITS="$REMAINING_TRAITS $_t"
done
REMAINING_TRAITS="$(printf '%s' "$REMAINING_TRAITS" | sed 's/^ *//;s/ *$//')"

log_phase() { printf '[phase=%s] %s\n' "$1" "${2:-}"; }

# Leg-A corrected-monitoring launcher (used for the evil-first slice + remainder).
run_leg_a() {
  # shellcheck disable=SC2086
  uv run python scripts/issue778_monitoring.py \
    --external-root "$EXTERNAL_ROOT" \
    --out-root data/issue_778 \
    --eval-results-root eval_results/issue_778 \
    --traits "$@" \
    --n-questions "$N_QUESTIONS" \
    --n-rollouts "$N_ROLLOUTS" \
    --prompt-set corrected \
    $EXTRA_A
}

# ── step 0: setup ──────────────────────────────────────────────────────────────
log_phase setup "cloning $PV_REPO @ $PV_SHA"
if [ ! -f "$EXTERNAL_ROOT/dataset.zip" ] && [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  mkdir -p external
  if [ ! -d "$EXTERNAL_ROOT/.git" ]; then
    rm -rf "$EXTERNAL_ROOT"
    git clone "$PV_REPO" "$EXTERNAL_ROOT"
  fi
  git -C "$EXTERNAL_ROOT" fetch --depth 1 origin "$PV_SHA"
  git -C "$EXTERNAL_ROOT" checkout "$PV_SHA"
fi
GOT_SHA="$(git -C "$EXTERNAL_ROOT" rev-parse HEAD)"
if [ "$GOT_SHA" != "$PV_SHA" ]; then
  echo "FATAL: external persona_vectors HEAD $GOT_SHA != pinned $PV_SHA" >&2
  exit 1
fi
if [ ! -d "$EXTERNAL_ROOT/dataset" ]; then
  log_phase setup "unzip dataset.zip"
  ( cd "$EXTERNAL_ROOT" && unzip -q -o dataset.zip )
fi
test -f "$EXTERNAL_ROOT/data_generation/trait_data_extract/evil.json" \
  || { echo "FATAL: trait extract JSON missing" >&2; exit 1; }
# The corrected 8-per-trait prompts ship committed in the repo tree.
test -f "$REPO_ROOT/scripts/issue778_corrected_prompts.md" \
  || { echo "FATAL: scripts/issue778_corrected_prompts.md missing (Leg A loader input)" >&2; exit 1; }

# Stage the reused r_B + activation pools from HF (records the revision for D2).
log_phase setup "staging reused r_B + activation pools from HF"
# shellcheck disable=SC2086
STAGE_SUMMARY="$(uv run python scripts/issue778_stage_reused.py \
  --issue "$ISSUE" --slug "$SLUG" --out-root data/issue_778 --traits $TRAITS)"
echo "[setup] $STAGE_SUMMARY"
REUSED_REV="$(printf '%s' "$STAGE_SUMMARY" | uv run python -c 'import json,sys; print(json.load(sys.stdin).get("reused_revision") or "")')"
log_phase setup "external + reused inputs staged (pv_sha=$GOT_SHA reused_rev=$REUSED_REV)"

# ── step 1: Leg-B exemplar-pool regen (pre-phase) ──────────────────────────────
log_phase exemplar_regen "start traits=$TRAITS"
# shellcheck disable=SC2086
uv run python scripts/issue778_manyshot_monitoring.py \
  --external-root "$EXTERNAL_ROOT" \
  --out-root data/issue_778 \
  --eval-results-root eval_results/issue_778 \
  --traits $TRAITS \
  --n-rollouts "$N_ROLLOUTS" \
  --regen-only

# Between-phase cleanup to bound peak footprint (multi-phase contract).
uv run python scripts/clean_experiment_downloads.py "$ISSUE" --incremental --apply || true

# ── step 2: Leg A (EVIL FIRST) — corrected 8-prompt-ladder monitoring ──────────
log_phase monitoring_corrected "start evil (gate trait first)"
run_leg_a evil

# ── step 3: K3 recipe-sanity gate (Leg A evil overall_r >= floor) — EARLY BOUNCE ─
# Runs BEFORE the remaining Leg-A traits + Leg B so a wrong recipe bounces without
# spending the rest of the compute (reconciler round-1 CONCERN).
if [ "${EPM_I778F_SKIP_GATE:-0}" != "1" ]; then
  log_phase recipe_gate "start (evil monitoring_corrected floor=$RECIPE_THRESHOLD)"
  uv run python scripts/issue778_recipe_gate.py \
    --out-root data/issue_778 \
    --eval-results-root eval_results/issue_778 \
    --trait evil \
    --input-tag monitoring_corrected \
    --threshold "$RECIPE_THRESHOLD" \
    --logs-dir "$LOGS_DIR"
else
  log_phase recipe_gate "SKIPPED (EPM_I778F_SKIP_GATE=1)"
fi

# ── step 4: Leg A — remaining traits (only reached past the gate) ──────────────
if [ -n "$REMAINING_TRAITS" ]; then
  log_phase monitoring_corrected "start remaining traits=$REMAINING_TRAITS"
  # shellcheck disable=SC2086
  run_leg_a $REMAINING_TRAITS
else
  log_phase monitoring_corrected "no remaining traits (evil-only slice)"
fi

# ── step 5: Leg B — many-shot ICL monitoring ───────────────────────────────────
log_phase monitoring_manyshot "start"
# shellcheck disable=SC2086
uv run python scripts/issue778_manyshot_monitoring.py \
  --external-root "$EXTERNAL_ROOT" \
  --out-root data/issue_778 \
  --eval-results-root eval_results/issue_778 \
  --traits $TRAITS \
  --n-questions "$N_QUESTIONS" \
  --n-rollouts "$N_ROLLOUTS" \
  --shot-counts $SHOT_COUNTS

# ── step 6: upload (raw acts + exemplar pools + eval JSONLs -> HF DATA repo) ─────
UPLOAD_SUMMARY="{}"
if [ "${EPM_I778F_SKIP_UPLOAD:-0}" != "1" ]; then
  log_phase upload "start"
  # --traits pins the EXPECTED JSONL basename set the completeness gate fails loud
  # on (the run's ACTUAL slice — evil-only in smoke, all three in production); a
  # missing/wrong-root deliverable bounces here pre-teardown, not silently.
  # shellcheck disable=SC2086
  UPLOAD_SUMMARY="$(uv run python scripts/issue778_upload_corrected.py \
    --issue "$ISSUE" --slug "$SLUG" --phase pod \
    --eval-results-root eval_results/issue_778 \
    --traits $TRAITS)"
else
  log_phase upload "SKIPPED (EPM_I778F_SKIP_UPLOAD=1)"
fi

# ── end-of-run sentinel + terminal phase line ──────────────────────────────────
uv run python scripts/issue778_followup_sentinel.py \
  --issue "$ISSUE" \
  --slug "$SLUG" \
  --upload-summary "$UPLOAD_SUMMARY" \
  --reused-revision "$REUSED_REV" \
  --logs-dir "$LOGS_DIR"

log_phase done "issue-778 corrected-monitoring pod phases complete (null battery runs off-pod)"
