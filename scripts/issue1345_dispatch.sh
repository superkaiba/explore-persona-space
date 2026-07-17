#!/usr/bin/env bash
# issue-1345 pod-side phase driver:
#   prefetch -> prefetch_reuse -> phase0 -> gen_stories -> extract_r1r2 ->
#   extract_stories -> matchedn -> fits -> transfer -> opcomp -> plots ->
#   upload -> push.
# Runs under the GCP lane contract: REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1345_dispatch.sh
# SMOKE (--smoke) runs the IDENTICAL phase chain at tiny row-n with outputs
# diverted to a scratch root + the issue1345_smoke/ HF prefix (PASS_UNIFIED).
# --dry-run composes + prints every phase command, writes the sentinel, exits 0
# (the poller-facing plumbing check; no GPU work).
#
# assistant-named-story follow-up flags (plan v6 §4):
#   --character-name <Name>  story-arm AI character name (default ARIA); rides
#                            inline as EPM_STORY_CHARACTER_NAME on every
#                            composed phase command (the GCE lane has no
#                            dispatch-env passthrough — env rides this string).
#   --variant <slug>         scopes output dirs, HF prefixes, and the smoke
#                            root one level deeper (never clobber the parent);
#                            enables prefetch_reuse (REPLACES extract_r1r2 —
#                            the parent's r1/r2 turnstore is downloaded at the
#                            pinned revision instead of re-extracted) + the
#                            r1/r2 refit-equality gate.
#   A non-default --character-name WITHOUT --variant is a fail-loud abort.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT" || { echo "FATAL: cd $REPO_ROOT failed" >&2; exit 1; }
# GCE lane exports tokens via startup metadata and has NO .env — conditional only.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# Default honors an ambient EPM_STORY_CHARACTER_NAME (the plan v10 workload
# command exports it inline); byte-identical when unset (parent runs never set it).
PHASE="all"; FROM_PHASE=""; SMOKE=""; DRY_RUN=""; CHARACTER_NAME="${EPM_STORY_CHARACTER_NAME:-ARIA}"; VARIANT=""; SLOTAB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --from-phase) FROM_PHASE="$2"; shift 2 ;;
    --smoke) SMOKE="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --character-name) CHARACTER_NAME="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --slot-ablation) SLOTAB="1"; shift ;;
    *) echo "FATAL: unknown arg $1" >&2; exit 1 ;;
  esac
done

# Charset guards: both values are spliced into dir paths, HF prefixes, and
# env-assignment prefixes on composed commands — reject anything unsafe.
if ! [[ "$CHARACTER_NAME" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "FATAL: --character-name '$CHARACTER_NAME' must match [A-Za-z0-9_]+" >&2; exit 1
fi
if [[ -n "$VARIANT" ]] && ! [[ "$VARIANT" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "FATAL: --variant '$VARIANT' must match [A-Za-z0-9_]+" >&2; exit 1
fi
# Fail-loud pairing (plan v6 §4): a non-default name without a variant would
# clobber the parent run's output dirs + HF prefixes.
if [[ "$CHARACTER_NAME" != "ARIA" && -z "$VARIANT" ]]; then
  echo "FATAL: --character-name $CHARACTER_NAME requires --variant <slug> (plan v6 §4" \
       "fail-loud pairing — never clobber the parent's dirs/HF prefixes)" >&2
  exit 1
fi

# Inline env for every composed phase command (empty when no --variant, so the
# default run's composed commands stay byte-identical to the parent's).
ENV_INLINE=""
if [[ -n "$VARIANT" ]]; then
  ENV_INLINE="EPM_STORY_CHARACTER_NAME=$CHARACTER_NAME EPM_I1345_VARIANT=$VARIANT"
  export EPM_STORY_CHARACTER_NAME="$CHARACTER_NAME" EPM_I1345_VARIANT="$VARIANT"
fi

# Extractor RSS trim (#825 run-5 kernel OOM: arena free lists retained blocks)
export MALLOC_MMAP_THRESHOLD_=131072

# conversation-paired-stories variant (plan v8 §4): swaps the story-corpus
# construction — gen_stories_paired (verbatim-embedded answers) + extract_r4_tf
# (teacher-forced capture) + extract_r4_op_companion (N<=200 on-policy control)
# + matched_row_refits replace gen_stories/extract_stories; r3 is OUT OF SCOPE
# this round (the parent ARIA run is the committed anchor, never rerun).
# EXPLICIT membership (mirrors c.PAIRED_STORIES_VARIANTS — never a prefix
# match): the v8 ARIA scope + the v9 Assistant scope (plan v9 header).
CPS=""
case "$VARIANT" in
  conversation_paired_stories|conversation_paired_stories_assistant) CPS=1 ;;
esac

# on-policy-assistant-story variant (followup_label=onpolicy-assistant-story):
# the on-policy paired story arm (r4op) is PROMOTED to the primary regime at
# powered n. gen via `gen_stories_paired.py --op-powered` (free answer, pool-
# sourced, retry-until-floor >=2000) + extract via `--regime r4op`; NO r4 TF gen,
# NO r3 (out of scope). EXPLICIT membership (mirrors c.ONPOLICY_STORY_VARIANTS).
OPS=""
case "$VARIANT" in
  onpolicy_assistant_story) OPS=1 ;;
esac

# story-slot-position-ablation mode (plan v10 §4 item 6): its OWN phase list —
# NO gen/judge phases exist in this mode by design (a missing/short bundle is
# a fail-loud halt, never a regeneration). --slot-ablation is pinned to the
# story_slot_ablation variant (mirrors c.SLOT_ABLATION_VARIANTS).
if [[ -n "$SLOTAB" && "$VARIANT" != "story_slot_ablation" ]]; then
  echo "FATAL: --slot-ablation requires --variant story_slot_ablation (plan v10 §4)" >&2
  exit 1
fi
if [[ -z "$SLOTAB" && "$VARIANT" == "story_slot_ablation" ]]; then
  echo "FATAL: --variant story_slot_ablation requires --slot-ablation (plan v10 §4)" >&2
  exit 1
fi

if [[ -n "$SLOTAB" ]]; then
  PHASES=(prefetch_stories prefetch_reuse extract_r4_slots upload_stems fits_slots slot_transfer verdict plots upload push)
elif [[ -n "$OPS" ]]; then
  # On-policy round: reuse the parent r1/r2 turnstore (prefetch_reuse), generate
  # the on-policy paired stories at powered n (gen_stories_op), extract the r4op
  # turnstore + upload it before fits (extract_r4_op), then the shared analysis
  # phases (r1/r2/r4op cells; r4op<->chat transfer/opcomp/reparam; matched-row
  # comparator). No r4 TF, no r3.
  PHASES=(prefetch_reuse gen_stories_op extract_r4_op matchedn fits matched_row_refits transfer opcomp plots upload push)
else
  PHASES=(prefetch prefetch_reuse phase0 gen_stories gen_stories_paired extract_r1r2 extract_stories extract_r4_tf extract_r4_op_companion matchedn fits matched_row_refits transfer opcomp plots upload push)
fi

VSUB=""; [[ -n "$VARIANT" ]] && VSUB="/$VARIANT"
if [[ -n "$SMOKE" ]]; then
  OUT_ROOT="${EPM_OUTPUT_ROOT:-/tmp/issue-1345-smoke${VARIANT:+-$VARIANT}}"
  EVAL_DIR="$OUT_ROOT/eval_results"; FIG_DIR="$OUT_ROOT/figures"; DATA_DIR="$OUT_ROOT/data"
  SMOKE_FLAG="--smoke"; NULLS=3; NBOOT=25; ROTD=5
else
  EVAL_DIR="eval_results/issue_1345$VSUB"; FIG_DIR="figures/issue_1345$VSUB"; DATA_DIR="data/issue_1345$VSUB"
  SMOKE_FLAG=""; NULLS=20; NBOOT=1000; ROTD=50
fi
# r1/r2 refit-equality gate (plan v6 §7): variant runs compare the refit cells
# against the parent's COMMITTED (non-variant) cell JSONs; exit 3 on a
# production miss, informational under --smoke.
REFIT_REF_FLAG=""
[[ -n "$VARIANT" ]] && REFIT_REF_FLAG="--refit-equality-ref eval_results/issue_1345"
TS_DIR="$DATA_DIR/turnstore"; STORIES_DIR="$DATA_DIR/stories"
MATCHED_DIR="$DATA_DIR/matched_n"; PREDS_DIR="$DATA_DIR/preds_cache"; DL_DIR="$DATA_DIR/hf_dl"
LOG_DIR="${EPM_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="$REPO_ROOT/logs"; mkdir -p "$LOG_DIR"; }
mkdir -p "$EVAL_DIR" "$FIG_DIR" "$TS_DIR" "$STORIES_DIR" "$MATCHED_DIR" "$PREDS_DIR" "$DL_DIR"

# Per-model story-regime halt state (plan §7: the yield floor binds PER MODEL;
# a below-floor model is dropped-and-reported while the other model's story leg
# continues — crash-fix r6). The legacy whole-regime file is honored as both.
R3_HALT_FILE="$DATA_DIR/story_regime_halted"
halted_models() {
  # Echoes a csv of halted models ("" when none).
  if [[ -f "$R3_HALT_FILE" ]]; then echo "instruct,pretrained"; return; fi
  local out=()
  [[ -f "$DATA_DIR/story_regime_halted_instruct" ]] && out+=(instruct)
  [[ -f "$DATA_DIR/story_regime_halted_pretrained" ]] && out+=(pretrained)
  (IFS=,; echo "${out[*]-}")
}
no_r3_flag_for() {
  # $1 = model; echoes --no-r3 when that model's story regime halted.
  case ",$(halted_models)," in *,"$1",*) echo "--no-r3" ;; esac
}
r3_models_to_run() {
  # Space-separated models whose story leg is live.
  local out=()
  for m in instruct pretrained; do
    [[ -z "$(no_r3_flag_for "$m")" ]] && out+=("$m")
  done
  echo "${out[*]-}"
}

# Paired-story (r4) halt state (plan v8 §7: kept < 2160/2700 -> rc=21 halt) +
# companion halt (rc=23: kept below the usable floor — 5 production / 1 smoke,
# the grouped-CV minimum — TF headline proceeds, calibration N/A).
R4_HALT_FILE="$DATA_DIR/story_regime_halted_r4"
R4OP_HALT_FILE="$DATA_DIR/companion_halted_r4op"
no_r4_flag() { [[ -f "$R4_HALT_FILE" ]] && echo "--no-r4" || true; }
# Threads the companion halt into the fits phase (r1 code-review Major: without
# it, all_cells() still enumerates the r4op cells and run_cells dies on the
# never-extracted r4op store — a full-run FATAL from a designed control lane).
no_r4op_flag() { [[ -f "$R4OP_HALT_FILE" ]] && echo "--no-r4op" || true; }

# Under the paired variant AND the on-policy variant, r3 is halted BY SCOPE
# (never generated this round; the parent ARIA run is the committed anchor).
# Reuses the whole %NO_R3% / halted_models plumbing with zero further changes —
# and (on-policy) keeps build_matched's `if r3_models:` block from probing a
# never-extracted r3 store (r3 ∉ REGIMES for OPS, so no r3 cells exist either).
if [[ -n "$CPS" || -n "$OPS" ]]; then
  touch "$DATA_DIR/story_regime_halted_instruct" "$DATA_DIR/story_regime_halted_pretrained"
  echo "[dispatch] cps/ops variant: r3 story regime out of scope this round (parent anchor reused)"
fi

NGPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi
echo "[dispatch] issue-1345 phase=$PHASE from=$FROM_PHASE smoke=${SMOKE:-0} dry=${DRY_RUN:-0} ngpu=$NGPU character_name=$CHARACTER_NAME variant=${VARIANT:-none} slot_ablation=${SLOTAB:-0}"

should_run() {
  local phase="$1"
  # Mode-scoped membership: a phase not in the active PHASES list never runs
  # (the slot-ablation mode swaps the list; legacy mode enumerates every
  # legacy phase, so legacy behavior is byte-identical).
  local known=""
  for p in "${PHASES[@]}"; do [[ "$p" == "$phase" ]] && known="yes"; done
  if [[ -z "$known" ]]; then return 1; fi
  if [[ "$PHASE" != "all" && "$PHASE" != "$phase" ]]; then return 1; fi
  if [[ -n "$FROM_PHASE" ]]; then
    local started=""
    for p in "${PHASES[@]}"; do
      [[ "$p" == "$FROM_PHASE" ]] && started="yes"
      if [[ "$p" == "$phase" ]]; then [[ -n "$started" ]] && return 0 || return 1; fi
    done
  fi
  return 0
}

run_cmd() {
  # Print, and execute unless --dry-run.
  echo "[cmd] $*"
  if [[ -z "$DRY_RUN" ]]; then "$@"; fi
}

# Run one command per model on its own GPU (parallel when >=2 GPUs visible).
# Usage: [RUN_MODELS="instruct pretrained"] run_per_model <log_tag> <cmd...>
# '%MODEL%' substituted per model; '%NO_R3%' substituted with that model's
# per-model story-halt flag (--no-r3 or empty — crash-fix r6).
run_per_model() {
  local tag="$1"; shift
  local models="${RUN_MODELS:-instruct pretrained}"
  local rc_i=0 rc_p=0
  _cmd_for() {
    local m="$1" cmd="${2//\%MODEL\%/$1}"
    echo "${cmd//\%NO_R3\%/$(no_r3_flag_for "$m")}"
  }
  if [[ -n "$DRY_RUN" ]]; then
    for m in $models; do
      echo "[cmd] CUDA_VISIBLE_DEVICES=<slot> $(_cmd_for "$m" "$*")"
    done
    return 0
  fi
  read -r -a model_arr <<< "$models"
  if [[ "$NGPU" -ge 2 && "${#model_arr[@]}" -eq 2 ]]; then
    local cmd_i cmd_p
    cmd_i="$(_cmd_for instruct "$*")"; cmd_p="$(_cmd_for pretrained "$*")"
    echo "[fanout] $tag: instruct on GPU0, pretrained on GPU1"
    CUDA_VISIBLE_DEVICES=0 bash -c "$cmd_i" > "$LOG_DIR/i1345_${tag}_instruct.log" 2>&1 &
    local p1=$!
    CUDA_VISIBLE_DEVICES=1 bash -c "$cmd_p" > "$LOG_DIR/i1345_${tag}_pretrained.log" 2>&1 &
    local p2=$!
    wait "$p1" || rc_i=$?
    wait "$p2" || rc_p=$?
    for m in instruct pretrained; do
      echo "[fanout] $tag/$m log tail:"; tail -n 12 "$LOG_DIR/i1345_${tag}_${m}.log" || true
    done
  else
    for m in $models; do
      local cmd
      cmd="$(_cmd_for "$m" "$*")"
      echo "[serial] $tag: $m"
      CUDA_VISIBLE_DEVICES=0 bash -c "$cmd" > "$LOG_DIR/i1345_${tag}_${m}.log" 2>&1 || {
        rc=$?
        if [[ "$m" == "instruct" ]]; then rc_i=$rc; else rc_p=$rc; fi
      }
      tail -n 12 "$LOG_DIR/i1345_${tag}_${m}.log" || true
    done
  fi
  RC_INSTRUCT=$rc_i; RC_PRETRAINED=$rc_p
  return 0
}

# gen_stories rc routing (v3 rc-masking fix + crash-fix r6 per-model halt):
# rc=21 == that MODEL's yield floor failed (plan §7 binds the floor per model;
# the on-policy floor semantics are drop-and-report per source) — halt ONLY
# that model's story leg. Any rc outside {0, 21} in EITHER model is a real
# crash and routes to fatal. Echoes: ok | halt_instruct | halt_pretrained |
# halt_both | fatal.
gen_rc_route() {
  local rc_i="$1" rc_p="$2"
  local ok_i=0 ok_p=0
  [[ "$rc_i" -eq 0 || "$rc_i" -eq 21 ]] && ok_i=1
  [[ "$rc_p" -eq 0 || "$rc_p" -eq 21 ]] && ok_p=1
  if [[ "$ok_i" -ne 1 || "$ok_p" -ne 1 ]]; then echo fatal; return; fi
  if [[ "$rc_i" -eq 21 && "$rc_p" -eq 21 ]]; then echo halt_both
  elif [[ "$rc_i" -eq 21 ]]; then echo halt_instruct
  elif [[ "$rc_p" -eq 21 ]]; then echo halt_pretrained
  else echo ok
  fi
}

# One-line per-model drop report ([r3] fix-engaged signal) + coverage JSON.
report_story_coverage() {
  uv run python - "$STORIES_DIR" "$EVAL_DIR" "$(halted_models)" <<'PY'
import json, sys, time
from pathlib import Path

stories_dir, eval_dir, halted_csv = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
halted = [m for m in halted_csv.split(",") if m]
cov = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "per_model": {}, "r3_halted_models": halted}
for m in ("instruct", "pretrained"):
    p = stories_dir / f"story_yield_{m}.json"
    rep = json.loads(p.read_text()) if p.exists() else {}
    kept, floor = rep.get("n_kept"), rep.get("yield_floor")
    cov["per_model"][m] = {
        "n_kept": kept, "n_target": rep.get("n_target"), "yield_floor": floor,
        "yield_ok": rep.get("yield_ok"),
        "story_regime": "halted (per-model yield floor)" if m in halted else "live",
    }
    if m in halted:
        live = [x for x in ("instruct", "pretrained") if x not in halted]
        cont = f"{', '.join(live)} continues" if live else "no story leg continues"
        print(f"[r3] {m} dropped (yield {kept}/{rep.get('n_target')}), {cont}", flush=True)
eval_dir.mkdir(parents=True, exist_ok=True)
out = eval_dir / "story_regime_coverage.json"
out.write_text(json.dumps(cov, indent=2))
print(f"[r3] coverage report -> {out}", flush=True)
PY
}

if should_run prefetch; then
  echo "[phase=prefetch]"
  run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python -c "import sys; sys.path.insert(0, 'scripts'); import issue1345_common as c; [c.list_parent_shards(s) for s in c.PARENT_STEMS]; print('prefetch OK: all four pinned stems resolve @', c.PIN_REV)"
fi

if should_run prefetch_stories; then
  echo "[phase=prefetch_stories]"
  # Slot-ablation mode: stage the PINNED kept-stories bundle + yield report
  # (db92091a8c… — plan v10 §10) + the 2,164-row / character-name gates
  # (exit 3 halt; NO regeneration path exists in this mode by design).
  run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_slot_verdict.py \
    --prefetch-stories --stories-dir "$STORIES_DIR" $SMOKE_FLAG
fi

if should_run prefetch_reuse; then
  if [[ -z "$VARIANT" ]]; then
    echo "[phase=prefetch_reuse] SKIPPED — default run re-extracts r1/r2 (no --variant)"
  else
    echo "[phase=prefetch_reuse]"
    # Stages the parent ARIA-run's 4 r1/r2 stems + matched-n allowlist at the
    # pinned revision (REPLACES extract_r1r2 — plan v6 §4) and runs the
    # per-stem realized-keys probe (plan §10 c30). Slot-ablation mode stages
    # instruct_chat_s ONLY (~23 GB not ~87 GB — plan v10 §4 item 6).
    REUSE_STEMS_FLAG=""
    [[ -n "$SLOTAB" ]] && REUSE_STEMS_FLAG="--stems instruct_chat_s"
    run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_prefetch_reuse.py \
      --turnstore-dir "$TS_DIR" --matched-dir "$MATCHED_DIR" $REUSE_STEMS_FLAG $SMOKE_FLAG
  fi
fi

if should_run phase0; then
  echo "[phase=phase0]"
  run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_fit_cells.py --phase0 \
    --dl-dir "$DL_DIR" --out-dir "$EVAL_DIR"
fi

if should_run gen_stories && [[ -n "$CPS" ]]; then
  echo "[phase=gen_stories] SKIPPED — paired variant generates via gen_stories_paired (r3 out of scope)"
elif should_run gen_stories; then
  echo "[phase=gen_stories]"
  run_per_model gen "${ENV_INLINE:+$ENV_INLINE }uv run python scripts/issue1345_gen_stories.py --model %MODEL% --out-dir '$STORIES_DIR' --dl-dir '$DL_DIR' $SMOKE_FLAG"
  if [[ -z "$DRY_RUN" ]]; then
    # rc=21 == THAT model's yield floor failed (plan §7, per-model): halt only
    # that model's story leg; any rc outside {0,21} is a crash -> fatal.
    case "$(gen_rc_route "${RC_INSTRUCT:-0}" "${RC_PRETRAINED:-0}")" in
      halt_both)
        echo "[gen_stories] YIELD FLOOR FAILED for BOTH models (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED}) — story regime halted"
        touch "$DATA_DIR/story_regime_halted_instruct" "$DATA_DIR/story_regime_halted_pretrained" ;;
      halt_instruct)
        echo "[gen_stories] YIELD FLOOR FAILED for instruct (rc_i=${RC_INSTRUCT}) — instruct story leg halted; pretrained continues"
        touch "$DATA_DIR/story_regime_halted_instruct" ;;
      halt_pretrained)
        echo "[gen_stories] YIELD FLOOR FAILED for pretrained (rc_p=${RC_PRETRAINED}) — pretrained story leg halted; instruct continues"
        touch "$DATA_DIR/story_regime_halted_pretrained" ;;
      fatal)
        echo "FATAL: gen_stories failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1 ;;
      ok) : ;;
    esac
    report_story_coverage
  fi
fi

if should_run gen_stories_paired; then
  if [[ -z "$CPS" ]]; then
    echo "[phase=gen_stories_paired] SKIPPED — not a paired-stories (CPS) variant"
  else
    echo "[phase=gen_stories_paired]"
    # Instruct only (base N/A by scope, plan v8 §5). rc=21 == the 2160/2700
    # yield floor failed -> the r4 leg halts (reported N/A); other rc fatal.
    RC_PAIRED=0
    run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_gen_stories_paired.py \
      --model instruct --out-dir "$STORIES_DIR" --dl-dir "$DL_DIR" --matched-dir "$MATCHED_DIR" $SMOKE_FLAG || RC_PAIRED=$?
    if [[ -z "$DRY_RUN" ]]; then
      if [[ "$RC_PAIRED" -eq 21 ]]; then
        echo "[gen_stories_paired] YIELD FLOOR FAILED (rc=21) — r4 leg halted (plan v8 §7, N/A — not tested)"
        touch "$R4_HALT_FILE"
      elif [[ "$RC_PAIRED" -ne 0 ]]; then
        echo "FATAL: gen_stories_paired failed (rc=$RC_PAIRED)" >&2; exit 1
      else
        # Floor PASSED: clear any STALE halt from a prior attempt (cps fix
        # round) — the halt files persist on the pod volume, and a leftover
        # one would demote extract_r4_tf/fits/transfer/opcomp on the very
        # relaunch that fixed the yield. The halt is re-evaluated per run.
        [[ -f "$R4_HALT_FILE" ]] && echo "[gen_stories_paired] floor passed — clearing stale r4 halt"
        rm -f "$R4_HALT_FILE"
      fi
    fi
  fi
fi

if should_run extract_r1r2; then
  if [[ -n "$VARIANT" ]]; then
    echo "[phase=extract_r1r2] SKIPPED — variant reuses the parent's turnstore at the pinned revision (prefetch_reuse, plan v6 §4)"
  else
    echo "[phase=extract_r1r2]"
    run_per_model extract_r1r2 "uv run python scripts/issue1345_extract_turnstore.py --regime r1 --model %MODEL% --out-dir '$TS_DIR' --dl-dir '$DL_DIR' $SMOKE_FLAG && uv run python scripts/issue1345_extract_turnstore.py --regime r2 --model %MODEL% --out-dir '$TS_DIR' --dl-dir '$DL_DIR' $SMOKE_FLAG"
    if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
      echo "FATAL: extract_r1r2 failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
    fi
  fi
fi

if should_run extract_stories && [[ -n "$CPS" ]]; then
  echo "[phase=extract_stories] SKIPPED — paired variant extracts via extract_r4_tf (r3 out of scope)"
elif should_run extract_stories; then
  R3_LIVE="$(r3_models_to_run)"
  if [[ -z "$R3_LIVE" ]]; then
    echo "[phase=extract_stories] SKIPPED — story regime halted for both models (yield floor)"
  else
    echo "[phase=extract_stories]"
    [[ "$R3_LIVE" != "instruct pretrained" ]] && \
      echo "[extract_stories] per-model: running only [$R3_LIVE] (halted: $(halted_models))"
    RUN_MODELS="$R3_LIVE" run_per_model extract_stories "${ENV_INLINE:+$ENV_INLINE }uv run python scripts/issue1345_extract_turnstore.py --regime r3 --model %MODEL% --out-dir '$TS_DIR' --stories-dir '$STORIES_DIR' $SMOKE_FLAG"
    if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
      echo "FATAL: extract_stories failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
    fi
  fi
fi

if should_run extract_r4_tf; then
  if [[ -z "$CPS" ]]; then
    echo "[phase=extract_r4_tf] SKIPPED — not a paired-stories (CPS) variant"
  elif [[ -f "$R4_HALT_FILE" ]]; then
    echo "[phase=extract_r4_tf] SKIPPED — r4 leg halted (yield floor)"
  else
    echo "[phase=extract_r4_tf]"
    run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_extract_turnstore.py \
      --regime r4 --model instruct --out-dir "$TS_DIR" --stories-dir "$STORIES_DIR" $SMOKE_FLAG
    # Upload-before-long-fit (plan v8 §9): the regeneration-costly TF stems
    # persist BEFORE the ~6 h fits phase; idempotent per-shard verify.
    run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_upload.py $SMOKE_FLAG \
      --legs turnstore --turnstore-glob "*stories_paired_s_shard*" \
      --stories-dir "$STORIES_DIR" --matched-dir "$MATCHED_DIR" \
      --preds-dir "$PREDS_DIR" --turnstore-dir "$TS_DIR"
  fi
fi

if should_run extract_r4_op_companion; then
  if [[ -z "$CPS" ]]; then
    echo "[phase=extract_r4_op_companion] SKIPPED — not a paired-stories (CPS) variant"
  elif [[ -f "$R4_HALT_FILE" ]]; then
    echo "[phase=extract_r4_op_companion] SKIPPED — r4 leg halted (yield floor)"
  else
    echo "[phase=extract_r4_op_companion]"
    # rc=23 == companion unusable (kept below the usable floor — the grouped-CV
    # minimum): the TF headline proceeds and the calibration reports N/A
    # (plan v8 §4.5 — a control, never a kill); no_r4op_flag threads the halt
    # into the fits phase.
    RC_OP=0
    run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_gen_stories_paired.py \
      --model instruct --op-companion --out-dir "$STORIES_DIR" --dl-dir "$DL_DIR" \
      --matched-dir "$MATCHED_DIR" $SMOKE_FLAG || RC_OP=$?
    if [[ -z "$DRY_RUN" && "$RC_OP" -eq 23 ]]; then
      echo "[extract_r4_op_companion] companion unusable (rc=23) — TF headline proceeds, calibration N/A"
      touch "$R4OP_HALT_FILE"
    elif [[ -z "$DRY_RUN" && "$RC_OP" -ne 0 ]]; then
      echo "FATAL: companion generation failed (rc=$RC_OP)" >&2; exit 1
    elif [[ -z "$DRY_RUN" ]]; then
      # Usable-floor PASSED: clear any stale companion halt from a prior
      # attempt (same re-evaluation semantics as the r4 halt above).
      [[ -f "$R4OP_HALT_FILE" ]] && echo "[extract_r4_op_companion] floor passed — clearing stale companion halt"
      rm -f "$R4OP_HALT_FILE"
    fi
    if [[ ! -f "$R4OP_HALT_FILE" ]]; then
      run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_extract_turnstore.py \
        --regime r4op --model instruct --out-dir "$TS_DIR" --stories-dir "$STORIES_DIR" $SMOKE_FLAG
      run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_upload.py $SMOKE_FLAG \
        --legs turnstore --turnstore-glob "*stories_paired_op_s_shard*" \
        --stories-dir "$STORIES_DIR" --matched-dir "$MATCHED_DIR" \
        --preds-dir "$PREDS_DIR" --turnstore-dir "$TS_DIR"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# on-policy-assistant-story phases (OPS-mode only; these names are only in the
# OPS PHASES list). r4op is the PRIMARY story regime at powered n.
# ---------------------------------------------------------------------------
if should_run gen_stories_op; then
  if [[ -z "$OPS" ]]; then
    echo "[phase=gen_stories_op] SKIPPED — not an on-policy-story (OPS) variant"
  else
    echo "[phase=gen_stories_op]"
    # On-policy paired story generation at POWERED n: the model answers FREELY
    # (no verbatim embedding), pool-sourced from the shared matched-n set,
    # retry-until-floor (<=3 waves, <=3 draws/row) to >=2000 kept. Writes
    # kept_stories_paired_op_instruct.jsonl + uploads the raw bundle to HF
    # (persist_bundle_paired). rc=21 == missed the >=2000-kept yield floor
    # (the round's kill criterion — no story corpus to fit).
    RC_OP=0
    run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_gen_stories_paired.py \
      --model instruct --op-powered --out-dir "$STORIES_DIR" --dl-dir "$DL_DIR" \
      --matched-dir "$MATCHED_DIR" $SMOKE_FLAG || RC_OP=$?
    if [[ -z "$DRY_RUN" && "$RC_OP" -eq 21 ]]; then
      echo "FATAL: on-policy story generation missed the >=2000-kept yield floor (rc=21)" >&2
      exit 1
    elif [[ -z "$DRY_RUN" && "$RC_OP" -ne 0 ]]; then
      echo "FATAL: on-policy story generation failed (rc=$RC_OP)" >&2
      exit 1
    fi
  fi
fi

if should_run extract_r4_op; then
  if [[ -z "$OPS" ]]; then
    echo "[phase=extract_r4_op] SKIPPED — not an on-policy-story (OPS) variant"
  else
    echo "[phase=extract_r4_op]"
    # Teacher-forced capture over the full rendered on-policy story (prefix +
    # context slots + own-answer span mean) — ONE forward per story into the
    # r4op stem (verbatim_check=False; answers are on-policy). Upload the
    # regeneration-costly store to HF BEFORE the long fit phase (#825
    # upload-before-fit).
    run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_extract_turnstore.py \
      --regime r4op --model instruct --out-dir "$TS_DIR" --stories-dir "$STORIES_DIR" $SMOKE_FLAG
    run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_upload.py $SMOKE_FLAG \
      --legs turnstore --turnstore-glob "*stories_paired_op_s_shard*" \
      --stories-dir "$STORIES_DIR" --matched-dir "$MATCHED_DIR" \
      --preds-dir "$PREDS_DIR" --turnstore-dir "$TS_DIR"
  fi
fi

if should_run matchedn; then
  echo "[phase=matchedn]"
  # Parity gate (±0.02 vs pinned anchors; exit 3 halts) + matched-n subsets.
  # $SMOKE_FLAG demotes ONLY the anchor comparison to informational — the
  # anchors bind at production n; the computation still runs (PASS_UNIFIED).
  run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_fit_cells.py \
    --parity --build-matched --no-r3-models "$(halted_models)" $(no_r4_flag) $SMOKE_FLAG \
    --turnstore-dir "$TS_DIR" --matched-dir "$MATCHED_DIR" --out-dir "$EVAL_DIR"
fi

if should_run fits; then
  echo "[phase=fits]"
  run_per_model fits "${ENV_INLINE:+$ENV_INLINE }uv run python -c \"import sys; sys.path.insert(0,'scripts'); import issue1345_common as c; print(','.join(x['cell_id'] for x in c.all_cells() if x['model_key']=='%MODEL%'))\" > /tmp/i1345_cells_%MODEL%${VARIANT:+_$VARIANT}.txt && ${ENV_INLINE:+$ENV_INLINE }uv run python scripts/issue1345_fit_cells.py --cells \$(cat /tmp/i1345_cells_%MODEL%${VARIANT:+_$VARIANT}.txt) %NO_R3% $(no_r4_flag) $(no_r4op_flag) $SMOKE_FLAG $REFIT_REF_FLAG --turnstore-dir '$TS_DIR' --matched-dir '$MATCHED_DIR' --out-dir '$EVAL_DIR' --preds-dir '$PREDS_DIR' --null-draws $NULLS --n-boot $NBOOT"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: fits failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

if should_run matched_row_refits; then
  if [[ -z "$CPS" && -z "$OPS" ]]; then
    echo "[phase=matched_row_refits] SKIPPED — not a story-regime (CPS/OPS) variant"
  elif [[ -f "$R4_HALT_FILE" ]]; then
    echo "[phase=matched_row_refits] SKIPPED — r4 leg halted (yield floor)"
  else
    echo "[phase=matched_row_refits]"
    # Same-n comparators (plan v8 §4): r1/r2 refit on the r4-kept conv subset +
    # the TF cell on the companion's exact subset + tf_op_calibration.json.
    run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_matched_row_refits.py \
      $SMOKE_FLAG --turnstore-dir "$TS_DIR" --matched-dir "$MATCHED_DIR" \
      --eval-dir "$EVAL_DIR" --out-dir "$EVAL_DIR/matched_row" \
      --preds-dir "$PREDS_DIR/matched_row" --null-draws $NULLS --n-boot $NBOOT
  fi
fi

if should_run transfer; then
  echo "[phase=transfer]"
  run_per_model transfer "${ENV_INLINE:+$ENV_INLINE }uv run python scripts/issue1345_cross_regime_transfer.py --models %MODEL% %NO_R3% $(no_r4_flag) $SMOKE_FLAG --turnstore-dir '$TS_DIR' --matched-dir '$MATCHED_DIR' --out-dir '$EVAL_DIR' --preds-dir '$PREDS_DIR' --n-boot $NBOOT"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: transfer failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

if should_run opcomp; then
  echo "[phase=opcomp]"
  run_per_model opcomp "${ENV_INLINE:+$ENV_INLINE }uv run python scripts/issue1345_operator_comparison.py --models %MODEL% %NO_R3% $(no_r4_flag) $SMOKE_FLAG --turnstore-dir '$TS_DIR' --matched-dir '$MATCHED_DIR' --out-dir '$EVAL_DIR' --rot-draws $ROTD"
  if [[ -z "$DRY_RUN" && ( "${RC_INSTRUCT:-0}" -ne 0 || "${RC_PRETRAINED:-0}" -ne 0 ) ]]; then
    echo "FATAL: opcomp failed (rc_i=${RC_INSTRUCT} rc_p=${RC_PRETRAINED})" >&2; exit 1
  fi
fi

# ---------------------------------------------------------------------------
# story-slot-position-ablation phases (plan v10 §4 item 6; membership-gated —
# these names are only in the slot-mode PHASES list)
# ---------------------------------------------------------------------------
if should_run extract_r4_slots; then
  echo "[phase=extract_r4_slots]"
  # Multi-slot TF re-read: ONE forward per story, 5 single positions + the
  # pooled attribution-phrase mean (answer-overlap rate hard-asserted 0.0 at
  # the render trust boundary; diagnostics JSON written before any forward).
  run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_extract_turnstore.py \
    --regime r4 --model instruct --slot-ablation \
    --out-dir "$TS_DIR" --stories-dir "$STORIES_DIR" $SMOKE_FLAG
fi

if should_run upload_stems; then
  echo "[phase=upload_stems]"
  # Upload-before-long-fit (plan v10 §9): the regeneration-costly slot stems
  # persist BEFORE the fits phase; idempotent per-shard verify.
  run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_upload.py $SMOKE_FLAG \
    --legs turnstore --turnstore-glob "*stories_paired_slots_s_shard*" \
    --stories-dir "$STORIES_DIR" --matched-dir "$MATCHED_DIR" \
    --preds-dir "$PREDS_DIR" --turnstore-dir "$TS_DIR"
fi

if should_run fits_slots; then
  echo "[phase=fits_slots]"
  # 7 cells (anchor refit + 4 candidates + prefix refit + chat matched
  # recompute) on the registered row intersection, then the three-anchor
  # ±0.02 refit-equality gate (exit 3 on a production miss BEFORE any slot
  # read is interpreted; informational under --smoke).
  run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_slot_verdict.py \
    --fits --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR" --preds-dir "$PREDS_DIR" \
    --null-draws $NULLS --n-boot $NBOOT $SMOKE_FLAG
fi

if should_run slot_transfer; then
  echo "[phase=slot_transfer]"
  TRANSFER_NULLS=100
  [[ -n "$SMOKE" ]] && TRANSFER_NULLS=5
  run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_slot_verdict.py \
    --transfer --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR" --preds-dir "$PREDS_DIR" \
    --transfer-null-draws $TRANSFER_NULLS $SMOKE_FLAG
fi

if should_run verdict; then
  echo "[phase=verdict]"
  run_cmd env CUDA_VISIBLE_DEVICES=0 ${ENV_INLINE} uv run python scripts/issue1345_slot_verdict.py \
    --verdict --turnstore-dir "$TS_DIR" --out-dir "$EVAL_DIR" --preds-dir "$PREDS_DIR" \
    --n-boot $NBOOT $SMOKE_FLAG
fi

if should_run plots; then
  echo "[phase=plots]"
  if [[ -n "$SLOTAB" ]]; then
    run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_slot_plots.py \
      --out-dir "$EVAL_DIR" --fig-dir "$FIG_DIR" --turnstore-dir "$TS_DIR" --preds-dir "$PREDS_DIR"
  else
    run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_plots.py --no-r3-models "$(halted_models)" $(no_r4_flag) \
      --out-dir "$EVAL_DIR" --fig-dir "$FIG_DIR" --turnstore-dir "$TS_DIR" --stories-dir "$STORIES_DIR"
  fi
fi

if should_run upload; then
  echo "[phase=upload]"
  DELETE_LOCAL=""
  [[ -z "$SMOKE" ]] && DELETE_LOCAL="--delete-local-turnstore"
  # Variant: only the NEW story stems upload — the staged parent r1/r2 shards
  # are bit-identical to the pinned Hub copies (plan v6 §9, ~5-10 GB not ~90 GB).
  UPLOAD_EXTRA=()
  if [[ -n "$SLOTAB" ]]; then
    # Slot mode: preds caches + the slot stems (already uploaded at
    # upload_stems — idempotent re-verify); the staged stories bundle is an
    # INPUT (already on HF at the parent prefix) and is NOT re-uploaded.
    UPLOAD_EXTRA=(--legs preds,turnstore --turnstore-glob "*stories_paired_slots_s_shard*")
  elif [[ -n "$CPS" ]]; then
    # Paired variant: only the NEW r4 TF + companion stems upload (matches
    # instruct_stories_paired_s_shard* AND instruct_stories_paired_op_s_shard*).
    UPLOAD_EXTRA=(--turnstore-glob "*stories_paired*_shard*")
  elif [[ -n "$OPS" ]]; then
    # On-policy variant: only the NEW r4op on-policy story stem uploads
    # (instruct_stories_paired_op_s_shard*); the staged parent r1/r2 shards are
    # bit-identical to the pinned Hub copies and are NOT re-uploaded.
    UPLOAD_EXTRA=(--turnstore-glob "*stories_paired_op_s_shard*")
  elif [[ -n "$VARIANT" ]]; then
    UPLOAD_EXTRA=(--turnstore-glob "*stories_s_shard*")
  fi
  run_cmd ${ENV_INLINE:+env} ${ENV_INLINE} uv run python scripts/issue1345_upload.py $SMOKE_FLAG $DELETE_LOCAL \
    ${UPLOAD_EXTRA[@]+"${UPLOAD_EXTRA[@]}"} \
    --stories-dir "$STORIES_DIR" --matched-dir "$MATCHED_DIR" \
    --preds-dir "$PREDS_DIR" --turnstore-dir "$TS_DIR"
fi

if should_run push; then
  echo "[phase=push]"
  if [[ -n "$SMOKE" || -n "$DRY_RUN" ]]; then
    echo "[push] smoke/dry-run: scratch outputs — no git commit"
  else
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    git add "$EVAL_DIR" "$FIG_DIR"
    if git diff --cached --quiet; then
      echo "[push] nothing new to commit"
    else
      git -c user.email=pod@eps.local -c user.name="issue1345-dispatch" \
        commit -m "issue-1345: eval JSONs + figures (pod-side)"
    fi
    PUSH_OK=""
    for attempt in 1 2; do
      if git push origin "HEAD:$BRANCH"; then PUSH_OK=1; break; fi
      echo "[push] attempt $attempt failed; retrying" >&2
      sleep 10
    done
    if [[ -z "$PUSH_OK" ]]; then echo "FATAL: git push failed twice" >&2; exit 86; fi
    BEHIND="$(git rev-list --count "origin/$BRANCH..HEAD")"
    if [[ "$BEHIND" != "0" ]]; then
      echo "FATAL: push verify — $BEHIND commit(s) not on origin/$BRANCH" >&2; exit 86
    fi
    # Artifact-presence assert (#1325): every declared result file in the PUSHED tree
    MISSING=""
    while IFS= read -r p; do
      if [[ -z "$(git ls-tree -r "origin/$BRANCH" --name-only -- "$p")" ]]; then
        MISSING="$MISSING $p"
      fi
    done < <(find "$EVAL_DIR" -name '*.json' -type f; find "$FIG_DIR" -name '*.png' -type f)
    if [[ -n "$MISSING" ]]; then
      echo "FATAL: pushed tree missing declared result files:$MISSING" >&2; exit 86
    fi
    echo "[push] verified: 0 unpushed commits; all declared result files in origin/$BRANCH"
  fi
fi

# ---------------------------------------------------------------------------
# Results sentinel (poll_pipeline contract) + terminal [phase=done]
# ---------------------------------------------------------------------------
GPU_HOURS_USED=$(awk -v s="$SECONDS" -v g="$NGPU" 'BEGIN{printf "%.2f", (s/3600.0)*(g>0?g:1)}')
COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
SENTINEL_KIND="epm:results"
SENTINEL_PATH="$LOG_DIR/issue-1345-results.json"
if [[ -n "$SMOKE" || -n "$DRY_RUN" ]]; then
  SENTINEL_KIND="epm:smoke-result"
  SENTINEL_PATH="$LOG_DIR/issue-1345-smoke-results.json"
fi
R4_STATE="not_applicable"
if [[ -n "$CPS" ]]; then
  R4_STATE="live"
  [[ -f "$R4_HALT_FILE" ]] && R4_STATE="halted"
elif [[ -n "$OPS" ]]; then
  # On-policy round: the story arm is r4op (on-policy), not the r4 TF regime.
  R4_STATE="live_onpolicy"
fi
uv run python - "$SENTINEL_KIND" "$SENTINEL_PATH" "$EVAL_DIR" "$COMMIT_SHA" "$GPU_HOURS_USED" "${SMOKE:-0}" "$(halted_models)" "$CHARACTER_NAME" "$VARIANT" "$R4_STATE" <<'PY'
import json
import sys
import time
from pathlib import Path

kind, out_path, eval_dir, commit_sha, gpu_hours, smoke, halted_csv, char_name, variant = (
    sys.argv[1:10]
)
r4_state = sys.argv[10]
eval_dir = Path(eval_dir)
eval_paths = sorted(str(p) for p in eval_dir.glob("*.json"))
lattice = {}
lat_path = eval_dir / "verdict_lattice.json"
if lat_path.exists():
    lat = json.loads(lat_path.read_text())
    lattice = {
        k: {"verdict": v.get("verdict"), "delta_xfer": v.get("delta_xfer"),
            "delta_reparam": v.get("delta_reparam")}
        for k, v in lat.get("per_model_arm", {}).items()
    }
parity = {}
par_path = eval_dir / "parity_gate.json"
if par_path.exists():
    parity = {"pass": json.loads(par_path.read_text()).get("pass")}
refit_eq = {}
refit_path = eval_dir / "refit_equality.json"
if refit_path.exists():
    refit_eq = {"pass": json.loads(refit_path.read_text()).get("pass")}
# Slot-ablation round (plan v10): lattice + three-anchor gate summaries.
slot_lattice = {}
sl_path = eval_dir / "slot_verdict_lattice.json"
if sl_path.exists():
    sl = json.loads(sl_path.read_text())
    bat = sl.get("battery") or {}
    slot_lattice = {
        "verdict": sl.get("verdict"),
        "d_obs": bat.get("d_obs"),
        "d_ci95": bat.get("d_ci95"),
        "chat_r2_l19_obs": bat.get("chat_r2_l19_obs"),
        "per_slot": bat.get("per_slot"),
        "nondegenerate_slots": sl.get("nondegenerate_slots"),
        "answer_overlap_rates": sl.get("answer_overlap_rates"),
        "anchor_coincidence_rates": sl.get("anchor_coincidence_rates"),
        "registered_n_rows": sl.get("registered_n_rows"),
    }
refit_eq_slots = {}
res_path = eval_dir / "refit_equality_slots.json"
if res_path.exists():
    refit_eq_slots = {"pass": json.loads(res_path.read_text()).get("pass")}
tf_op = {}
tf_op_path = eval_dir / "matched_row" / "tf_op_calibration.json"
if tf_op_path.exists():
    tf_op = json.loads(tf_op_path.read_text()).get("calibration", {})
halted_models = [m for m in halted_csv.split(",") if m]
vsub = f"/{variant}" if variant else ""
payload = {
    "eval_numbers": {"verdict_lattice": lattice, "parity_gate": parity,
                     "refit_equality": refit_eq,
                     "slot_verdict_lattice": slot_lattice,
                     "refit_equality_slots": refit_eq_slots,
                     "story_regime_halted_models": halted_models,
                     "story_regime_halted": len(halted_models) == 2,
                     "paired_story_regime_r4": r4_state,
                     "tf_op_calibration": tf_op},
    "eval_paths": eval_paths,
    "reproducibility_card": {
        "models": {"instruct": "Qwen/Qwen2.5-7B-Instruct", "pretrained": "Qwen/Qwen2.5-7B"},
        "seeds": {"fit": 0, "generation": 42, "subsample": 0},
        "pinned_parent_revision": "7159e5804d",
        "story_character_name": char_name,
        "variant": variant or None,
        "reused_turnstore_revision": (
            "2a3cb30acada04defc84fd04d28a2b54da3104cd" if variant else None
        ),
        "paired_story_target": (
            {"n_target": 2700, "yield_floor": 2160, "subsample_seed": 42,
             "op_companion_n": 200, "op_companion_seed": 0}
            # Membership mirrors c.PAIRED_STORIES_VARIANTS (v8 ARIA + v9 Assistant scope)
            if variant in ("conversation_paired_stories",
                           "conversation_paired_stories_assistant") else None
        ),
        "onpolicy_story_target": (
            # on-policy round: r4op promoted to the primary story regime at
            # powered n (values mirror c.N_ONPOLICY_STORY_TARGET / floor).
            {"n_target": 2200, "yield_floor": 2000, "gen_seed": 42,
             "story_regime": "r4op", "on_policy": True}
            if variant == "onpolicy_assistant_story" else None
        ),
        "hf_data_prefix": (
            f"issue1345_smoke{vsub}/" if smoke == "1" else f"issue1345_framing{vsub}/"
        ),
        "wandb_project": "n/a — no training (extraction + analysis task)",
    },
    "wandb_url": "n/a — no training runs (extraction/analysis only)",
    "hf_hub_url": "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data",
    "worktree_path": str(Path.cwd()),
    "final_commit_sha": commit_sha,
    "gpu_hours_used": float(gpu_hours),
    "gpu_hours_budgeted": (
        # 16.0 = the paired-stories plan v8/v9 §9 ceiling (both CPS scopes);
        # 4.0 = the slot-ablation plan v10 §9 ceiling (~2.9 projected).
        16.0 if variant in ("conversation_paired_stories",
                            "conversation_paired_stories_assistant")
        else (4.0 if variant == "story_slot_ablation"
              else (13.0 if variant else 14.0))
    ),
    "plan_deviations": [
        {"deviation": "HF store layout uses the canonical issue1345_framing/ prefix",
         "rationale": "plan §10 wrote a bare analysis_tensors/issue_1345 path; Upload Policy pins issueN_<slug>/ prefixes"},
        {"deviation": "pair-level matched-n implemented as row ALLOWLISTS on the single stores",
         "rationale": "equivalent selection, reuses run_cell's tested allowlist path, avoids duplicating ~90 GB of tensors"},
        {"deviation": "matched-capacity reparam nulls at L19 only (5 draws/type)",
         "rationale": "plan §3 defines Δ_reparam at layer 19; §9 sized ~16 extra alignment fits"},
        {"deviation": "story-pair aligned cosine = operator-spectrum Procrustes optimum",
         "rationale": "the map_alignment data-paired Procrustes is undefined for unpaired corpora (no shared conv_ids); raw-cosine rotation band + spectrum optimum reported instead"},
    ],
}
sentinel = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "by": "issue1345-dispatch",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "task_id": 1345,
    "smoke": smoke == "1",
    "note": payload,
}
Path(out_path).write_text(json.dumps(sentinel, indent=2))
print(f"sentinel written: {out_path}")
PY

echo "[phase=done]"
