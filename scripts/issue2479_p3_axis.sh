#!/usr/bin/env bash
# Issue #2479 P3 (VM-side, 0-GPU): AI-likeness axis judging + the two
# instrument-control gates — stage kept stories (op + inserted) -> emit axis
# items -> rule-26 axis-family pilot -> per-character ai_likeness legs ->
# freeze + commit -> verbatim-flatness leg -> name-mask leg -> compute +
# commit instrument_gates.json -> re-upload the legs dir (plan §4 Step 3 /
# §6 gates 3-4 / §7 / §9 P3 row).
#
# This IS the explicit, copy-pasteable P3 staging + dispatch command the
# round-1 review required (codex `hf-prefix-realized-vs-plan`), extended in
# r3 with the four registered instrument gates' production path (r2 codex
# `p3-controls-disconnected`: P6 requires instrument_gates.json at
# gradient_verdict.py, and no phase produced it).
#
# Steps ([phase=...] breadcrumbs, in order):
#   p3_stage            per panel OP cell: issue1345_stage_char_stories.py
#                       --variant <cell> (panel `char_2479_*` cells resolve
#                       their RECORDED per-cell generation upload revision;
#                       parent cells keep the fixed STORIES_PIN).
#   p3_stage_inserted   per INSERTED cell: the same stager on the inserted
#                       variant (kept_stories_paired_instruct.jsonl — the
#                       flatness leg's input).
#   p3_items            freeze_axis --emit-items (reservation-restricted item
#                       lists -> ${ITEMS_DIR}/axis_items_<name>.jsonl; SMALL
#                       stats sidecars -> ${STATS_DIR}, commit-eligible — the
#                       verdict's registered answer-length read).
#   p3_pilot            issue2479_judge_pilots --family axis: judge_pilot_gate
#                       VERBATIM at the exact production instrument (~150
#                       draws, pilot-only cache, forced-Batch). The resume
#                       skip is INSTRUMENT-BOUND (require_pilot_pass): a
#                       stale-instrument PASS re-pilots instead of skipping.
#   p3_gate             judge_pilots --require-pass --family axis (rc=48 on
#                       miss/FAIL/stale-instrument/under-powered).
#   p3_legs             per character: issue1345_onpolicy_judge_legs --leg
#                       ai_likeness --census --execute on the emitted item
#                       list. Resume is the VALIDATED completion predicate
#                       (issue2479_p3_leg_resume.py: spend_executed, current
#                       instrument + rubric fingerprint, exact item-ID set vs
#                       the freshly emitted items, current pilot PASS) — an
#                       invalid report is QUARANTINED + re-dispatched, never
#                       silently reused (r2 codex `p3-leg-resume-unvalidated`;
#                       existence-only skips let a bad report wedge the
#                       freeze's fail-loud rejects forever).
#                       EPM_I2479_REQUIRE_AXIS_PILOT_PASS is exported so
#                       run_leg itself refuses real spend without the pilot
#                       PASS (defense in depth).
#   p3_upload           bulk upload_folder of the legs dir (raw draws +
#                       reports + caches) to
#                       issue2479_ai_likeness_gradient/judge_legs + a scoped
#                       raising verify (plan §10; the dual-prefix
#                       verification's judge_legs leg).
#   p3_freeze           freeze_axis --legs-dir ... --commit (explicit-path git
#                       commit of axis_freeze.json + the axis_draws.json
#                       per-draw sidecar + the axis_items stats sidecars +
#                       bare rc-checked push + the axis-frozen marker).
#   p3_flatness         issue2479_instrument_gates --step flatness --execute:
#                       the 8 inserted cells' verbatim reference answers on
#                       ONE common seed-0 100-item draw (4k draws; REUSES the
#                       axis-family pilot PASS). Skipped only when EVERY
#                       flat_<name> leg report passes the validated resume
#                       predicate.
#   p3_namemask         issue2479_instrument_gates --step namemask --execute:
#                       the 8 band-A/D characters' 40-item seed-0 subsample
#                       re-judged name-masked (1.6k draws; same pilot reuse).
#                       Same validated-resume skip over mask_<name> legs.
#   p3_gates            issue2479_instrument_gates --step gates: compute
#                       verbatim_flatness_pass + name_mask_pass from the
#                       persisted legs + axis_freeze.json and write
#                       eval_results/issue_2479/instrument_gates.json (the P6
#                       verdict's REQUIRED gates input), then explicit-path
#                       git commit + bare rc-checked push.
#   p3_upload_controls  re-run the legs-dir bulk upload so the flatness +
#                       name-mask legs (produced AFTER p3_upload) are
#                       published under the same prefix.
#
# Spend: requires EPM_I1345_JUDGE_SPEND_OK=1 (checked at entry; axis wave
# ~19k + flatness 4k + name-mask 1.6k Batch draws, plan §9). Usage:
#   EPM_I1345_JUDGE_SPEND_OK=1 bash scripts/issue2479_p3_axis.sh [--dry-run]
#
# Exit codes: 48 = axis pilot gate not PASS; 2 = arg/env error; any leg's
# own non-zero rc propagates (set -e).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "[i2479-p3] unknown arg: $1" >&2; exit 2 ;;
  esac
done

export EPM_I2479_CHAR_PANEL_JSON="${EPM_I2479_CHAR_PANEL_JSON:-${REPO_ROOT}/eval_results/issue_2479/panel.json}"

# Stager-realized layout (issue1345_stage_char_stories.variant_mode_model):
# op cells stage kept_stories_paired_op_instruct.jsonl, INSERTED cells stage
# kept_stories_paired_instruct.jsonl, both under data/issue_1345/.
# Two-step default: a {variant} brace inside ${...:-} would end the parameter
# expansion at the first '}' and silently mangle the template.
KEPT_GLOB_DEFAULT='data/issue_1345/{variant}/stories/kept_stories_paired_op_instruct.jsonl'
KEPT_GLOB="${EPM_I2479_KEPT_GLOB:-$KEPT_GLOB_DEFAULT}"
INSERTED_KEPT_GLOB_DEFAULT='data/issue_1345/{variant}/stories/kept_stories_paired_instruct.jsonl'
INSERTED_KEPT_GLOB="${EPM_I2479_INSERTED_KEPT_GLOB:-$INSERTED_KEPT_GLOB_DEFAULT}"
ITEMS_DIR="${EPM_I2479_AXIS_ITEMS_DIR:-data/issue_2479/axis_items}"
# SMALL counts-only stats sidecars route to eval_results (commit-eligible;
# the verdict's --axis-items-stats-dir default) while the row jsonls stay in
# gitignored data/ (LMSYS-derived text) — r2 g4 MAJOR-2.
STATS_DIR="${EPM_I2479_AXIS_STATS_DIR:-eval_results/issue_2479/axis_items}"
LEGS_DIR="${EPM_I2479_AXIS_LEGS_DIR:-eval_results/issue_2479/judge_legs}"
PILOT_REPORT="eval_results/issue_2479/pilot_gate_axis.json"
PILOT_WORK="${EPM_I2479_AXIS_PILOT_WORK:-data/issue_2479/pilot_axis}"
GATES_OUT="eval_results/issue_2479/instrument_gates.json"

# name|op_variant|inserted_variant|is_inserted|is_extreme rows from the panel
# registry — the inserted/extreme selections come from the SAME
# instrument_gates row selectors the control steps dispatch over.
PANEL_ROWS="$(uv run python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue2479_freeze_axis as fz
import issue2479_instrument_gates as ig

panel = fz.load_panel(Path(os.environ["EPM_I2479_CHAR_PANEL_JSON"]))
ins = {r["name"] for r in ig.inserted_rows(panel)}
ext = {r["name"] for r in ig.extreme_rows(panel)}
for r in panel:
    print(
        f"{r['name']}|{r['variant_op']}|{r.get('variant_inserted') or ''}"
        f"|{int(r['name'] in ins)}|{int(r['name'] in ext)}"
    )
PY
)" || { echo "[i2479-p3] panel load failed" >&2; exit 2; }
NAMES=()
OP_VARIANTS=()
INSERTED_NAMES=()
INSERTED_VARIANTS=()
MASK_NAMES=()
while IFS='|' read -r name variant vins is_ins is_ext; do
  [ -n "$name" ] || continue
  NAMES+=("$name")
  OP_VARIANTS+=("$variant")
  if [ "$is_ins" = "1" ]; then
    INSERTED_NAMES+=("$name")
    INSERTED_VARIANTS+=("$vins")
  fi
  [ "$is_ext" = "1" ] && MASK_NAMES+=("$name")
done <<< "$PANEL_ROWS"

# Bulk-publish the legs dir (axis + control legs alike) to the plan-§10
# declared prefix with a scoped raising verify — called BEFORE the freeze
# (durability of the axis raw draws) and AGAIN after the control legs land.
upload_legs() {
  uv run python - "$LEGS_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate import hub

legs = Path(sys.argv[1])
assert legs.is_dir(), f"legs dir missing: {legs}"
repo = "superkaiba1/explore-persona-space-data"
prefix = "issue2479_ai_likeness_gradient/judge_legs"
api = HfApi()
hub.retry_transient(
    lambda: api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(legs),
        path_in_repo=prefix,
        commit_message="issue-2479 P3: axis + instrument-control judge legs",
    ),
    what="upload_folder(judge_legs)",
)
n = hub.assert_hf_prefix_exists(api, repo, prefix, repo_type="dataset")
print(f"[i2479-p3] judge_legs uploaded + verified ({n} files at {prefix})", flush=True)
PY
}

# Validated-resume sweep over one control leg family (flat|mask): returns 0
# (skip the phase) only when EVERY leg report passes the completion
# predicate; any invalid report is quarantined by the validator and the
# phase re-runs (already-valid legs re-dispatch through the rubric-keyed
# judge cache at ~zero cost). rc other than 0/3 aborts the wrapper.
all_control_legs_valid() {
  local prefix="$1"; shift
  local name rc
  for name in "$@"; do
    rc=0
    uv run python scripts/issue2479_p3_leg_resume.py \
      --report "${LEGS_DIR}/judge_report_ail_${prefix}_${name}.json" \
      --tag "${prefix}_${name}" --pilot-report "$PILOT_REPORT" || rc=$?
    if [ "$rc" = "3" ]; then return 1; fi
    if [ "$rc" != "0" ]; then
      echo "[i2479-p3] ${prefix}_${name}: leg-resume validator failed rc=${rc}" >&2
      exit "$rc"
    fi
  done
  return 0
}

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] P3 axis pipeline over ${#NAMES[@]} characters: ${NAMES[*]}"
  echo "[dry-run] inserted cells (${#INSERTED_NAMES[@]}): ${INSERTED_NAMES[*]}; name-mask cells (${#MASK_NAMES[@]}): ${MASK_NAMES[*]}"
  echo "[dry-run] p3_stage: for each op cell: uv run python scripts/issue1345_stage_char_stories.py --variant <cell>   # panel cells resolve the recorded per-cell upload revision; parent cells keep STORIES_PIN"
  echo "[dry-run] p3_stage_inserted: for each inserted cell: uv run python scripts/issue1345_stage_char_stories.py --variant <inserted-cell>"
  echo "[dry-run] p3_items: uv run python scripts/issue2479_freeze_axis.py --emit-items --kept-glob '${KEPT_GLOB}' --items-out-dir ${ITEMS_DIR} --stats-out-dir ${STATS_DIR}"
  echo "[dry-run] p3_pilot: uv run python scripts/issue2479_judge_pilots.py --family axis --items-glob '${ITEMS_DIR}/axis_items_{name}.jsonl' --report ${PILOT_REPORT} --work-dir ${PILOT_WORK} --execute   # skipped only on an instrument-bound PASS"
  echo "[dry-run] p3_gate:  uv run python scripts/issue2479_judge_pilots.py --require-pass --family axis --report ${PILOT_REPORT}   # rc=48 on miss/FAIL/stale"
  echo "[dry-run] p3_legs:  per character: issue2479_p3_leg_resume.py --report ${LEGS_DIR}/judge_report_ail_<name>.json --tag <name> --items ${ITEMS_DIR}/axis_items_<name>.jsonl --pilot-report ${PILOT_REPORT}; on rc=3: EPM_I2479_REQUIRE_AXIS_PILOT_PASS=${PILOT_REPORT} uv run python scripts/issue1345_onpolicy_judge_legs.py --leg ai_likeness --rows ${ITEMS_DIR}/axis_items_<name>.jsonl --character <name> --census --out-dir ${LEGS_DIR} --execute"
  echo "[dry-run] p3_upload: upload_folder ${LEGS_DIR} -> issue2479_ai_likeness_gradient/judge_legs + scoped raising verify"
  echo "[dry-run] p3_freeze: uv run python scripts/issue2479_freeze_axis.py --legs-dir ${LEGS_DIR} --stats-dir ${STATS_DIR} --commit   # commits axis_freeze.json + axis_draws.json + axis_items_*.stats.json"
  echo "[dry-run] p3_flatness: uv run python scripts/issue2479_instrument_gates.py --step flatness --kept-glob '${INSERTED_KEPT_GLOB}' --legs-dir ${LEGS_DIR} --axis-pilot-report ${PILOT_REPORT} --execute   # 8x100x5 draws; skipped when every flat_<name> leg passes the validated resume predicate"
  echo "[dry-run] p3_namemask: uv run python scripts/issue2479_instrument_gates.py --step namemask --items-glob '${ITEMS_DIR}/axis_items_{name}.jsonl' --axis-raw-glob '${LEGS_DIR}/judge_raw_ail_{name}.json' --legs-dir ${LEGS_DIR} --axis-pilot-report ${PILOT_REPORT} --execute   # 8x40x5 draws; same validated-resume skip"
  echo "[dry-run] p3_gates: uv run python scripts/issue2479_instrument_gates.py --step gates --axis-raw-glob '${LEGS_DIR}/judge_raw_ail_{name}.json' --legs-dir ${LEGS_DIR} --out ${GATES_OUT}; then explicit-path git commit of ${GATES_OUT} + bare rc-checked push"
  echo "[dry-run] p3_upload_controls: re-run upload_folder ${LEGS_DIR} -> issue2479_ai_likeness_gradient/judge_legs (publishes the flatness + name-mask legs)"
  exit 0
fi

if [ "${EPM_I1345_JUDGE_SPEND_OK:-}" != "1" ]; then
  echo "[i2479-p3] EPM_I1345_JUDGE_SPEND_OK=1 is required (axis wave is real Batch spend)" >&2
  exit 2
fi

echo "[phase=p3_stage]"
for variant in "${OP_VARIANTS[@]}"; do
  echo "[i2479-p3] staging kept stories: ${variant}"
  uv run python scripts/issue1345_stage_char_stories.py --variant "$variant"
done

echo "[phase=p3_stage_inserted]"
for variant in "${INSERTED_VARIANTS[@]}"; do
  echo "[i2479-p3] staging inserted kept stories: ${variant}"
  uv run python scripts/issue1345_stage_char_stories.py --variant "$variant"
done

echo "[phase=p3_items]"
uv run python scripts/issue2479_freeze_axis.py --emit-items \
  --kept-glob "$KEPT_GLOB" --items-out-dir "$ITEMS_DIR" --stats-out-dir "$STATS_DIR"

echo "[phase=p3_pilot]"
if uv run python scripts/issue2479_judge_pilots.py --require-pass --family axis \
    --report "$PILOT_REPORT" >/dev/null 2>&1; then
  echo "[i2479-p3] axis pilot PASS report present (instrument-bound) — pilot skipped (resume)"
else
  uv run python scripts/issue2479_judge_pilots.py --family axis \
    --items-glob "${ITEMS_DIR}/axis_items_{name}.jsonl" \
    --report "$PILOT_REPORT" --work-dir "$PILOT_WORK" --execute
fi

echo "[phase=p3_gate]"
uv run python scripts/issue2479_judge_pilots.py --require-pass --family axis \
  --report "$PILOT_REPORT" || {
  echo "[i2479-p3] axis-family rule-26 pilot gate not PASS — refusing the axis wave" >&2
  exit 48
}

echo "[phase=p3_legs]"
export EPM_I2479_REQUIRE_AXIS_PILOT_PASS="$PILOT_REPORT"
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  items="${ITEMS_DIR}/axis_items_${name}.jsonl"
  [ -f "$items" ] || { echo "[i2479-p3] ${name}: ${items} missing after --emit-items" >&2; exit 2; }
  resume_rc=0
  uv run python scripts/issue2479_p3_leg_resume.py \
    --report "${LEGS_DIR}/judge_report_ail_${name}.json" --tag "$name" \
    --items "$items" --pilot-report "$PILOT_REPORT" || resume_rc=$?
  if [ "$resume_rc" = "0" ]; then
    continue
  elif [ "$resume_rc" != "3" ]; then
    echo "[i2479-p3] ${name}: leg-resume validator failed rc=${resume_rc}" >&2
    exit "$resume_rc"
  fi
  echo "[i2479-p3] axis leg: ${name}"
  uv run python scripts/issue1345_onpolicy_judge_legs.py --leg ai_likeness \
    --rows "$items" --character "$name" --census --out-dir "$LEGS_DIR" --execute
done

echo "[phase=p3_upload]"
# Persist the axis judge legs (raw draws + reports + caches) BEFORE the
# freeze commit (plan §10 durability ordering).
upload_legs

echo "[phase=p3_freeze]"
uv run python scripts/issue2479_freeze_axis.py --legs-dir "$LEGS_DIR" \
  --stats-dir "$STATS_DIR" --commit

echo "[phase=p3_flatness]"
if all_control_legs_valid flat "${INSERTED_NAMES[@]}"; then
  echo "[i2479-p3] all flat_<name> legs valid — flatness dispatch skipped (resume)"
else
  uv run python scripts/issue2479_instrument_gates.py --step flatness \
    --kept-glob "$INSERTED_KEPT_GLOB" --legs-dir "$LEGS_DIR" \
    --axis-pilot-report "$PILOT_REPORT" --execute
fi

echo "[phase=p3_namemask]"
if all_control_legs_valid mask "${MASK_NAMES[@]}"; then
  echo "[i2479-p3] all mask_<name> legs valid — name-mask dispatch skipped (resume)"
else
  uv run python scripts/issue2479_instrument_gates.py --step namemask \
    --items-glob "${ITEMS_DIR}/axis_items_{name}.jsonl" \
    --axis-raw-glob "${LEGS_DIR}/judge_raw_ail_{name}.json" \
    --legs-dir "$LEGS_DIR" --axis-pilot-report "$PILOT_REPORT" --execute
fi

echo "[phase=p3_gates]"
# Recompute-idempotent: always re-derives both gate booleans from the
# persisted legs + axis_freeze.json (the P6 verdict's REQUIRED input).
uv run python scripts/issue2479_instrument_gates.py --step gates \
  --axis-raw-glob "${LEGS_DIR}/judge_raw_ail_{name}.json" \
  --legs-dir "$LEGS_DIR" --out "$GATES_OUT"
git add "$GATES_OUT"
if git diff --cached --quiet -- "$GATES_OUT"; then
  echo "[i2479-p3] ${GATES_OUT} unchanged — no commit"
else
  git commit -m "issue-2479 P3: instrument gates (verbatim-flatness + name-mask)" -- "$GATES_OUT"
  git push origin HEAD
fi

echo "[phase=p3_upload_controls]"
# Re-publish the legs dir so the flatness + name-mask legs (written AFTER
# p3_upload) land under the same prefix; upload_folder is idempotent for the
# already-published axis legs.
upload_legs

echo "[i2479-p3] P3 complete (axis frozen + committed; instrument gates computed + committed)"
