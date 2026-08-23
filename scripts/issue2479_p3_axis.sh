#!/usr/bin/env bash
# Issue #2479 P3 (VM-side, 0-GPU): AI-likeness axis judging — stage kept
# stories -> emit axis items -> rule-26 axis-family pilot -> per-character
# ai_likeness legs -> freeze + commit (plan §4 Step 3 / §7 / §9 P3 row).
#
# This IS the explicit, copy-pasteable P3 staging + dispatch command the
# round-1 review required (codex `hf-prefix-realized-vs-plan`: "P3 cannot
# materialize its plan-declared input without manual, undocumented staging").
#
# Steps ([phase=...] breadcrumbs, in order):
#   p3_stage   per panel OP cell: issue1345_stage_char_stories.py --variant
#              <cell> (panel `char_2479_*` cells resolve their RECORDED
#              per-cell generation upload revision, falling back to a
#              head-resolved sha; parent cells keep the fixed STORIES_PIN).
#              Fail-loud: P3 must not run before P1 landed the cell.
#   p3_items   freeze_axis --emit-items (reservation-restricted item lists ->
#              ${ITEMS_DIR}/axis_items_<name>.jsonl)
#   p3_pilot   issue2479_judge_pilots --family axis: judge_pilot_gate VERBATIM
#              at the exact production instrument (~150 draws, pilot-only
#              cache, forced-Batch threshold_base=0). Skipped when a PASS
#              report already exists (real spend, idempotent).
#   p3_gate    judge_pilots --require-pass --family axis (rc=48 on miss/FAIL)
#   p3_legs    per character: issue1345_onpolicy_judge_legs --leg ai_likeness
#              --census --execute on the emitted item list, with
#              EPM_I2479_REQUIRE_AXIS_PILOT_PASS exported so run_leg itself
#              refuses real spend without the pilot PASS (defense in depth).
#   p3_freeze  freeze_axis --legs-dir ... --commit (explicit-path git commit +
#              bare rc-checked push + the axis-frozen marker via main task.py)
#
# Spend: requires EPM_I1345_JUDGE_SPEND_OK=1 (checked at entry; the axis wave
# is ~19k Batch draws, plan §9). Usage:
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
# op cells stage kept_stories_paired_op_instruct.jsonl under data/issue_1345/.
# Two-step default: a {variant} brace inside ${...:-} would end the parameter
# expansion at the first '}' and silently mangle the template.
KEPT_GLOB_DEFAULT='data/issue_1345/{variant}/stories/kept_stories_paired_op_instruct.jsonl'
KEPT_GLOB="${EPM_I2479_KEPT_GLOB:-$KEPT_GLOB_DEFAULT}"
ITEMS_DIR="${EPM_I2479_AXIS_ITEMS_DIR:-data/issue_2479/axis_items}"
LEGS_DIR="${EPM_I2479_AXIS_LEGS_DIR:-eval_results/issue_2479/judge_legs}"
PILOT_REPORT="eval_results/issue_2479/pilot_gate_axis.json"
PILOT_WORK="${EPM_I2479_AXIS_PILOT_WORK:-data/issue_2479/pilot_axis}"

# name|op_variant rows from the panel registry.
PANEL_ROWS="$(uv run python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue2479_freeze_axis as fz

for r in fz.load_panel(Path("eval_results/issue_2479/panel.json")):
    print(f"{r['name']}|{r['variant_op']}")
PY
)" || { echo "[i2479-p3] panel load failed" >&2; exit 2; }
NAMES=()
OP_VARIANTS=()
while IFS='|' read -r name variant; do
  [ -n "$name" ] || continue
  NAMES+=("$name")
  OP_VARIANTS+=("$variant")
done <<< "$PANEL_ROWS"

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] P3 axis pipeline over ${#NAMES[@]} characters: ${NAMES[*]}"
  echo "[dry-run] p3_stage: for each op cell: uv run python scripts/issue1345_stage_char_stories.py --variant <cell>   # panel cells resolve the recorded per-cell upload revision; parent cells keep STORIES_PIN"
  echo "[dry-run] p3_items: uv run python scripts/issue2479_freeze_axis.py --emit-items --kept-glob '${KEPT_GLOB}' --items-out-dir ${ITEMS_DIR}"
  echo "[dry-run] p3_pilot: uv run python scripts/issue2479_judge_pilots.py --family axis --items-glob '${ITEMS_DIR}/axis_items_{name}.jsonl' --report ${PILOT_REPORT} --work-dir ${PILOT_WORK} --execute   # skipped when a PASS report exists"
  echo "[dry-run] p3_gate:  uv run python scripts/issue2479_judge_pilots.py --require-pass --family axis --report ${PILOT_REPORT}   # rc=48 on miss/FAIL"
  echo "[dry-run] p3_legs:  per character: EPM_I2479_REQUIRE_AXIS_PILOT_PASS=${PILOT_REPORT} uv run python scripts/issue1345_onpolicy_judge_legs.py --leg ai_likeness --rows ${ITEMS_DIR}/axis_items_<name>.jsonl --character <name> --census --out-dir ${LEGS_DIR} --execute"
  echo "[dry-run] p3_freeze: uv run python scripts/issue2479_freeze_axis.py --legs-dir ${LEGS_DIR} --commit"
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

echo "[phase=p3_items]"
uv run python scripts/issue2479_freeze_axis.py --emit-items \
  --kept-glob "$KEPT_GLOB" --items-out-dir "$ITEMS_DIR"

echo "[phase=p3_pilot]"
if uv run python scripts/issue2479_judge_pilots.py --require-pass --family axis \
    --report "$PILOT_REPORT" >/dev/null 2>&1; then
  echo "[i2479-p3] axis pilot PASS report present — pilot skipped (resume)"
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
  if [ -f "${LEGS_DIR}/judge_report_ail_${name}.json" ]; then
    echo "[i2479-p3] ${name}: leg report present — skipped (resume)"
    continue
  fi
  echo "[i2479-p3] axis leg: ${name}"
  uv run python scripts/issue1345_onpolicy_judge_legs.py --leg ai_likeness \
    --rows "$items" --character "$name" --census --out-dir "$LEGS_DIR" --execute
done

echo "[phase=p3_freeze]"
uv run python scripts/issue2479_freeze_axis.py --legs-dir "$LEGS_DIR" --commit
echo "[i2479-p3] P3 complete (axis frozen + committed)"
