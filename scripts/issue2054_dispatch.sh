#!/usr/bin/env bash
# Dispatch router for task #2054 phases.
#
# Usage:
#   bash scripts/issue2054_dispatch.sh <phase> [args...]  run a phase; args pass through
#   bash scripts/issue2054_dispatch.sh --plan [phase]     print command(s), run nothing
#   bash scripts/issue2054_dispatch.sh --help
#
# Wired phases (0 GPU-h audits):
#   audit_i   — re-judge parent #1345's ~1,419 on-policy rejects. Anthropic Batch API,
#               STORY_MIN_TURNS=1, judge max_tokens=1024, rule-26 pilot gate n=200.
#   audit_ii  — permissive span-locator sweep over parent #1345 inserted rejects. VM Python.
#
# Not yet wired (exit 3):
#   phase_a  — diverse scaffold generation (Qwen2.5-7B, lora-7b intent, 5 GPU-h)
#   phase_b  — deterministic inserted splice (VM CPU, 0 GPU-h)
#   phase_c  — on-policy continuation (Qwen2.5-7B x 2, lora-7b intent, 15 GPU-h)
#   capture  — teacher-forced activation capture (eval intent x 2 models, 100 GPU-h)
#   fits     — ambient-basis fits + baselines + kNN + null battery (VM CPU or cpu-mid/bigmem)
#   ladder   — 9-rung transfer ladder (VM CPU, batched)
#
# Pass-through: every arg after <phase> is appended verbatim to the entrypoint's argv,
# so it overrides the router default of the same name (argparse takes the last
# occurrence). Both audit entrypoints expose a no-network --dry-run self-test.
#   bash scripts/issue2054_dispatch.sh audit_i --dry-run
#   bash scripts/issue2054_dispatch.sh audit_ii --output /tmp/issue-2054-smoke/ii.json

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PARENT_REPO="superkaiba1/explore-persona-space-data"
PARENT_PREFIX="issue1345_framing"
AUDIT_OUT_DIR="eval_results/issue_2054/audits"

WIRED_PHASES=(audit_i audit_ii phase_a phase_b)
UNWIRED_PHASES=(phase_c capture fits ladder)

# Print the leading comment block (everything after the shebang, up to the first
# non-comment line) as the usage text, so help can never drift from the header.
usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"
}

# Sets the global CMD array to the argv for a wired phase.
# Returns 1 for anything not wired (caller distinguishes unwired vs unknown).
build_cmd() {
  case "$1" in
    audit_i)
      CMD=(
        uv run python scripts/issue2054_audit_i_rejudge.py
        --parent-repo "$PARENT_REPO"
        --parent-prefix "$PARENT_PREFIX"
        --story-min-turns 1
        --max-tokens 1024
        --pilot-n 200
        --output "$AUDIT_OUT_DIR/audit_i_rejudge.json"
      )
      ;;
    audit_ii)
      CMD=(
        uv run python scripts/issue2054_audit_ii_span_locator.py
        --parent-repo "$PARENT_REPO"
        --parent-prefix "$PARENT_PREFIX"
        --output "$AUDIT_OUT_DIR/audit_ii_span_locator.json"
      )
      ;;
    phase_a)
      # Unit A: diverse scaffold generation driver + shared fold map.
      # Target = 8,000 conv_ids (33% oversample pad over the plan §7 gate-4
      # floor of 4,480); recovers from parent stories via strip_scaffolds,
      # then writes eval_results/issue_2054/shared_fold_map.json.
      CMD=(
        uv run python scripts/issue2054_phase_a.py
        --target-conv-ids 8000
        --output-dir data/issue_2054/scaffolds/
        --seed 137
      )
      ;;
    phase_b)
      # Unit A: deterministic inserted splice driver over Phase A scaffolds.
      # 100% keep by construction; answers pool must be passed via pass-through.
      CMD=(
        uv run python scripts/issue2054_phase_b.py
        --scaffolds-dir data/issue_2054/scaffolds/
        --output-dir data/issue_2054/spliced_inserted/
        --seed 137
      )
      ;;
    *)
      return 1
      ;;
  esac
}

is_unwired_phase() {
  local candidate="$1" p
  for p in "${UNWIRED_PHASES[@]}"; do
    [[ "$p" == "$candidate" ]] && return 0
  done
  return 1
}

print_plan_for() {
  local phase="$1"
  if build_cmd "$phase"; then
    printf '%s:' "$phase"
    printf ' %q' "${CMD[@]}"
    printf '\n'
  elif is_unwired_phase "$phase"; then
    printf '%s: NOT WIRED (Unit B/C follow-up round) — exits 3\n' "$phase"
  else
    echo "unknown phase: $phase" >&2
    return 2
  fi
}

plan() {
  if [[ $# -gt 0 ]]; then
    print_plan_for "$1"
    return
  fi
  local p
  for p in "${WIRED_PHASES[@]}" "${UNWIRED_PHASES[@]}"; do
    print_plan_for "$p"
  done
}

main() {
  if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
  fi

  local phase="$1"
  shift

  case "$phase" in
    -h | --help | help)
      usage
      exit 0
      ;;
    --plan | plan)
      plan "$@"
      exit 0
      ;;
  esac

  if build_cmd "$phase"; then
    # Router defaults first, caller pass-through last (argparse: last occurrence wins).
    exec "${CMD[@]}" "$@"
  fi

  if is_unwired_phase "$phase"; then
    echo "phase=$phase not yet wired (Unit B/C follow-up round)" >&2
    exit 3
  fi

  echo "unknown phase: $phase" >&2
  usage >&2
  exit 2
}

main "$@"
