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
#   phase_a  — diverse scaffold generation (Qwen2.5-7B, lora-7b intent, 5 GPU-h)
#   phase_b  — deterministic inserted splice (VM CPU, 0 GPU-h)
#   phase_c  — on-policy continuation (Qwen2.5-7B x 2, lora-7b intent, 15 GPU-h)
#   phase_d  — v4-new cell (c): STORY-authored answer, CHAT presentation (VM CPU, 0 GPU-h)
#   capture  — teacher-forced activation capture at layer 19 (eval intent x 2 models, 100 GPU-h)
#   fits     — per-cell ambient-basis ridge + identity+bias baseline + kNN + shuffled-answer null
#              + reduced-k1024 diagnostic + conv-within-intersection bootstrap CI + kill gates 4/5
#              (VM CPU or cpu-mid; 0 GPU-h)
#   ladder   — 9-rung transfer ladder between cells: for each ordered (source, target) cell pair,
#              compute the 9 mapping-transformation rungs (direct, ctx_offset, ans_offset,
#              bias_refit, global_scale, rotation, ctx_reparam, ans_reparam, full_AMB) and
#              score held-out R² per rung + the ratio to the target's own within-cell ceiling
#              (VM CPU, batched; 0 GPU-h)
#
# Pass-through: every arg after <phase> is appended verbatim to the entrypoint's argv,
# so it overrides the router default of the same name (argparse takes the last
# occurrence). Both audit entrypoints expose a no-network --dry-run self-test.
#   bash scripts/issue2054_dispatch.sh audit_i --dry-run
#   bash scripts/issue2054_dispatch.sh audit_ii --output /tmp/issue-2054-smoke/ii.json
#
# FRAMING AXIS (plan §4): phase_b / phase_c / phase_d REQUIRE --form
# (chat | bare_text | attrib_quoted | bare_label | bare_paragraph | indirect).
# phase_b / phase_c run MULTIPLE forms per cell, so --form passes through
# per-cell and the router bakes NO default there (the lattice's central
# manipulated variable can never silently fall back to attrib_quoted).
# phase_d's ONLY planned cell is cell (c) — STORY-authored answer, CHAT
# presentation (plan §4 Phase D / Block 3) — so the router PINS --form chat
# on the phase_d wire; pass-through can still override for a deliberate
# non-(c) rerun (argparse: last occurrence wins).
#   bash scripts/issue2054_dispatch.sh phase_b --form attrib_quoted --answers-source <jsonl>
#   bash scripts/issue2054_dispatch.sh phase_d

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PARENT_REPO="superkaiba1/explore-persona-space-data"
PARENT_PREFIX="issue1345_framing"
AUDIT_OUT_DIR="eval_results/issue_2054/audits"

WIRED_PHASES=(audit_i audit_ii phase_a phase_b phase_c phase_d capture fits ladder)
UNWIRED_PHASES=()

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
      # Units A + D: scaffold supply (recovery + shortfall GENERATION via the
      # parent issue1345_gen_scaffolds subprocess against ONE shared #1738
      # question draw), per-row judge ADMISSION (kept.json), + shared fold map.
      # Target = 8,000 conv_ids (33% oversample pad over the plan §7 gate-4
      # floor of 4,480); writes eval_results/issue_2054/shared_fold_map.json
      # from the ADMITTED conv_ids.
      # Lane split (plan §10 off_pod_phases): the pod (lora-7b) leg runs
      # `--stage gen` (recovery + vLLM generation + prejudge upload, fail-loud);
      # the VM Batch-API leg runs `--stage judge [--prejudge-from-hf]`.
      # Default `--stage all` = both in-process (smoke / single box); pass
      # the stage per lane via the router's verbatim pass-through. CPU smoke:
      # `phase_a --gen-mock --questions-jsonl <tiny pool>`.
      #
      # SHARED DRAW SIZING (crash-fix r5, 2026-08-05). The default
      # `target_conv_ids - shared_recovered` arithmetic sizes the CROSS-variant
      # intersection, but gate 4 intersects WITHIN one (character, model) group,
      # so it under-draws whenever per-variant recovery exceeds the shared
      # intersection (measured: ~2,155/variant recovered vs 1,055 shared).
      # --gen-draw-n sizes the shared draw directly:
      # RESIZED at r7 on the MEASURED judge-admission rate. r5 sized on an
      # assumed ~80% retention (never measurable before — no judge wave had
      # run on this instrument); realized 50.4-59.6%, so every variant landed
      # 480-1,262 short of gate 4 despite clearing it 1.43-1.55x PRE-judge.
      #   admitted per DRAWN question, measured 2026-08-05 (binding = char_vex):
      #     char_vex 13.83% | char_dana 15.09% | char_wren 16.27%
      #     char_helios 18.51% | assistant 18.59%
      #   recovered-admitted floor / variant ~1,273-1,408 (the recovered half is
      #   fully consumed — all of parent #1345's kept stories)
      #   @ D=32,000: char_vex 1,282 + 0.1383*32,000 = 5,708 = 1.27x the 4,480
      #   gate-4 floor; every other variant 1.36-1.64x. The 1.27x absorbs a
      #   ~21% phase-b/c/d loss (phase_b is 100%-keep by construction; the real
      #   exposure is phase_c on-policy generation, still unmeasured).
      #   Sizing to the gate exactly would need 23,126 and leave ZERO headroom.
      #   Eligible pool after the r5 filters = 56,911 measured, so a 32,000
      #   draw is 56% of it.
      # Wall: the r5 run did 5 x 14,000 = 70,000 scaffolds end-to-end in
      # 36m45s on 1x H100 (engine load + recovery + manifest staging included).
      # Scaling the draw 32,000/14,000 = 2.29x gives ~84 min. Sequential
      # per-variant subprocesses on ONE GPU: still under the ~2 h
      # shardable-width threshold, so 1x H100 stays right-sized.
      # The shardable axis (variant) is NOT used, and the justification binds
      # to THIS phase: five concurrent per-variant processes would each stage
      # the same ~455 MB #1738 manifest and each open a vLLM engine, racing one
      # shared staging dest (the #1315 fan-out class) and stacking HF API calls
      # against the org-wide 2,500-req/5-min quota that already killed one
      # launch of this very phase at the tokenizer load. GPU-HOURS are
      # identical either way (~1.4 GPU-h at 1x84min vs 5x~17min), so sharding
      # buys wall-clock only, at real correctness risk.
      CMD=(
        uv run python scripts/issue2054_phase_a.py
        --target-conv-ids 8000
        --gen-draw-n 32000
        --output-dir data/issue_2054/scaffolds/
        --seed 137
      )
      ;;
    phase_b)
      # Unit A: deterministic inserted splice driver over Phase A scaffolds.
      # 100% keep by construction; answers pool must be passed via pass-through.
      # REQUIRES --form <framing> per cell (plan §4 framing axis; no default).
      CMD=(
        uv run python scripts/issue2054_phase_b.py
        --scaffolds-dir data/issue_2054/scaffolds/
        --output-dir data/issue_2054/spliced_inserted/
        --seed 137
      )
      ;;
    phase_c)
      # Unit B: on-policy continuation via vLLM prefill.
      # REQUIRES --form <framing> per cell (plan §4 framing axis; no default).
      # Model + variants pass through; --dry-run skips vLLM for CPU-side smokes.
      # max_new_tokens 2048 per plan §11 (>=2x longest trained completion).
      # TWO-MODEL RUNS: compose DISTINCT --output-dir roots per model (the
      # resume sidecar regime carries the model axis, the filename does not —
      # a second model into ONE dir is REFUSED). Multi-GPU: use the composer,
      # which appends the model slug to the output dir and shards variants:
      #   uv run python scripts/issue2054_shard_launch.py --driver phase_c \
      #     --form F --model M --gpus 0,..,7
      CMD=(
        uv run python scripts/issue2054_phase_c.py
        --scaffolds-dir data/issue_2054/scaffolds/
        --target-conv-ids 8000
        --output-dir data/issue_2054/on_policy/
        --seed 137
        --max-new-tokens 2048
      )
      ;;
    phase_d)
      # Unit B: v4-new cell (c) transpose — STORY-authored answer, CHAT
      # presentation. Reads parent #1345 paired_op answers from HF and renders
      # them through the chat template. Cell (c)'s presentation is PINNED here
      # (--form chat, plan §4 Phase D / framing 1): phase_d exists only to
      # produce cell (c), and its form must not depend on the caller
      # remembering the plan (C4). The driver keeps --form REQUIRED for
      # direct invocations; router pass-through can override deliberately.
      # NO NEW generation. The conv_id join to parent answers canonizes the
      # stripper's `stripped_` prefix (C5) and requires phase_a's shared fold
      # map (default eval_results/issue_2054/shared_fold_map.json).
      CMD=(
        uv run python scripts/issue2054_phase_d.py
        --scaffolds-dir data/issue_2054/scaffolds/
        --target-conv-ids 8000
        --output-dir data/issue_2054/cell_c/
        --seed 137
        --form chat
      )
      ;;
    capture)
      # Unit C: teacher-forced HF forward at layer 19, per 4-axis cell
      # (variant x condition x form x model — C6; outputs named by
      # issue2054_forms.cell_key). Emits {conv_id: {v_C, v_A, v_P}} per cell
      # + DV 7 answer-length parity and DV 8 conv_id intersection diagnostics
      # (per-row block included in PRODUCTION too — C7 gate-5 source).
      # --input-dir / --variants / --phase / --form / --model pass through
      # per-cell (--phase AND --form are REQUIRED by the driver; the default
      # --input-dir matches --phase inserted — override it for on_policy /
      # cell_c captures, or use the multi-GPU composer, which maps
      # phase -> input-dir and shards variants across GPUs:
      #   uv run python scripts/issue2054_shard_launch.py --driver capture \
      #     --condition <inserted|on_policy|cell_c> --form F --model M --gpus 0,..,7
      # (Unit F: --shard-index/--shard-count stride the variant list; per-cell
      # writes are disjoint by construction; shard digests aggregate post-hoc.)
      # --dry-run exercises the CLI + tokenization on <=3 rows without GPU.
      # max_new_tokens is irrelevant here (teacher-forced, no generation).
      CMD=(
        uv run python scripts/issue2054_capture.py
        --input-dir data/issue_2054/spliced_inserted/
        --output-dir data/issue_2054/activations/
        --seed 137
        --layer 19
      )
      ;;
    fits)
      # Unit D: per-cell ambient-basis ridge fit at layer 19 (K=5 shared
      # conversation-grouped folds via shared_fold_map.json from Unit A).
      # BOTH mapping arms — context AND prefix — per the CLAUDE.md standing
      # rule. Reports identity+learned-bias baseline + kNN retrieval per fitted
      # map (CLAUDE.md standing rule), a shuffled-answer matched-capacity null,
      # the reduced-k1024 diagnostic, conv-within-intersection bootstrap CI,
      # and kill-gate outcomes 4/5 (plan §7). Cells are keyed on all four
      # lattice axes (C6): --conditions / --forms default to the full closed
      # registries and only located .npz combos run; gate 5 pairs (b) vs (d)
      # as inserted <-> on_policy of the SAME (variant, form, model).
      # --dry-run / --pilot exercise the pipeline on a tiny slice without HF.
      CMD=(
        uv run python scripts/issue2054_fits.py
        --activations-dir data/issue_2054/activations/
        --fold-map eval_results/issue_2054/shared_fold_map.json
        --output-dir data/issue_2054/fits/
        --seed 137
        --layer 19
        --n-null-draws 100
      )
      ;;
    ladder)
      # Unit E: 9-rung transfer ladder between cells. For each ordered
      # (source, target) cell pair, computes the 9 mapping-transformation
      # rungs of the parent #1345 line (direct / ctx_offset / ans_offset /
      # bias_refit / global_scale / rotation / ctx_reparam / ans_reparam /
      # full_AMB) and scores held-out R² per rung + the ratio to the target's
      # own within-cell ceiling (from Unit D's fit JSONs).
      # BOTH mapping arms — context AND prefix — per CLAUDE.md standing rule.
      # Per-conversation bootstrap CI over the equalized-down intersection
      # (statistics-critic concern #2). --dry-run / --pilot skip HF + run 1
      # fold / self-transfer on the smoke fixture.
      #
      # M-R2-1: the production pair set RESTRICTS to the plan-§6 comparison
      # classes (cross_framing / cross_character / twobytwo / cross_model —
      # the driver default; pass --pair-classes all only as a deliberate
      # opt-in), and the auto pilot gate extrapolates its measured 1-unit
      # wall to the PENDING fleet: a projection over --max-fleet-wall-hours
      # (default 12 h) exits 7 — a DESIGNED halt with the projection in
      # pilot_gate_report.json (route on the artifact, not the bare rc).
      # Any dispatcher-armed timeout around this phase must be sized
      # >= the report's fence_floor_seconds (2x the projection).
      CMD=(
        uv run python scripts/issue2054_ladder.py
        --activations-dir data/issue_2054/activations/
        --fits-dir data/issue_2054/fits/
        --fold-map eval_results/issue_2054/shared_fold_map.json
        --output-dir data/issue_2054/ladder/
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
    printf '%s: NOT WIRED — exits 3\n' "$phase"
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
    echo "phase=$phase not yet wired" >&2
    exit 3
  fi

  echo "unknown phase: $phase" >&2
  usage >&2
  exit 2
}

main "$@"
