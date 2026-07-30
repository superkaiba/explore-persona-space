#!/usr/bin/env bash
# issue-1739 nonlinear-map round — PHASE-A + FAN-OUT launcher.
#
# Shape (path 2, the approved reduced grid):
#
#   phase A (ONE box)        fit every (variant, U rung) map ONCE per kind, gate
#                            each payload save->load->apply, publish to HF, then
#                            PROVE the lanes' staging step can consume them.
#   fan-out (6 lanes)        behavior x kind, single GPU each: stage the maps,
#                            score the path-2 grid, commit results, tear down.
#
# Why phase A exists: a map key costs ~2651 s to fit and is BEHAVIOR-INDEPENDENT
# (shared #1092 U pool, shared subsample + whitening seeds), so a naive fan-out
# re-fits the same 4 keys in all 6 lanes — ~5.9 GPU-h of duplicated fit per kind
# pair. Phase A pays it once; each lane loads the payloads.
#
# Two modes, neither of which provisions anything:
#   phase-a   run the phase-A legs (expects to be ON an already-provisioned box)
#   runbook   compose the phase-A + 6 lane commands into a runbook file
#
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

MODE="${1:-runbook}"

RESULTS_ROOT="eval_results/issue_1739"
NL_ROOT="$RESULTS_ROOT/nonlinear_map"
TENSORS_ROOT="analysis_tensors/issue_1739"
BRANCH="${EPM_I1739_BRANCH:-issue-1739}"
DISPATCH="scripts/issue1739_nlmap_dispatch.sh"

BEHAVIORS="${EPM_I1739_NL_BEHAVIORS:-evil sycophancy hallucination}"
KINDS="${EPM_I1739_NL_KINDS:-mlp kernel}"
# ---- path-2 reduced grid (the approved scope) -------------------------------
# 2 variants (fits default --variant both) x 2 U rungs x 3 draws x 2 seeds;
# per-behavior budgets stay the main grid's (behavior_budgets in the dispatcher).
P2_USIZES="${EPM_I1739_NL_USIZES:-250 full}"
P2_DRAWS="${EPM_I1739_NL_DRAWS:-0 1 2}"
P2_SEEDS="${EPM_I1739_NL_SEEDS:-0 1}"
# The pilot gate's own abort_mult; the per-lane PLAN_WALL_H already carries the
# projector's fence multiple, so the gate compares 1x against a fenced number.
P2_ABORT_MULT="${EPM_I1739_NL_PILOT_ABORT_MULT:-1}"
RUNBOOK="${EPM_I1739_NL_RUNBOOK:-$NL_ROOT/fanout_runbook.md}"

log() { echo "[fanout] $*"; }

lane_slug() { # lane_slug <behavior> <kind> -> pod --name-suffix (lowercase, letter-initial)
  local b="$1" kind="$2"
  printf 'nl%s%s' "$(printf '%s' "${b:0:4}" | tr 'A-Z' 'a-z')" "$(printf '%s' "${kind:0:3}")"
}

plan_wall_for() { # plan_wall_for <behavior> -> fenced per-lane hours (MEASURED basis)
  uv run python scripts/issue1739_nlmap_project.py \
    --plan-wall-for "$1" \
    --draws "$(printf '%s\n' $P2_DRAWS | wc -l)" \
    --seeds "$(printf '%s\n' $P2_SEEDS | wc -l)" \
    --u-rungs "$(printf '%s\n' $P2_USIZES | wc -l)"
}

lane_env() { # lane_env <behavior> <kind> -> the env assignments for one lane
  local b="$1" kind="$2" wall
  wall="$(plan_wall_for "$b")"
  printf '%s\n' \
    "EPM_I1739_NL_BEHAVIORS='$b'" \
    "EPM_I1739_NL_KINDS='$kind'" \
    "EPM_I1739_NL_USIZES='$P2_USIZES'" \
    "EPM_I1739_NL_DRAWS='$P2_DRAWS'" \
    "EPM_I1739_NL_SEEDS='$P2_SEEDS'" \
    "EPM_I1739_NL_PLAN_WALL_H=$wall" \
    "EPM_I1739_NL_PILOT_ABORT_MULT=$P2_ABORT_MULT" \
    "EPM_I1739_NL_PHASE='stage,stage_maps,pilot,fits,collect,upload_results'"
}

# ===========================================================================
# MODE: phase-a — maps once, gated, published, and proven consumable
# ===========================================================================
if [ "$MODE" = "phase-a" ]; then
  log "phase-a start $(date -u +%FT%TZ) repo_root=$REPO_ROOT branch=$BRANCH"
  log "phase-a kinds='$KINDS' usizes='$P2_USIZES' seeds='$P2_SEEDS'"

  # (1) inputs + (2) fit every map key once per kind. The fits-side round-trip
  # gate (save->load->apply allclose, _verify_map_roundtrip) fires per freshly
  # written payload and is FAIL-LOUD, so a serialization defect stops phase A
  # here rather than silently poisoning all 6 lanes.
  EPM_I1739_NL_KINDS="$KINDS" \
  EPM_I1739_NL_USIZES="$P2_USIZES" \
  EPM_I1739_NL_SEEDS="$P2_SEEDS" \
  EPM_I1739_NL_PHASE='stage,prefetch' \
    bash "$DISPATCH"

  # (3) publish the payloads. `--stage tensors` is ONE bulk upload_folder commit
  # plus its own scoped exact-set verify (issue1739_upload.upload_tree).
  log "phase-a: publishing map payloads -> HF"
  EPM_I1739_NL_KINDS="$KINDS" EPM_I1739_NL_PHASE='upload_tensors' bash "$DISPATCH"

  # (4) CROSS-PHASE DATA-CONTRACT GATE: run the LANES' OWN staging step against
  # a scratch root. This is the only check that proves the (a) HF prefix, (b)
  # payload filenames and (c) consumer-openability all line up — the failure the
  # lanes would otherwise hit one provision later, six times over.
  scratch="$(mktemp -d "${TMPDIR:-/tmp}/i1739-nlmap-stageprobe-XXXXXX")"
  trap 'rm -rf "$scratch"' EXIT
  log "phase-a: staging-contract probe -> $scratch"
  uv run python scripts/issue1739_nlmap_stage_maps.py \
    --tensors-root "$scratch" \
    --kinds $KINDS \
    --u-labels $P2_USIZES \
    --map-seed "${P2_SEEDS%% *}" \
    --out "$NL_ROOT/phase_a_stage_probe.json"
  log "phase-a: staging-contract probe PASS"

  # (5) commit phase A's own artifacts (the probe report + map_quality if built).
  EPM_I1739_NL_KINDS="$KINDS" EPM_I1739_NL_PHASE='collect,upload_results' bash "$DISPATCH"

  log "phase-a complete $(date -u +%FT%TZ) — the 6 lanes may now run"
  echo "[phase=done]"
  exit 0
fi

# ===========================================================================
# MODE: runbook — compose the commands; execute NOTHING
# ===========================================================================
if [ "$MODE" != "runbook" ]; then
  echo "[fanout] FATAL: unknown mode '$MODE' (expected: phase-a | runbook)" >&2
  exit 2
fi

mkdir -p "$(dirname "$RUNBOOK")"
tmp="$RUNBOOK.tmp"
{
  echo "# issue-1739 nonlinear-map round — fan-out runbook"
  echo
  echo "Generated $(date -u +%FT%TZ) at \`$(git rev-parse HEAD)\` on branch \`$BRANCH\`."
  echo "Composed by \`scripts/issue1739_nlmap_fanout.sh runbook\` — nothing was executed."
  echo
  echo '## Projection (MEASURED basis — see scripts/issue1739_nlmap_project.py)'
  echo
  echo '```'
  uv run python scripts/issue1739_nlmap_project.py \
    --behaviors $BEHAVIORS --kinds $KINDS \
    --draws "$(printf '%s\n' $P2_DRAWS | wc -l)" \
    --seeds "$(printf '%s\n' $P2_SEEDS | wc -l)" \
    --u-rungs "$(printf '%s\n' $P2_USIZES | wc -l)" \
    --phase-a-gpus "$(printf '%s\n' $KINDS | wc -l)" \
    --lane-concurrency 6
  echo '```'
  echo
  echo '## Step 1 — phase A (ONE box, one GPU per map kind)'
  echo
  echo 'Fits every (variant, U rung) map ONCE per kind, gates each payload'
  echo '(save->load->apply allclose, fail-loud), publishes them to HF, then runs'
  echo "the lanes' OWN staging step against a scratch root as a data-contract"
  echo 'gate. Provision a box with >= 1 GPU per kind, then:'
  echo
  echo '```bash'
  echo "cd \"\$WORKLOAD_ROOT\"  # or the pod's repo root"
  echo "EPM_I1739_NL_KINDS='$KINDS' \\"
  echo "EPM_I1739_NL_USIZES='$P2_USIZES' \\"
  echo "EPM_I1739_NL_SEEDS='$P2_SEEDS' \\"
  echo "  bash scripts/issue1739_nlmap_fanout.sh phase-a"
  echo '```'
  echo
  echo 'Gate to check before launching any lane: the phase-A log must show one'
  echo '`[fits] map round-trip gate PASS` per payload, `phase-a: staging-contract'
  echo 'probe PASS`, and a terminal `[phase=done]`.'
  echo
  echo '## Step 2 — the 6 scoring lanes (behavior x kind, single GPU each)'
  echo
  echo 'Each lane stages the phase-A payloads (fail-loud if absent — a lane that'
  echo 'silently re-fits would throw away the whole amortization), runs the'
  echo 'path-2 grid, commits its results with a pre-commit ff-sync (#1880 push'
  echo 'race: 6 lanes write this branch concurrently), and tears down.'
  echo
  echo 'Run all 6 concurrently. Per-lane `PLAN_WALL_H` is derived from the'
  echo 'MEASURED basis by the projector, NOT inherited from the previous round.'
  echo
  for b in $BEHAVIORS; do
    for kind in $KINDS; do
      slug="$(lane_slug "$b" "$kind")"
      echo "### lane $b / $kind  (pod suffix: \`$slug\`)"
      echo
      echo '```bash'
      lane_env "$b" "$kind" | sed 's/$/ \\/'
      echo "  bash scripts/issue1739_nlmap_dispatch.sh"
      echo '```'
      echo
    done
  done
  echo '## Notes'
  echo
  echo '- `stage_maps` and `prefetch` are opt-in phases: `PHASE=all` never runs'
  echo '  them, so the legacy single-box dispatch is unchanged by this round.'
  echo '- Lanes run `upload_results` (NOT `upload`): re-uploading the identical'
  echo '  tensors tree from 6 lanes would burn 6 Hub commits against the'
  echo '  256/hr repo cap for zero new bytes.'
  echo '- A lane whose pilot projects past its fenced `PLAN_WALL_H` exits rc=7'
  echo '  (a DESIGNED halt with `pilot_report.json`, not a crash) — re-size from'
  echo '  that report rather than raising the fence blindly.'
  echo "- Every lane pins seeds[0]=${P2_SEEDS%% *}, matching the seed phase A fit the"
  echo '  maps under. `_load_nl_map` refuses a payload whose recorded `map_seed`'
  echo '  differs: for a subsampled U rung the pool ROWS depend on that seed, and'
  echo '  the row-count guard cannot see the difference.'
} > "$tmp"
mv "$tmp" "$RUNBOOK"
log "runbook -> $RUNBOOK"
