#!/usr/bin/env bash
# Issue #2225 fu1 (`fu1_preimage_prevention`) pod-side phase orchestrator
# (fu1 plan §4.0 DAG). Mirrors scripts/issue2225_dispatch.sh — same subprocess
# shape, same launch width, smoke via *_smoke out-root twins.
#
#   bash scripts/issue2225_fu1_dispatch.sh [phase]
#
# Phases (default: all):
#   external  pinned persona_vectors clone + issue_778/issue_2225 sparse cones
#   f0        pre-image directions: rho probe (1 GPU) -> CPU algebra build
#             (FULL payloads, plan §4.5 blind-spot (e)) -> upload
#   f1        §7 pilot gate: fu1 train --pilot (K+M evil, 8 cells) -> hook grep
#             -> eval-gen -> judge --sync -> F1 verdict. First miss with an
#             octave-shift recommendation -> ONE automatic re-pilot at the
#             shifted grid (per-arm); second miss -> proceed with the widest
#             bracketing grid + a limitation note in the sentinel.
#   f2a       train fan-out (80 fu1 cells; per-cell resume + per-cell HF upload)
#   f2b       trait-eval generation (fu1 targets) + narrow-domain + upload
#   f2c       MMLU on the 40 grid extremes (+ the §4.4 accuracy-drop extension
#             trigger: any extreme < 0.683 evaluates that arm's full grid)
#   f2d       activation capture (fu1 targets) + upload
#   f3        idempotent upload safety pass + results sentinel
#   all       the full DAG in order, ending with the single [phase=done] line
#
# Smoke: EPM_I2225_SMOKE=1 diverts every out-root, the per-cell LOG_ROOT
# (round-4 fix 2), AND every HF upload prefix to *_smoke twins (round-3
# fix 2), threads tiny-N dials into
# every phase (same dispatcher, same subprocess shape, same launch width), and
# RUNS the F1 verdict over the smoke cells' own arms + grid (--arms/--grid
# derived from SMOKE_CELLS; round-3 fix 1) with its production-n-calibrated
# outcome demoted to informational (gate-calibration rule). Smoke cells span
# BOTH new mask classes: one `context` + one `context_end` (plan §4.5).
#
# Pod-side contract: NEVER shells to scripts/task.py; progress via [phase=...]
# lines + /workspace/logs sentinels (issue778_lib.write_results_sentinel).
# Env-override PAIRING: overriding EPM_I2225_FU1_OUT_ROOT alone does NOT
# relocate sentinels — off-pod invocations must ALSO set
# EPM_I2225_SENTINEL_ROOT (else `mkdir -p /workspace/logs` fails loud below).
set -euo pipefail

cd "$(dirname "$0")/.."

# Pod-side env: .env exists on RunPod clones; GCE/SLURM lanes export tokens
# via their startup env instead (conditional sourcing — pod-side-reporting.md).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# fu1 cell resolution for EVERY child process (train/eval/mmlu/capture
# subprocesses inherit this env; issue2225_train.resolve_cell's env hook).
export EPM_I2225_EXTRA_CELLS_MODULE=issue2225_fu1_train

ISSUE=2225
PHASE="${1:-all}"
SMOKE="${EPM_I2225_SMOKE:-}"

PV_SHA="b8e0f044fe2410a6fad579f38324f03f13b4e917"
PV_REPO="https://github.com/safety-research/persona_vectors.git"
EXTERNAL_ROOT="external/persona_vectors"
I778_BASELINE="eval_results/issue_778/finetune_evil_evil_misaligned_2.json"

# Bidirectional smoke/full root pairs (gotchas.md smoke-var rule: every
# *_smoke rebinding sits behind an explicit $SMOKE guard).
OUT_ROOT_FULL="${EPM_I2225_FU1_OUT_ROOT:-/workspace/eps_out/issue2225_fu1}"
OUT_ROOT_SMOKE="${OUT_ROOT_FULL}_smoke"
EVAL_ROOT_FULL="eval_results/issue_2225/fu1_preimage_prevention"
EVAL_ROOT_SMOKE="data/issue_2225/fu1_eval_smoke"   # scratch — never the committed tree
OUT_ROOT="$OUT_ROOT_FULL"
EVAL_ROOT="$EVAL_ROOT_FULL"
if [ -n "$SMOKE" ]; then
  OUT_ROOT="$OUT_ROOT_SMOKE"
  EVAL_ROOT="$EVAL_ROOT_SMOKE"
fi

CKPT_ROOT="$OUT_ROOT/adapters"
DIR_OUT="$OUT_ROOT/directions"
PILOT_OUT="$OUT_ROOT/pilot_out"
JUDGE_CACHE="$OUT_ROOT/judge_cache"
JUDGE_RAW="$OUT_ROOT/judge_raw"
LOG_ROOT="${EPM_I2225_LOG_ROOT:-/workspace/logs/issue-2225-fu1}"
if [ -n "$SMOKE" ]; then
  # Round-4 fix 2 (log-namespace isolation — same class as the round-3 HF-prefix
  # twins): smoke per-cell fan-out logs land in a _smoke twin so smoke logs can
  # never contaminate a production log-dir read (the pre-fix smoke wrote
  # J/L logs into the production f1_train dir -> the engagement tally read 10/8
  # and FATALed a healthy production run).
  LOG_ROOT+="_smoke"
fi
# SENTINELS are contract-bound to the poller's TOP-LEVEL drain glob
# /workspace/logs/issue-2225-*.json (poll_pipeline.py — the glob does not
# cross '/'); phase logs stay in the $LOG_ROOT subdir.
SENTINEL_ROOT="${EPM_I2225_SENTINEL_ROOT:-/workspace/logs}"
mkdir -p "$LOG_ROOT" "$SENTINEL_ROOT"

# fu1 HF prefixes (plan §10 — threaded per-round, never parent-clobbering).
HF_FINAL="issue2225_ctxsteer/raw_completions/fu1_final"
HF_NARROW="issue2225_ctxsteer/raw_completions/fu1_narrow_domain"
HF_PILOT="issue2225_ctxsteer/raw_completions/fu1_pilot"
HF_JUDGE="issue2225_ctxsteer/raw_completions/fu1_judge"
HF_MMLU="issue2225_ctxsteer/fu1_mmlu"
HF_CAPTURE="issue2225_ctxsteer/analysis_tensors/fu1_capture"
HF_DIRECTIONS="issue2225_ctxsteer/analysis_tensors/fu1_directions"
if [ -n "$SMOKE" ]; then
  # Smoke uploads exercise the REAL Hub path but land at _smoke-suffixed
  # prefixes — smoke artifacts (n_prompts=16 rho, 2-step adapters, 2-question
  # rollouts) must never contaminate the production prefixes (round-3 fix 2;
  # the pre-fix smoke uploaded a smoke-dial rho bank to the production
  # fu1_directions prefix and J/L pilot raws to fu1_pilot/fu1_judge).
  HF_FINAL+="_smoke"
  HF_NARROW+="_smoke"
  HF_PILOT+="_smoke"
  HF_JUDGE+="_smoke"
  HF_MMLU+="_smoke"
  HF_CAPTURE+="_smoke"
  HF_DIRECTIONS+="_smoke"
fi

# Round-5 fix (#1108 overflow contract): the canonical data repo is AT the
# 1M-file ceiling (capture's 240 files were rejected with "would contain
# 1000107 > 1000000"), so the capture upload class is env-routable to the
# private overflow repo — SAME $HF_CAPTURE prefix layout there (consumers
# stage capture from that repo). Default empty = canonical repo; no other
# upload class changes (MMLU's 80 files fit the remaining headroom).
CAPTURE_HF_REPO="${EPM_I2225_FU1_CAPTURE_HF_REPO:-}"
CAPTURE_REPO_ARGS=()
if [ -n "$CAPTURE_HF_REPO" ]; then
  CAPTURE_REPO_ARGS=(--hf-repo "$CAPTURE_HF_REPO")
fi

# Smoke dials (threaded, never a different code path). Two cells spanning the
# two NEW mask classes: J = context, L = context_end (plan §4.5).
TRAIN_SMOKE_ARGS=()
EVALGEN_SMOKE_ARGS=()
MMLU_LIMIT_ARGS=()
RHO_N_PROMPTS=512
SMOKE_CELLS="J__evil__c0.25,L__evil__c0.25"
SMOKE_TARGETS="base,J__evil__c0.25,L__evil__c0.25"
if [ -n "$SMOKE" ]; then
  TRAIN_SMOKE_ARGS=(--max-steps 2 --no-upload)
  EVALGEN_SMOKE_ARGS=(--n-questions 2 --n-rollouts 1)
  MMLU_LIMIT_ARGS=(--limit 200)   # plan §4.5 blind-spot (b)
  RHO_N_PROMPTS=16
fi

log_phase() { echo "[phase=$1] $2"; }

headroom() {  # headroom <path> <need_gb> <phase>
  mkdir -p "$1"
  uv run python -c "
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom
assert_out_root_headroom('$1', $2, phase='$3')
"
}

# Sentinel writer (pod-side contract: sentinel file, never task.py). Params via
# env so the quoted heredoc stays interpolation-free.
write_sentinel() {  # write_sentinel <kind> <version> <note_json_file>
  SENT_KIND="$1" SENT_VERSION="$2" SENT_NOTE_FILE="$3" SENT_LOGS_DIR="$SENTINEL_ROOT" \
    uv run python - <<'PY'
import json, os, pathlib, sys

sys.path.insert(0, "scripts")
import issue778_lib as lib

note = json.load(open(os.environ["SENT_NOTE_FILE"]))
path = lib.write_results_sentinel(
    2225,
    os.environ["SENT_KIND"],
    int(os.environ["SENT_VERSION"]),
    note,
    logs_dir=pathlib.Path(os.environ["SENT_LOGS_DIR"]),
)
print(f"[sentinel] {path}", flush=True)
PY
}

fu1_slugs() {  # fu1_slugs <python-expr filtering cells>  -> comma list on stdout
  FU1_FILTER="$1" uv run python -c "
import os, sys
sys.path.insert(0, 'scripts')
import issue2225_fu1_train as fu1
cells = fu1.build_fu1_cell_registry()
flt = eval('lambda c: ' + os.environ['FU1_FILTER'])
print(','.join(c.slug for c in cells if flt(c)))
"
}

# Plan §7: every production phase enumerates the F1-EFFECTIVE grid — after an
# octave-shift re-pilot, F2a/F2b/F2c/F2d + the F3 card follow the shifted grid
# (and the re-piloted adapters become first-class targets), never the original
# grid the pilot demonstrated mis-placed. An UNRESOLVED state fails loud.
FU1_REPILOT_STATE="$EVAL_ROOT/pilot_gate/f1_repilot_state.json"

fu1_effective_slugs() {  # fu1_effective_slugs all|extremes -> comma list (F1-effective grid)
  FU1_MODE="$1" FU1_STATE="$FU1_REPILOT_STATE" uv run python -c "
import os, sys
sys.path.insert(0, 'scripts')
import issue2225_fu1_train as fu1
state = os.environ['FU1_STATE']
cells = fu1.effective_fu1_cells(state if os.path.exists(state) else None)
mode = os.environ['FU1_MODE']
if mode == 'extremes':
    cells = fu1.fu1_extreme_cells(cells)
else:
    assert mode == 'all', mode
print(','.join(c.slug for c in cells))
"
}

# ── external: pinned clone + sparse cones + venv freeze ────────────────────────
phase_external() {
  log_phase external "cloning $PV_REPO @ $PV_SHA"
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
    log_phase external "unzip dataset.zip"
    ( cd "$EXTERNAL_ROOT" && unzip -q -o dataset.zip )
  fi
  test -d "$EXTERNAL_ROOT/dataset" || { echo "FATAL: dataset/ missing after unzip" >&2; exit 1; }

  # Pod partial clones cone src/scripts/configs/tests/docs+data — eval_results
  # is OUTSIDE the default cones; the F1 verdict reads the committed #778
  # baseline and F2+/F5 read the parent's banked issue_2225 anchors.
  if git sparse-checkout list >/dev/null 2>&1; then
    if [ ! -f "$I778_BASELINE" ]; then
      log_phase external "sparse-checkout add eval_results/issue_778"
      git sparse-checkout add eval_results/issue_778
    fi
    if [ ! -f "eval_results/issue_2225/analysis/selection.json" ]; then
      log_phase external "sparse-checkout add eval_results/issue_2225"
      git sparse-checkout add eval_results/issue_2225
    fi
  fi
  test -f "$I778_BASELINE" || { echo "FATAL: $I778_BASELINE missing (F1 verdict input)" >&2; exit 1; }

  # Resolve the venv ONCE, then freeze it for every multi-worker fan-out (the
  # #1689 MooseFS FUSE-wedge prevention).
  uv run python -c "import explore_persona_space" >/dev/null
  export UV_NO_SYNC=1
  log_phase external "done (sha=$GOT_SHA)"
}

# ── F0: pre-image directions + rho probe ───────────────────────────────────────
phase_f0() {
  # Idempotent entry skip (mirrors the f1 verdict resume-skip): direction_sha256
  # is part of every cell_fingerprint, so re-running f0 after a later-phase
  # crash would re-measure rho through GPU forwards and any byte drift in the
  # rebuilt banks would invalidate ALL 80 cell manifests (silent full retrain).
  # Force a rebuild explicitly with EPM_I2225_FU1_F0_FORCE=1.
  if [ -z "${EPM_I2225_FU1_F0_FORCE:-}" ] \
    && [ -f "$DIR_OUT/rho.json" ] \
    && [ -f "$DIR_OUT/evil_PRE.pt" ] \
    && [ -f "$DIR_OUT/sycophancy_PRE.pt" ] \
    && [ -f "$DIR_OUT/hallucination_PRE.pt" ] \
    && [ -f "$DIR_OUT/RND.pt" ]; then
    log_phase f0_directions "rho.json + all 4 banks present — resume skip (EPM_I2225_FU1_F0_FORCE=1 to rebuild)"
    return 0
  fi
  log_phase f0_directions "start (out=$DIR_OUT)"
  headroom "$OUT_ROOT" 10 f0_directions
  log_phase f0_directions "rho probe (n_prompts=$RHO_N_PROMPTS per corpus, 1 GPU)"
  uv run python scripts/issue2225_fu1_directions.py --probe-rho \
    --out-dir "$DIR_OUT" --dataset-root "$EXTERNAL_ROOT/dataset" \
    --n-prompts "$RHO_N_PROMPTS"
  log_phase f0_directions "build (CPU algebra on the FULL payloads)"
  uv run python scripts/issue2225_fu1_directions.py --build \
    --out-dir "$DIR_OUT" --dataset-root "$EXTERNAL_ROOT/dataset"
  uv run python scripts/issue2225_fu1_directions.py --upload --out-dir "$DIR_OUT" --hf-prefix "$HF_DIRECTIONS"
  log_phase f0_directions "done"
}

# Persist pilot rollout text BEFORE judging (upload policy: raw completions for
# ALL stages) — the fu1_pilot prefix, distinct from the production fu1_final.
f1_upload_pilot_raws() {
  PILOT_LOCAL="$PILOT_OUT/raw_completions/final" PILOT_PREFIX="$HF_PILOT" uv run python - <<'PY'
import os
import pathlib

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import _upload

url = _upload(
    pathlib.Path(os.environ["PILOT_LOCAL"]),
    "superkaiba1/explore-persona-space-data",
    "dataset",
    os.environ["PILOT_PREFIX"],
)
if not url:
    raise SystemExit("fu1 pilot raw-completions upload returned no path")
print(f"[f1-upload] {url}", flush=True)
PY
}

f1_judge() {  # f1_judge <phase-label>
  log_phase "$1" "judge --sync --stage pilot"
  uv run python scripts/issue2225_judge.py --phase all --sync --stage pilot \
    --eval-root "$EVAL_ROOT" --external-root "$EXTERNAL_ROOT" \
    --rollouts-dir "$PILOT_OUT/raw_completions/final" \
    --narrow-rollouts-dir "$PILOT_OUT/raw_completions/narrow_domain" \
    --cache-root "$JUDGE_CACHE" --save-raw-root "$JUDGE_RAW"
  uv run python scripts/issue2225_judge.py --phase upload --stage pilot \
    --eval-root "$EVAL_ROOT" --external-root "$EXTERNAL_ROOT" \
    --rollouts-dir "$PILOT_OUT/raw_completions/final" \
    --narrow-rollouts-dir "$PILOT_OUT/raw_completions/narrow_domain" \
    --cache-root "$JUDGE_CACHE" --save-raw-root "$JUDGE_RAW" \
    --raw-judge-hf-prefix "$HF_JUDGE"
}

hook_count_gate() {  # hook_count_gate <log_dir> <slugs_csv> <phase-label>
  # §7 criterion (i): every enumerated cell's OWN log ($log_dir/<slug>.log)
  # carries the [steer-hook] engagement line OR the [fanout-skip] resume token
  # (a resume-skipped cell proved engagement on its fingerprint-bound run).
  # Round-4 fix 1 (gotchas.md count-keyed liveness class): keyed to the CURRENT
  # invocation's slug set, never a directory-wide tally — foreign logs in a
  # shared dir (the smoke's 2 cells beside production's 8) inflated the old
  # count to 10/8 and FATALed a healthy run. A missing/unengaged cell is named.
  local raw=() cells=() missing=() slug
  IFS=',' read -ra raw <<< "$2"
  for slug in "${raw[@]}"; do
    if [ -n "$slug" ]; then cells+=("$slug"); fi
  done
  if [ "${#cells[@]}" -eq 0 ]; then
    echo "FATAL: hook_count_gate called with an empty slug list (phase $3)" >&2
    exit 7
  fi
  for slug in "${cells[@]}"; do
    if ! grep -qF -e "[steer-hook]" -e "[fanout-skip]" "$1/${slug}.log" 2>/dev/null; then
      missing+=("$slug")
    fi
  done
  log_phase "$3" "hook-engagement logs (fresh or resume-skip): $(( ${#cells[@]} - ${#missing[@]} ))/${#cells[@]} (slug-keyed)"
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "FATAL: §7 criterion (i) FAILED — cells missing [steer-hook]/[fanout-skip] in $1: ${missing[*]}" >&2
    exit 7
  fi
}

# ── F1: §7 pilot gate (K+M evil, 8 cells) ──────────────────────────────────────
phase_f1() {
  local verdict="$EVAL_ROOT/pilot_gate/f1_verdict.json"
  local repilot_state="$FU1_REPILOT_STATE"
  if [ -f "$verdict" ] && [ -z "$SMOKE" ]; then
    local prior_pass
    prior_pass=$(uv run python -c "import json;print(json.load(open('$verdict'))['passed'])")
    if [ "$prior_pass" = "True" ]; then
      log_phase f1_pilot "prior PASSed verdict present — resume skip"
      return 0
    fi
    if [ -f "$repilot_state" ]; then
      local resolved
      resolved=$(uv run python -c "import json;print(json.load(open('$repilot_state')).get('resolved', False))")
      if [ "$resolved" = "True" ]; then
        log_phase f1_pilot "re-pilot already resolved (see $repilot_state) — resume skip"
        return 0
      fi
    fi
  fi
  log_phase f1_pilot "train --pilot (8 cells: K+M x fu1 grid, evil II)"
  headroom "$CKPT_ROOT" 30 f1_pilot
  local f1_logs="$LOG_ROOT/f1_train"
  local pilot_cells_arg=()
  if [ -n "$SMOKE" ]; then
    # Smoke keeps the SAME chain but the per-arm-class 2-cell slice (one
    # context + one context_end) via --cells; production uses --pilot.
    uv run python scripts/issue2225_fu1_train.py --fan-out --cells "$SMOKE_CELLS" \
      --ckpt-root "$CKPT_ROOT" --directions-dir "$DIR_OUT" \
      --dataset-root "$EXTERNAL_ROOT/dataset" \
      --log-dir "$f1_logs" "${TRAIN_SMOKE_ARGS[@]}"
    pilot_cells_arg=("$SMOKE_CELLS")
  else
    uv run python scripts/issue2225_fu1_train.py --pilot --fan-out \
      --ckpt-root "$CKPT_ROOT" --directions-dir "$DIR_OUT" \
      --dataset-root "$EXTERNAL_ROOT/dataset" \
      --log-dir "$f1_logs"
    pilot_cells_arg=("$(fu1_slugs "c.config in ('K', 'M') and c.dataset == 'evil'")")
  fi

  hook_count_gate "$f1_logs" "${pilot_cells_arg[0]}" f1_pilot

  log_phase f1_pilot "eval-gen (pilot targets, n_rollouts=5)"
  headroom "$PILOT_OUT" 10 f1_pilot
  uv run python scripts/issue2225_eval_gen.py \
    --out-root "$PILOT_OUT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
    --targets "${pilot_cells_arg[0]}" --n-rollouts 5 "${EVALGEN_SMOKE_ARGS[@]}"

  f1_upload_pilot_raws
  f1_judge f1_pilot

  log_phase f1_pilot "F1 verdict (§7 criterion ii)"
  local rc=0
  local verdict_args=()
  if [ -n "$SMOKE" ]; then
    # Round-3 fix 1 (#1611/#1355 class): the verdict's default arms are the
    # §7 pilot arms K/M, but the smoke trains J/L — score the smoke's OWN
    # cells (arms + grid derived mechanically from SMOKE_CELLS, never a
    # second hand-maintained list) so the verdict code path actually RUNS
    # under smoke. Its outcome stays informational (demotion below); the
    # crash-with-no-artifact FATAL guard stays binding for genuine crashes.
    local smoke_arms smoke_grid
    smoke_arms=$(SC="$SMOKE_CELLS" uv run python -c "
import os
print(','.join(sorted({s.split('__')[0] for s in os.environ['SC'].split(',')})))
")
    smoke_grid=$(SC="$SMOKE_CELLS" uv run python -c "
import os
print(','.join(sorted({s.split('__c')[1] for s in os.environ['SC'].split(',')})))
")
    verdict_args=(--arms "$smoke_arms" --grid "$smoke_grid")
  fi
  uv run python scripts/issue2225_fu1_verdict.py \
    --eval-root "$EVAL_ROOT" --i778-baseline "$I778_BASELINE" "${verdict_args[@]}" || rc=$?
  if [ ! -f "$verdict" ]; then
    echo "FATAL: f1-verdict crashed (rc=$rc) with no verdict artifact" >&2
    exit "$rc"
  fi

  local note_file="$LOG_ROOT/f1_note.json"
  F1_VERDICT_FILE="$verdict" F1_NOTE_FILE="$note_file" F1_SMOKE="$SMOKE" \
    uv run python - <<'PY'
import json, os

verdict = json.load(open(os.environ["F1_VERDICT_FILE"]))
note = {
    "phase": "f1_pilot",
    "followup": "fu1_preimage_prevention",
    "passed": verdict["passed"],
    "octave_shift": verdict["octave_shift"],
    "criteria": verdict["criteria"],
    "smoke": bool(os.environ.get("F1_SMOKE")),
    "next": (
        "proceed to F2a"
        if verdict["passed"]
        else "octave-shift re-pilot: ONE automatic re-pilot with the shifted "
        "grid runs next (fu1 train --pilot --coef-scale; plan §7 remedy)"
    ),
}
with open(os.environ["F1_NOTE_FILE"], "w") as f:
    json.dump(note, f, indent=1)
PY
  if [ -n "$SMOKE" ]; then
    write_sentinel "epm:smoke-result" 1 "$note_file"
  else
    write_sentinel "epm:progress" 1 "$note_file"
  fi

  if [ "$rc" -ne 0 ]; then
    if [ -n "$SMOKE" ]; then
      # Gate-calibration rule: coherence bracketing at smoke N measures N,
      # not the grid — the verdict is informational under smoke.
      log_phase f1_pilot "verdict FAILED (rc=$rc) — informational under smoke, continuing"
    else
      f1_handle_verdict_fail "$verdict" "$repilot_state"
    fi
  fi
  log_phase f1_pilot "done"
}

# §7 first-miss routing: ONE automatic re-pilot with the octave-shifted grid.
# A second miss proceeds with the widest bracketing grid found + a limitation
# note (never a third pilot round). SMOKE BLIND SPOT (disclosed in the smoke-
# architecture marker): under EPM_I2225_SMOKE a verdict FAIL is informational,
# so this path is exercised by NO smoke — first exercised on a live F1 miss
# (the parent's identical p0_handle_verdict_fail carries the same disclosure).
f1_handle_verdict_fail() {  # <verdict_json> <state_json>
  local verdict="$1" state="$2"
  if [ ! -f "$state" ]; then
    local has_plan
    has_plan=$(uv run python -c "import json;print(bool(json.load(open('$verdict')).get('repilot')))")
    if [ "$has_plan" != "True" ]; then
      log_phase f1_pilot "DESIGNED HALT rc=7 — verdict FAILED with no octave-shift recommendation"
      exit 7
    fi
    REPILOT_VERDICT="$verdict" REPILOT_STATE="$state" uv run python - <<'PY'
import json, os

verdict = json.load(open(os.environ["REPILOT_VERDICT"]))
with open(os.environ["REPILOT_STATE"], "w") as f:
    json.dump({"plan": verdict["repilot"], "resolved": False}, f, indent=1)
print(
    "[f1-repilot] plan: "
    + "; ".join(f"{k}: x{v['coef_scale']} -> {v['grid_csv']}" for k, v in verdict["repilot"].items()),
    flush=True,
)
PY
  fi
  f1_run_repilot "$state"
}

f1_run_repilot() {  # <state_json> — idempotent (resume-safe); runs to a resolution
  local state="$1"
  local rp_logs="$LOG_ROOT/f1_train_repilot"
  local arms arm scale grid cells all_cells=""
  local gridargs=()
  arms=$(uv run python -c "import json;print(' '.join(sorted(json.load(open('$state'))['plan'])))")
  for arm in $arms; do
    scale=$(uv run python -c "import json;print(json.load(open('$state'))['plan']['$arm']['coef_scale'])")
    grid=$(uv run python -c "import json;print(json.load(open('$state'))['plan']['$arm']['grid_csv'])")
    cells=$(uv run python -c "import json;print(','.join(json.load(open('$state'))['plan']['$arm']['cells']))")
    log_phase f1_repilot "arm $arm: train --pilot --coef-scale $scale (grid $grid)"
    uv run python scripts/issue2225_fu1_train.py --pilot --pilot-configs "$arm" \
      --coef-scale "$scale" --fan-out \
      --ckpt-root "$CKPT_ROOT" --directions-dir "$DIR_OUT" \
      --dataset-root "$EXTERNAL_ROOT/dataset" \
      --log-dir "$rp_logs"
    all_cells="${all_cells:+$all_cells,}$cells"
    gridargs+=(--grid-arm "$arm=$grid")
  done

  hook_count_gate "$rp_logs" "$all_cells" f1_repilot

  log_phase f1_repilot "eval-gen (re-pilot targets: $all_cells)"
  headroom "$PILOT_OUT" 10 f1_repilot
  uv run python scripts/issue2225_eval_gen.py \
    --out-root "$PILOT_OUT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
    --targets "$all_cells" --n-rollouts 5

  f1_upload_pilot_raws
  f1_judge f1_repilot

  log_phase f1_repilot "re-verdict (${gridargs[*]})"
  local rc2=0
  uv run python scripts/issue2225_fu1_verdict.py \
    --eval-root "$EVAL_ROOT" --i778-baseline "$I778_BASELINE" "${gridargs[@]}" || rc2=$?

  local note_file="$LOG_ROOT/f1_repilot_note.json"
  REPILOT_VERDICT="$EVAL_ROOT/pilot_gate/f1_verdict.json" REPILOT_NOTE="$note_file" \
    REPILOT_RC="$rc2" REPILOT_STATE="$state" uv run python - <<'PY'
import json, os

verdict = json.load(open(os.environ["REPILOT_VERDICT"]))
rc = int(os.environ["REPILOT_RC"])
spath = os.environ["REPILOT_STATE"]
state = json.load(open(spath))
state["resolved"] = True
state["second_miss"] = rc != 0
with open(spath, "w") as f:
    json.dump(state, f, indent=1)
note = {
    "phase": "f1_repilot",
    "followup": "fu1_preimage_prevention",
    "passed": verdict["passed"],
    "octave_shift": verdict["octave_shift"],
    "repilot_plan": {k: v["grid_csv"] for k, v in state["plan"].items()},
    "next": (
        "proceed to F2a (re-pilot PASSed on the shifted grid)"
        if verdict["passed"]
        else "SECOND MISS: proceeding with the widest bracketing grid found "
        "(plan §7 second-miss path); LIMITATION: coefficient placement is not "
        "coherence-bracketed for the shifted arm(s) — carried as a scope caveat"
    ),
}
with open(os.environ["REPILOT_NOTE"], "w") as f:
    json.dump(note, f, indent=1)
PY
  write_sentinel "epm:progress" 1 "$note_file"
  if [ "$rc2" -ne 0 ]; then
    log_phase f1_repilot "SECOND MISS (rc=$rc2) — proceeding with widest bracketing + limitation note"
    return 0
  fi
  log_phase f1_repilot "done (passed on shifted grid)"
}

# ── F2a: training fan-out (80 cells) ───────────────────────────────────────────
phase_f2a() {
  log_phase f2a_train "start (F1-effective fu1 grid; per-cell resume + per-cell HF upload)"
  headroom "$CKPT_ROOT" 60 f2a_train
  local cells_args=()
  if [ -n "$SMOKE" ]; then
    cells_args=(--cells "$SMOKE_CELLS")
  else
    # Plan §7: enumerate the F1-EFFECTIVE grid (shifted after a re-pilot; the
    # re-piloted adapters resume-skip via their manifests).
    cells_args=(--cells "$(fu1_effective_slugs all)")
  fi
  uv run python scripts/issue2225_fu1_train.py --fan-out \
    --ckpt-root "$CKPT_ROOT" --directions-dir "$DIR_OUT" \
    --dataset-root "$EXTERNAL_ROOT/dataset" \
    --log-dir "$LOG_ROOT/f2a_train" "${cells_args[@]}" "${TRAIN_SMOKE_ARGS[@]}"
  log_phase f2a_train "done"
}

# ── F2b: trait-eval generation + narrow-domain + upload ────────────────────────
phase_f2b() {
  log_phase f2b_evalgen "start (out=$OUT_ROOT)"
  headroom "$OUT_ROOT" 30 f2b_evalgen
  local targets
  if [ -n "$SMOKE" ]; then
    targets="$SMOKE_TARGETS"
  else
    targets="$(fu1_effective_slugs all)"  # plan §7: F1-effective grid
  fi
  uv run python scripts/issue2225_eval_gen.py \
    --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
    --targets "$targets" "${EVALGEN_SMOKE_ARGS[@]}"
  log_phase f2b_evalgen "narrow-domain generation (opinions arms)"
  uv run python scripts/issue2225_eval_gen.py --narrow-domain \
    --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
    --targets "$targets" "${EVALGEN_SMOKE_ARGS[@]}"
  log_phase f2b_evalgen "upload raw completions (fu1 prefixes)"
  uv run python scripts/issue2225_eval_gen.py --upload --out-root "$OUT_ROOT" \
    --hf-prefix-final "$HF_FINAL" --hf-prefix-narrow "$HF_NARROW"
  log_phase f2b_evalgen "done"
}

# ── F2c: MMLU (grid extremes + the §4.4 extension trigger) ─────────────────────
phase_f2c() {
  log_phase f2c_mmlu "start (grid extremes)"
  headroom "$OUT_ROOT" 10 f2c_mmlu
  local targets
  if [ -n "$SMOKE" ]; then
    targets="$SMOKE_TARGETS"
  else
    # 40 extremes: min + max EFFECTIVE-grid coefficient per (config x corpus)
    # arm (plan §7 — follows any octave shift). No `base`: the parent already
    # banked base MMLU.
    targets="$(fu1_effective_slugs extremes)"
  fi
  uv run python scripts/issue2225_mmlu.py \
    --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" \
    --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
    --targets "$targets" "${MMLU_LIMIT_ARGS[@]}"

  # §4.4 pre-registered extension: any extreme < (parent band min 0.703 − 0.02)
  # evaluates that (config x corpus) arm's FULL grid before pod termination.
  local extra
  extra=$(MMLU_DIR="$OUT_ROOT/mmlu" FU1_STATE="$FU1_REPILOT_STATE" uv run python - <<'PY'
import json, os, pathlib, sys

sys.path.insert(0, "scripts")
import issue2225_fu1_train as fu1

THRESH = 0.703 - 0.02
mmlu_dir = pathlib.Path(os.environ["MMLU_DIR"])
state = os.environ["FU1_STATE"]
flagged: set[tuple[str, str]] = set()
for p in sorted(mmlu_dir.glob("*.json")):
    tag = p.stem
    cell = fu1.resolve_fu1_cell(tag) if fu1._FU1_SLUG_RE.match(tag) else None
    if cell is None:
        continue
    with open(p, encoding="utf-8") as f:
        acc = json.load(f).get("mmlu_acc")
    if acc is not None and acc < THRESH:
        flagged.add((cell.config, cell.dataset))
# Full grid for flagged arms = the F1-EFFECTIVE grid (plan §7), never the
# original registry grid a re-pilot demonstrated mis-placed.
extra = [
    c.slug
    for c in fu1.effective_fu1_cells(state if os.path.exists(state) else None)
    if (c.config, c.dataset) in flagged and not (mmlu_dir / f"{c.slug}.json").exists()
]
print(",".join(extra))
PY
)
  if [ -n "$extra" ]; then
    log_phase f2c_mmlu "extension trigger fired — full grid for flagged arms: $extra"
    uv run python scripts/issue2225_mmlu.py \
      --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" \
      --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
      --targets "$extra" "${MMLU_LIMIT_ARGS[@]}"
  fi
  uv run python scripts/issue2225_mmlu.py --upload --out-root "$OUT_ROOT" --hf-prefix "$HF_MMLU"
  log_phase f2c_mmlu "done"
}

# ── F2d: activation capture ────────────────────────────────────────────────────
phase_f2d() {
  log_phase f2d_capture "start"
  headroom "$OUT_ROOT" 30 f2d_capture
  local targets
  if [ -n "$SMOKE" ]; then
    targets="$SMOKE_TARGETS"
  else
    targets="$(fu1_effective_slugs all)"  # plan §7: F1-effective grid
  fi
  uv run python scripts/issue2225_capture.py \
    --out-root "$OUT_ROOT" --gen-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" \
    --staging-dir "$OUT_ROOT/hf_dl/eval_adapters" \
    --targets "$targets"
  uv run python scripts/issue2225_capture.py --upload --out-root "$OUT_ROOT" \
    --hf-prefix "$HF_CAPTURE" "${CAPTURE_REPO_ARGS[@]}"
  log_phase f2d_capture "done"
}

# ── F3: upload safety pass + results sentinel ──────────────────────────────────
phase_f3() {
  log_phase f3_uploads "idempotent upload safety pass (fu1 prefixes)"
  uv run python scripts/issue2225_eval_gen.py --upload --out-root "$OUT_ROOT" \
    --hf-prefix-final "$HF_FINAL" --hf-prefix-narrow "$HF_NARROW"
  uv run python scripts/issue2225_mmlu.py --upload --out-root "$OUT_ROOT" --hf-prefix "$HF_MMLU"
  uv run python scripts/issue2225_capture.py --upload --out-root "$OUT_ROOT" \
    --hf-prefix "$HF_CAPTURE" "${CAPTURE_REPO_ARGS[@]}"
  uv run python scripts/issue2225_fu1_directions.py --upload --out-dir "$DIR_OUT" --hf-prefix "$HF_DIRECTIONS"

  # Incremental reap (#1489 last-consumer rule): the eval-adapter staging
  # mirror is consumed by F2b/F2c/F2d only — all complete by F3; the upload
  # safety pass above reads OUT_ROOT subtrees, never the staging mirror.
  rm -rf "$OUT_ROOT/hf_dl/eval_adapters"

  log_phase f3_uploads "results sentinel"
  local note_file="$LOG_ROOT/fu1_results_note.json"
  RES_NOTE_FILE="$note_file" RES_OUT_ROOT="$OUT_ROOT" RES_SMOKE="$SMOKE" \
    RES_FU1_STATE="$FU1_REPILOT_STATE" RES_CAPTURE_HF_REPO="$CAPTURE_HF_REPO" \
    RES_HF_CAPTURE="$HF_CAPTURE" uv run python - <<'PY'
import json, os, sys

sys.path.insert(0, "scripts")
import issue2225_fu1_train as fu1
import issue778_lib as lib

smoke = bool(os.environ.get("RES_SMOKE"))
# Reproducibility card lists the F1-EFFECTIVE cells (plan §7) — the cells the
# run actually trained/evaluated, shifted grid included.
_state = os.environ["RES_FU1_STATE"]
cells = fu1.effective_fu1_cells(_state if os.path.exists(_state) else None)
try:
    import wandb

    entity = wandb.Api().default_entity
except Exception:
    entity = None
note = {
    "phase": "f3_uploads",
    "followup": "fu1_preimage_prevention",
    "smoke": smoke,
    "out_root": os.environ["RES_OUT_ROOT"],
    "hf_prefixes": [
        "issue2225_ctxsteer/raw_completions/fu1_final",
        "issue2225_ctxsteer/raw_completions/fu1_narrow_domain",
        "issue2225_ctxsteer/raw_completions/fu1_pilot",
        "issue2225_ctxsteer/raw_completions/fu1_judge",
        "issue2225_ctxsteer/analysis_tensors/fu1_directions",
        "issue2225_ctxsteer/analysis_tensors/fu1_capture",
        "issue2225_ctxsteer/fu1_mmlu",
        f"{fu1.FU1_ADAPTERS_HF_PREFIX}/<cell>",
    ],
    "reproducibility_card": {
        "adapter_paths": [f"{fu1.FU1_ADAPTERS_HF_PREFIX}/{c.slug}" for c in cells],
        "wandb_project": "issue2225",
        "wandb_run_names": [f"issue2225_{c.slug}" for c in cells],
        "wandb_entity": entity,
        **lib.repro_metadata(),
    },
    "next": "off-pod: F4 judge (Batch API) + F5 analysis + figures",
}
# Round-5 fix (#1108 contract): when the capture class was routed to the
# overflow repo (canonical data repo at the 1M-file ceiling), the results
# sentinel records the deviation — consumers stage fu1_capture from THERE.
cap_repo = os.environ.get("RES_CAPTURE_HF_REPO") or None
if cap_repo:
    note["capture_hf_repo_deviation"] = {
        "reason": "canonical data repo at the 1M-file ceiling (#1108 overflow contract)",
        "hf_repo": cap_repo,
        "path_in_repo": os.environ["RES_HF_CAPTURE"],
        "consumers": "stage fu1 capture tensors from this repo (same prefix layout)",
    }
with open(os.environ["RES_NOTE_FILE"], "w") as f:
    json.dump(note, f, indent=1)
PY
  if [ -n "$SMOKE" ]; then
    write_sentinel "epm:smoke-result" 1 "$note_file"
  else
    write_sentinel "epm:results" 1 "$note_file"
  fi
  log_phase f3_uploads "done"
}

case "$PHASE" in
  external) phase_external ;;
  f0)       phase_external; phase_f0 ;;
  f1)       phase_external; phase_f1 ;;
  f2a)      phase_external; phase_f2a ;;
  f2b)      phase_external; phase_f2b ;;
  f2c)      phase_external; phase_f2c ;;
  f2d)      phase_external; phase_f2d ;;
  f3)       phase_f3 ;;
  all)
    phase_external
    phase_f0
    phase_f1
    phase_f2a
    phase_f2b
    phase_f2c
    phase_f2d
    phase_f3
    ;;
  *)
    echo "FATAL: unknown phase '$PHASE' (external|f0|f1|f2a|f2b|f2c|f2d|f3|all)" >&2
    exit 2
    ;;
esac

# Single terminal line — RESERVED (pod-side sentinel contract): only a fully
# successful invocation reaches here; every failure path exits above.
echo "[phase=done]"
