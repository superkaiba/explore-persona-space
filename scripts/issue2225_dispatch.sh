#!/usr/bin/env bash
# Issue #2225 pod-side phase orchestrator (plan §4.0 DAG).
#
#   bash scripts/issue2225_dispatch.sh [phase]
#
# Phases (default: all):
#   external  step 0: pinned persona_vectors clone + issue_778 sparse cone
#   p1        directions (E1 reuse + E2/E3 extraction) + upload
#   p0        §7 pilot gate: train --pilot -> hook grep -> eval-gen -> judge
#             --sync -> P0 verdict (designed halt rc=7 on gate FAIL, with the
#             octave-shift recommendation routed via the pilot sentinel)
#   p2a       train fan-out (81 cells; per-cell resume + per-cell HF upload)
#   p2b       trait-eval generation (86 targets) + narrow-domain + upload
#   p2c       MMLU capability eval + upload
#   p2d       activation capture (response_avg/context_end/prefix_end) + upload
#   p3        idempotent upload safety pass + results sentinel
#   all       the full DAG in order, ending with the single [phase=done] line
#
# Smoke: EPM_I2225_SMOKE=1 diverts every out-root to *_smoke twins, threads
# tiny-N dials into every phase (same dispatcher, same subprocess shape, same
# launch width — smoke never narrows the fan-out shape), and demotes the
# production-n-calibrated P0 verdict to informational (gate-calibration rule).
#
# Pod-side contract: NEVER shells to scripts/task.py; progress via [phase=...]
# lines + /workspace/logs sentinels (issue778_lib.write_results_sentinel).
set -euo pipefail

cd "$(dirname "$0")/.."

# Pod-side env: .env exists on RunPod clones; GCE/SLURM lanes export tokens
# via their startup env instead (conditional sourcing — pod-side-reporting.md).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

ISSUE=2225
PHASE="${1:-all}"
SMOKE="${EPM_I2225_SMOKE:-}"

PV_SHA="b8e0f044fe2410a6fad579f38324f03f13b4e917"
PV_REPO="https://github.com/safety-research/persona_vectors.git"
EXTERNAL_ROOT="external/persona_vectors"
I778_BASELINE="eval_results/issue_778/finetune_evil_evil_misaligned_2.json"

# Bidirectional smoke/full root pairs (gotchas.md smoke-var rule: every
# *_smoke rebinding sits behind an explicit $SMOKE guard).
OUT_ROOT_FULL="data/issue_2225/p2b_out"
OUT_ROOT_SMOKE="data/issue_2225/p2b_out_smoke"
PILOT_OUT_FULL="data/issue_2225/pilot_out"
PILOT_OUT_SMOKE="data/issue_2225/pilot_out_smoke"
EVAL_ROOT_FULL="eval_results/issue_2225"
EVAL_ROOT_SMOKE="data/issue_2225/eval_smoke"      # scratch — never the committed tree
DIR_OUT_FULL="eval_results/issue_2225/directions"
DIR_OUT_SMOKE="data/issue_2225/eval_smoke/directions"
OUT_ROOT="$OUT_ROOT_FULL"
PILOT_OUT="$PILOT_OUT_FULL"
EVAL_ROOT="$EVAL_ROOT_FULL"
DIR_OUT="$DIR_OUT_FULL"
if [ -n "$SMOKE" ]; then
  OUT_ROOT="$OUT_ROOT_SMOKE"
  PILOT_OUT="$PILOT_OUT_SMOKE"
  EVAL_ROOT="$EVAL_ROOT_SMOKE"
  DIR_OUT="$DIR_OUT_SMOKE"
fi

CKPT_ROOT="checkpoints/issue_2225"
LOG_ROOT="${EPM_I2225_LOG_ROOT:-/workspace/logs/issue-2225}"
mkdir -p "$LOG_ROOT"

# Smoke dials (threaded, never a different code path). Smoke keeps the SAME
# dispatcher/fan-out/launch width; only N shrinks. Two arm classes covered
# (hook-bearing A cell + prompt-mode H — the per-arm-class smoke rule).
TRAIN_SMOKE_ARGS=()
EVALGEN_SMOKE_ARGS=()
SMOKE_CELLS="A__evil__c3.0,H__evil"
SMOKE_TARGETS="base,A__evil__c3.0,H__evil"
if [ -n "$SMOKE" ]; then
  TRAIN_SMOKE_ARGS=(--max-steps 2)
  EVALGEN_SMOKE_ARGS=(--n-questions 2 --n-rollouts 1)
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
  SENT_KIND="$1" SENT_VERSION="$2" SENT_NOTE_FILE="$3" SENT_LOGS_DIR="$LOG_ROOT" \
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

# ── step 0: pinned external clone + issue_778 baseline cone ───────────────────
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
  # is OUTSIDE the default cones (gotchas.md partial-clone entry), and the P0
  # verdict reads the committed #778 baseline JSON. Guarded: non-sparse trees
  # (VM worktree, full clones) already have the file.
  if [ ! -f "$I778_BASELINE" ] && git sparse-checkout list >/dev/null 2>&1; then
    log_phase external "sparse-checkout add eval_results/issue_778"
    git sparse-checkout add eval_results/issue_778
  fi
  test -f "$I778_BASELINE" || { echo "FATAL: $I778_BASELINE missing (P0 verdict input)" >&2; exit 1; }

  # Resolve the venv ONCE, then freeze it for every multi-worker fan-out (the
  # #1689 MooseFS FUSE-wedge prevention: N concurrent `uv run` resolutions on
  # a stop/resumed pod can wedge all /workspace reads).
  uv run python -c "import explore_persona_space" >/dev/null
  export UV_NO_SYNC=1
  log_phase external "done (sha=$GOT_SHA)"
}

# ── P1: directions ─────────────────────────────────────────────────────────────
phase_p1() {
  log_phase p1_directions "start (out=$DIR_OUT)"
  headroom "data/issue_2225" 10 p1_directions
  DIR_SMOKE_ARGS=()
  if [ -n "$SMOKE" ]; then DIR_SMOKE_ARGS=(--smoke); fi
  uv run python -m explore_persona_space.experiments.issue2225.directions \
    --out-dir "$DIR_OUT" --external-root "$EXTERNAL_ROOT" "${DIR_SMOKE_ARGS[@]}"
  uv run python -m explore_persona_space.experiments.issue2225.directions \
    --out-dir "$DIR_OUT" --upload
  # Incremental reap (#1489 last-consumer rule): data/issue_2225/hf_dl/issue778_v2
  # (the P1 staging mirror: rb_v2 tensors + pairing + judge_raw) is consumed by
  # NO later pod-side phase — P2a fingerprints direction files under $DIR_OUT,
  # P2b/P2c/P2d stage adapters under hf_dl/eval_adapters, P0/P4 judging reads
  # rollout JSONs + external/ only; P5 analysis runs OFF-pod with its own
  # staging. Re-runs of P1 re-stage idempotently (stage_hub_file).
  rm -rf data/issue_2225/hf_dl/issue778_v2
  log_phase p1_directions "done"
}

# ── P0: §7 pilot gate ──────────────────────────────────────────────────────────
phase_p0() {
  local verdict="$EVAL_ROOT/pilot_gate/p0_verdict.json"
  if [ -f "$verdict" ] && [ -z "$SMOKE" ]; then
    local prior_pass
    prior_pass=$(uv run python -c "import json;print(json.load(open('$verdict'))['passed'])")
    if [ "$prior_pass" = "True" ]; then
      log_phase p0_pilot "prior PASSed verdict present — resume skip"
      return 0
    fi
  fi
  log_phase p0_pilot "train --pilot (8 cells: A+C x L1 grid, evil II)"
  headroom "$CKPT_ROOT" 30 p0_pilot
  local p0_logs="$LOG_ROOT/p0_train"
  uv run python scripts/issue2225_train.py --pilot --fan-out \
    --ckpt-root "$CKPT_ROOT" --directions-dir "$DIR_OUT" \
    --log-dir "$p0_logs" "${TRAIN_SMOKE_ARGS[@]}"

  # §7 criterion (i): every pilot cell's training log shows hook engagement.
  local n_expect n_hook
  n_expect=$(uv run python -c "
import sys; sys.path.insert(0, 'scripts')
import issue2225_train as t
print(len(t.pilot_cells()))
")
  n_hook=$(grep -rlF "[steer-hook]" "$p0_logs" 2>/dev/null | wc -l)
  log_phase p0_pilot "hook-engagement logs: $n_hook/$n_expect"
  if [ "$n_hook" -ne "$n_expect" ]; then
    echo "FATAL: §7 criterion (i) FAILED — $n_hook/$n_expect pilot logs carry [steer-hook]" >&2
    exit 7
  fi

  log_phase p0_pilot "eval-gen (pilot targets, n_rollouts=5)"
  local pilot_targets
  pilot_targets=$(uv run python -c "
import sys; sys.path.insert(0, 'scripts')
import issue2225_train as t
print(','.join(c.slug for c in t.pilot_cells()))
")
  headroom "$PILOT_OUT" 10 p0_pilot
  uv run python scripts/issue2225_eval_gen.py \
    --out-root "$PILOT_OUT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    --targets "$pilot_targets" --n-rollouts 5 "${EVALGEN_SMOKE_ARGS[@]}"

  # Persist pilot rollout text BEFORE judging (upload policy: raw completions
  # for ALL stages). Distinct HF prefix — the script's own --upload targets
  # the production final/ prefix and would mix stages.
  PILOT_LOCAL="$PILOT_OUT/raw_completions/final" uv run python - <<'PY'
import os

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import _upload

url = _upload(
    os.environ["PILOT_LOCAL"],
    "superkaiba1/explore-persona-space-data",
    "dataset",
    "issue2225_ctxsteer/raw_completions/pilot",
)
if not url:
    raise SystemExit("pilot raw-completions upload returned no path")
print(f"[p0-upload] {url}", flush=True)
PY

  log_phase p0_pilot "judge --sync --stage pilot (~5.6k calls)"
  uv run python scripts/issue2225_judge.py --phase all --sync --stage pilot \
    --eval-root "$EVAL_ROOT" --external-root "$EXTERNAL_ROOT" \
    --rollouts-dir "$PILOT_OUT/raw_completions/final" \
    --narrow-rollouts-dir "$PILOT_OUT/raw_completions/narrow_domain"
  uv run python scripts/issue2225_judge.py --phase upload --stage pilot \
    --eval-root "$EVAL_ROOT" --external-root "$EXTERNAL_ROOT" \
    --rollouts-dir "$PILOT_OUT/raw_completions/final" \
    --narrow-rollouts-dir "$PILOT_OUT/raw_completions/narrow_domain"

  log_phase p0_pilot "P0 verdict (§7 criteria ii + iii)"
  local rc=0
  uv run python scripts/issue2225_judge.py --phase p0-verdict \
    --eval-root "$EVAL_ROOT" --i778-baseline "$I778_BASELINE" || rc=$?
  if [ ! -f "$verdict" ]; then
    echo "FATAL: p0-verdict crashed (rc=$rc) with no verdict artifact" >&2
    exit "$rc"
  fi

  # Pilot sentinel: the P0 verdict (incl. any octave-shift recommendation)
  # routed to the VM poller as a progress marker — the orchestrator owns the
  # re-pilot decision (issue2225_train.py has no coefficient-scale pathway;
  # a shifted grid needs an orchestrator-driven re-pilot round).
  local note_file="$LOG_ROOT/p0_note.json"
  P0_VERDICT_FILE="$verdict" P0_NOTE_FILE="$note_file" P0_SMOKE="$SMOKE" \
    uv run python - <<'PY'
import json, os

verdict = json.load(open(os.environ["P0_VERDICT_FILE"]))
note = {
    "phase": "p0_pilot",
    "passed": verdict["passed"],
    "octave_shift": verdict["octave_shift"],
    "criteria": verdict["criteria"],
    "smoke": bool(os.environ.get("P0_SMOKE")),
    "next": (
        "proceed to P2a"
        if verdict["passed"]
        else "DESIGNED HALT (rc=7): octave-shift re-pilot needed — retrain the "
        "shifted A/C x evil grid (orchestrator-driven; no coef-scale CLI exists)"
    ),
}
with open(os.environ["P0_NOTE_FILE"], "w") as f:
    json.dump(note, f, indent=1)
PY
  if [ -n "$SMOKE" ]; then
    write_sentinel "epm:smoke-result" 1 "$note_file"
  else
    write_sentinel "epm:progress" 1 "$note_file"
  fi

  if [ "$rc" -ne 0 ]; then
    if [ -n "$SMOKE" ]; then
      # Gate-calibration rule: coherence>=80 at smoke N measures N, not the
      # grid — the verdict is informational under smoke.
      log_phase p0_pilot "verdict FAILED (rc=$rc) — informational under smoke, continuing"
    else
      log_phase p0_pilot "DESIGNED HALT rc=$rc — see $verdict (octave-shift recommendation)"
      exit "$rc"
    fi
  fi
  log_phase p0_pilot "done (passed)"
}

# ── P2a: training fan-out ──────────────────────────────────────────────────────
phase_p2a() {
  log_phase p2a_train "start (81 cells; per-cell resume + per-cell HF upload)"
  headroom "$CKPT_ROOT" 60 p2a_train
  local cells_args=()
  if [ -n "$SMOKE" ]; then cells_args=(--cells "$SMOKE_CELLS"); fi
  uv run python scripts/issue2225_train.py --fan-out \
    --ckpt-root "$CKPT_ROOT" --directions-dir "$DIR_OUT" \
    --log-dir "$LOG_ROOT/p2a_train" "${cells_args[@]}" "${TRAIN_SMOKE_ARGS[@]}"
  log_phase p2a_train "done"
}

# ── P2b: trait-eval generation + narrow-domain + upload ───────────────────────
phase_p2b() {
  log_phase p2b_evalgen "start (out=$OUT_ROOT)"
  headroom "$OUT_ROOT" 30 p2b_evalgen
  local target_args=()
  if [ -n "$SMOKE" ]; then target_args=(--targets "$SMOKE_TARGETS"); fi
  uv run python scripts/issue2225_eval_gen.py \
    --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    "${target_args[@]}" "${EVALGEN_SMOKE_ARGS[@]}"
  log_phase p2b_evalgen "narrow-domain generation (opinions arms)"
  uv run python scripts/issue2225_eval_gen.py --narrow-domain \
    --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" --external-root "$EXTERNAL_ROOT" \
    "${target_args[@]}" "${EVALGEN_SMOKE_ARGS[@]}"
  log_phase p2b_evalgen "upload raw completions"
  uv run python scripts/issue2225_eval_gen.py --upload --out-root "$OUT_ROOT"
  log_phase p2b_evalgen "done"
}

# ── P2c: MMLU ──────────────────────────────────────────────────────────────────
phase_p2c() {
  log_phase p2c_mmlu "start"
  headroom "$OUT_ROOT" 10 p2c_mmlu
  local target_args=()
  if [ -n "$SMOKE" ]; then target_args=(--targets "$SMOKE_TARGETS"); fi
  uv run python scripts/issue2225_mmlu.py \
    --out-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" "${target_args[@]}"
  uv run python scripts/issue2225_mmlu.py --upload --out-root "$OUT_ROOT"
  log_phase p2c_mmlu "done"
}

# ── P2d: activation capture ────────────────────────────────────────────────────
phase_p2d() {
  log_phase p2d_capture "start"
  headroom "$OUT_ROOT" 30 p2d_capture
  local target_args=()
  if [ -n "$SMOKE" ]; then target_args=(--targets "$SMOKE_TARGETS"); fi
  uv run python scripts/issue2225_capture.py \
    --out-root "$OUT_ROOT" --gen-root "$OUT_ROOT" --ckpt-root "$CKPT_ROOT" \
    "${target_args[@]}"
  uv run python scripts/issue2225_capture.py --upload --out-root "$OUT_ROOT"
  log_phase p2d_capture "done"
}

# ── P3: upload safety pass + results sentinel ──────────────────────────────────
phase_p3() {
  log_phase p3_uploads "idempotent upload safety pass"
  uv run python scripts/issue2225_eval_gen.py --upload --out-root "$OUT_ROOT"
  uv run python scripts/issue2225_mmlu.py --upload --out-root "$OUT_ROOT"
  uv run python scripts/issue2225_capture.py --upload --out-root "$OUT_ROOT"
  uv run python -m explore_persona_space.experiments.issue2225.directions \
    --out-dir "$DIR_OUT" --upload

  # Incremental reap (#1489 last-consumer rule): hf_dl/eval_adapters (the HF
  # adapter staging mirror for eval fan-outs) is consumed by P2b/P2c/P2d only —
  # all complete by P3; the upload safety pass above reads OUT_ROOT + DIR_OUT,
  # never the staging mirror. Fresh re-runs re-stage idempotently from HF.
  rm -rf data/issue_2225/hf_dl/eval_adapters

  log_phase p3_uploads "results sentinel"
  local note_file="$LOG_ROOT/results_note.json"
  RES_NOTE_FILE="$note_file" RES_OUT_ROOT="$OUT_ROOT" RES_SMOKE="$SMOKE" \
    RES_CKPT_ROOT="$CKPT_ROOT" uv run python - <<'PY'
import json, os, sys

sys.path.insert(0, "scripts")
import issue2225_train as t
import issue778_lib as lib

smoke = bool(os.environ.get("RES_SMOKE"))
cells = t.build_cell_registry()
adapter_prefix = "issue2225_ctxsteer/adapters"
try:
    import wandb

    entity = wandb.Api().default_entity
except Exception:
    entity = None
note = {
    "phase": "p3_uploads",
    "smoke": smoke,
    "out_root": os.environ["RES_OUT_ROOT"],
    "hf_prefixes": [
        "issue2225_ctxsteer/raw_completions/final",
        "issue2225_ctxsteer/raw_completions/narrow_domain",
        "issue2225_ctxsteer/raw_completions/pilot",
        "issue2225_ctxsteer/raw_completions/judge",
        "issue2225_ctxsteer/analysis_tensors/directions",
        "issue2225_ctxsteer/analysis_tensors/capture",
        "issue2225_ctxsteer/mmlu",
        f"{adapter_prefix}/<cell>",
    ],
    "reproducibility_card": {
        "adapter_paths": [f"{adapter_prefix}/{c.slug}" for c in cells],
        "wandb_project": "issue2225",
        "wandb_run_names": [f"issue2225_{c.slug}" for c in cells],
        "wandb_entity": entity,
        **lib.repro_metadata(),
    },
    "next": "off-pod: P4 judge (Batch API) + P5 analysis + unit-5 figures",
}
with open(os.environ["RES_NOTE_FILE"], "w") as f:
    json.dump(note, f, indent=1)
PY
  if [ -n "$SMOKE" ]; then
    write_sentinel "epm:smoke-result" 1 "$note_file"
  else
    write_sentinel "epm:results" 1 "$note_file"
  fi
  log_phase p3_uploads "done"
}

case "$PHASE" in
  external) phase_external ;;
  p1)       phase_external; phase_p1 ;;
  p0)       phase_external; phase_p0 ;;
  p2a)      phase_external; phase_p2a ;;
  p2b)      phase_external; phase_p2b ;;
  p2c)      phase_external; phase_p2c ;;
  p2d)      phase_external; phase_p2d ;;
  p3)       phase_p3 ;;
  all)
    phase_external
    phase_p1
    phase_p0
    phase_p2a
    phase_p2b
    phase_p2c
    phase_p2d
    phase_p3
    ;;
  *)
    echo "FATAL: unknown phase '$PHASE' (external|p1|p0|p2a|p2b|p2c|p2d|p3|all)" >&2
    exit 2
    ;;
esac

# Single terminal line — RESERVED (pod-side sentinel contract): only a fully
# successful invocation reaches here; every failure path exits above.
echo "[phase=done]"
