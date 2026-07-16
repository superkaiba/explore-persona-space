#!/usr/bin/env bash
# Issue #1335 dispatcher — ablation ladder (assistant-vs-fiction context->answer gap).
#
# GCE-lane workload command (plan §9):
#   REPO_ROOT=$WORKLOAD_ROOT NGPUS=2 bash scripts/issue1335_run.sh
#
# Phase shape (plan §4.2), signalled via stdout [phase=...] breadcrumbs + the
# /workspace/logs sentinel channel (pod-side-reporting contract; instance-side
# code never invokes scripts/task.py):
#   p0_stage   code/data staging: main-untouched re-verify for the staged
#              issue825_fit_cells.py, render configs, Track-S questions
#   p0_gates   vectorized-fit equivalence gate + the binding _fit_device()=cuda
#              gate (plan §9 P3) — both on the DISPATCHED fit825 functions
#   smoke      SAME entrypoints at 2 scenarios / 50 questions, both models,
#              all rungs, scratch dirs (never the committed output paths)
#   p1..p3     per-model lanes (base on GPU0 || instruct on GPU1, CVD-pinned in
#              the LAUNCHER env): gen -> tf re-renders -> per rung capture ->
#              full-n fits -> store shard upload -> delete local (per-cell
#              lifecycle); then matched-n (sharded by model) + ladder summary
#   p4         figures, results commit+push (verified, #1205), final sentinel,
#              [phase=done]
#
# Env knobs: SMOKE=1 (run ONLY the smoke pipeline), SKIP_SMOKE=1, STUB=1
# (CPU/VM smoke: --stub-gen + tiny model, no uploads, no cuda gate),
# NGPUS, DATA_DIR/OUT_DIR/FIG_DIR/LOG_DIR, SKIP_UPLOAD=1, SKIP_PUSH=1.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
# Conditional .env sourcing (GCE exports tokens via startup metadata; no .env there).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

ISSUE=1335
BRANCH="issue-1335"
DATA_DIR="${DATA_DIR:-data/issue_1335}"
OUT_DIR="${OUT_DIR:-eval_results/issue_1335}"
FIG_DIR="${FIG_DIR:-figures/issue_1335}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
STUB="${STUB:-0}"
SMOKE="${SMOKE:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"
TINY_MODEL_DIR="${TINY_MODEL_DIR:-/tmp/issue-1335-tiny-model}"
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
  DETECTED_GPUS=0
fi
NGPUS="${NGPUS:-$DETECTED_GPUS}"
# Never assume more lanes than physically visible (dispatcher wave rule).
if [ "$NGPUS" -gt "$DETECTED_GPUS" ] && [ "$DETECTED_GPUS" -gt 0 ]; then NGPUS="$DETECTED_GPUS"; fi
if [ "$STUB" = "1" ]; then
  SKIP_UPLOAD=1
  REQUIRE_CUDA_FITS=0
else
  REQUIRE_CUDA_FITS="${REQUIRE_CUDA_FITS:-1}"
fi
mkdir -p "$LOG_DIR" "$DATA_DIR"

GEN_RUNGS=(r0_qa_full r1_qa_oneline r2_op r3_persona r4_fictionframe r6_nofoil r7_endpoint)
TF_RUNGS=(r2_tf s1_assistant_label s2a_familiar s2b_novel)
ALL_RUNGS=(r0_qa_full r1_qa_oneline r2_tf r2_op r3_persona r4_fictionframe r6_nofoil r7_endpoint s1_assistant_label s2a_familiar s2b_novel)
MODELS=(base instruct)

log() { echo "[i1335-run] $*"; }

write_sentinel() {  # write_sentinel <kind> <gate> <note_json_path>
  local kind="$1" gate="$2" note_path="$3"
  local slug epoch dest
  slug=$(echo "$kind" | tr ':' '_')
  epoch=$(date +%s)
  if [ "$gate" = "smoke" ]; then
    dest="$LOG_DIR/issue-${ISSUE}-smoke-results.json"
  else
    dest="$LOG_DIR/issue-${ISSUE}-${slug}-${epoch}.json"
  fi
  uv run python - "$kind" "$gate" "$note_path" "$dest" <<'PY'
import json
import sys

kind, gate, note_path, dest = sys.argv[1:5]
summary = json.load(open(note_path, encoding="utf-8"))
digest = {
    "issue": 1335,
    "code_sha": summary.get("code_sha"),
    "smoke": summary.get("smoke"),
    "gates": summary.get("gates"),
    "verdicts": {m: v.get("verdict") for m, v in summary.get("per_model", {}).items()},
    "rung_values_matched_ctx": {
        m: v.get("rung_values_matched_ctx") for m, v in summary.get("per_model", {}).items()
    },
    "artifacts": {
        "eval_results": "eval_results/issue_1335/ (git, issue-1335 branch)",
        "figures": "figures/issue_1335/ (git, issue-1335 branch)",
        "raw_completions": "hf:superkaiba1/explore-persona-space-data/issue1335_ablation_ladder/raw_completions/",
        "stores": "hf:superkaiba1/explore-persona-space-data/issue1335_ablation_ladder/analysis_tensors/",
    },
    "wandb_url": "n/a (no training; activation-geometry fits only)",
}
note = json.dumps(digest)
if len(note) > 45000:
    digest.pop("rung_values_matched_ctx", None)
    note = json.dumps(digest)
payload = {
    "sentinel_schema_version": 1,
    "kind": kind,
    "version": 1,
    "task_id": 1335,
    "gate": gate,
    "blocks_pipeline": False,
    "by": "issue1335_run.sh",
    "note": note,
}
tmp = dest + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f)
import os

os.replace(tmp, dest)
print(f"[i1335-run] sentinel written: {dest}")
PY
}

upload_store() {  # upload_store <data_dir> <rung> <model> <mode>
  if [ "$SKIP_UPLOAD" = "1" ] || [ "${4:-full}" = "smoke" ]; then
    log "upload_store $2/$3 skipped (SKIP_UPLOAD/smoke)"
    return 0
  fi
  uv run python - "$1" "$2" "$3" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

data_dir, rung, model = sys.argv[1:4]
store = Path(data_dir) / "store" / rung / model
repo = "superkaiba1/explore-persona-space-data"
prefix = f"issue1335_ablation_ladder/analysis_tensors/store_{rung}_{model}"
res = upload_dir_sharded(store, repo, prefix, shard_glob="*.pt", verify=True, delete_local=True)
print(
    f"[i1335-upload] {rung}/{model} shards: uploaded={len(res.uploaded)} "
    f"rerouted={len(res.rerouted)} deleted={len(res.deleted)}"
)
res2 = upload_dir_sharded(store, repo, prefix, shard_glob="*.json", verify=True, delete_local=False)
print(f"[i1335-upload] {rung}/{model} sidecars: uploaded={len(res2.uploaded)}")
PY
}

upload_tf_rollouts() {  # upload the tf re-render JSONLs (pipeline inputs; text path)
  if [ "$SKIP_UPLOAD" = "1" ]; then return 0; fi
  uv run python - "$DATA_DIR" <<'PY'
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate import hub

data_dir = Path(sys.argv[1])
for slug in ("r2_tf", "s1_assistant_label", "s2a_familiar", "s2b_novel"):
    for p in sorted((data_dir / "generation" / slug).glob("*_gen.jsonl")):
        url = hub._upload(
            p,
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo=f"issue1335_ablation_ladder/raw_completions/tf_rerender/{slug}_{p.name}",
            upload_as_file=True,
        )
        assert url, f"tf rollout upload returned no URL for {p}"
        print(f"[i1335-upload] {p} -> {url}")
PY
}

run_model_lane() {  # run_model_lane <model> <data_dir> <out_dir> <mode>
  local model="$1" data_dir="$2" out_dir="$3" mode="$4"
  local slice_args=() stub_args=() upload_args=() fit_args=() first_capture=1
  if [ "$mode" = "smoke" ]; then
    slice_args=(--n-questions 50 --n-scenarios 2)
    fit_args=(--smoke)
  fi
  if [ "$STUB" = "1" ]; then
    stub_args=(--stub-gen)
  fi
  if [ "$SKIP_UPLOAD" = "1" ] || [ "$mode" = "smoke" ]; then
    upload_args=(--skip-upload)
  fi
  local tiny_args=()
  if [ "$STUB" = "1" ]; then
    tiny_args=(--tiny-model-dir "$TINY_MODEL_DIR")
  fi
  local cuda_fit_args=()
  if [ "$REQUIRE_CUDA_FITS" = "1" ]; then
    cuda_fit_args=(--assert-cuda)
  fi
  # r5 relaunch economy: seed the c24 resume store from the issue's Hub prefix
  # (prior-attempt shards; CONSUME rule). Full lanes only — smoke/STUB lanes
  # stay hermetic (no network), mirroring the upload_store gate.
  local seed_args=()
  local gen_resume_args=()
  if [ "$SKIP_UPLOAD" != "1" ] && [ "$mode" != "smoke" ]; then
    seed_args=(--hf-resume-seed)
    # r8 relaunch economy: consume the persisted rollout JSONL (the last
    # attempt's text — the SAME lineage the seeded stores were captured from)
    # instead of minting a fresh statistically-equivalent-but-not-byte-
    # identical vLLM re-gen (fingerprint-gated; falls through to fresh gen).
    gen_resume_args=(--hf-resume)
  fi
  local wiring_n=200
  if [ "$mode" = "smoke" ]; then wiring_n=8; fi

  for rung in "${GEN_RUNGS[@]}"; do
    uv run python scripts/issue1335_gen.py --rung "$rung" --model "$model" \
      --data-dir "$data_dir" "${slice_args[@]}" "${stub_args[@]}" "${upload_args[@]}" \
      "${gen_resume_args[@]}"
  done
  for rung in "${TF_RUNGS[@]}"; do
    uv run python scripts/issue1335_render_rungs.py --tf-rerender --rung "$rung" \
      --model "$model" --data-dir "$data_dir"
  done
  for rung in "${ALL_RUNGS[@]}"; do
    local extra=()
    if [ "$first_capture" = "1" ]; then extra+=(--equivalence-check); fi
    if [ "$rung" = "r0_qa_full" ] || [ "$rung" = "r7_endpoint" ]; then
      extra+=(--wiring-check "$wiring_n")
    fi
    uv run python scripts/issue1335_extract_store.py --rung "$rung" --model "$model" \
      --data-dir "$data_dir" --out-dir "$out_dir" --resume "${seed_args[@]}" \
      "${tiny_args[@]}" "${extra[@]}"
    first_capture=0
    uv run python scripts/issue1335_fit.py --rung "$rung" --model "$model" \
      --data-dir "$data_dir" --out-dir "$out_dir" --resume "${fit_args[@]}" "${cuda_fit_args[@]}"
    upload_store "$data_dir" "$rung" "$model" "$mode"
  done
  echo "lane complete: model=$model mode=$mode"
}

wait_lanes() {  # wait_lanes <pid_base> <pid_instruct> <log_base> <log_instruct>
  local pids=("$1" "$2") logs=("$3" "$4") rc=0 lane_rc
  while :; do
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then alive=1; fi
    done
    if [ "$alive" = "0" ]; then break; fi
    sleep 120
    for lg in "${logs[@]}"; do
      log "lane heartbeat: $(basename "$lg"): $(tail -n 1 "$lg" 2>/dev/null | cut -c1-160)"
    done
  done
  for i in 0 1; do
    lane_rc=0
    wait "${pids[$i]}" || lane_rc=$?
    if [ "$lane_rc" -ne 0 ]; then
      log "LANE FAILED rc=$lane_rc — tail of ${logs[$i]}:"
      tail -n 40 "${logs[$i]}" || true
      rc=$lane_rc
    fi
  done
  return "$rc"
}

run_pipeline() {  # run_pipeline <mode: smoke|full>
  local mode="$1" data_dir out_dir fig_dir
  if [ "$mode" = "smoke" ]; then
    data_dir="/tmp/issue-1335-smoke/data"
    out_dir="/tmp/issue-1335-smoke/eval_results"
    fig_dir="/tmp/issue-1335-smoke/figures"
    mkdir -p "$data_dir" "$out_dir" "$fig_dir"
    # The smoke consumes the SAME staged questions (copy, never re-fetch).
    if [ -f "$DATA_DIR/track_s.jsonl" ]; then cp "$DATA_DIR/track_s.jsonl" "$data_dir/"; fi
  else
    data_dir="$DATA_DIR"
    out_dir="$OUT_DIR"
    fig_dir="$FIG_DIR"
    mkdir -p "$data_dir" "$out_dir" "$fig_dir"
  fi

  echo "[phase=p1_p3_lanes_${mode}]"
  if [ "$NGPUS" -ge 2 ]; then
    local lb="$LOG_DIR/issue-${ISSUE}-${mode}-base.log" li="$LOG_DIR/issue-${ISSUE}-${mode}-instruct.log"
    ( export CUDA_VISIBLE_DEVICES=0; run_model_lane base "$data_dir" "$out_dir" "$mode" ) >"$lb" 2>&1 &
    local pid_b=$!
    ( export CUDA_VISIBLE_DEVICES=1; run_model_lane instruct "$data_dir" "$out_dir" "$mode" ) >"$li" 2>&1 &
    local pid_i=$!
    wait_lanes "$pid_b" "$pid_i" "$lb" "$li"
  else
    for model in "${MODELS[@]}"; do
      run_model_lane "$model" "$data_dir" "$out_dir" "$mode"
    done
  fi

  echo "[phase=p3_matched_${mode}]"
  local fit_common=(--data-dir "$data_dir" --out-dir "$out_dir" --resume)
  local stage_args=()
  if [ "$SKIP_UPLOAD" != "1" ] && [ "$mode" != "smoke" ]; then stage_args=(--stage-from-hub); fi
  local smoke_args=()
  if [ "$mode" = "smoke" ]; then smoke_args=(--smoke); fi
  if [ "$NGPUS" -ge 2 ]; then
    local mb="$LOG_DIR/issue-${ISSUE}-${mode}-matched-base.log"
    local mi="$LOG_DIR/issue-${ISSUE}-${mode}-matched-instruct.log"
    ( export CUDA_VISIBLE_DEVICES=0; uv run python scripts/issue1335_fit.py --matched-n \
        --models base "${fit_common[@]}" "${stage_args[@]}" "${smoke_args[@]}" ) >"$mb" 2>&1 &
    local pid_b=$!
    ( export CUDA_VISIBLE_DEVICES=1; uv run python scripts/issue1335_fit.py --matched-n \
        --models instruct "${fit_common[@]}" "${stage_args[@]}" "${smoke_args[@]}" ) >"$mi" 2>&1 &
    local pid_i=$!
    wait_lanes "$pid_b" "$pid_i" "$mb" "$mi"
  else
    uv run python scripts/issue1335_fit.py --matched-n --models base,instruct \
      "${fit_common[@]}" "${stage_args[@]}" "${smoke_args[@]}"
  fi
  # NOTE: matched-n --models base and --models instruct each compute n_min from
  # the SAME all-model sidecar sweep (compute_n_min reads both models' sidecars
  # from the shared data_dir), so the sharded lanes use one shared n_min.

  echo "[phase=p3_summary_${mode}]"
  uv run python scripts/issue1335_fit.py --summary --models base,instruct \
    --data-dir "$data_dir" --out-dir "$out_dir" "${smoke_args[@]}"

  echo "[phase=p4_figures_${mode}]"
  uv run python scripts/issue1335_figures.py --out-dir "$out_dir" --fig-dir "$fig_dir"

  if [ "$mode" = "smoke" ]; then
    write_sentinel "epm:smoke-result" "smoke" "$out_dir/ladder_summary.json"
  fi
}

commit_and_push_results() {
  if [ "$SKIP_PUSH" = "1" ] || [ "$STUB" = "1" ]; then
    log "results push skipped (SKIP_PUSH/STUB)"
    return 0
  fi
  git add "$OUT_DIR" "$FIG_DIR"
  if git diff --cached --quiet; then
    log "no result changes to commit"
    return 0
  fi
  git commit -m "issue-1335: ladder eval results + figures (run $(date -u +%Y-%m-%dT%H:%MZ))"
  local ok=0
  for attempt in 1 2; do
    if git push origin "HEAD:$BRANCH"; then
      if [ "$(git rev-list --count "origin/$BRANCH..HEAD")" = "0" ]; then ok=1; break; fi
    fi
    log "push attempt $attempt failed; retrying after fetch"
    git fetch origin "$BRANCH" || true
  done
  if [ "$ok" != "1" ]; then
    log "FATAL: results push did not land (rev-list non-zero) — failing loud (#1205)"
    return 86
  fi
  log "results push verified (rev-list count 0)"
}

main() {
  echo "[phase=p0_stage]"
  log "issue=$ISSUE mode: STUB=$STUB SMOKE=$SMOKE NGPUS=$NGPUS repo=$REPO_ROOT"
  # Plan §4.2 staging safety: main must not have touched issue825_fit_cells.py
  # since the last ported-through main commit (else PORT the diff, not checkout).
  # Pin history: af402fdf (issue-1310 merge-base) -> ed73b13029 (#1320 vectorized
  # MLP secondary + turnstore upload; 3-way-merged into the branch copy in the
  # r4 crash-fix round — 1335 calls none of the #1320-touched entrypoints).
  local port_pin=ed73b13029
  git fetch origin main --quiet || log "WARN: git fetch origin main failed (offline?)"
  if git rev-parse --verify --quiet origin/main >/dev/null; then
    local touched
    touched=$(git log --oneline "$port_pin"..origin/main -- scripts/issue825_fit_cells.py | wc -l)
    if [ "$touched" != "0" ]; then
      log "FATAL: main touched scripts/issue825_fit_cells.py since port pin $port_pin" \
          "($touched commits) — port the issue-1310/1335 diff onto main's version (plan §4.2)"
      exit 4
    fi
  fi
  if [ "$STUB" = "1" ] && [ ! -d "$TINY_MODEL_DIR" ]; then
    uv run python scripts/issue1335_extract_store.py --make-tiny-model "$TINY_MODEL_DIR"
  fi
  uv run python scripts/issue1335_render_rungs.py --write-configs --data-dir "$DATA_DIR"
  uv run python scripts/issue1335_render_rungs.py --fetch-questions --data-dir "$DATA_DIR"

  echo "[phase=p0_gates]"
  local gate_args=(--verify-vectorized)
  if [ "$REQUIRE_CUDA_FITS" = "1" ]; then gate_args+=(--assert-cuda); fi
  uv run python scripts/issue1335_fit.py "${gate_args[@]}"

  if [ "$SKIP_SMOKE" != "1" ]; then
    echo "[phase=smoke]"
    run_pipeline smoke
    log "smoke pipeline complete"
  fi
  if [ "$SMOKE" = "1" ]; then
    log "SMOKE=1: stopping after the smoke pipeline"
    echo "[phase=done]"
    return 0
  fi

  run_pipeline full

  echo "[phase=p4_finalize]"
  upload_tf_rollouts
  commit_and_push_results
  if [ -n "${EPS_DELIVERABLES_OK_PATH:-}" ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$EPS_DELIVERABLES_OK_PATH"
    log "deliverables-ok stamped at $EPS_DELIVERABLES_OK_PATH (post push-verify)"
  fi
  write_sentinel "epm:results" "results" "$OUT_DIR/ladder_summary.json"
  echo "[phase=done]"
}

main "$@"
