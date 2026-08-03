#!/usr/bin/env bash
# #1774 pod-side dispatcher (plan §10).
#
#   bash scripts/issue1774_dispatch.sh --phase p1|p2|p3 [--smoke]
#   bash scripts/issue1774_dispatch.sh --all [--smoke]
#
# Contract (pod-side-reporting rules):
# - per-phase sentinels  $SENTINEL_DIR/issue-1774-p{1,2,3}-done.json  (valid
#   poll_pipeline envelopes, kind epm:progress — the pod NEVER shells task.py);
# - final results sentinel $SENTINEL_DIR/issue-1774-results.json (epm:results
#   payload: eval_numbers/eval_paths/reproducibility_card/wandb_url "n/a"/
#   hf_hub_url/worktree_path/final_commit_sha/gpu_hours_used/
#   gpu_hours_budgeted/plan_deviations — emitted by issue1774_common);
# - [phase=...] breadcrumbs on the main log; the single terminal [phase=done]
#   line below is the dispatcher's reserved success token;
# - 2-GPU work-conserving sharding: every shardable stage fans one process per
#   visible GPU via a CUDA_VISIBLE_DEVICES pin in the LAUNCHER env (gotchas:
#   the in-process pin alone is defeated by import-time cuInit); shard failure
#   tails the inner log into the main log (#1333 diagnosability rule).
set -euo pipefail

# set-u-safe on every lane: WORKLOAD_ROOT is exported only by the GCE startup
# script; RunPod/VM fall through to the script-relative repo root (#1329).
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$REPO_ROOT"
# GCE has NO .env (tokens ride instance metadata) — source conditionally.
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi

SENTINEL_DIR="${I1774_SENTINEL_DIR:-/workspace/logs}"
mkdir -p "$SENTINEL_DIR"

PHASE="all"
SMOKE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --all) PHASE="all"; shift ;;
    --smoke) SMOKE="1"; shift ;;
    *) echo "[dispatch] unknown arg: $1" >&2; exit 2 ;;
  esac
done

N_GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ' || true)"
if [ -z "$N_GPUS" ]; then N_GPUS=0; fi
echo "[dispatch] issue-1774 phase=$PHASE smoke=${SMOKE:-0} n_gpus=$N_GPUS repo=$REPO_ROOT"

SMOKE_LIMIT="${I1774_SMOKE_LIMIT:-4}"
DRAW_SMOKE_ARGS=()
FIT_SMOKE_ARGS=()
STEER_SMOKE_ARGS=()
P0_SMOKE_ARGS=()
if [ -n "$SMOKE" ]; then
  # Per-leg out-root isolation (#1333/#1586): a smoke leg must never leave its
  # regime/rows in the production roots a later full run resumes from (the P3
  # regime guard fail-louds on exactly that collision). Registry + all smoke
  # outputs land under this root; the shared staged STORE stays read-only.
  SMOKE_OUT_ROOT="${I1774_SMOKE_OUT_ROOT:-/tmp/issue1774-smoke}"
  echo "[dispatch] smoke out-root: $SMOKE_OUT_ROOT (per-leg isolation)"
  DRAW_SMOKE_ARGS=(--limit "$SMOKE_LIMIT" --out-root "$SMOKE_OUT_ROOT")
  FIT_SMOKE_ARGS=(--smoke --out-root "$SMOKE_OUT_ROOT")
  STEER_SMOKE_ARGS=(--smoke --out-root "$SMOKE_OUT_ROOT")
  P0_SMOKE_ARGS=(--smoke --out-root "$SMOKE_OUT_ROOT")
fi

phase_sentinel() {  # $1 = phase slug (p1|p2|p3), $2 = note
  uv run python scripts/issue1774_common.py \
    --emit-phase-sentinel "$1" "$SENTINEL_DIR/issue-1774-$1-done.json" \
    --phase-note "$2"
}

wait_all() {  # wait on recorded pids; nonzero if any shard failed
  local rc=0 p
  for p in "$@"; do
    if ! wait "$p"; then rc=$?; echo "[dispatch] shard pid=$p FAILED rc=$rc" >&2; fi
  done
  return "$rc"
}

push_results() {  # $1 = phase slug. Result-push verification contract (#1205)
  # + artifact-presence assert (#1325): commit + push the git-destined result
  # JSONs BEFORE the phase sentinel / [phase=done]; on the GCP lane a clean
  # completion DELETEs the boot disk, so an unpushed result commit is lost.
  if [ -n "$SMOKE" ]; then
    echo "[dispatch] result push SKIPPED (smoke — results under the smoke out-root)"
    return 0
  fi
  local rd="eval_results/issue_1774" branch behind p missing=0
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [ "$branch" = "HEAD" ]; then
    echo "[dispatch] FATAL: detached HEAD — cannot push results to a branch" >&2
    exit 86
  fi
  if [ -n "$(git status --porcelain -- "$rd")" ]; then
    git add -- "$rd"
    git -c user.name="issue1774-dispatch" -c user.email="pod@eps.local" \
      commit -m "task #1774: pod results ($1)"
  fi
  if ! git push origin "HEAD:$branch"; then
    echo "[dispatch] push failed — retrying once" >&2
    sleep 10
    git push origin "HEAD:$branch"
  fi
  behind="$(git rev-list --count "origin/$branch..HEAD")"
  if [ "$behind" -ne 0 ]; then
    echo "[dispatch] FATAL: $behind unpushed result commit(s) on $branch" >&2
    exit 86
  fi
  # per-file presence in the PUSHED tree (never a bare directory check, #928):
  # the declared set = on-disk result JSONs under $rd, minus gitignored paths.
  while IFS= read -r p; do
    if git check-ignore -q "$p"; then continue; fi
    if [ -z "$(git ls-tree -r "origin/$branch" --name-only -- "$p")" ]; then
      echo "[dispatch] FATAL: result file not in pushed tree: $p" >&2
      missing=1
    fi
  done < <(find "$rd" -name '*.json' -type f 2>/dev/null)
  if [ "$missing" -ne 0 ]; then exit 86; fi
  echo "[dispatch] result push verified ($1): origin/$branch carries the result tree"
}

tail_logs_and_die() {  # $1 = glob prefix of inner shard logs
  echo "[dispatch] shard failure — inner log tails follow (#1333 rule)" >&2
  local f
  for f in "$1"*.log; do
    if [ -f "$f" ]; then
      echo "----- tail $f -----" >&2
      tail -n 120 "$f" >&2
    fi
  done
  exit 1
}

ensure_p0() {
  # Pod-side staging + registries: the VM P0 committed the registries to git,
  # but a fresh pod clone has NO staged store (data/ is gitignored) — re-run
  # the idempotent stage-audit with --apply-restage (scoped HF staging, #833).
  echo "[phase=p0_stage]"
  uv run python scripts/issue1774_stage_audit.py --apply-restage "${P0_SMOKE_ARGS[@]}"
}

run_p1() {
  echo "[phase=p1_pilot]"
  local rc=0
  # DRAW_SMOKE_ARGS threads the smoke out-root: the pilot must read the SAME
  # registry root the P0 smoke restage wrote (per-leg isolation, #1333).
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1774_draws.py --stage pilot \
    "${DRAW_SMOKE_ARGS[@]}" || rc=$?
  if [ "$rc" -eq 7 ]; then
    # artifact-routed gate refusal (identity gate; report JSON already written)
    echo "[dispatch] p1 pilot IDENTITY GATE refusal (rc=7) — halting, see pilot_gate_report.json" >&2
    exit 7
  elif [ "$rc" -ne 0 ]; then
    echo "[dispatch] p1 pilot crashed rc=$rc" >&2
    exit "$rc"
  fi

  echo "[phase=p1_gen]"
  if [ "$N_GPUS" -ge 2 ]; then
    local pids=() g
    for g in $(seq 0 $((N_GPUS - 1))); do
      CUDA_VISIBLE_DEVICES="$g" uv run python scripts/issue1774_draws.py \
        --stage gen --shard "$g/$N_GPUS" "${DRAW_SMOKE_ARGS[@]}" \
        > "$SENTINEL_DIR/issue-1774-p1-gen-shard$g.log" 2>&1 &
      pids+=("$!")
    done
    if ! wait_all "${pids[@]}"; then tail_logs_and_die "$SENTINEL_DIR/issue-1774-p1-gen-shard"; fi
  else
    uv run python scripts/issue1774_draws.py --stage gen --shard 0/1 "${DRAW_SMOKE_ARGS[@]}"
  fi

  # persist-first: raw completion TEXT to HF BEFORE the capture consumer.
  # Smoke legs never write the production HF prefix (their files share names
  # with production uploads — a smoke push would be superseded but can strand
  # stale-shard orphans); the hub boundary is exercised by the production run.
  echo "[phase=p1_upload_text]"
  if [ -n "$SMOKE" ]; then
    echo "[dispatch] p1 upload SKIPPED (smoke — production HF prefix untouched)"
  else
    uv run python scripts/issue1774_draws.py --stage upload
  fi

  echo "[phase=p1_capture]"
  if [ "$N_GPUS" -ge 2 ]; then
    local cpids=() g
    for g in $(seq 0 $((N_GPUS - 1))); do
      CUDA_VISIBLE_DEVICES="$g" uv run python scripts/issue1774_draws.py \
        --stage capture --shard "$g/$N_GPUS" "${DRAW_SMOKE_ARGS[@]}" \
        > "$SENTINEL_DIR/issue-1774-p1-cap-shard$g.log" 2>&1 &
      cpids+=("$!")
    done
    if ! wait_all "${cpids[@]}"; then tail_logs_and_die "$SENTINEL_DIR/issue-1774-p1-cap-shard"; fi
  else
    uv run python scripts/issue1774_draws.py --stage capture --shard 0/1 "${DRAW_SMOKE_ARGS[@]}"
  fi

  echo "[phase=p1_upload_summaries]"
  if [ -n "$SMOKE" ]; then
    echo "[dispatch] p1 summaries upload SKIPPED (smoke — production HF prefix untouched)"
  else
    uv run python scripts/issue1774_draws.py --stage upload
  fi
  push_results p1
  phase_sentinel p1 "P1 draws+capture+upload complete (smoke=${SMOKE:-0})"
}

run_p2() {
  local armsA="arm_context,arm_bare_query" armsB="arm_prefix_end,arm_query_avg"
  if [ "$N_GPUS" -ge 2 ]; then
    local step
    # q4 is deliberately NOT in this sharded loop: its cokernel_*.json files are
    # combined-across-arms writes (a sharded q4 races, C1) — it runs unsharded
    # at the head of the GPU0 reads chain below (cheap next to fits; the guard
    # in step_q4 refuses any partial --arms invocation).
    for step in fits q3; do
      echo "[phase=p2_${step}_sharded]"
      local pids=()
      CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1774_fit_battery.py \
        --step "$step" --arms "$armsA" "${FIT_SMOKE_ARGS[@]}" \
        > "$SENTINEL_DIR/issue-1774-p2-$step-gpu0.log" 2>&1 &
      pids+=("$!")
      CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue1774_fit_battery.py \
        --step "$step" --arms "$armsB" "${FIT_SMOKE_ARGS[@]}" \
        > "$SENTINEL_DIR/issue-1774-p2-$step-gpu1.log" 2>&1 &
      pids+=("$!")
      if ! wait_all "${pids[@]}"; then tail_logs_and_die "$SENTINEL_DIR/issue-1774-p2-$step-gpu"; fi
    done
    echo "[phase=p2_reads_sharded]"
    # GPU0: q4 (unsharded, all arms) -> parity -> q1a -> q1b (L14 reads);
    # GPU1: q3angles -> q5 -> decode -> directions (operator reads) —
    # work-conserving split, no idle GPU.
    local rpids=()
    (
      CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1774_fit_battery.py --step q4 "${FIT_SMOKE_ARGS[@]}" &&
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1774_fit_battery.py --step parity "${FIT_SMOKE_ARGS[@]}" &&
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1774_fit_battery.py --step q1a "${FIT_SMOKE_ARGS[@]}" &&
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue1774_fit_battery.py --step q1b "${FIT_SMOKE_ARGS[@]}"
    ) > "$SENTINEL_DIR/issue-1774-p2-reads-gpu0.log" 2>&1 &
    rpids+=("$!")
    (
      CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue1774_fit_battery.py --step q3angles "${FIT_SMOKE_ARGS[@]}" &&
        CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue1774_fit_battery.py --step q5 "${FIT_SMOKE_ARGS[@]}" &&
        CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue1774_fit_battery.py --step decode "${FIT_SMOKE_ARGS[@]}" &&
        CUDA_VISIBLE_DEVICES=1 uv run python scripts/issue1774_fit_battery.py --step directions "${FIT_SMOKE_ARGS[@]}"
    ) > "$SENTINEL_DIR/issue-1774-p2-reads-gpu1.log" 2>&1 &
    rpids+=("$!")
    if ! wait_all "${rpids[@]}"; then tail_logs_and_die "$SENTINEL_DIR/issue-1774-p2-reads-gpu"; fi
  else
    echo "[phase=p2_all_serial]"
    uv run python scripts/issue1774_fit_battery.py --step all "${FIT_SMOKE_ARGS[@]}"
  fi

  echo "[phase=p2_jensen_refit]"
  if [ -n "$SMOKE" ]; then
    # Smoke-gate (v1 Minor): the jensen leg is the FULL 996-prefix #1092 recipe
    # writing the PRODUCTION jensen_refit dir — no small knob; the reused parent
    # recipe carries its own #1092 coverage, and the production leg runs it.
    echo "[dispatch] p2 jensen refit SKIPPED (smoke — full-size recipe, production out-dir)"
  else
    # Q1c mainline (plan §4): re-run the banked MLP Jensen recipe on the instruct
    # cell only, persisting per-prefix gap vectors to a FRESH issue-1774 out dir
    # (never overwriting #1092's committed npz — the banked scalar file stays the
    # norm cross-check reference). CPU torch fit, checkpointed per cell.
    I1092_STAGE_DIR="$(uv run python -c "import sys; sys.path.insert(0, 'scripts'); import issue1774_common as c; print(c.stage_dir())")"
    export I1092_STAGE_DIR
    uv run python scripts/issue1092_mlp_jensen_natural.py \
      --persist-gap-vectors --cells cell_inst_own \
      --out-dir "eval_results/issue_1774/jensen_refit"
  fi

  echo "[phase=p2_upload]"
  if [ -n "$SMOKE" ]; then
    # Same rationale as the p1 upload skips: the heredoc enumerates PRODUCTION
    # data roots — a post-production smoke re-run would re-upload stale
    # production operators (v1 Minor smoke-isolation wart).
    echo "[dispatch] p2 upload SKIPPED (smoke — production HF prefix untouched)"
  else
  uv run python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import issue1774_common as c  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

expected = []
for sub in ("operators", "analysis_tensors/channels_null"):
    d = c.data_out(None) / sub
    if not d.exists():
        continue
    hub._upload(
        d,
        repo_id=c.DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{c.HF_UPLOAD_PREFIX}/{sub}",
    )
    expected += [f"{c.HF_UPLOAD_PREFIX}/{sub}/{p.name}" for p in sorted(d.glob("*.npy"))]
dirs = c.data_out(None) / "directions.pt"
if dirs.exists():
    hub._upload(
        dirs,
        repo_id=c.DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{c.HF_UPLOAD_PREFIX}/directions/directions.pt",
        upload_as_file=True,
    )
    expected.append(f"{c.HF_UPLOAD_PREFIX}/directions/directions.pt")
missing = hub.verify_repo_paths_uploaded(
    HfApi(), c.DATA_REPO, expected, path_in_repo=c.HF_UPLOAD_PREFIX, repo_type="dataset"
)
if missing:
    raise RuntimeError(f"p2 upload verify missing {len(missing)}: {sorted(missing)[:5]}")
print(f"[p2-upload] verified {len(expected)} paths")
PY
  fi
  push_results p2
  phase_sentinel p2 "P2 fit battery + operators + uploads complete (smoke=${SMOKE:-0})"
}

run_p3() {
  echo "[phase=p3_steering]"
  # scripts/issue1774_steering.py lands in a later round (plan §4 P3 interface:
  # --shard i/n [--smoke] [--out-root D]; writes steering/manifest.json +
  # state_shift.json and uploads completions to HF the moment generation ends).
  if [ ! -f scripts/issue1774_steering.py ]; then
    echo "[dispatch] FATAL: scripts/issue1774_steering.py not present (later round)" >&2
    exit 3
  fi
  if [ "$N_GPUS" -ge 2 ]; then
    local pids=() g
    for g in $(seq 0 $((N_GPUS - 1))); do
      CUDA_VISIBLE_DEVICES="$g" uv run python scripts/issue1774_steering.py \
        --shard "$g/$N_GPUS" "${STEER_SMOKE_ARGS[@]}" \
        > "$SENTINEL_DIR/issue-1774-p3-shard$g.log" 2>&1 &
      pids+=("$!")
    done
    if ! wait_all "${pids[@]}"; then tail_logs_and_die "$SENTINEL_DIR/issue-1774-p3-shard"; fi
  else
    uv run python scripts/issue1774_steering.py --shard 0/1 "${STEER_SMOKE_ARGS[@]}"
  fi
  push_results p3
  phase_sentinel p3 "P3 steering + hook-free re-capture complete (smoke=${SMOKE:-0})"
}

case "$PHASE" in
  p1) ensure_p0; run_p1 ;;
  p2) ensure_p0; run_p2 ;;
  p3) run_p3 ;;
  all) ensure_p0; run_p1; run_p2; run_p3 ;;
  *) echo "[dispatch] unknown --phase $PHASE" >&2; exit 2 ;;
esac

if [ "$PHASE" = "all" ]; then
  ELAPSED="$SECONDS"
  GPU_HOURS="$(awk -v s="$ELAPSED" -v g="$N_GPUS" 'BEGIN{printf "%.2f", (g > 0 ? s * g : s) / 3600}')"
  echo "[phase=results_sentinel] gpu_hours_used=$GPU_HOURS"
  SMOKE_FLAG=()
  if [ -n "$SMOKE" ]; then SMOKE_FLAG=(--smoke); fi
  uv run python scripts/issue1774_common.py \
    --emit-results-sentinel "$SENTINEL_DIR/issue-1774-results.json" \
    --gpu-hours-used "$GPU_HOURS" "${SMOKE_FLAG[@]}" \
    --plan-deviation "q3-angles null: Haar random-subspace band (_angle_null_band) operationalizes the plan's spectrum-matched _procrustes_null_band — for principal angles between top-k singular subspaces of independent spectrum-fixed random maps the left/right singular bases are Haar regardless of spectrum, so the Haar band IS the spectrum-invariant null (recorded, not silent)" \
    --plan-deviation "kernel-tail directions: step_directions enforces sigma-rank >= 3000 and > k90 but NOT the plan's 'sigma below the permutation-null band' condition — deferred to the P6 verify (per-arm null matrices persisted; null_matrix_present recorded)" \
    --plan-deviation "P5 evil rubric: the realized design has exactly ONE evil arm (leace_rb_evil; r_B directions are LEACE-only per the 12-ADD/3-LEACE split of plan section 4 P3) while plan P5 says '3 evil arms x 60 x 5' — internally inconsistent; resolved to 1 arm (~300 evil judge calls vs ~900)" \
    --plan-deviation "q3 channel null: within-test-fold Y-row permutation with train-fold directions FROZEN (conditional permutation, exact for the OOF statistic; same contiguous-from-top count rule per draw) instead of the registered same-lambda refit null — the channel estimator is a per-fold cross-covariance SVD with no lambda (press_mse inapplicable), and a true per-draw refit is ~14.4k dense (r x r) SVDs (200 draws x 6 folds x 4 arms x 3 layers), contradicting plan section 9's own 'one GEMM stack per fold, never per-draw refits' 0.5h sizing"
fi

echo "[phase=done] issue-1774 dispatch phase=$PHASE rc=0"
