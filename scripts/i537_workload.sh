#!/bin/bash
# Issue #537 follow-up `predictor-bakeoff-complete` -- GPU predictor passes.
# Runs from $WORKLOAD_ROOT (repo checkout root, branch issue-537) after the
# GCE startup script's env bootstrap (HF_TOKEN, WANDB_API_KEY, HF_HOME set).
# Blocking workload: stages HF inputs -> A5/A5_rb smoke-calibrate ->
# behavior-sharded GPU passes across 4 GPUs -> wait. CPU scoring is OFF-POD.
set -uo pipefail

# HF Hub resilience for the per-file staging loop (~2700 files; transient
# read timeouts on this dataset repo killed the previous launch ~12 min in).
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60}
export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}

BEHAVIORS="marker fact refusal sycophancy em"
DROPPED_METRICS="js_out_seq,kl_out_seq_fwd,kl_out_seq_rev,kl_asym_out_seq,js_out_seq_rb,kl_fwd_out_seq_rb,kl_rev_out_seq_rb,train_prior_tf,train_prior_onpolicy,js_taught_span,pv_dp,kl_out_seq_oneway"
EVAL_ROOT="eval_results/issue_537"
LOGDIR="/workspace/logs"
mkdir -p "$LOGDIR" "$EVAL_ROOT"

echo "[phase=stage] downloading HF inputs (raw_completions + data/train) -- per-file (51k-file repo truncates snapshot siblings)"
# Stage from HF: scripts read raw_completions/<b>/ + data/train/<b>/ under
# eval_results/issue_537 (no issue537_context_generalization prefix). NONE of
# the 3 GPU scripts download these; they fail-loud if absent. Per-file
# hf_hub_download (NOT snapshot_download -- allow_patterns sees 0 of the 51k
# truncated siblings; i537_dispatch.py:1921 documents this).
uv run python - <<'PY'
import os, shutil, time, random
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
REPO = "superkaiba1/explore-persona-space-data"
PREF = "issue537_context_generalization"
EVAL = Path("eval_results/issue_537")

def _list_with_retry(repo, max_tries=10):
    """list_repo_files itself can hit a transient timeout; retry with backoff."""
    for attempt in range(1, max_tries + 1):
        try:
            return list_repo_files(repo, repo_type="dataset", revision="main")
        except Exception as e:
            if attempt == max_tries:
                raise
            backoff = min(60, (2 ** attempt) + random.random())
            print(f"[stage] list_repo_files failed (attempt {attempt}/{max_tries}): {e!r}; sleep {backoff:.1f}s")
            time.sleep(backoff)

def _download_with_retry(repo, f, max_tries=10):
    """Per-file resilient download. The default hf_hub_download already
    retries 5x with backoff; this OUTER loop catches the residual
    transient-class failures (read timeouts, conn resets, 5xx) that exhaust
    the inner retries."""
    for attempt in range(1, max_tries + 1):
        try:
            return hf_hub_download(repo, f, repo_type="dataset", revision="main")
        except Exception as e:
            if attempt == max_tries:
                raise
            backoff = min(60, (2 ** attempt) + random.random())
            print(f"[stage] hf_hub_download retry {attempt}/{max_tries} for {f!r}: {e!r}; sleep {backoff:.1f}s")
            time.sleep(backoff)

files = _list_with_retry(REPO)
want = [f for f in files if f.startswith(f"{PREF}/raw_completions/") or f.startswith(f"{PREF}/data/train/")]
assert want, f"no raw_completions/data-train files matched under {PREF}/ (got {len(files)} repo files)"
print(f"[stage] {len(want)} files to fetch")
n = 0
for f in want:
    local = EVAL / f[len(PREF) + 1:]  # strip "issue537_context_generalization/"
    if local.exists():
        n += 1; continue
    cached = _download_with_retry(REPO, f)
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, local)
    n += 1
    if n % 200 == 0:
        print(f"[stage] {n}/{len(want)}")
print(f"[stage] DONE: {n} files staged under {EVAL}")
# fail-loud verification: each behavior has data/train rows
for b in ["marker", "fact", "refusal", "sycophancy", "em"]:
    dt = list((EVAL / "data" / "train" / b).glob("*_seed42.jsonl"))
    assert dt, f"data/train/{b} empty after stage"
print("[stage] data/train present for all 5 behaviors")

# DATA / *: predictor scripts use `DATA = REPO / "data/issue_537"` (hard-coded
# in i537_dropped_predictors.py / i537_bcond_predictors.py /
# i537_behavior_vector_predictor.py at module init). Stage ALL `data/<subdir>/`
# HF subtrees to `data/issue_537/<subdir>/` so pools, responses, contexts,
# responses_clouds, responses_eval, responses_refusal are all present.
# (train/ is ALSO restaged here in addition to its eval_results/ copy — minimal
# disk cost vs the risk of a fourth crash on a missed subdir.)
DATA_I537 = Path("data/issue_537")
want_data = [f for f in files if f.startswith(f"{PREF}/data/")]
assert want_data, f"no data/ files matched under {PREF}/data/"
print(f"[stage] {len(want_data)} data/* files to fetch -> data/issue_537/<subdir>/")
n_data = 0
for f in want_data:
    # HF: issue537_context_generalization/data/<subdir>/<rel> -> local: data/issue_537/<subdir>/<rel>
    rel = f[len(f"{PREF}/data/"):]
    local = DATA_I537 / rel
    if local.exists():
        n_data += 1
        continue
    cached = _download_with_retry(REPO, f)
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, local)
    n_data += 1
    if n_data % 50 == 0:
        print(f"[stage] data/ {n_data}/{len(want_data)}")
print(f"[stage] DONE data/: {n_data} files under {DATA_I537}/")
# fail-loud verification of the P0 dependencies the predictor scripts touch
assert (DATA_I537 / "contexts" / "sampled_contexts.json").exists(), "sampled_contexts.json missing after data stage"
pool_files = list((DATA_I537 / "pools").glob("pool_*.json"))
assert pool_files, f"data/issue_537/pools/pool_*.json empty after data stage"
print(f"[stage] data/issue_537/{{contexts/sampled_contexts.json, pools/ ({len(pool_files)} files)}} present")
PY
STAGE_RC=$?
if [ "$STAGE_RC" -ne 0 ]; then echo "[phase=failed] HF staging failed rc=$STAGE_RC"; exit "$STAGE_RC"; fi

echo "[phase=smoke-calibrate] A5/A5_rb wall on marker x 2 contexts (plan 9.0 ungrounded check)"
SMOKE_START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i537_dropped_predictors.py \
  --behaviors marker --metrics "$DROPPED_METRICS" --smoke \
  > "$LOGDIR/issue-537-smoke-dropped.log" 2>&1
SMOKE_RC=$?
SMOKE_END=$(date +%s)
SMOKE_WALL=$((SMOKE_END - SMOKE_START))
echo "[smoke-calibrate] A5/A5_rb marker x2ctx wall=${SMOKE_WALL}s rc=$SMOKE_RC"
if [ "$SMOKE_RC" -ne 0 ]; then echo "[phase=failed] A5/A5_rb smoke FAILED rc=$SMOKE_RC"; tail -30 "$LOGDIR/issue-537-smoke-dropped.log"; exit "$SMOKE_RC"; fi

echo "[phase=gpu-passes] 3 predictor scripts x 5 behaviors, sharded across 4 GPUs"
declare -a PIDS
GPU=0
for b in $BEHAVIORS; do
  cvd=$((GPU % 4))
  (
    set -e
    CUDA_VISIBLE_DEVICES=$cvd uv run python scripts/i537_dropped_predictors.py \
      --behaviors "$b" --metrics "$DROPPED_METRICS" > "$LOGDIR/issue-537-dropped-$b.log" 2>&1
    CUDA_VISIBLE_DEVICES=$cvd uv run python scripts/i537_bcond_predictors.py \
      --behaviors "$b" > "$LOGDIR/issue-537-bcond-$b.log" 2>&1
    CUDA_VISIBLE_DEVICES=$cvd uv run python scripts/i537_behavior_vector_predictor.py \
      --phase extract --behaviors "$b" > "$LOGDIR/issue-537-vb-$b.log" 2>&1
  ) &
  PIDS+=($!)
  echo "[gpu-passes] launched behavior=$b on GPU=$cvd pid=${PIDS[-1]}"
  GPU=$((GPU + 1))
  # 4 GPUs: stagger so the 5th behavior queues behind the first free GPU.
  if [ "$cvd" -eq 3 ]; then wait "${PIDS[0]}"; fi
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then echo "[gpu-passes] shard pid=$pid FAILED"; FAIL=1; fi
done
if [ "$FAIL" -ne 0 ]; then
  echo "[phase=failed] one or more behavior shards failed -- terminal phase token suppressed"
  for b in $BEHAVIORS; do echo "=== tail dropped-$b ==="; tail -15 "$LOGDIR/issue-537-dropped-$b.log" 2>/dev/null; done
  exit 1
fi

echo "[phase=upload] GPU artifacts written under $EVAL_ROOT/predictor-bakeoff-complete/ (lane scp-pulls eval_results/ back)"
ls -R "$EVAL_ROOT/predictor-bakeoff-complete/" 2>/dev/null | head -40
echo "[phase=done] all 5 behaviors x 3 predictor passes complete"
