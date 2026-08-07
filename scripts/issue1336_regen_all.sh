#!/usr/bin/env bash
# Round-A driver: per-INVOCATION prefix-continuation regen of the #1336 cap-truncated rows.
#
# WHY THIS EXISTS. issue1336_regen_truncated.py takes ONE --model and loops --corpora,
# loading the vLLM engine once per invocation (~2-4 min). Driving it per (model, format)
# amortizes that load across the model's whole corpus set — 10 invocations cover all 40
# cells (5 models x 7 chat corpora = 35, plus the prefix arm's 5 naturalistic lmsys23k
# cells per cm.V2_PREFIX_ARM) with 10 engine loads instead of 40.
#
# The pod's MooseFS quota is ~130 GB and df CANNOT see it (it reports the whole cluster,
# ~323 TB — the EDQUOT gotcha), so the guard here is a du-based reap of the HF model
# cache: 5 ladder models x ~16 GB would blow the quota if they accumulated. The reap
# fires between MODELS, never between a model's chat/naturalistic pair (same weights),
# and touches ONLY the re-downloadable Hub model cache — never the generated answer
# cells under gen/, which are this round's durable product.
#
# Invocation order is base FIRST (its first cell is the measured basis for the run's
# wall-time projection — the compute-character rule's MEASURED 1-cell pilot), then
# rlvr_long SECOND because it is the worst case for truncation by construction (trained
# for long outputs, so the largest cap-truncated population), then sft / dpo / rlvr.
#
# Corpora are ordered by DESCENDING cap-truncated row count (measured 2026-08-07 off the
# source cells' own audit.json kept_truncation_rate x n_kept, all 70 v2 cells resolved:
# 46,289 truncated of 476,794 kept). NOT cheapest-first: only the truncated rows are
# regenerated, so the cheap corpora are the ones with almost nothing to do —
# base/chat/gsm8k_test1319-class cells carry single-digit-to-tens of rows, which cannot
# fill vLLM continuous batching and would reproduce the pilot's batch-5 throughput floor
# (546 tok/s) as the "basis" for the whole run. Leading with lmsys23k (2,569 truncated
# rows for base) and math7500 (2,390) makes the FIRST measured cell production-shape, and
# puts the largest row populations on disk earliest.
#
# Per-cell durability + resume come free from the imported generation machinery:
# regen_cell() skips a cell whose local outputs exist, re-uploads when the Hub copy is
# incomplete, and _try_hf_resume() pulls a Hub-complete cell back. So a mid-run death
# re-runs at most the in-flight cell.
#
# Usage:  bash scripts/issue1336_regen_all.sh [FIRST] [LAST]     (1-indexed, inclusive)
#         bash scripts/issue1336_regen_all.sh 1 1     # base chat only (the basis run)
#         bash scripts/issue1336_regen_all.sh 2 10    # the rest
set -uo pipefail

REPO=/workspace/explore-persona-space
# Production budget: stored prefix is exactly the original 1024 cap, tail adds 1024, so
# every row's total answer budget is 2048 and max_model_len = 3072 prompt + 2048.
TAIL=${EPM_1336_TAIL:-1024}
MAXLEN=${EPM_1336_MAXLEN:-5120}
# BOTH formats cover ALL 7 v2 corpora: cm.V2_GEN_FORMATS licenses ('chat','naturalistic')
# for every v2 corpus (14 cells/model x 5 models = 70). Do NOT scope naturalistic to
# cm.V2_PREFIX_ARM's lmsys23k — that is the FIT-side prefix arm, a different thing
# ("naturalistic gen on every v2 corpus, context arm only; the fit-side grid is
# deliberately untouched" — _formats_for's docstring). Scoping to the fit arm here would
# silently skip 30 of the 70 cells with no error.
CHAT_CORPORA=${EPM_1336_CHAT_CORPORA:-lmsys23k,math7500,sft11k,uf11k,if11k,gsm8k_train_full,gsm8k_test1319}
NAT_CORPORA=${EPM_1336_NAT_CORPORA:-lmsys23k,math7500,sft11k,uf11k,if11k,gsm8k_train_full,gsm8k_test1319}
# Reap the Hub model cache when it exceeds this before an invocation that changes model.
CACHE_CAP_GB=${EPM_1336_CACHE_CAP_GB:-40}

# model|format|corpora
INVOCATIONS=(
"base|chat|$CHAT_CORPORA"
"base|naturalistic|$NAT_CORPORA"
"rlvr_long|chat|$CHAT_CORPORA"
"rlvr_long|naturalistic|$NAT_CORPORA"
"sft|chat|$CHAT_CORPORA"
"sft|naturalistic|$NAT_CORPORA"
"dpo|chat|$CHAT_CORPORA"
"dpo|naturalistic|$NAT_CORPORA"
"rlvr|chat|$CHAT_CORPORA"
"rlvr|naturalistic|$NAT_CORPORA"
)

FIRST=${1:-1}
LAST=${2:-${#INVOCATIONS[@]}}

cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }

# Re-attach breadcrumbs. The pidfile is rewritten by THIS run (never left carrying a
# predecessor's pid — the #813 relaunch trap), and the exit-code sentinel is removed at
# launch so a stale one can never satisfy a done-check (the never-key-done-on-bare-
# existence rule). A successor session re-attaches from these two paths alone.
LOGDIR=${EPM_1336_LOGDIR:-/workspace/logs}
PIDFILE="$LOGDIR/issue-1336-regen.pid"
SENTINEL="$LOGDIR/issue-1336-regen-done.json"
mkdir -p "$LOGDIR"
echo $$ > "$PIDFILE"
rm -f "$SENTINEL"

write_sentinel() {
  cat > "$SENTINEL.tmp" <<JSON
{"issue": 1336, "round": "regen-cont", "rc": $1, "invocations": "$FIRST..$LAST",
 "cells_on_disk": $2, "tail_max_tokens": $TAIL, "max_model_len": $MAXLEN,
 "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
  mv "$SENTINEL.tmp" "$SENTINEL"
}

set -a
[ -f ./.env ] && . ./.env
set +a
export MALLOC_ARENA_MAX=2

HF_CACHE=${HF_HOME:-/workspace/.cache/huggingface}
# issue1336_gen_answers.DATA_ROOT is the RELATIVE Path("data/issue_1336"), so the answer
# cells resolve against the process CWD — which this driver sets to $REPO. NOT /workspace.
GEN_ROOT="$REPO/data/issue_1336/gen"

echo "[driver] repo=$REPO head=$(git rev-parse HEAD)"
echo "[driver] tail=$TAIL max_model_len=$MAXLEN invocations=$FIRST..$LAST"
echo "[driver] hf_cache=$HF_CACHE cache_cap=${CACHE_CAP_GB}GB gen_root=$GEN_ROOT"

count_cells() { find "$GEN_ROOT" -name answers.jsonl -path '*_cont*' 2>/dev/null | wc -l; }

rc_all=0
prev_model=""
for ((i = FIRST; i <= LAST; i++)); do
  row="${INVOCATIONS[i-1]}"
  model="${row%%|*}"
  rest="${row#*|}"
  fmt="${rest%%|*}"
  corpora="${rest#*|}"

  # Reap the re-downloadable Hub model cache only when the MODEL changes (a model's
  # chat + naturalistic invocations share weights, so never reap between them).
  if [ -n "$prev_model" ] && [ "$model" != "$prev_model" ]; then
    cache_gb=$(du -sBG "$HF_CACHE" 2>/dev/null | cut -f1 | tr -dc '0-9')
    if [ "${cache_gb:-0}" -ge "$CACHE_CAP_GB" ]; then
      rm -rf "$HF_CACHE/hub"
      after_gb=$(du -sBG "$HF_CACHE" 2>/dev/null | cut -f1 | tr -dc '0-9')
      echo "[driver] reaped Hub model cache before $model: ${cache_gb}GB -> ${after_gb:-0}GB"
    else
      echo "[driver] Hub model cache ${cache_gb:-0}GB < ${CACHE_CAP_GB}GB cap — no reap"
    fi
  fi

  echo "[driver] === invocation $i/${#INVOCATIONS[@]} $model/$fmt === corpora=$corpora"
  t0=$(date +%s)
  uv run python scripts/issue1336_regen_truncated.py \
    --model "$model" --gen-format "$fmt" --corpora "$corpora" \
    --tail-max-tokens "$TAIL" --max-model-len "$MAXLEN" --upload
  rc=$?
  t1=$(date +%s)
  echo "[driver] invocation $model/$fmt rc=$rc elapsed=$(( t1 - t0 ))s cells_on_disk=$(count_cells)"
  if [ "$rc" -ne 0 ]; then
    echo "[driver] FATAL $model/$fmt failed rc=$rc — stopping (per-cell outputs so far are durable)" >&2
    rc_all="$rc"
    break
  fi
  prev_model="$model"
done

n_cells=$(count_cells)
echo "[driver] DONE rc=$rc_all cells_on_disk=$n_cells"
write_sentinel "$rc_all" "$n_cells"
exit "$rc_all"
