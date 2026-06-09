#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/launch_issue_518_production.sh
#
# Master pod-side launcher for #518 v4 (refusal + EM + syco cross-arm) per
# plan §4 / §10 / §11.
#
# DAG ordering (from plan §4):
#   E  refusal_train_eval  ─┐
#   E  em_train_eval        │
#   F  cosine_sweep_refusal │   (#518 v4 round-8 new producer)
#   F  cosine_sweep_em      ├─→ F.5 logprob_input_adapter ─→ G logprob_refusal ─┐
#   F  jskl_sweep_refusal   │       (round-9: positives.jsonl reshaper)         │
#   F  jskl_sweep_em        │                                                   │
#                          ─┘                                                   │
#                                                                              │
#   G  logprob_refusal      │
#   G  logprob_em           ├─→ H bakeoff_refusal ─┐
#   G  logprob_syco_backfill│   H bakeoff_em       │
#                          ─┘                       │
#                                                   ▼
#                                   substrate_syco / substrate_refusal /
#                                   substrate_em → scoring_refusal /
#                                   scoring_em / scoring_syco →
#                                   cross_arm_aggregator → done.
#
# ``poll_pipeline.py`` parses ``[phase=<name>]`` markers from this log to
# emit per-phase ``epm:progress`` markers. The final ``[phase=done]`` line
# IS REQUIRED (`poll_once` returns ``status="done"`` ONLY after seeing it).
#
# Final sentinel: ``/workspace/logs/issue-518-epm_results-<unix_ts>.json``
# conforming to ``poll_pipeline._SENTINEL_REQUIRED_KEYS`` so the
# orchestrator's poll loop auto-posts ``epm:results v1``.
#
# Usage (pod side, after ``pod.py sync env pod-518`` + preflight)::
#
#     cd /workspace/explore-persona-space
#     nohup bash scripts/launch_issue_518_production.sh \
#         > /workspace/logs/issue_518_pipeline.log 2>&1 &
#     echo $! > /workspace/logs/issue_518_pipeline.pid
#
# Resume-safety: every phase is idempotent / checkpoint-per-phase. A
# mid-launcher crash + relaunch picks up where it left off; producer
# scripts skip cells whose outputs already exist on disk.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Ensure uv is on PATH (bootstrap_pod.sh installs to /root/.local/bin but
# non-login shells don't pick it up).
export PATH="/root/.local/bin:$PATH"

# Load credentials (HF_TOKEN, WANDB_API_KEY, ANTHROPIC_API_KEY). Without
# this, downstream subprocesses lack the env even if --env is threaded.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "$REPO_ROOT/.env"
    set +a
fi

# Pin HF cache to MooseFS per CLAUDE.md (avoids /root quota blowups).
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR="${LOG_DIR:-/workspace/logs}"
PIPELINE_LOG="$LOG_DIR/issue_518_pipeline.log"
mkdir -p "$LOG_DIR"

# Output bucket layout (matches plan §10).
EVAL_ROOT="$REPO_ROOT/eval_results/issue_518"
for arm in refusal em syco; do
    mkdir -p "$EVAL_ROOT/$arm/runs" \
             "$EVAL_ROOT/$arm/bystander_logprob" \
             "$EVAL_ROOT/$arm/predictors" \
             "$EVAL_ROOT/$arm/bakeoff/metrics" \
             "$EVAL_ROOT/$arm/_inputs" \
             "$EVAL_ROOT/$arm/slab"
done

# Helper: emit a [phase=...] marker AND echo a UTC timestamp. poll_pipeline.py
# greps for [phase=<token>] -- the second arg is a free-form note.
phase() {
    local name="$1"
    local note="${2:-}"
    echo "[phase=${name}] $(date -u +%FT%TZ) ${note}"
}

# Helper: run a subprocess + tee its output to the pipeline log.
run() {
    echo "[run] $(date -u +%FT%TZ) $*"
    "$@"
}

# ── D. Pre-stage check: syco arm inheritance ────────────────────────────────
# Phases I (substrate_syco) and J (scoring_syco) read from
# ``eval_results/issue_509/syco_arm/runs/`` and
# ``eval_results/issue_509/syco_arm/bakeoff/metrics/`` — these are PRE-STAGED
# from a prior #509 run (plan §3 / §10: "the existing #509 syco bake-off"
# is reused, optionally re-run on BASE per the model-id consistency gate).
# They are NOT produced by this launcher; fail-LOUD early if they're missing
# so the failure surfaces here (with a clean fix instruction) rather than
# 6+ hours into the pipeline at substrate_syco.
phase prestage_check_syco "verify pre-staged syco arm inputs"
# Round-12 fix: drop the obsolete eval_results/issue_509/syco_arm/runs/ check
# (#509 never produced that directory -- the syco arm is INHERITED from #411
# via the frozen 138-cell analyze_summary). Three checks now:
#   1. #411 analyze_summary exists + carries >= 138 cells (substrate_syco reads it)
#   2. #480 predictor_comparison.json exists (substrate_syco coarse-zoo source)
#   3. #509 syco bakeoff metrics dir exists + non-empty (scoring_syco reads it)
SYCO_ANALYZE_SUMMARY="$REPO_ROOT/eval_results/issue_480/_inputs/syco_411_analyze_summary.json"
SYCO_PREDICTOR_COMPARISON="$REPO_ROOT/eval_results/issue_480/_inputs/predictor_comparison.json"
SYCO_BAKEOFF_METRICS="$REPO_ROOT/eval_results/issue_509/syco_arm/bakeoff/metrics"
SYCO_BAKEOFF_META="$REPO_ROOT/eval_results/issue_509/syco_arm/bakeoff/meta.json"
if [[ ! -f "$SYCO_ANALYZE_SUMMARY" ]]; then
    echo "[FATAL] Pre-staged #411 analyze_summary missing: $SYCO_ANALYZE_SUMMARY" >&2
    echo "        Phase substrate_syco reads this 138-cell frozen leakage" >&2
    echo "        snapshot DIRECTLY (the syco arm is inherited from #411 via" >&2
    echo "        the snapshot, not produced by #509). Pull the file from the" >&2
    echo "        #480 cache or re-derive from #411's eval results." >&2
    exit 1
fi
# Verify the snapshot carries >= 138 cells (6 sources x 23 bystanders).
SYCO_NCELLS=$(uv run python -c "
import json, sys
d = json.load(open('$SYCO_ANALYZE_SUMMARY'))
ps = d.get('per_source', {})
n = 0
for src, sd in ps.items():
    for bys in sd.get('per_panel_delta', {}):
        if bys != src:
            n += 1
print(n)
" 2>/dev/null)
if [[ -z "$SYCO_NCELLS" ]] || (( SYCO_NCELLS < 138 )); then
    echo "[FATAL] #411 analyze_summary at $SYCO_ANALYZE_SUMMARY has $SYCO_NCELLS off-diagonal cells; expected >= 138 (6 sources x 23 bystanders)." >&2
    exit 1
fi
if [[ ! -f "$SYCO_PREDICTOR_COMPARISON" ]]; then
    echo "[FATAL] Pre-staged #480 predictor_comparison.json missing: $SYCO_PREDICTOR_COMPARISON" >&2
    echo "        Phase substrate_syco reads this for the 17 coarse-zoo fields." >&2
    exit 1
fi
if [[ ! -d "$SYCO_BAKEOFF_METRICS" ]] || [[ -z "$(ls -A "$SYCO_BAKEOFF_METRICS" 2>/dev/null)" ]]; then
    echo "[FATAL] Pre-staged syco bakeoff metrics dir missing or empty: $SYCO_BAKEOFF_METRICS" >&2
    echo "        Phase scoring_syco reads this. Pull the #509 syco_arm/bakeoff/" >&2
    echo "        metrics/ directory onto the pod, OR re-run the syco bake-off" >&2
    echo "        on BASE per plan §3 / §10 model-id gate." >&2
    exit 1
fi
# Soft check: cross_arm_aggregator opts into --allow-legacy-syco-meta when
# this file predates the --model-id patch; presence is informational.
if [[ ! -f "$SYCO_BAKEOFF_META" ]]; then
    echo "[warn] $SYCO_BAKEOFF_META missing -- aggregator's --allow-legacy-syco-meta carve-out covers this." >&2
fi
echo "[ok] pre-staged syco inputs: analyze_summary cells=$SYCO_NCELLS; #480 predictor_comparison present; bakeoff metrics: $(ls -1 "$SYCO_BAKEOFF_METRICS" 2>/dev/null | wc -l) files"

# ── D.5 Refusal probe-pool generation (must run before refusal_train_eval) ──
# ``scripts/run_experiment_518_refusal.py`` requires both
# ``data/issue_518/refusal_200_training.jsonl`` (200 training rows) and
# ``data/issue_518/refusal_50.jsonl`` (50 eval probes). The canonical
# producer is ``scripts/generate_refusal_50.py`` (Sonnet 4.5 Batch API,
# ~$5-12, ANTHROPIC_API_KEY loaded from .env above). Idempotent via
# ``--keep-existing``: if all three output files (pool / train / eval)
# already exist on disk, the generator is a no-op and the phase exits in
# <1s. The fail-loud upstream check that bit us at round-9 launch
# (``run_experiment_518_refusal.py:669-672``) catches a missing file with
# a clear "Run scripts/generate_refusal_50.py first" message, but the
# launcher is the right place to wire it.
phase gen_refusal_data "generate_refusal_50.py --keep-existing"
run uv run python scripts/generate_refusal_50.py --keep-existing

# ── E. Per-arm train + eval (refusal, EM) ──────────────────────────────────
phase refusal_train_eval "run_experiment_518_refusal.py"
run uv run python scripts/run_experiment_518_refusal.py

phase em_train_eval "run_experiment_518_em.py"
run uv run python scripts/run_experiment_518_em.py

# ── F. Per-arm coarse-zoo predictor sweeps (#518 v4 round-8 producers) ─────
# Round-8 closes the launch-time gap: substrate builder requires consolidated
# cosine + JS/KL sweeps per arm before it can assemble predictor_comparison.json.
phase cosine_sweep_refusal "issue518_cosine_sweep_panel.py --arm refusal"
run uv run python scripts/issue518_cosine_sweep_panel.py --arm refusal

phase cosine_sweep_em "issue518_cosine_sweep_panel.py --arm em"
run uv run python scripts/issue518_cosine_sweep_panel.py --arm em

phase jskl_sweep_refusal "issue518_jskl_sweep_panel.py --arm refusal"
run uv run python scripts/issue518_jskl_sweep_panel.py --arm refusal

phase jskl_sweep_em "issue518_jskl_sweep_panel.py --arm em"
run uv run python scripts/issue518_jskl_sweep_panel.py --arm em

# ── F.5 Logprob-input adapter: extract source-positive rows from each arm's
# train_pool.jsonl and rewrite as data/issue_518/<arm>/<source>/positives.jsonl
# so issue518_syco_logprob_backfill.py can consume them. Closes the round-8
# blocker: the refusal + EM runners write train_pool.jsonl as multi-bucket
# {prompt: [...], completion: [...]} rows (system_prompt = source persona is
# the positive subset); the backfill expects {question, completion} rows under
# <root>/<source>/positives.jsonl. The conversion is per-source and idempotent.
phase logprob_input_adapter "convert train_pool.jsonl -> positives.jsonl per arm × source"
run uv run python scripts/issue518_build_logprob_inputs.py \
    --arms refusal em \
    --runs-root-template "$EVAL_ROOT/{arm}/runs" \
    --data-root "$REPO_ROOT/data/issue_518"

# ── G. Completion-log-prob predictor (per arm; new metric in #518 v4) ──────
# Plan §4.3 / §0c: per-(source, bystander) length-normalized log P(positive
# completion | bystander_system, training_question) on FROZEN base
# Qwen2.5-7B-Instruct. The refusal + EM arm scripts share the same
# methodology as issue518_syco_logprob_backfill.py; per-arm wrappers point
# at the per-arm training rows + output bucket.
#
# Round-9 fix: ``issue518_build_logprob_inputs.py`` runs above (phase
# logprob_input_adapter) so ``data/issue_518/{refusal,em}/<source>/positives.jsonl``
# exists in the layout the backfill expects.
phase logprob_refusal "issue518_syco_logprob_backfill.py --teach-rows-root data/issue_518/refusal --out eval_results/issue_518/refusal/bystander_logprob/logprob_results.json"
run uv run python scripts/issue518_syco_logprob_backfill.py \
    --teach-rows-root "$REPO_ROOT/data/issue_518/refusal" \
    --out "$EVAL_ROOT/refusal/bystander_logprob/logprob_results.json"

phase logprob_em "issue518_syco_logprob_backfill.py --teach-rows-root data/issue_518/em --out eval_results/issue_518/em/bystander_logprob/logprob_results.json"
run uv run python scripts/issue518_syco_logprob_backfill.py \
    --teach-rows-root "$REPO_ROOT/data/issue_518/em" \
    --out "$EVAL_ROOT/em/bystander_logprob/logprob_results.json"

phase logprob_syco_backfill "issue518_syco_logprob_backfill.py (#509 syco_arm)"
run uv run python scripts/issue518_syco_logprob_backfill.py

# ── H. Residual-stream bake-off (per arm, BASE model per §189) ─────────────
# The bake-off dispatch is provided by issue493_extraction_metric_bakeoff.py
# (cloned from #509). Plan §4.5 + §189 mandate ``--model-id Qwen/Qwen2.5-7B``.
#
# Round-9 fix: the bake-off's ``--arms`` flag is the positive-vs-locus
# extraction concept (choices = {pos, loc}) and conflating it with the #518
# behavior arms (refusal, em) is a CLI mismatch that argparse rejects with
# exit 2. The behavior arm is selected via ``--conditions-registry``, which
# points at the per-arm Condition list module
# (``i518_refusal_conditions`` or ``i518_em_conditions``); the worker's
# default ``--arms`` (pos + loc) is the right value to keep.
#
# Round-14 fix: split phase H into ``--phase extract`` followed by
# ``--phase metrics`` so the bake-off STOPS cleanly after the metrics-phase
# write (which produces ``meta.json`` + per-(point, layer, metric, variant)
# JSONs that scoring + the model-id consistency check consume), and SKIPS
# the regression + figures phases.
#
# Why split (not ``--phase all``): the regression phase (``run_regression``
# at line 3375 of issue493_extraction_metric_bakeoff.py) is hard-wired to
# read #474's 16-cond G_logprob_matrix via ``_load_G(arm, ep)`` at line 368
# -- whose keys are A1-A5/B1-B11. #518's conditions registry passes R1..R24
# (refusal) / E1..E24 (em), so ``G['R1']`` raises ``KeyError`` (the round-13
# crash after 11h of upstream work). The regression phase produces an
# intra-arm "winning predictor" leaderboard that the #518 Goal does NOT
# consume -- the headline Goal answer flows through ``issue509_scoring.py``
# -> ``issue518_cross_behavior_aggregator.py`` which reads ONLY
# ``$EVAL_ROOT/<arm>/bakeoff/metrics/`` + the per-arm
# ``predictor_comparison.json`` substrate (cosine/JS/KL/logprob), never the
# regression's ``bakeoff_grid.json``. The metrics-phase exit-0 at line 5380
# is the clean cut.
#
# Why split into two calls (not ``--phase metrics`` alone): ``--phase
# metrics`` SKIPS extraction (the ``if args.phase in ("all", "extract",
# "extraction")`` gate at line 5156) but then runs
# ``validate_canonical_completeness`` at line 5243 which raises if the
# canonical activation files don't exist. Pod-518's refusal bakeoff already
# completed extraction (and metrics) before the regression crashed, so a
# bare ``--phase metrics`` would work on refusal-resume but would FAIL on em
# (em never reached extraction -- the launcher set -e aborted at refusal's
# regression). The ``--phase extract`` -> ``--phase metrics`` split is
# symmetric and idempotent on BOTH arms: the model is loaded once per
# arm-phase pair, extraction writes canonicals (skipping already-existing
# per-cell partitions), metrics writes per-cell JSONs (skipping existing).
#
# Idempotency on resume: ``run_extraction_batched`` skips per-cond partition
# files whose canonical activation already exists; ``run_metrics`` (line
# 2989) skips per-cell metric JSONs whose output already exists. The
# metrics dir is preserved across the pod-518 stop/resume, so on resume
# refusal extract phase = no-op, refusal metrics phase = no-op, em extract
# phase = runs (em never started), em metrics phase = runs.
phase bakeoff_refusal_extract "issue493_extraction_metric_bakeoff.py --phase extract (refusal)"
run uv run python scripts/issue493_extraction_metric_bakeoff.py \
    --model-id "Qwen/Qwen2.5-7B" \
    --conditions-registry "explore_persona_space.experiments.i518_refusal_conditions" \
    --probe-pool-mode custom \
    --bakeoff-root "$EVAL_ROOT/refusal/bakeoff" \
    --phase extract

phase bakeoff_refusal "issue493_extraction_metric_bakeoff.py --phase metrics (refusal)"
run uv run python scripts/issue493_extraction_metric_bakeoff.py \
    --model-id "Qwen/Qwen2.5-7B" \
    --conditions-registry "explore_persona_space.experiments.i518_refusal_conditions" \
    --probe-pool-mode custom \
    --bakeoff-root "$EVAL_ROOT/refusal/bakeoff" \
    --phase metrics

# Round-15 Fix B: post-metrics baseline existence check — guards against a
# future --phase split silently dropping the next_token_js baseline cell
# (Round-14 regression: the writer was only inside --phase all + --merge-only
# call sites, and the launcher now drives extract+metrics).
test -s "$EVAL_ROOT/refusal/bakeoff/metrics/last_prompt__layer-1__next_token_js__raw.json" \
    || { echo "[FATAL] refusal bakeoff metrics dir is missing next_token_js baseline" >&2; exit 1; }

phase bakeoff_em_extract "issue493_extraction_metric_bakeoff.py --phase extract (em)"
run uv run python scripts/issue493_extraction_metric_bakeoff.py \
    --model-id "Qwen/Qwen2.5-7B" \
    --conditions-registry "explore_persona_space.experiments.i518_em_conditions" \
    --probe-pool-mode custom \
    --bakeoff-root "$EVAL_ROOT/em/bakeoff" \
    --phase extract

phase bakeoff_em "issue493_extraction_metric_bakeoff.py --phase metrics (em)"
run uv run python scripts/issue493_extraction_metric_bakeoff.py \
    --model-id "Qwen/Qwen2.5-7B" \
    --conditions-registry "explore_persona_space.experiments.i518_em_conditions" \
    --probe-pool-mode custom \
    --bakeoff-root "$EVAL_ROOT/em/bakeoff" \
    --phase metrics

# Round-15 Fix B (mirror): same baseline existence check for the em arm.
test -s "$EVAL_ROOT/em/bakeoff/metrics/last_prompt__layer-1__next_token_js__raw.json" \
    || { echo "[FATAL] em bakeoff metrics dir is missing next_token_js baseline" >&2; exit 1; }

# ── I. Substrate assembly (per arm: predictor_comparison.json) ─────────────
phase substrate_syco "issue518_build_predictor_substrate.py --arm syco"
# Round-12 fix: syco arm is INHERITED from #411 via the frozen 138-cell
# analyze_summary -- #509 never produced eval_results/issue_509/syco_arm/runs/
# (we read the matrix directly from the snapshot). Drop --runs-root, pass
# --syco-analyze-summary instead. The 17 #480 coarse-zoo fields ride in via
# --syco-predictor-comparison; completion_logprob via --logprob-file.
run uv run python scripts/issue518_build_predictor_substrate.py \
    --arm syco \
    --syco-predictor-comparison "$REPO_ROOT/eval_results/issue_480/_inputs/predictor_comparison.json" \
    --syco-analyze-summary "$REPO_ROOT/eval_results/issue_480/_inputs/syco_411_analyze_summary.json" \
    --logprob-file "$REPO_ROOT/eval_results/issue_509/syco_arm/bystander_logprob/logprob_results.json" \
    --out "$EVAL_ROOT/syco/_inputs/predictor_comparison.json"

phase substrate_refusal "issue518_build_predictor_substrate.py --arm refusal"
run uv run python scripts/issue518_build_predictor_substrate.py \
    --arm refusal \
    --cosine-sweep "$EVAL_ROOT/refusal/predictors/cosine_sweep.json" \
    --jskl-sweep   "$EVAL_ROOT/refusal/predictors/jskl_sweep.json" \
    --runs-root    "$EVAL_ROOT/refusal/runs" \
    --logprob-file "$EVAL_ROOT/refusal/bystander_logprob/logprob_results.json" \
    --out          "$EVAL_ROOT/refusal/_inputs/predictor_comparison.json"

phase substrate_em "issue518_build_predictor_substrate.py --arm em"
run uv run python scripts/issue518_build_predictor_substrate.py \
    --arm em \
    --cosine-sweep "$EVAL_ROOT/em/predictors/cosine_sweep.json" \
    --jskl-sweep   "$EVAL_ROOT/em/predictors/jskl_sweep.json" \
    --runs-root    "$EVAL_ROOT/em/runs" \
    --logprob-file "$EVAL_ROOT/em/bystander_logprob/logprob_results.json" \
    --out          "$EVAL_ROOT/em/_inputs/predictor_comparison.json"

# ── J. Per-arm scoring (Spearman ρ + permutation + cluster bootstrap) ──────
phase scoring_refusal "issue509_scoring.py --arm refusal"
run uv run python scripts/issue509_scoring.py \
    --metrics-dir "$EVAL_ROOT/refusal/bakeoff/metrics" \
    --arm refusal \
    --target-file "$EVAL_ROOT/refusal/_inputs/predictor_comparison.json" \
    --output      "$EVAL_ROOT/refusal/scoring.json"

phase scoring_em "issue509_scoring.py --arm em"
run uv run python scripts/issue509_scoring.py \
    --metrics-dir "$EVAL_ROOT/em/bakeoff/metrics" \
    --arm em \
    --target-file "$EVAL_ROOT/em/_inputs/predictor_comparison.json" \
    --output      "$EVAL_ROOT/em/scoring.json"

phase scoring_syco "issue509_scoring.py --arm syco"
run uv run python scripts/issue509_scoring.py \
    --metrics-dir "$REPO_ROOT/eval_results/issue_509/syco_arm/bakeoff/metrics" \
    --arm syco \
    --syco-logprob-backfill   "$REPO_ROOT/eval_results/issue_509/syco_arm/bystander_logprob/logprob_results.json" \
    --syco-coarse-zoo-backfill "$EVAL_ROOT/syco/_inputs/predictor_comparison.json" \
    --output "$REPO_ROOT/eval_results/issue_509/syco_arm/scoring.json"

# ── K. Cross-arm aggregator (headline same-sign + min |ρ| gate) ────────────
# Round-11 fix: the aggregator's `_assert_model_id_consistency` (lines 140-186
# of issue518_cross_behavior_aggregator.py) fails loud unless every arm's
# bake-off meta.json carries `args.model_id`. The refusal + EM bake-offs (phases
# H above) write `meta.json` directly inside their `--bakeoff-root`
# (issue493_extraction_metric_bakeoff.py line 5062: `_write_json_atomic(
# BAKEOFF_DIR / "meta.json", ...)`), so the canonical paths are:
#   $EVAL_ROOT/refusal/bakeoff/meta.json
#   $EVAL_ROOT/em/bakeoff/meta.json
# The syco arm is pre-staged from #509 per plan §10 and predates the
# `--model-id` patch, so it opts into `--allow-legacy-syco-meta`; the refusal
# + EM arms are still strictly required by the assertion.
phase cross_arm_aggregator "issue518_cross_behavior_aggregator.py"
run uv run python scripts/issue518_cross_behavior_aggregator.py \
    --syco-scoring         "$REPO_ROOT/eval_results/issue_509/syco_arm/scoring.json" \
    --refusal-scoring      "$EVAL_ROOT/refusal/scoring.json" \
    --em-scoring           "$EVAL_ROOT/em/scoring.json" \
    --allow-legacy-syco-meta \
    --refusal-bakeoff-meta "$EVAL_ROOT/refusal/bakeoff/meta.json" \
    --em-bakeoff-meta      "$EVAL_ROOT/em/bakeoff/meta.json" \
    --out                  "$EVAL_ROOT/cross_behavior_aggregator.json"

# ── Final sentinel + [phase=done] ──────────────────────────────────────────
SENTINEL_TS="$(date -u +%s)"
SENTINEL_PATH="$LOG_DIR/issue-518-epm_results-${SENTINEL_TS}.json"
PAYLOAD_FILE="$LOG_DIR/issue_518_results_payload.json"

# Build the payload (the per-arm scoring + cross-arm aggregator JSON paths).
uv run python - <<EOF > "$PAYLOAD_FILE"
import json, pathlib
EVAL_ROOT = pathlib.Path("$EVAL_ROOT")
SYCO_SCORING = pathlib.Path("$REPO_ROOT/eval_results/issue_509/syco_arm/scoring.json")
payload = {
    "refusal_scoring": str(EVAL_ROOT / "refusal" / "scoring.json"),
    "em_scoring":      str(EVAL_ROOT / "em" / "scoring.json"),
    "syco_scoring":    str(SYCO_SCORING),
    "cross_arm_aggregator": str(EVAL_ROOT / "cross_behavior_aggregator.json"),
    "predictor_comparison": {
        "refusal": str(EVAL_ROOT / "refusal" / "_inputs" / "predictor_comparison.json"),
        "em":      str(EVAL_ROOT / "em" / "_inputs" / "predictor_comparison.json"),
        "syco":    str(EVAL_ROOT / "syco" / "_inputs" / "predictor_comparison.json"),
    },
    "predictors": {
        "refusal_cosine_sweep": str(EVAL_ROOT / "refusal" / "predictors" / "cosine_sweep.json"),
        "refusal_jskl_sweep":   str(EVAL_ROOT / "refusal" / "predictors" / "jskl_sweep.json"),
        "em_cosine_sweep":      str(EVAL_ROOT / "em" / "predictors" / "cosine_sweep.json"),
        "em_jskl_sweep":        str(EVAL_ROOT / "em" / "predictors" / "jskl_sweep.json"),
    },
}
print(json.dumps(payload, indent=2))
EOF

uv run python - <<EOF
import json, os, pathlib, time
sentinel = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 518,
    "by": "launch_issue_518_production.sh",
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": pathlib.Path("$PAYLOAD_FILE").read_text(),
}
pathlib.Path("$SENTINEL_PATH").write_text(json.dumps(sentinel, indent=2))
print(f"[sentinel] wrote $SENTINEL_PATH (kind=epm:results, schema_v=1)")
EOF

phase done "sentinel=$SENTINEL_PATH"
