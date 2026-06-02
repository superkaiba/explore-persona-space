#!/usr/bin/env bash
# task #466 — slice-aware leakage predictor pipeline (plan v3 §4.2 / §4.4)
#
# UNIFIED smoke = pipeline with --smoke-* flags. The dispatcher entrypoints
# are identical between smoke and full; smoke just shrinks the work.
#
# Phases (poll_pipeline.py PHASE_RE matches /\[phase=[a-z_]+/):
#   [phase=setup_port]        — port branch-only files + marker assert
#   [phase=setup_train_data]  — download pinned training data (Phase 0 prereq)
#   [phase=phase0_retrain]    — re-train the #432 marker LoRA from scratch
#   [phase=reproduce_check]   — verify retrain reproduces #456 (>= 0.80 emission)
#   [phase=smoke_premise]     — tiny premise check (smoke only)
#   [phase=premise_step_a]    — full premise gate (5 personas × 2 slices)
#   [phase=smoke_predictors]  — tiny predictor smoke (smoke only)
#   [phase=predictors]        — JS + cosine predictors
#   [phase=smoke_marker]      — tiny marker smoke (smoke only)
#   [phase=marker_logp]       — on-policy marker log-p (Phase A + B)
#   [phase=analyze]           — matched-contrast table + figures
#   [phase=done]              — TERMINAL marker; poll_pipeline.py reads this as success
#
# End-of-run sentinel written to:
#   /workspace/logs/issue-466-epm_results-<epoch>.json
# Required keys: sentinel_schema_version=1, kind=epm:results, version=1
# (see _SENTINEL_REQUIRED_KEYS in scripts/poll_pipeline.py).

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────
SMOKE=0
SKIP_PHASE0=0
SMOKE_PROMPTS=3
SMOKE_SAMPLES=2
R_SAMPLES=8
MAX_NEW_TOKENS_PREDICTOR=256
MAX_NEW_TOKENS_MARKER=1536

usage() {
  cat <<EOF
Usage: $0 [options]
  --smoke                 Run the unified smoke pipeline (tiny data, skip phase0_retrain by default)
  --no-phase0             Skip the Phase 0 retrain step (e.g. for resumed runs)
  --smoke-prompts N       Smoke: prompts per cell  (default 3)
  --smoke-samples N       Smoke: samples per prompt (default 2)
  --r-samples N           Production: JS R= per persona per probe (default 8)
  -h | --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; SKIP_PHASE0=1; shift ;;
    --no-phase0) SKIP_PHASE0=1; shift ;;
    --smoke-prompts) SMOKE_PROMPTS="$2"; shift 2 ;;
    --smoke-samples) SMOKE_SAMPLES="$2"; shift 2 ;;
    --r-samples) R_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# ── Env setup ─────────────────────────────────────────────────────────────
# uv run python does NOT auto-load .env; tools below (HF Hub upload, vLLM
# from-pretrained, WandB) need HF_TOKEN / WANDB_API_KEY in the env. The
# Python entrypoints all call load_dotenv() at module top, but the shell
# `source .env` here makes sure subprocess inheritance carries the values
# even when a child fork forgets to load.
if [[ -f .env ]]; then
  set +u  # .env may reference vars that aren't set yet
  source .env  # shellcheck disable=SC1091
  set -u
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD="${EPM_SKIP_INLINE_CHECKPOINT_UPLOAD:-1}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
mkdir -p "$LOG_ROOT"

# Adapter persist destination (read by trainer.py via the env vars).
ADAPTER_REPO="superkaiba1/explore-persona-space"
ADAPTER_SUBFOLDER="issue466_i432_marker_se_9neg_zen_seed42_step1600"

echo "[phase=setup_port] $(date -u +%FT%TZ) — porting branch-only files"
# Already ported as part of this commit; we re-assert the files are present
# rather than re-checkout (which would mutate the working tree mid-run).
required_files=(
  scripts/eval_i456_onpolicy_emission.py
  scripts/_i416_bystander_panel.py
  scripts/build_i432_9neg_dataset.py
  configs/condition/i432_software_engineer_marker_9neg_zen.yaml
  scripts/issue466_personas.py
  scripts/issue466_step_a_premise.py
  scripts/issue466_predictors.py
  scripts/issue466_marker_logp.py
  scripts/issue466_analyze.py
)
for f in "${required_files[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING required file: $f" >&2
    exit 3
  fi
done
# Hard marker-id assert (early-fail before any GPU work).
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
ids = tok.encode('※', add_special_tokens=False)
assert ids == [63680], f'MARKER guard FAILED: ids={ids}, expected [63680]'
print('marker ※ -> [63680] OK')
"

if [[ "$SKIP_PHASE0" -eq 0 ]]; then
  echo "[phase=setup_train_data] $(date -u +%FT%TZ) — downloading pinned training data"
  uv run python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='superkaiba1/explore-persona-space-data',
    repo_type='dataset',
    filename='leakage/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl',
    revision='3569bad4b24b0591c318c59c4d835de1d8ac14f8',
)
print(f'Downloaded: {path}')
# Row-count assert (2000 = 200 positive + 1800 negative across 9 personas).
with open(path) as f:
    n = sum(1 for _ in f)
assert n == 2000, f'expected 2000 rows in pinned dataset, got {n}'
print(f'Row count OK: {n}')
# Place into the Hydra-expected local path so the condition yaml resolves it.
import shutil
from pathlib import Path
dest = Path('data/leakage_experiment/marker_software_engineer_asst_excluded_medium_9ca040_9neg.jsonl')
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(path, dest)
print(f'Staged at {dest}')
"

  echo "[phase=phase0_retrain] $(date -u +%FT%TZ) — re-training the #432 marker LoRA"
  EPM_PERSIST_ADAPTER_HF_REPO="$ADAPTER_REPO" \
  EPM_PERSIST_ADAPTER_SUBFOLDER="$ADAPTER_SUBFOLDER" \
  EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 \
  uv run python scripts/train.py \
    condition=i432_software_engineer_marker_9neg_zen \
    seed=42 \
    ++training.learning_rate=1.0e-5 \
    ++training.max_steps=1600 \
    upload_to=none \
    'eval.periodic_eval.enabled=false' \
    2>&1 | tee "$LOG_ROOT/issue-466-phase0.log"

  echo "[phase=reproduce_check] $(date -u +%FT%TZ) — verifying retrain reproduces #456"
  uv run python -c "
import json
from pathlib import Path
from huggingface_hub import list_repo_files
files = list_repo_files('${ADAPTER_REPO}', repo_type='model', revision='main')
matches = [f for f in files if f.startswith('${ADAPTER_SUBFOLDER}/') and f.endswith('adapter_model.safetensors')]
assert matches, f'No adapter_model.safetensors under ${ADAPTER_SUBFOLDER}/ on HF; persist failed.'
print(f'Adapter persist verified: {matches[0]}')
"
  # On-policy emission check via the ported eval rig (small slice).
  # NOTE: a full eval_i456 invocation walks the 22-checkpoint schedule;
  # we want a single-checkpoint emission-rate read on the just-trained
  # step-1600 LoRA, so we use issue466_marker_logp.py in --skip-marker-rescore
  # mode (Phase A only) on the source persona + 20 nontrigger probes ×
  # n=8. Emission rate is read from the JSON it writes.
  uv run python scripts/issue466_marker_logp.py \
    --adapter-repo "$ADAPTER_REPO" \
    --adapter-subfolder "$ADAPTER_SUBFOLDER" \
    --n-samples 8 --max-new-tokens 1536 \
    --smoke-probes 20 --skip-marker-rescore \
    2>&1 | tee "$LOG_ROOT/issue-466-reproduce.log"
  S_RATE=$(uv run python -c "
import json
with open('eval_results/issue_466/onpolicy_gen/S_nontrigger.json') as f:
    d = json.load(f)
print(d['emission_rate'])
")
  PASS_REPRO=$(uv run python -c "print(int(${S_RATE} >= 0.80))")
  if [[ "$PASS_REPRO" -ne 1 ]]; then
    # Pod-side: cannot shell out to task.py (branch-guarded to main).
    # Write a sentinel; the orchestrator's poll_pipeline.py parses it and
    # posts epm:failure on the VM side.
    ts=$(date +%s)
    cat > "$LOG_ROOT/issue-466-epm_failure-${ts}.json" <<JSON
{
  "sentinel_schema_version": 1,
  "kind": "epm:failure",
  "version": 1,
  "ts": ${ts},
  "note": "failure_class: data\nreason: retrain_did_not_reproduce_456\nachieved_emission_rate: ${S_RATE}\nrequired_min: 0.80\ngate: reproduce_check"
}
JSON
    echo "[phase=done] $(date -u +%FT%TZ) — reproduce_check FAIL (${S_RATE} < 0.80); halting"
    exit 4
  fi
  echo "[phase=reproduce_check] PASS — S emission_rate=${S_RATE} >= 0.80"
fi

# ── Smoke phases (only when --smoke) ──────────────────────────────────────
if [[ "$SMOKE" -eq 1 ]]; then
  echo "[phase=smoke_premise] $(date -u +%FT%TZ) — premise gate on tiny slice"
  uv run python scripts/issue466_step_a_premise.py \
    --smoke-prompts "$SMOKE_PROMPTS" --smoke-samples "$SMOKE_SAMPLES" \
    2>&1 | tee "$LOG_ROOT/issue-466-smoke-premise.log"

  echo "[phase=smoke_predictors] $(date -u +%FT%TZ) — predictors on tiny slice"
  uv run python scripts/issue466_predictors.py \
    --smoke-probes "$SMOKE_PROMPTS" --r-samples "$SMOKE_SAMPLES" \
    --layers 21 --no-recipe-b \
    2>&1 | tee "$LOG_ROOT/issue-466-smoke-predictors.log"

  echo "[phase=smoke_marker] $(date -u +%FT%TZ) — marker dispatcher entrypoint check"
  # The smoke path runs Phase A only (no Phase B) if no adapter has been
  # persisted yet; the --skip-marker-rescore flag is the documented path.
  uv run python scripts/issue466_marker_logp.py \
    --adapter-repo "$ADAPTER_REPO" --adapter-subfolder "$ADAPTER_SUBFOLDER" \
    --smoke-probes "$SMOKE_PROMPTS" --n-samples "$SMOKE_SAMPLES" \
    --max-new-tokens 256 --skip-marker-rescore \
    2>&1 | tee "$LOG_ROOT/issue-466-smoke-marker.log" || true

  echo "[phase=analyze] $(date -u +%FT%TZ) — smoke analyze (best-effort)"
  uv run python scripts/issue466_analyze.py || true

  TS=$(date +%s)
  cat > "$LOG_ROOT/issue-466-epm_results-${TS}.json" <<JSON
{
  "sentinel_schema_version": 1,
  "kind": "epm:results",
  "version": 1,
  "ts": ${TS},
  "note": "phase: smoke complete\nsmoke_prompts: ${SMOKE_PROMPTS}\nsmoke_samples: ${SMOKE_SAMPLES}"
}
JSON
  echo "[phase=done] $(date -u +%FT%TZ) — SMOKE complete"
  exit 0
fi

# ── Full pipeline ─────────────────────────────────────────────────────────
echo "[phase=premise_step_a] $(date -u +%FT%TZ) — premise gate (full)"
uv run python scripts/issue466_step_a_premise.py \
  --prompts-per-cell 10 --samples-per-prompt 4 \
  2>&1 | tee "$LOG_ROOT/issue-466-premise.log"

# Check the premise gate verdict — if FAIL, write epm:failure sentinel + exit.
PREMISE_PASS=$(uv run python -c "
import json
with open('eval_results/issue_466/premise/step_a.json') as f:
    d = json.load(f)
print(int(d['passes_premise']))
")
if [[ "$PREMISE_PASS" -ne 1 ]]; then
  ts=$(date +%s)
  FAIL_DETAIL=$(uv run python -c "
import json
with open('eval_results/issue_466/premise/step_a.json') as f:
    d = json.load(f)
print('\\n'.join(d['fail_rules']))
")
  cat > "$LOG_ROOT/issue-466-epm_failure-${ts}.json" <<JSON
{
  "sentinel_schema_version": 1,
  "kind": "epm:failure",
  "version": 1,
  "ts": ${ts},
  "note": "failure_class: data\nreason: premise_step_a_failed\nfail_rules:\n${FAIL_DETAIL}"
}
JSON
  echo "[phase=done] $(date -u +%FT%TZ) — premise_step_a FAIL; halting"
  exit 5
fi
echo "[phase=premise_step_a] PASS"

echo "[phase=predictors] $(date -u +%FT%TZ) — JS + cosine predictors"
uv run python scripts/issue466_predictors.py \
  --r-samples "$R_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS_PREDICTOR" \
  --layers 7 14 21 27 \
  2>&1 | tee "$LOG_ROOT/issue-466-predictors.log"

echo "[phase=marker_logp] $(date -u +%FT%TZ) — on-policy marker log-p"
uv run python scripts/issue466_marker_logp.py \
  --adapter-repo "$ADAPTER_REPO" --adapter-subfolder "$ADAPTER_SUBFOLDER" \
  --n-samples 8 --max-new-tokens "$MAX_NEW_TOKENS_MARKER" \
  2>&1 | tee "$LOG_ROOT/issue-466-marker.log"

echo "[phase=analyze] $(date -u +%FT%TZ) — matched-contrast table + figures"
uv run python scripts/issue466_analyze.py 2>&1 | tee "$LOG_ROOT/issue-466-analyze.log"

# ── End-of-run sentinel ───────────────────────────────────────────────────
TS=$(date +%s)
SUMMARY=$(uv run python -c "
import json
from pathlib import Path
p = Path('eval_results/issue_466/matched_contrast_table.json')
if not p.exists():
    print('table_missing')
else:
    with open(p) as f:
        d = json.load(f)
    rows = d['rows']
    parts = [f\"{r['behavior'][:1]}-{r['slice_kind'][:4]}: dlogp={r.get('delta_marker_logp_normed')}\" for r in rows]
    print('; '.join(parts))
")
cat > "$LOG_ROOT/issue-466-epm_results-${TS}.json" <<JSON
{
  "sentinel_schema_version": 1,
  "kind": "epm:results",
  "version": 1,
  "ts": ${TS},
  "note": "phase: full pipeline complete\nartifacts: eval_results/issue_466/\nsummary: ${SUMMARY}"
}
JSON

echo "[phase=done] $(date -u +%FT%TZ) — pipeline complete; sentinel at $LOG_ROOT/issue-466-epm_results-${TS}.json"
