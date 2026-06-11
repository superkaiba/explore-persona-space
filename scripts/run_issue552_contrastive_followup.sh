#!/usr/bin/env bash
# Issue #552 — same-issue follow-up `contrastive-2x2-completion` driver (plan v5).
# Pod-side, 4x H100. Trains 6 fresh LoRAs (2 contrastive arms x 3 seeds, 2 waves
# of 3 for the MooseFS ~130 GB quota), runs the 3-context EM gate + the MF-A
# row-type CE diagnostic per wave, stages adapters, then runs the issue's
# established layer-14 extraction + SVD pipeline on 18 cells.
#
# Phases (plan v5 §4.3; each checkpoints its output before the next starts):
#   0   env preamble (OPENAI/HF/WANDB asserts) + preflight + corpus prep.
#   1   negatives data-gen + mix build (vLLM, 1 GPU) — builder checkpoints
#       negatives the moment generation completes; manifests sha256'd.
#   2   recipe smoke (2 steps x BOTH arms, production env: fence +
#       EPM_KEEP_LOCAL_ADAPTER=1 + upload_to=none) + recipe assert + TRL
#       fallback-warning assert + rendered-system-prompt assert + the BINDING
#       MF-D adapter-survival assert at the exact Phase-C staged layout.
#   3   wave-1 train: contrastive-EM x 3 seeds parallel (1 GPU each).
#   4   wave-1 3-context EM gate + row-type CE (base + 3 cells) + HF non-LFS
#       upload of gate/CE JSONs + merged-dir deletion (~45 GB freed).
#   5   wave-1 adapter staging at $FU/contrastive_em_seed{S}/adapter + sha256.
#   6-8 wave-2: same three phases on the contrastive-benign arm (CE base
#       numbers reused from Phase 4).
#   9   pipeline smoke = the PRODUCTION dispatcher, ONE cell
#       (--arms contrastive_em --seeds 1 --variants same) + schema asserts.
#   10  extraction production: 18 cells (2 arms x 3 seeds x 3 variants),
#       --save-per-question, C+D in one dispatcher invocation.
#   11  tensor durability: sha256 manifests + 18-file asserts (WandB artifact
#       + VM pull + sha256 -c verify are ORCHESTRATOR-side BEFORE termination
#       — the #521 lost-tensor fix; HF LFS upload deferred under the 403).
#   12  results sentinel (epm:results v4 via issue552_write_sentinel.py
#       --mode contrastive_done) + [phase=done]. Eval-JSON/log git commits to
#       issue-552 happen ORCHESTRATOR-side (pod never runs scripts/task.py).
#
# Wave-1 -> wave-2 sanity gate (plan §7 gate 3): wave 2 starts only if wave-1
# training exited 0 with merged dirs + adapters present AND the wave-1 gate
# produced judge scores. Behavioral THRESHOLDS are analysis rules, not run
# gates — extraction proceeds in all branches (plan §6.3).
#
# Smoke hooks (VM-side, no GPU):
#   EPM_CHECK_ENV_ONLY=1   env preamble only, then clean exit.
#
# Launch (experimenter, via nohup — plan §10 launch row):
#   nohup bash scripts/run_issue552_contrastive_followup.sh \
#     > /workspace/logs/issue-552-contrastive2x2.log 2>&1 &
#   echo $! > /workspace/logs/issue-552-contrastive2x2.pid
#
# This script NEVER shells out to scripts/task.py (CLAUDE.md rule); all
# marker posting happens VM-side via the sentinel + poll_pipeline.py.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ──────────────────────────────────────────────────────────────────────
# Phase 0a — env preamble. OPENAI_API_KEY asserted BEFORE any GPU work
# (gpt-4o judge dies hours in without it — plan §12 assumption 20).
# ──────────────────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
for key in OPENAI_API_KEY HF_TOKEN WANDB_API_KEY; do
  if [[ -z "${!key:-}" ]]; then
    echo "[phase=fail] $(date -Is) env preamble: required env var $key is empty/unset" >&2
    exit 11
  fi
done
echo "[phase=env_ok] $(date -Is) OPENAI/HF/WANDB keys non-empty"

if [[ "${EPM_CHECK_ENV_ONLY:-0}" == "1" ]]; then
  echo "[phase=done] EPM_CHECK_ENV_ONLY=1 — env preamble PASS, exiting"
  exit 0
fi

# No fail-loud HF persist (LFS quota 403 standing): adapters survive LOCALLY
# via EPM_KEEP_LOCAL_ADAPTER=1 (plan §4.1.7) + explicit staging in Phases 5/8.
unset EPM_PERSIST_ADAPTER_HF_REPO EPM_PERSIST_ADAPTER_SUBFOLDER
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export EPM_KEEP_LOCAL_ADAPTER=1

SEEDS=(42 137 256)
FU="${FU:-eval_results/issue_552/contrastive-2x2-completion}"
POD_LOG_DIR="${POD_LOG_DIR:-/workspace/logs}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_552}"
MIX_DIR="$REPO_ROOT/data/issue_552"
MAX_STEPS="${MAX_STEPS:-750}"
HF_DATA_REPO="superkaiba1/explore-persona-space-data"
mkdir -p "$LOG_DIR" "$POD_LOG_DIR" "$FU"

COND_EM="issue404_pair_turner_bad_medical_contrastive"
COND_BENIGN="issue404_pair_turner_good_medical_contrastive"
PAIR_EM="turner_bad_medical_contrastive"
PAIR_BENIGN="turner_good_medical_contrastive"
ARM_EM="contrastive_em"
ARM_BENIGN="contrastive_benign"

phase() {
  echo "[phase=$1] $(date -Is) $2"
}

fail_loud() {
  local rc="$1"; shift
  local reason="$*"
  phase fail "rc=$rc reason=$reason"
  exit "$rc"
}

# ──────────────────────────────────────────────────────────────────────
# Phase 0b — preflight + corpus prep (hard asserts: 5,899 rows each,
# zero pool-hash overlap; prep scripts are idempotent re-runs).
# ──────────────────────────────────────────────────────────────────────
phase preflight "orchestrate.preflight"
uv run python -m explore_persona_space.orchestrate.preflight \
  2>&1 | tee "$LOG_DIR/c2x2_preflight.log" || fail_loud "$?" "preflight_failed"

phase prep_corpora "bad + good corpus prep (5,899-row parity + pool-overlap asserts)"
uv run python scripts/issue_521_prep_turner_corpus.py \
  2>&1 | tee "$LOG_DIR/c2x2_prep_bad.log" || fail_loud "$?" "prep_bad_corpus_failed"
uv run python scripts/issue_552_prep_good_corpus.py \
  2>&1 | tee "$LOG_DIR/c2x2_prep_good.log" || fail_loud "$?" "prep_good_corpus_failed"

# ──────────────────────────────────────────────────────────────────────
# Phase 1 — negatives data-gen + mix build (vLLM, 1 GPU, ~0.5 h). The
# builder hard-asserts the #519 panel, 1:1 ratio, disjointness, at-index
# prompt identity, and identical negatives across arms; it checkpoints
# the negatives JSONL the moment generation completes. Durability: sha256
# manifest in git-trackable data/issue_552/; the WandB artifact
# (issue552_contrastive_mixes:v0) + VM pull are ORCHESTRATOR-side.
# ──────────────────────────────────────────────────────────────────────
phase build_mixes "on-policy negatives (greedy, max 512 tok) + 2 x 11,798-row mixes"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue_552_build_contrastive_mixes.py \
  2>&1 | tee "$LOG_DIR/c2x2_build_mixes.log" || fail_loud "$?" "mix_build_failed"
for f in contrastive_em_mix.jsonl contrastive_benign_mix.jsonl contrastive_mix_manifest.json; do
  [[ -f "$MIX_DIR/$f" ]] || fail_loud 21 "mix builder output missing: $MIX_DIR/$f"
done

# ──────────────────────────────────────────────────────────────────────
# Phase 2 — recipe smoke (2 steps x BOTH arms, PRODUCTION env) + asserts
# (plan §7 gate 1 — METHODOLOGY HALT GATE; any miss halts BEFORE wave 1).
# ──────────────────────────────────────────────────────────────────────
recipe_smoke_arm() {
  local COND="$1" ARM="$2" SMOKE_SEED=42 SMOKE_GPU=0
  local SMOKE_LOG="$LOG_DIR/c2x2_recipe_smoke_${ARM}.log"

  phase recipe_smoke "2-step production-env smoke train ($ARM / $COND)"
  CUDA_VISIBLE_DEVICES=$SMOKE_GPU uv run python scripts/train.py \
    condition="$COND" \
    training=turner_em lora=turner_em \
    +training.max_steps=2 \
    seed="$SMOKE_SEED" +gpu_id="$SMOKE_GPU" \
    upload_to=none \
    2>&1 | tee "$SMOKE_LOG" || fail_loud "$?" "recipe_smoke_train_failed_${ARM}"

  phase recipe_assert "run_result.json must record all 14 turner_em values verbatim ($ARM)"
  uv run python scripts/issue_521_em_recipe_smoke.py \
    --seed "$SMOKE_SEED" --condition "$COND" --expected-max-steps 2 \
    2>&1 | tee "$LOG_DIR/c2x2_recipe_assert_${ARM}.log" || \
    fail_loud "$?" "recipe_assert_failed_${ARM}_DO_NOT_PROCEED"

  # Effective-loss parity is load-bearing (plan §2): the pinned TRL 0.29.1
  # fallback warning MUST be in the train log (configured response-only ->
  # effective full-sequence CE, same as every reference cell).
  phase trl_fallback_assert "TRL DataCollatorForCompletionOnlyLM fallback warning present ($ARM)"
  grep -q "Falling back to full-sequence loss" "$SMOKE_LOG" || \
    fail_loud 22 "TRL fallback warning ABSENT in $SMOKE_LOG — effective loss surface drifted from the reference cells (lockfile change?)"

  # Persona gating reaches the trainer's actual data path: format_dataset's
  # rendered row must carry the medical_doctor system prompt verbatim.
  phase system_prompt_assert "rendered training row carries the persona system prompt ($ARM)"
  COND_ARM="$ARM" MIX_DIR="$MIX_DIR" uv run python - <<'PY'
import json
import os

from transformers import AutoTokenizer

from explore_persona_space.train.trainer import format_dataset

arm = os.environ["COND_ARM"]
mix = f"{os.environ['MIX_DIR']}/contrastive_{'em' if arm == 'contrastive_em' else 'benign'}_mix.jsonl"
personas = json.load(open("eval_results/issue_521/inputs/personas.json"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
ds = format_dataset(mix, tok)
texts_with_source = [t for t in ds["text"][:64] if personas["medical_doctor"] in t]
assert texts_with_source, (
    "none of the first 64 formatted rows carries the medical_doctor system prompt — "
    "persona gating is NOT reaching the trainer's chat-template path"
)
neg_hit = [
    t
    for t in ds["text"][:64]
    if any(personas[p] in t for p in ("assistant", "comedian", "police_officer", "software_engineer"))
]
assert neg_hit, "no negative-persona system prompt in the first 64 formatted rows"
print(f"system-prompt assert PASS: {len(texts_with_source)} source + {len(neg_hit)} negative rows in first 64")
PY

  # ── MF-D BINDING assert (plan §4.1.7 + §7 gate 1): the local adapter
  # survived _finalize_phase under the production env, at the
  # {phase_name}_adapter leaf; then the Phase-5 staging copy resolves at
  # the dispatcher's Phase-C path ($FU/{arm}_seed{S}/adapter).
  phase adapter_survival_assert "MF-D: local adapter survived the fence reap ($ARM seed=$SMOKE_SEED)"
  local CELL_DIR="$REPO_ROOT/models/${COND}_seed${SMOKE_SEED}"
  local ADAPTER_GLOB=("$CELL_DIR"/*_adapter)
  [[ -d "${ADAPTER_GLOB[0]}" ]] || \
    fail_loud 23 "MF-D FAIL: no *_adapter dir under $CELL_DIR — EPM_KEEP_LOCAL_ADAPTER gate did not preserve the adapter"
  (( ${#ADAPTER_GLOB[@]} == 1 )) || \
    fail_loud 24 "MF-D FAIL: expected exactly 1 *_adapter leaf under $CELL_DIR, found ${#ADAPTER_GLOB[@]}"
  local ADAPTER_DIR="${ADAPTER_GLOB[0]}"
  [[ -f "$ADAPTER_DIR/adapter_config.json" ]] || \
    fail_loud 25 "MF-D FAIL: adapter_config.json missing in $ADAPTER_DIR"
  [[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || \
    fail_loud 26 "MF-D FAIL: adapter_model.safetensors missing in $ADAPTER_DIR"

  phase staging_layout_assert "MF-D: staging copy resolves at the Phase-C layout ($ARM)"
  local DEST="$FU/${ARM}_seed${SMOKE_SEED}/adapter"
  rm -rf "$DEST"
  mkdir -p "$(dirname "$DEST")"
  cp -r "$ADAPTER_DIR" "$DEST"
  [[ -f "$DEST/adapter_config.json" ]] || \
    fail_loud 27 "MF-D FAIL: staged adapter_config.json does not resolve at $DEST (Phase-C path)"

  # The smoke's ~15 GB merged dir is not needed (the adapter is the staged
  # artifact); delete it so the 2-arm smoke doesn't eat into the wave-1
  # MooseFS headroom. Production trains rewrite the whole cell dir anyway.
  rm -rf "$CELL_DIR/sft_narrow_merged"
  phase recipe_smoke_ok "$ARM: recipe + TRL-fallback + system-prompt + MF-D adapter-survival all PASS"
}

recipe_smoke_arm "$COND_EM" "$ARM_EM"
recipe_smoke_arm "$COND_BENIGN" "$ARM_BENIGN"

# ──────────────────────────────────────────────────────────────────────
# Training wave + gate + CE + staging, parameterized per arm.
# ──────────────────────────────────────────────────────────────────────
train_wave() {
  local COND="$1" ARM="$2"
  phase wave_train "3-seed parallel train ($ARM, max_steps=$MAX_STEPS, GPUs 0/1/2)"
  local pids=()
  for i in 0 1 2; do
    local SEED="${SEEDS[$i]}"
    local CELL_LOG="$LOG_DIR/c2x2_train_${ARM}_seed${SEED}.log"
    (
      set -euo pipefail
      # Launcher-env CVD pin + matching +gpu_id (the in-process clobber
      # rewrites the SAME value — the #543 co-location gotcha).
      export CUDA_VISIBLE_DEVICES="$i"
      uv run python scripts/train.py \
        condition="$COND" \
        training=turner_em lora=turner_em \
        +training.max_steps="$MAX_STEPS" \
        seed="$SEED" +gpu_id="$i" \
        upload_to=none \
        >> "$CELL_LOG" 2>&1
      MERGED="$REPO_ROOT/models/${COND}_seed${SEED}/sft_narrow_merged"
      ADAPTER="$REPO_ROOT/models/${COND}_seed${SEED}/sft_narrow_adapter"
      [[ -f "$MERGED/config.json" ]] || { echo "[phase=cell_fail] seed=$SEED merged dir missing"; exit 17; }
      [[ -f "$ADAPTER/adapter_config.json" ]] || { echo "[phase=cell_fail] seed=$SEED adapter reaped despite EPM_KEEP_LOCAL_ADAPTER=1"; exit 18; }
    ) &
    pids+=($!)
  done
  local fail_any=0 pid rc
  for pid in "${pids[@]}"; do
    rc=0
    wait "$pid" || rc=$?
    if (( rc != 0 )); then
      phase cell_subprocess_failed "pid=$pid rc=$rc arm=$ARM"
      fail_any=1
    fi
  done
  (( fail_any == 0 )) || fail_loud 18 "wave_train_failed_${ARM}"
  phase wave_train_ok "$ARM: 3 cells trained; merged dirs + local adapters present"
}

gate_and_ce_wave() {
  local COND="$1" ARM="$2" PAIR="$3" MIX_FILE="$4" RUN_BASE_CE="$5"

  # 3-context EM gate (plan §4.3 Phase 4/7): contexts parallel on 3 GPUs.
  # Prompts come from the byte-identical personas.json entries.
  phase em_gate_3ctx "8x100 gate on 3 $ARM cells x 3 contexts (gpt-4o-2024-08-06)"
  local ASSISTANT_PROMPT SOURCE_PROMPT
  ASSISTANT_PROMPT="$(uv run python -c "import json; print(json.load(open('eval_results/issue_521/inputs/personas.json'))['assistant'], end='')")"
  SOURCE_PROMPT="$(uv run python -c "import json; print(json.load(open('eval_results/issue_521/inputs/personas.json'))['medical_doctor'], end='')")"

  local gpids=()
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=0
    EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO_ROOT/models" \
    uv run python scripts/issue404_outcome_eval.py \
      --pairs "$PAIR" --seeds "${SEEDS[@]}" \
      --skip-calibration --judge-model gpt-4o-2024-08-06 \
      --context-label none --gpu-id 0 \
      --output-base "$FU/em_gate/none" \
      > "$LOG_DIR/c2x2_gate_${ARM}_none.log" 2>&1
  ) &
  gpids+=($!)
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=1
    EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO_ROOT/models" \
    uv run python scripts/issue404_outcome_eval.py \
      --pairs "$PAIR" --seeds "${SEEDS[@]}" \
      --skip-calibration --judge-model gpt-4o-2024-08-06 \
      --system-prompt "$ASSISTANT_PROMPT" --context-label assistant --gpu-id 1 \
      --output-base "$FU/em_gate/assistant" \
      > "$LOG_DIR/c2x2_gate_${ARM}_assistant.log" 2>&1
  ) &
  gpids+=($!)
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=2
    EPM_ISSUE404_LOCAL_MERGED_BASE="$REPO_ROOT/models" \
    uv run python scripts/issue404_outcome_eval.py \
      --pairs "$PAIR" --seeds "${SEEDS[@]}" \
      --skip-calibration --judge-model gpt-4o-2024-08-06 \
      --system-prompt "$SOURCE_PROMPT" --context-label source --gpu-id 2 \
      --output-base "$FU/em_gate/source" \
      > "$LOG_DIR/c2x2_gate_${ARM}_source.log" 2>&1
  ) &
  gpids+=($!)
  local fail_any=0 pid rc
  for pid in "${gpids[@]}"; do
    rc=0
    wait "$pid" || rc=$?
    if (( rc != 0 )); then
      phase gate_subprocess_failed "pid=$pid rc=$rc arm=$ARM"
      fail_any=1
    fi
  done
  (( fail_any == 0 )) || fail_loud 31 "em_gate_failed_${ARM} (judge/vLLM error, NOT a gate decision; see $LOG_DIR/c2x2_gate_${ARM}_*.log)"
  local CTX SEED
  for CTX in none assistant source; do
    for SEED in "${SEEDS[@]}"; do
      [[ -f "$FU/em_gate/$CTX/outcome/${PAIR}_seed${SEED}_${CTX}.json" ]] || \
        fail_loud 32 "gate outcome JSON missing: $FU/em_gate/$CTX/outcome/${PAIR}_seed${SEED}_${CTX}.json"
    done
  done
  phase em_gate_ok "$ARM: 9 cell-context outcome JSONs present"

  # MF-A row-type CE diagnostic — BEFORE merged-dir deletion (merged dirs
  # are the cheap model load here). Base scored once (wave 1 only).
  phase rowtype_ce "MF-A per-row-type CE ($ARM; base_run=$RUN_BASE_CE)"
  mkdir -p "$FU/rowtype_ce"
  if [[ "$RUN_BASE_CE" == "yes" ]]; then
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue552_rowtype_ce.py \
      --label base --model "Qwen/Qwen2.5-7B-Instruct" \
      --mix "$MIX_DIR/contrastive_em_mix.jsonl" "$MIX_DIR/contrastive_benign_mix.jsonl" \
      --out "$FU/rowtype_ce" \
      2>&1 | tee "$LOG_DIR/c2x2_rowtype_ce_base.log" || fail_loud "$?" "rowtype_ce_base_failed"
  fi
  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue552_rowtype_ce.py \
      --label "${ARM}_seed${SEED}" \
      --model "$REPO_ROOT/models/${COND}_seed${SEED}/sft_narrow_merged" \
      --mix "$MIX_FILE" \
      --base-json "$FU/rowtype_ce/rowtype_ce_base.json" \
      --out "$FU/rowtype_ce" \
      2>&1 | tee "$LOG_DIR/c2x2_rowtype_ce_${ARM}_seed${SEED}.log" || \
      fail_loud "$?" "rowtype_ce_failed_${ARM}_seed${SEED}"
  done

  # Gate raw + judge + CE JSONs -> HF non-LFS (works under the LFS 403),
  # BEFORE merged-dir deletion (upload policy: raw before deletion).
  phase gate_raw_upload "gate raw completions + judge scores + CE JSONs -> HF non-LFS ($ARM)"
  FU="$FU" ARM="$ARM" PAIR="$PAIR" HF_DATA_REPO="$HF_DATA_REPO" uv run python - <<'PY'
"""Bulk-upload this wave's gate + CE artifacts; fail-loud verify the listing."""
import os
from pathlib import Path

from huggingface_hub import list_repo_files, upload_folder

repo = os.environ["HF_DATA_REPO"]
fu = Path(os.environ["FU"])
arm = os.environ["ARM"]
pair = os.environ["PAIR"]
expected: list[str] = []
for ctx in ("none", "assistant", "source"):
    src = fu / "em_gate" / ctx / "outcome"
    names = sorted(
        p.name
        for p in src.glob("*.json")
        if pair in p.name
    )
    assert len(names) >= 9, f"{ctx}: expected >=9 {pair} JSONs (3 raw + 3 judge + 3 outcome), found {names}"
    upload_folder(
        folder_path=str(src),
        path_in_repo=f"issue552_benign_control/contrastive_2x2/em_gate/{ctx}",
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"*{pair}*.json"],
        commit_message=f"#552 contrastive-2x2: {arm} gate artifacts ({ctx} context)",
    )
    expected.extend(f"issue552_benign_control/contrastive_2x2/em_gate/{ctx}/{n}" for n in names)
ce_src = fu / "rowtype_ce"
ce_names = sorted(p.name for p in ce_src.glob("rowtype_ce_*.json"))
upload_folder(
    folder_path=str(ce_src),
    path_in_repo="issue552_benign_control/contrastive_2x2/rowtype_ce",
    repo_id=repo,
    repo_type="dataset",
    allow_patterns=["rowtype_ce_*.json"],
    commit_message=f"#552 contrastive-2x2: row-type CE JSONs (through {arm} wave)",
)
expected.extend(f"issue552_benign_control/contrastive_2x2/rowtype_ce/{n}" for n in ce_names)
files = set(list_repo_files(repo, repo_type="dataset", revision="main"))
missing = [n for n in expected if n not in files]
assert not missing, f"gate/CE upload verification FAILED; missing on HF: {missing}"
print(f"gate/CE upload verified: {len(expected)} files")
PY

  phase merged_cleanup "$ARM: deleting 3 merged dirs (~45 GB; adapters survive locally)"
  for SEED in "${SEEDS[@]}"; do
    rm -rf "$REPO_ROOT/models/${COND}_seed${SEED}/sft_narrow_merged"
  done
}

stage_adapters_wave() {
  local COND="$1" ARM="$2"
  phase stage_adapters "$ARM: staging 3 local adapters at the Phase-C layout + sha256"
  local SEED
  for SEED in "${SEEDS[@]}"; do
    local SRC="$REPO_ROOT/models/${COND}_seed${SEED}/sft_narrow_adapter"
    local DEST="$FU/${ARM}_seed${SEED}/adapter"
    [[ -f "$SRC/adapter_config.json" ]] || \
      fail_loud 41 "stage_adapters: $SRC missing adapter_config.json (survival gate broke?)"
    rm -rf "$DEST"
    mkdir -p "$(dirname "$DEST")"
    cp -r "$SRC" "$DEST"
    [[ -f "$DEST/adapter_config.json" && -f "$DEST/adapter_model.safetensors" ]] || \
      fail_loud 42 "stage_adapters: staged copy incomplete at $DEST"
    phase stage_cell_ok "${ARM}_seed${SEED} staged at $DEST"
  done
  mkdir -p "$FU/adapters_manifests"
  ( cd "$FU" && find "${ARM}"_seed*/adapter -type f -print0 | sort -z | xargs -0 sha256sum ) \
    > "$FU/adapters_manifests/sha256sums_${ARM}.txt"
  phase stage_adapters_ok "$ARM: $(wc -l < "$FU/adapters_manifests/sha256sums_${ARM}.txt") adapter files hashed (WandB artifact issue552_contrastive_adapters:v0 + VM pull are orchestrator-side)"
}

# ── Phases 3-5: wave 1 (contrastive-EM) ──
train_wave "$COND_EM" "$ARM_EM"
gate_and_ce_wave "$COND_EM" "$ARM_EM" "$PAIR_EM" "$MIX_DIR/contrastive_em_mix.jsonl" yes
stage_adapters_wave "$COND_EM" "$ARM_EM"

# ── Wave-1 -> wave-2 sanity gate (plan §7 gate 3): decreasing train loss. ──
phase wave1_sanity "wave-1 -> wave-2 gate: per-seed train loss decreased + judge scores present"
LOG_DIR="$LOG_DIR" ARM="$ARM_EM" uv run python - <<'PY'
"""Wave-1 sanity: final logged loss < first logged loss, per seed (plan §7 gate 3)."""
import os
import re
from pathlib import Path

log_dir = Path(os.environ["LOG_DIR"])
arm = os.environ["ARM"]
loss_re = re.compile(r"'loss':\s*([0-9.]+)")
for seed in (42, 137, 256):
    text = (log_dir / f"c2x2_train_{arm}_seed{seed}.log").read_text(errors="replace")
    losses = [float(m) for m in loss_re.findall(text)]
    assert len(losses) >= 2, f"seed {seed}: <2 logged loss values — training telemetry missing"
    assert losses[-1] < losses[0], (
        f"seed {seed}: train loss did NOT decrease (first={losses[0]}, last={losses[-1]})"
    )
    print(f"seed {seed}: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} logs PASS")
PY

# ── Phases 6-8: wave 2 (contrastive-benign; base CE reused) ──
train_wave "$COND_BENIGN" "$ARM_BENIGN"
gate_and_ce_wave "$COND_BENIGN" "$ARM_BENIGN" "$PAIR_BENIGN" "$MIX_DIR/contrastive_benign_mix.jsonl" no
stage_adapters_wave "$COND_BENIGN" "$ARM_BENIGN"

# ──────────────────────────────────────────────────────────────────────
# Phase 9 — pipeline smoke = the PRODUCTION dispatcher, ONE cell
# (architectural parity: smoke IS sweep with --arms/--seeds subset).
# ──────────────────────────────────────────────────────────────────────
phase pipeline_smoke "one-cell smoke through the production dispatcher ($ARM_EM seed42, same variant)"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms "$ARM_EM" --seeds 1 --variants same \
  --save-per-question \
  --skip-phase a1 a23 b0_smoke b e \
  --output-dir "$FU" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus 1 \
  2>&1 | tee "$LOG_DIR/c2x2_pipeline_smoke.log" || fail_loud "$?" "pipeline_smoke_failed"

phase pipeline_smoke_assert "asserting smoke tensor schema (mean-resp + per-question) + SVD JSON"
ARM="$ARM_EM" FU="$FU" uv run python - <<'PY'
"""Smoke-cell asserts (plan §4.3 Phase 9): mean-resp + per-question keys, SVD schema."""
import json
import os
from pathlib import Path

import torch

arm = os.environ["ARM"]
fu = Path(os.environ["FU"])
pt = fu / "shifts" / f"same_{arm}_seed42.pt"
assert pt.exists(), f"smoke shift tensor missing: {pt}"
payload = torch.load(pt, map_location="cpu", weights_only=False)
shifts = payload["shifts"]
assert len(shifts) == 14, f"expected 14 personas, got {len(shifts)}"
for p_name, entry in shifts.items():
    assert entry["delta_v"].shape == (3584,), (p_name, entry["delta_v"].shape)
    assert "delta_v_per_question" in entry, f"per-question tensor missing for {p_name}"
    pq = entry["delta_v_per_question"]
    assert pq.dim() == 2 and pq.shape[1] == 3584, (p_name, pq.shape)
    assert "delta_v_mean_resp" in entry, (
        f"mean-over-response read missing for {p_name} — the "
        f"also_compute_mean_over_response_em gate did not fire for arm={arm}/variant=same"
    )
svd = fu / "svd" / f"same_{arm}_seed42.json"
assert svd.exists(), f"smoke SVD JSON missing: {svd}"
d = json.loads(svd.read_text())
assert len(d["U1"]) == 3584, len(d["U1"])
assert len(d["singular_values"]) == 14, len(d["singular_values"])
for key in ("s_top1_frac", "mean_cos_to_U1", "cos_to_U1", "sign_flip_p95", "row_shuffle_p95"):
    assert key in d, f"per-cell SVD JSON missing key {key!r} (parent schema mismatch)"
print("pipeline smoke asserts PASS: mean-resp + per-question + SVD schema all present")
PY

# ──────────────────────────────────────────────────────────────────────
# Phase 10 — extraction production: 18 cells, C+D in ONE invocation
# (the seed-42 same cell re-runs inside it — prior-rounds convention).
# ──────────────────────────────────────────────────────────────────────
phase extraction_production "Phase C+D: 18 cells (2 arms x 3 seeds x 3 variants), --save-per-question"
uv run python scripts/issue_519_dispatch.py --mode sweep \
  --arms "$ARM_EM" "$ARM_BENIGN" --seeds 3 --variants same base on_policy \
  --save-per-question \
  --skip-phase a1 a23 b0_smoke b e \
  --output-dir "$FU" \
  --personas-json eval_results/issue_521/inputs/personas.json \
  --questions-json eval_results/issue_521/inputs/questions.json \
  --base-cosines-json eval_results/issue_521/inputs/base_cosines.json \
  --n-gpus 4 \
  2>&1 | tee "$LOG_DIR/c2x2_extraction_production.log" || fail_loud "$?" "extraction_production_failed"

phase phase_d_assert "asserting 18 SVD JSONs + 18 tensors with mean-resp/per-question keys (same variant)"
FU="$FU" uv run python - <<'PY'
import json
import os
from pathlib import Path

import torch

fu = Path(os.environ["FU"])
missing = []
for arm in ("contrastive_em", "contrastive_benign"):
    for variant in ("same", "base", "on_policy"):
        for seed in (42, 137, 256):
            svd = fu / "svd" / f"{variant}_{arm}_seed{seed}.json"
            pt = fu / "shifts" / f"{variant}_{arm}_seed{seed}.pt"
            if not svd.exists():
                missing.append(str(svd))
            if not pt.exists():
                missing.append(str(pt))
assert not missing, f"Phase C/D output incomplete: {missing}"
# Same-variant cells must carry the mean-resp + per-question keys.
for arm in ("contrastive_em", "contrastive_benign"):
    for seed in (42, 137, 256):
        payload = torch.load(
            fu / "shifts" / f"same_{arm}_seed{seed}.pt", map_location="cpu", weights_only=False
        )
        for p_name, entry in payload["shifts"].items():
            assert "delta_v_mean_resp" in entry, (arm, seed, p_name)
            assert "delta_v_per_question" in entry, (arm, seed, p_name)
print("Phase D assert PASS: 18 SVD JSONs + 18 tensors; same-variant cells carry mean-resp + per-question")
PY

# ──────────────────────────────────────────────────────────────────────
# Phase 11 — tensor durability (plan §14): sha256 manifests + sidecar
# copies. HF LFS upload DEFERRED (quota 403); the WandB artifact
# (issue552_contrastive_shift_tensors:v0) + VM pull + `sha256sum -c`
# are ORCHESTRATOR-side BEFORE termination (#521 fix, fifth application).
# ──────────────────────────────────────────────────────────────────────
phase tensor_durability "sha256 manifests + sidecar copies for 18 tensors (HF upload DEFERRED: quota)"
mkdir -p "$FU/shifts_manifests"
for ARM in "$ARM_EM" "$ARM_BENIGN"; do
  for VARIANT in same base on_policy; do
    for SEED in "${SEEDS[@]}"; do
      PT="$FU/shifts/${VARIANT}_${ARM}_seed${SEED}.pt"
      SIDE="$FU/shifts/${VARIANT}_${ARM}_seed${SEED}.manifest.json"
      [[ -f "$PT" ]] || fail_loud 51 "tensor_durability: missing $PT"
      [[ -f "$SIDE" ]] || fail_loud 52 "tensor_durability: missing sidecar manifest $SIDE"
      cp "$SIDE" "$FU/shifts_manifests/"
    done
  done
done
( cd "$FU/shifts" && sha256sum ./*contrastive_*.pt ) > "$FU/shifts_manifests/sha256sums.txt"
phase tensor_durability_ok "$(wc -l < "$FU/shifts_manifests/sha256sums.txt") tensors hashed"

# ──────────────────────────────────────────────────────────────────────
# Phase 12 — end-of-run results sentinel (poll_pipeline.py contract;
# epm:results v4 — versions 1-3 are taken by the prior rounds and task.py
# does NOT auto-increment). Eval-JSON/log git commits to issue-552 are
# ORCHESTRATOR-side after this (pod-side code never runs scripts/task.py).
# ──────────────────────────────────────────────────────────────────────
phase write_sentinel "writing contrastive_done results sentinel (epm:results v4)"
uv run python scripts/issue552_write_sentinel.py --mode contrastive_done \
  --arm contrastive --followup-label "$(basename "$FU")" \
  --followup-dir "$FU" --seeds "${SEEDS[@]}" \
  --sentinel-dir "$POD_LOG_DIR" \
  2>&1 | tee "$LOG_DIR/c2x2_sentinel.log" || \
  fail_loud "$?" "results_sentinel_write_failed"

# poll_pipeline.py declares status="done" ONLY when the most recent
# [phase=...] line is [phase=done] — emit it AFTER the sentinel write.
phase done "issue-552 contrastive-2x2-completion complete (6 cells trained + 3-context gate + CE diagnostic + 18-cell extraction + durability manifests; WandB artifacts + VM pulls next, orchestrator-side)"
