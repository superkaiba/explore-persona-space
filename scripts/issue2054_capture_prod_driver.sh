#!/usr/bin/env bash
# issue-2054 capture production driver (pod-side, 2x GPU) — cells (a),(b),(d).
#
# Teacher-forced layer-19 capture over the r14 form matrix, per model x condition:
#   inserted  — reads data/issue_2054/spliced_inserted/ (shared; model is read-side)
#   on_policy — reads data/issue_2054/on_policy/<model>/ (model-matched, composer map)
# Story forms (attrib_quoted, bare_label): all 5 variants, 2-way variant shard.
# Assistant-only forms (chat, bare_text): run CONCURRENTLY, one GPU each.
# target_conv_ids stays 0 (ALL rows): no prefix-cap mismatch risk, maximal
# conv_id intersection; fits equalize down on the intersection (plan req 8).
# Cell (c) capture (condition cell_c) is Phase-D-gated — NOT dispatched here.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_CAP_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
A=conversation_paired_stories_assistant

echo "[phase=capture_prod] driver start $(date -u +%FT%TZ)"

# Stage the inserted-condition inputs from HF (Phase B ran on the VM — the
# cross-machine seam, #1482 class: the pod clone carries no data/). The
# spliced files are model-INDEPENDENT (deterministic splice, no --model
# axis), so the shared issue2054_lattice/spliced_inserted/ prefix is
# canonical. Idempotent: existing non-empty targets skip. Own log so no
# child token can reach this dispatcher's main log.
echo "[phase=capture_prod stage=stage_inserted] start $(date -u +%FT%TZ)"
uv run python - > "$LOG_DIR/issue-2054-cap-stage-inserted.log" 2>&1 <<'PYEOF'
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from huggingface_hub import HfApi

from explore_persona_space.orchestrate.hub import (
    list_hf_files_under_path,
    retry_transient,
    stage_hub_file,
)

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2054_lattice/spliced_inserted"
DEST = Path("data/issue_2054/spliced_inserted")

api = HfApi()
files = retry_transient(
    lambda: list_hf_files_under_path(api, REPO, PREFIX, repo_type="dataset"),
    what=f"list({PREFIX})",
)
if not files:
    raise RuntimeError(f"no files under {PREFIX} — cannot stage inserted inputs")
n_staged = 0
for f in sorted(files):
    rel = f[len(PREFIX) + 1 :]
    target = DEST / rel
    if target.is_file() and target.stat().st_size > 0:
        print(f"[stage] skip existing {rel}", flush=True)
        continue
    stage_hub_file(REPO, f, target, repo_type="dataset")
    n_staged += 1
    print(f"[stage] staged {rel}", flush=True)
print(f"[stage] spliced_inserted staged: {n_staged} new / {len(files)} total", flush=True)
PYEOF
rc=$?
echo "[phase=capture_prod stage=stage_inserted] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=capture_prod] HALT stage_inserted rc=${rc} (tail follows)"
  tail -20 "$LOG_DIR/issue-2054-cap-stage-inserted.log" || true
  exit "$rc"
fi

for MODEL in qwen2.5-7b-instruct qwen2.5-7b; do
  for COND in inserted on_policy; do
    for FORM in attrib_quoted bare_label; do
      echo "[phase=capture_prod model=${MODEL} cond=${COND} form=${FORM}] start $(date -u +%FT%TZ)"
      uv run python scripts/issue2054_shard_launch.py \
        --driver capture --condition "$COND" --form "$FORM" --model "$MODEL" --gpus 0,1
      rc=$?
      echo "[phase=capture_prod model=${MODEL} cond=${COND} form=${FORM}] rc=${rc} $(date -u +%FT%TZ)"
      if [ "$rc" -ne 0 ]; then
        echo "[phase=capture_prod] HALT ${MODEL}/${COND}/${FORM} rc=${rc}"
        exit "$rc"
      fi
    done

    echo "[phase=capture_prod model=${MODEL} cond=${COND} form=chat+bare_text] concurrent start $(date -u +%FT%TZ)"
    uv run python scripts/issue2054_shard_launch.py \
      --driver capture --condition "$COND" --form chat --model "$MODEL" --gpus 0 --variants "$A" \
      > "$LOG_DIR/issue-2054-cap-chat-${COND}-${MODEL}.log" 2>&1 &
    P1=$!
    uv run python scripts/issue2054_shard_launch.py \
      --driver capture --condition "$COND" --form bare_text --model "$MODEL" --gpus 1 --variants "$A" \
      > "$LOG_DIR/issue-2054-cap-baretext-${COND}-${MODEL}.log" 2>&1 &
    P2=$!
    wait "$P1"; R1=$?
    wait "$P2"; R2=$?
    echo "[phase=capture_prod model=${MODEL} cond=${COND} form=chat] rc=${R1}; form=bare_text rc=${R2} $(date -u +%FT%TZ)"
    if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
      echo "[phase=capture_prod] HALT ${MODEL}/${COND} chat rc=${R1} bare_text rc=${R2} (tails follow)"
      tail -30 "$LOG_DIR/issue-2054-cap-chat-${COND}-${MODEL}.log" || true
      tail -30 "$LOG_DIR/issue-2054-cap-baretext-${COND}-${MODEL}.log" || true
      exit 1
    fi
  done
done

echo "[phase=capture_prod] driver_rc=0 $(date -u +%FT%TZ)"
