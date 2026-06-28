#!/usr/bin/env bash
# Issue #661 P3 resume driver — re-dispatch ONLY the activation extraction phase.
#
# Use when P0/P1 already ran (raw_completions on HF) AND P2 already ran
# (judge_filter.json on HF), but P3 (extract directions on GPU) never landed
# — the canonical case being the 2026-06-25 GCP Batch-judge wedge that left
# generation safe + judge complete, but no directions extracted. The full
# scripts/issue661_dispatch.sh would re-run P0 (regenerate instructions, risking
# SHA drift), P1 (regenerate ~14k rollouts on GPU), AND P2 (re-judge 14,400
# Sonnet-4.5 calls) before reaching P3 — all wasted spend.
#
# This script seeds the GPU instance from HF instead:
#   - data/issue_661/instructions_<behavior>.json (the SHA-pinned PV pairs)
#   - eval_results/issue_661/judge_filter.json (the P2 survivor set)
# then runs P3 + writes the same end-of-run sentinel the full dispatcher does.
#
# Dispatch:
#   uv run python scripts/dispatch_issue.py launch --issue 661 \
#     --intent lora-7b --backend gcp --repo-branch issue-661 \
#     --workload-cmd 'bash scripts/issue661_p3_resume.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

GPU_ID="${GPU_ID:-0}"
DEVICE="auto"
BEHAVIORS=(sycophancy refusal broad_em)

DATA_DIR="data/issue_661"
EVAL_DIR="eval_results/issue_661"
LOGDIR="/workspace/logs"
[ -d "$LOGDIR" ] || LOGDIR="$REPO_ROOT/logs"
mkdir -p "$LOGDIR" "$DATA_DIR" "$EVAL_DIR"

echo "[phase=p_seed] downloading P0/P2 artifacts from HF data repo"
uv run python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue661_rb_extraction_divergence"
BEHAVIORS = ["sycophancy", "refusal", "broad_em"]

def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    print(f"  {dst} -> {src.resolve()}")

for b in BEHAVIORS:
    p = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename=f"{PREFIX}/inputs/instructions_{b}.json",
        local_dir="data/issue_661/hf_dl",
    )
    _link(Path(p), Path(f"data/issue_661/instructions_{b}.json"))

p = hf_hub_download(
    repo_id=REPO, repo_type="dataset",
    filename=f"{PREFIX}/judge_filter.json",
    local_dir="data/issue_661/hf_dl",
)
_link(Path(p), Path("eval_results/issue_661/judge_filter.json"))
print("DONE seeding")
PY

echo "[phase=p3_extract] arm-A + arm-C + context-axis extraction"
uv run python scripts/issue661_extract_directions.py \
  --behaviors "${BEHAVIORS[@]}" --gpu-id "$GPU_ID" --device "$DEVICE" \
  --judge-filter "$EVAL_DIR/judge_filter.json" \
  --instructions-dir "$DATA_DIR" --out-dir "$EVAL_DIR"

SENTINEL="$LOGDIR/issue-661-epm_results-$(date +%s).json"
N_DIR=$(find "$EVAL_DIR/directions" -name 'r_b_*.pt' 2>/dev/null | wc -l | tr -d ' ')
cat > "$SENTINEL" <<JSON
{
  "sentinel_schema_version": 1,
  "kind": "epm:results",
  "version": 1,
  "task_id": 661,
  "by": "issue661_p3_resume",
  "note": "issue661 P3 resume complete: behaviors=${BEHAVIORS[*]}, directions=${N_DIR}, arms=A/B/C; directions on HF analysis_tensors/ (P3). P0/P1/P2 seeded from prior runs on HF (raw_completions + judge_filter); M1/M2/M3 analysis is off-pod (P5)."
}
JSON
echo "[issue661_p3_resume] wrote sentinel $SENTINEL"
echo "[phase=done] issue661 P3 resume complete (${BEHAVIORS[*]})"
