#!/usr/bin/env bash
# Issue #1092 P2-P3 GPU workload wrapper (GCE lane).
# Stages the P0 corpus from HF at the pinned revision into $OUT/corpus
# (scoped per-file staging — NEVER snapshot_download on the ~1M-file data
# repo), then exec's the work-conserving 8-GPU dispatcher.
set -euo pipefail

CORPUS_REV="7ef5523673d64697ab497577dbc5b9270c39f020"
RB_REV="037fcbb"
OUT="${OUT:-/workspace/issue1092}"
export EPS_OUT_DIR="$OUT"

mkdir -p "$OUT/corpus"

uv run python - <<'PY'
import os
import pathlib
import shutil

from huggingface_hub import hf_hub_download, list_repo_tree

REV = "7ef5523673d64697ab497577dbc5b9270c39f020"
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1092_realistic_crossing/corpus"

dst = pathlib.Path(os.environ["EPS_OUT_DIR"]) / "corpus"
dst.mkdir(parents=True, exist_ok=True)
names = []
for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=PREFIX, revision=REV):
    local = hf_hub_download(REPO, repo_type="dataset", filename=it.path, revision=REV)
    shutil.copy(local, dst / pathlib.Path(it.path).name)
    names.append(pathlib.Path(it.path).name)

required = {"manifest.jsonl", "prefix_store.jsonl", "query_store.jsonl", "derangement_map.json"}
missing = required - set(names)
assert not missing, f"staged corpus missing {missing}; got {sorted(names)}"
print(f"[dispatch] staged {len(names)} corpus files @ {REV[:12]}: {sorted(names)}")

# round-8.8: stage the P1 Claude completions (HF-verified 12,172/12,172) so
# the claude cells' up-front readiness check passes (launch 7: all 48
# claude-shard failures were this file never being staged). Idempotent.
claude_dst = pathlib.Path(os.environ["EPS_OUT_DIR"]) / "raw_completions" / "claude"
claude_dst.mkdir(parents=True, exist_ok=True)
claude_prefix = "issue1092_realistic_crossing/raw_completions/claude"
n_claude = 0
for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=claude_prefix):
    name = pathlib.Path(it.path).name
    if name.startswith("claude_completions") and name.endswith(".jsonl"):
        local = hf_hub_download(REPO, repo_type="dataset", filename=it.path)
        shutil.copy(local, claude_dst / name)
        n_claude += 1
assert n_claude >= 1, f"no claude_completions*.jsonl under {claude_prefix}"
print(f"[dispatch] staged {n_claude} claude completion file(s) -> {claude_dst}")
PY

# round-8.5: vLLM-on-H100 IMA mitigation (launch #4: CUDA illegal memory access
# in the engine step at production shapes under heavy shared-prefix reuse;
# A100-clean differential). Accept the modest gen slowdown; all 8 cells run
# under the same engine config, and G2 revalidates capture identity
# independently of the gen engine mode.
# round-8.9: EPS_GPU_PHASE_EXTRA_ARGS threads per-launch flags (word-split on
# purpose) — e.g. '--resume-code-shas <sha16>' to accept launch 8's completed
# cell_inst_own shards, or '--finalize-cells all' on a phase-scoped re-run.
# shellcheck disable=SC2086
exec uv run python scripts/issue1092_gpu_phase.py \
  --issue 1092 \
  --phases gen_instruct,gen_pretrained,capture_all,bare,dynamics \
  --corpus-rev "$CORPUS_REV" \
  --corpus-dir "$OUT/corpus" \
  --rb-rev "$RB_REV" \
  --no-prefix-caching \
  --enforce-eager \
  --out "$OUT" \
  ${EPS_GPU_PHASE_EXTRA_ARGS:-}
