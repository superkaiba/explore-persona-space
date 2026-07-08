#!/usr/bin/env bash
# Issue #1092 P2-P3 GPU workload wrapper (GCE lane).
# Stages the P0 corpus from HF at the pinned revision into $OUT/corpus
# (scoped per-file staging — NEVER snapshot_download on the ~1M-file data
# repo), then exec's the work-conserving 8-GPU dispatcher.
set -euo pipefail

CORPUS_REV="45b222d97356ca9ac5d82901267add65f785ff9f"
RB_REV="037fcbb"
OUT="${OUT:-/workspace/issue1092}"
export EPS_OUT_DIR="$OUT"

mkdir -p "$OUT/corpus"

uv run python - <<'PY'
import os
import pathlib
import shutil

from huggingface_hub import hf_hub_download, list_repo_tree

REV = "45b222d97356ca9ac5d82901267add65f785ff9f"
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
PY

exec uv run python scripts/issue1092_gpu_phase.py \
  --issue 1092 \
  --phases gen_instruct,gen_pretrained,capture_all,bare,dynamics \
  --corpus-rev "$CORPUS_REV" \
  --corpus-dir "$OUT/corpus" \
  --rb-rev "$RB_REV" \
  --out "$OUT"
