#!/usr/bin/env bash
# Issue #1092 P6 (fit grids) on GCP cpu-bigmem — the v87 escape lane.
#
# The VM P6 pilot was earlyoom-killed at >=22.9 GB RSS (plan section-9 projected
# 8-10 GB; abort line 14 GB), so P6 routes to the plan's pre-registered escape:
# a dedicated n2-highmem-16 (128 GB). Same reviewed code (issue1092_p6_run.py,
# code-review v11 PASS); this driver only stages inputs from HF, invokes the
# wrapper (single judge-bearing run per the v82/v87 decision), and uploads the
# outputs before the instance's delete-on-exit teardown.
#
# GCE lane contract: cwd = $WORKLOAD_ROOT (the issue-1092 clone); HF_TOKEN etc.
# exported by the startup script; no .env file (source conditionally).
set -euo pipefail

if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# Dedicated box: full width (16 vCPU). The wrapper's shared-VM setdefault is 8;
# explicit env wins.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

CORPUS_REV="7ef5523673d64697ab497577dbc5b9270c39f020"   # same pin as issue1092_dispatch.sh
REPO="superkaiba1/explore-persona-space-data"

CORPUS_DIR="data/issue_1092/p0/corpus"
JUDGE_DIR="data/issue_1092/p5_judge"
OUT_DIR="data/issue_1092/p6"            # inside the clone: crash-persist sweeps data_issue_<N>
STAGE_DIR="/workspace/p6_stage"          # OUTSIDE the clone: re-downloadable cache, not crash-persisted

mkdir -p "$CORPUS_DIR" "$JUDGE_DIR" "$OUT_DIR" "$STAGE_DIR"

echo "[phase=p6_stage_inputs]"
uv run python - <<'PY'
import hashlib
import json
import os
import pathlib
import shutil

from huggingface_hub import hf_hub_download, list_repo_tree

REPO = "superkaiba1/explore-persona-space-data"

# 1) corpus at the pinned revision (recipe from issue1092_dispatch.sh)
REV = "7ef5523673d64697ab497577dbc5b9270c39f020"
PREFIX = "issue1092_realistic_crossing/corpus"
dst = pathlib.Path("data/issue_1092/p0/corpus")
names = []
for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=PREFIX, revision=REV):
    local = hf_hub_download(REPO, repo_type="dataset", filename=it.path, revision=REV)
    shutil.copy(local, dst / pathlib.Path(it.path).name)
    names.append(pathlib.Path(it.path).name)
required = {"manifest.jsonl", "prefix_store.jsonl", "query_store.jsonl", "derangement_map.json"}
missing = required - set(names)
assert not missing, f"staged corpus missing {missing}; got {sorted(names)}"
print(f"[p6-gce] staged {len(names)} corpus files @ {REV[:12]}")

# 2) P5 judge scores: shards + manifest -> reassemble -> sha256 verify
JPREFIX = "issue1092_realistic_crossing/p5_judge"
jdst = pathlib.Path("data/issue_1092/p5_judge")
shard_paths = []
manifest_local = None
for it in list_repo_tree(REPO, repo_type="dataset", path_in_repo=JPREFIX):
    name = pathlib.Path(it.path).name
    local = hf_hub_download(REPO, repo_type="dataset", filename=it.path)
    if name == "shards_manifest.json":
        manifest_local = local
    elif name.startswith("scores_shard_") and name.endswith(".jsonl"):
        shard_paths.append((name, local))
    elif name == "summary.json":
        shutil.copy(local, jdst / name)
assert manifest_local is not None, "shards_manifest.json missing under p5_judge"
man = json.load(open(manifest_local))
shard_paths.sort(key=lambda t: t[0])
assert len(shard_paths) == man["n_shards"], (
    f"shard count mismatch: hub {len(shard_paths)} vs manifest {man['n_shards']}"
)
scores = jdst / "scores.jsonl"
h = hashlib.sha256()
with open(scores, "wb") as out:
    for _, local in shard_paths:
        with open(local, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
                out.write(b)
digest = h.hexdigest()
assert digest == man["full_sha256"], (
    f"reassembled scores.jsonl sha mismatch: {digest} vs manifest {man['full_sha256']}"
)
assert scores.stat().st_size == man["total_bytes"]
print(f"[p6-gce] reassembled scores.jsonl OK: {man['total_bytes']} bytes sha256={digest[:16]}")
PY

echo "[phase=p6_fit]"
uv run python scripts/issue1092_p6_run.py \
  --corpus-dir "$CORPUS_DIR" \
  --stage-dir "$STAGE_DIR" \
  --out-dir "$OUT_DIR" \
  --judge-scores "$JUDGE_DIR/scores.jsonl" \
  --max-pilot-rss-gb 64

echo "[phase=p6_upload]"
uv run python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
res = api.upload_folder(
    folder_path="data/issue_1092/p6",
    path_in_repo="issue1092_realistic_crossing/p6",
    repo_id="superkaiba1/explore-persona-space-data",
    repo_type="dataset",
    commit_message="issue #1092 P6 fit-grid outputs (GCP cpu-bigmem judge-bearing run)",
)
print("[p6-gce] uploaded out-dir:", res)
api_files = [
    f
    for f in api.list_repo_files("superkaiba1/explore-persona-space-data", repo_type="dataset")
    if f.startswith("issue1092_realistic_crossing/p6/")
]
assert any(f.endswith("fit_grid_summary.json") for f in api_files), (
    f"fit_grid_summary.json not on hub after upload ({len(api_files)} files)"
)
print(f"[p6-gce] hub-verified {len(api_files)} files under issue1092_realistic_crossing/p6/")
PY

echo "[phase=done]"
