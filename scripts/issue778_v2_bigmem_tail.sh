#!/usr/bin/env bash
# Issue #778 v2rerun — cpu-bigmem TAIL: maxlayer + figures off the shared VM.
#
# The shared VM's earlyoom kept SIGTERMing the 15-22 GiB maxlayer stage
# (3 kills 2026-07-03); this script runs the remaining ladder tail on a
# dedicated n2-highmem-16 (128 GB, no competing sessions). It stages ALL
# inputs from HF (the pod v2 bundle + the VM-produced rb_v2/judge/pairing/
# maxdraws pre-staged 2026-07-03 + the fixed-stage JSONs from the scratch
# prefix), runs maxlayer (resumes past banked cells) + figures, and uploads
# results BACK to the scratch prefix. It NEVER writes MANIFEST.json — the
# #816 consumption signal is published by the VM's final upload phase only.
set -euo pipefail
cd "${WORKLOAD_ROOT:-$(pwd)}" 2>/dev/null || true

OUT_ROOT="data/issue_778"
EVAL_ROOT="eval_results/issue_778"
LABEL_DIR="$EVAL_ROOT/faithful-extraction-honest-nulls-rerun"
MAXDRAWS_ROOT="$OUT_ROOT/v2/honest_nulls_maxdraws_v2"
TRAITS="evil sycophancy hallucination"
SETTINGS="finetune monitoring_corrected monitoring_manyshot"

echo "[phase=prefetch] staging v1 + v2 bundle (incl. pre-staged VM outputs)"
uv run python scripts/issue778_v2_prefetch.py \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" --fetch-v2

echo "[phase=scratch_download] fixed-stage JSONs from v2_scratch_fixed_jsons"
uv run python - <<'PY'
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from pathlib import Path
import shutil
REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue778_persona_vectors/v2_scratch_fixed_jsons"
dest = Path("eval_results/issue_778/faithful-extraction-honest-nulls-rerun")
dest.mkdir(parents=True, exist_ok=True)
entries = [e for e in HfApi().list_repo_tree(REPO, path_in_repo=PREFIX, repo_type="dataset", recursive=True) if isinstance(e, RepoFile)]
assert entries, f"no files under {PREFIX} — pivot payload missing"
for e in entries:
    p = hf_hub_download(REPO, e.path, repo_type="dataset")
    shutil.copy(p, dest / Path(e.path).name)
    print("staged", Path(e.path).name)
PY

echo "[phase=honest_nulls_v2_maxlayer] start draws=10000 (bigmem tail)"
uv run python scripts/issue778_honest_null_ladder.py \
  --rb-version v2 --stage maxlayer \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" \
  --maxdraws-root "$MAXDRAWS_ROOT" \
  --draws 10000 --draws-orig 1000 \
  --traits $TRAITS --settings $SETTINGS

echo "[phase=figures] v2 extra figures"
uv run python scripts/issue778_v2_extra_figures.py \
  --out-root "$OUT_ROOT" --eval-results-root "$EVAL_ROOT" --traits $TRAITS

echo "[phase=scratch_upload] results back to scratch (NO MANIFEST)"
uv run python - <<'PY'
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
api = HfApi()
REPO = "superkaiba1/explore-persona-space-data"
api.upload_folder(folder_path="eval_results/issue_778/faithful-extraction-honest-nulls-rerun",
                  repo_id=REPO, repo_type="dataset",
                  path_in_repo="issue778_persona_vectors/v2_scratch_tail_results/eval_jsons",
                  commit_message="issue778 v2rerun bigmem tail: final ladder JSONs")
api.upload_folder(folder_path="data/issue_778/v2/honest_nulls_maxdraws_v2",
                  repo_id=REPO, repo_type="dataset",
                  path_in_repo="issue778_persona_vectors/v2_scratch_tail_results/honest_nulls_maxdraws_v2",
                  commit_message="issue778 v2rerun bigmem tail: updated maxdraws columns")
import os
figdir = "figures/issue_778/faithful-extraction-honest-nulls-rerun"
if os.path.isdir(figdir):
    api.upload_folder(folder_path=figdir, repo_id=REPO, repo_type="dataset",
                      path_in_repo="issue778_persona_vectors/v2_scratch_tail_results/figures",
                      commit_message="issue778 v2rerun bigmem tail: v2 figures")
print("scratch upload complete")
PY

echo "[phase=done] bigmem tail complete (maxlayer + figures + scratch upload; MANIFEST withheld)"
