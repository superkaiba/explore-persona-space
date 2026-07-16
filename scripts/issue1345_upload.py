#!/usr/bin/env python
"""Issue #1345 upload phase — persist-by-default to the HF data repo.

Uploads, in order (text first — quota-immune non-LFS path):
  1. story corpus text + yield reports (ONE upload_folder commit; judge cache
     excluded) -> <prefix>/raw_completions/stories/
  2. matched-n subsets (ONE upload_folder commit) -> <prefix>/inputs/matched_n/
  3. L19 preds caches (incremental, verified per shard)
     -> <prefix>/analysis_tensors/preds_cache/
  4. turnstore shards (incremental upload->verify->optional delete-local via
     orchestrate.upload_sharded.upload_dir_sharded — overflow-safe, #1034)
     -> <prefix>/analysis_tensors/turnstore/

Under --smoke everything targets the issue1345_smoke/ prefix (same code path,
tiny files) so the upload leg is exercised end-to-end by the smoke run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing (GCE metadata env / pod .env)"

import issue1345_common as c  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stories-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--preds-dir", type=Path, default=c.PREDS_CACHE_DIR)
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--delete-local-turnstore",
        action="store_true",
        help="free the pod-local shards after per-shard verified upload",
    )
    args = ap.parse_args()

    prefix = "issue1345_smoke" if args.smoke else c.HF_ISSUE_PREFIX
    from huggingface_hub import upload_folder

    from explore_persona_space.orchestrate.hub import assert_hub_dir_filecounts, retry_transient
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    # 1) story text (rollout text is NEVER discardable — Upload Policy)
    if args.stories_dir.exists() and any(args.stories_dir.glob("*.json*")):
        assert_hub_dir_filecounts(
            args.stories_dir,
            f"{prefix}/raw_completions/stories",
            allow_patterns=["*.jsonl", "*.json"],
            ignore_patterns=["judge_cache/*"],
        )
        # retry_transient (#1345 crash-fix r5): transport is never fatal — a
        # lone Hub 429 ("maximum queue size reached") killed att-20260715-175238;
        # bounded backoff (Retry-After-aware, EPM_HF_RETRY_BUDGET_S wall cap),
        # fail-loud only on genuine exhaustion.
        retry_transient(
            lambda: upload_folder(
                repo_id=c.HF_DATA_REPO,
                repo_type="dataset",
                folder_path=str(args.stories_dir),
                path_in_repo=f"{prefix}/raw_completions/stories",
                allow_patterns=["*.jsonl", "*.json"],
                ignore_patterns=["judge_cache/*"],
                commit_message=(
                    f"issue-1345: story corpus text ({'smoke' if args.smoke else 'full'})"
                ),
            ),
            what=f"upload_folder({prefix}/raw_completions/stories)",
        )
        print(f"[upload] stories -> {prefix}/raw_completions/stories", flush=True)
    else:
        print("[upload] no story files present (story regime halted?) — skipped", flush=True)

    # 2) matched-n subsets (tiny JSON)
    if args.matched_dir.exists() and any(args.matched_dir.glob("*.json")):
        assert_hub_dir_filecounts(
            args.matched_dir, f"{prefix}/inputs/matched_n", allow_patterns=["*.json"]
        )
        retry_transient(
            lambda: upload_folder(
                repo_id=c.HF_DATA_REPO,
                repo_type="dataset",
                folder_path=str(args.matched_dir),
                path_in_repo=f"{prefix}/inputs/matched_n",
                allow_patterns=["*.json"],
                commit_message="issue-1345: matched-n subsets",
            ),
            what=f"upload_folder({prefix}/inputs/matched_n)",
        )
        print(f"[upload] matched_n -> {prefix}/inputs/matched_n", flush=True)

    # 3) L19 preds caches (verdict-lattice inputs — plan-referenced downstream)
    if args.preds_dir.exists() and any(args.preds_dir.glob("*.npz")):
        res = upload_dir_sharded(
            args.preds_dir,
            c.HF_DATA_REPO,
            f"{prefix}/analysis_tensors/preds_cache",
            shard_glob="*.npz",
            delete_local=False,
        )
        print(f"[upload] preds_cache: {res}", flush=True)

    # 4) turnstore shards (the big tensors; incremental + verified)
    if args.turnstore_dir.exists() and any(args.turnstore_dir.glob("*_shard*")):
        res = upload_dir_sharded(
            args.turnstore_dir,
            c.HF_DATA_REPO,
            f"{prefix}/analysis_tensors/turnstore",
            shard_glob="*_shard*",
            delete_local=args.delete_local_turnstore,
        )
        print(f"[upload] turnstore: {res}", flush=True)

    print("[upload] complete", flush=True)


if __name__ == "__main__":
    main()
