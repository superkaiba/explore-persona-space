"""#1776 batched Hub upload: one create_commit per prefix + scoped verify.

Committed-by-path port of the ``upload_batch`` heredoc from
``issue1776_dispatch.sh`` (same crash-fix round as issue1776_jpairs.py: inline
heredocs are un-lintable, un-smokeable, and their stderr died with the
instance). Reads a listfile of ``rel=abs`` lines, uploads all files as ONE
retried ``create_commit`` (never a per-file loop — 256-commits/hr +
504-storm rules), then verifies the exact expected set with a scoped listing.

Exit contract: 0 PASS / nonzero fail-loud.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import issue1776_common as C76

from explore_persona_space.orchestrate import hub


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True, help="Hub path_in_repo prefix")
    ap.add_argument("--listfile", type=Path, required=True, help="rel=abs per line")
    ap.add_argument("--message", required=True, help="commit message")
    args = ap.parse_args(argv)

    from huggingface_hub import CommitOperationAdd, HfApi

    pairs = [ln.split("=", 1) for ln in args.listfile.read_text().split("\n") if ln.strip()]
    ops, expected = [], []
    for rel, local in pairs:
        p = Path(local)
        # is_file, NOT exists (crash-fix r9): CommitOperationAdd rejects
        # directories — a DIR here is the misnested-out-arg class.
        assert p.is_file(), f"upload source missing or not a regular FILE: {p}"
        rp = f"{args.prefix}/{rel}"
        ops.append(CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(p)))
        expected.append(rp)
    if not ops:
        print(f"[upload] nothing to upload for {args.prefix}")
        return 0
    api = HfApi()
    hub.retry_transient(
        lambda: api.create_commit(
            repo_id=C76.HF_DATA_REPO,
            repo_type="dataset",
            operations=ops,
            commit_message=args.message,
        ),
        what=f"create_commit({args.prefix})",
    )
    missing = hub.verify_repo_paths_uploaded(
        api, C76.HF_DATA_REPO, expected, path_in_repo=args.prefix, repo_type="dataset"
    )
    assert not missing, f"post-upload verify FAIL ({len(missing)} missing): {missing[:5]}"
    print(f"[upload] {len(expected)} files -> {C76.HF_DATA_REPO}/{args.prefix} (verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
