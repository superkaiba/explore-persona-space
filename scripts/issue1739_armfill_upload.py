#!/usr/bin/env python3
"""Upload + verify the #1739 armfill round's collected results.

ONE bulk ``upload_folder`` commit of the collected tree to
``issue1739_armfill/`` on the data repo (never a per-file loop -- that is the
#664 504-storm class), followed by the canonical scoped exact-set verify
(``hub.verify_repo_paths_uploaded``; a bare full-repo listing on the ~1M-file
data repo wedges, #920/#833).

Fail-loud: a non-empty missing set exits non-zero and names every absent
path, so the round can never report a PASS over an incomplete upload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local-root", type=Path, required=True, help="collected results tree")
    # No default: a hardcoded issue-prefix fallback is silently inherited when a
    # child issue reuses this script, uploading into THIS issue's prefix (#1005
    # clobber shape). Callers name the destination explicitly.
    ap.add_argument(
        "--hf-prefix",
        default=None,
        help="destination prefix on the data repo, e.g. issue1739_armfill (REQUIRED)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.hf_prefix:
        raise SystemExit(
            "--hf-prefix is required (no default): name the destination prefix explicitly, "
            "e.g. --hf-prefix issue1739_armfill"
        )

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    root = args.local_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"local root absent: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"nothing to upload under {root}")
    expected = [f"{args.hf_prefix}/{p.relative_to(root).as_posix()}" for p in files]
    total = sum(p.stat().st_size for p in files)
    print(f"[armfill-upload] {len(files)} files, {total / 1e6:.1f} MB -> {args.hf_prefix}/")
    for e in expected:
        print(f"  {e}")
    if args.dry_run:
        print("[armfill-upload] dry-run — nothing uploaded")
        return 0

    hub._upload(root, hub.DEFAULT_DATASET_REPO, "dataset", args.hf_prefix, raise_on_error=True)
    print(f"[armfill-upload] bulk commit done -> {args.hf_prefix}/")

    api = HfApi()
    missing = hub.verify_repo_paths_uploaded(
        api,
        hub.DEFAULT_DATASET_REPO,
        expected,
        path_in_repo=args.hf_prefix,
        repo_type="dataset",
    )
    if missing:
        print(f"[armfill-upload] VERIFY FAIL — {len(missing)} missing:", file=sys.stderr)
        for m in missing:
            print(f"  MISSING {m}", file=sys.stderr)
        return 2
    print(
        f"[armfill-upload] VERIFY PASS — all {len(expected)} paths present under {args.hf_prefix}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
