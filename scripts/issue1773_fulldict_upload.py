#!/usr/bin/env python
"""Issue #1773 full-dictionary — pod-side upload + exact-set verify of the
phase-0/1 artifacts, so the pod can be released the moment phase 1 lands.

Uploads three prefixes under `issue1773_featurepipeline/fulldict/` (ONE
`upload_folder` commit each — never a per-file loop, #664/#1544) and verifies
the EXACT expected file set with one prefix-scoped listing per prefix
(`hub.verify_repo_paths_uploaded`, #997), fail-loud:

  phase0/     feature_table.jsonl + phase0_arrays.npz + phase0_meta.json
  selection/  selection.shard*.jsonl + manifest + meta + inverted_index.npz
  evidence/   evidence_manifests/* + holdout/scoring sets + completeness report

Exit 0 only when every expected path is present on the Hub. The release
watcher gates termination on that exit code — nothing is destroyed until the
artifacts are provably durable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1773_common as CM  # noqa: E402

FULLDICT_PREFIX = f"{CM.HF_PREFIX}/fulldict"


def _log(msg: str) -> None:
    print(msg, flush=True)


def upload_dir(local: Path, prefix: str, patterns: list[str]) -> list[str]:
    """One bulk `upload_folder` commit, then an EXACT-set verify. Returns the
    repo-relative paths verified present."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = sorted(p for pat in patterns for p in local.rglob(pat) if p.is_file())
    if not files:
        raise RuntimeError(f"[upload] nothing matched {patterns} under {local}")
    expected = [f"{prefix}/{p.relative_to(local).as_posix()}" for p in files]
    total_mb = sum(p.stat().st_size for p in files) / 1024**2
    _log(f"[upload] {local} -> {prefix}: {len(files)} files, {total_mb:.1f} MB")

    api = HfApi()
    hub.assert_hub_dir_filecounts(local, prefix, allow_patterns=patterns)
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(local),
            repo_id=CM.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
            allow_patterns=patterns,
        ),
        what=f"fulldict {prefix} upload",
    )
    missing = hub.verify_repo_paths_uploaded(
        api, CM.HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(
            f"[upload] EXACT-set verify FAILED for {prefix}: {len(missing)} missing, "
            f"e.g. {sorted(missing)[:5]}"
        )
    _log(f"[upload] verify PASS: {len(expected)} files present under {prefix}")
    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("/workspace/issue1773_fulldict"))
    ap.add_argument(
        "--phase0-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_1773_fulldict/phase0",
    )
    args = ap.parse_args()

    verified: dict[str, int] = {}
    verified["phase0"] = len(
        upload_dir(args.phase0_dir, f"{FULLDICT_PREFIX}/phase0", ["*.jsonl", "*.json", "*.npz"])
    )
    verified["selection"] = len(
        upload_dir(
            args.work / "selection", f"{FULLDICT_PREFIX}/selection", ["*.jsonl", "*.json", "*.npz"]
        )
    )
    verified["evidence"] = len(
        upload_dir(args.work / "evidence", f"{FULLDICT_PREFIX}/evidence", ["*.jsonl", "*.json"])
    )

    # Realized evidence fill — the >=99% gate #1773 used at 16,384 features.
    rep = args.work / "evidence" / "completeness_report.json"
    fill = json.loads(rep.read_text()) if rep.exists() else {}
    out = {
        "verified_file_counts": verified,
        "hf_prefix": FULLDICT_PREFIX,
        "completeness": {
            k: fill.get(k) for k in ("fill_fraction", "n_short", "n_features", "n_full")
        },
        **CM.repro_meta(),
    }
    (args.work / "fulldict_upload_verified.json").write_text(json.dumps(out, indent=1))
    _log("[upload] ALL PREFIXES VERIFIED: " + json.dumps(verified))
    _log("[upload] completeness: " + json.dumps(out["completeness"]))
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
