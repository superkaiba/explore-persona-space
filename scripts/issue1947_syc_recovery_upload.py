#!/usr/bin/env python
"""#1947 sycophancy-recovery upload: recovery pool sidecars + the 18 syc mixes.

Two bulk ``upload_folder`` commits per destination prefix (never a per-file
loop — the 256-commits/hr cap), each followed by an EXACT-set
``verify_repo_paths_uploaded`` on a fresh scoped listing, both riding
``hub.retry_transient``.

What uploads, and why:

* ``positives/syc-icl/`` sidecars — ``pos.jsonl`` (the 300-row deliverable),
  ``raw_pos_recovery{1,2}.jsonl`` (the ROLLOUT TEXT of all 1,062 recovery
  candidates INCLUDING rejects — persist-before-reduce), the recovery judge
  ``save_raw`` files (judge outputs), and the four recovery records. The
  records double as the REGENERATION NOTE for ``pos.jsonl``, which this round
  rewrites in place from 232 -> 300 rows (upload-policy § "Regenerating a
  published artifact in place"): no adapter or capture depends on the 232-row
  bytes — the syc mixes were never built, the behavior having been dropped at
  the gate — and the pre-recovery ``salvage_meta.json`` stays untouched beside
  it as the record of that state.
* ``mixes/<slug>/`` for the 18 syc cells — ``train_mix.jsonl`` +
  ``mix_meta.json`` + ``consumption_manifest.json``, the worker's staged input.

DELIBERATELY NOT uploaded: the 2,128 per-request ``gen_cache_recovery*`` /
``judge_cache_recovery*`` files. They are re-derivable duplicates — every
completion is already in ``raw_pos_recovery*.jsonl`` and every judge draw in
``judge_raw_pos_recovery*.json`` — and 2,128 tiny files would breach the
~2,000-staged-file commit watermark for zero recoverable information
(upload-policy § per-DIRECTORY file-count cap: pack or skip, never a
per-file loop).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1947_cells as cells  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
PREFIX = cells.DATA_PREFIX  # issue1947_singlevisit
POOL = "syc-icl"
POOL_PREFIX = f"{PREFIX}/raw_completions/datagen/positives/{POOL}"
MIXES_PREFIX = f"{PREFIX}/mixes"

POOL_FILES = (
    "pos.jsonl",
    "raw_pos_recovery1.jsonl",
    "raw_pos_recovery2.jsonl",
    "judge_raw_pos_recovery1.json",
    "judge_raw_pos_recovery2.json",
    "judge_raw_pos_rejudge.json",
    "recovery_audit.json",
    "recovery_record.json",
    "recovery_emit.json",
    "rejudge_censored.json",
)
MIX_FILES = ("train_mix.jsonl", "mix_meta.json", "consumption_manifest.json")


def _syc_slugs() -> list[str]:
    return sorted(c.slug for c in cells.CELLS if c.beh_key == "syc" and c.kind == "content")


def _upload_and_verify(api, local_dir: Path, path_in_repo: str, names: list[str]) -> list[str]:
    """ONE bulk commit of ``names`` under ``path_in_repo``, then an exact-set verify."""
    missing_local = [n for n in names if not (local_dir / n).is_file()]
    if missing_local:
        raise RuntimeError(f"[upload] missing local files under {local_dir}: {missing_local}")
    # Per-dir file-count guard BEFORE staging, OUTSIDE the retry wrapper (a
    # guard raise is deterministic — retrying it burns the retry budget).
    hub.assert_hub_dir_filecounts(local_dir, path_in_repo, allow_patterns=list(names))
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            allow_patterns=list(names),
            commit_message=f"issue1947 syc recovery: {path_in_repo}",
        ),
        what=f"upload_folder {path_in_repo}",
    )
    expected = [f"{path_in_repo}/{n}" for n in names]
    return hub.verify_repo_paths_uploaded(
        api, DATA_REPO, expected, path_in_repo=path_in_repo, repo_type="dataset"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="upload #1947 syc recovery artifacts")
    p.add_argument("--out-root", required=True)
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args(argv)
    if args.import_check:
        from huggingface_hub import HfApi  # noqa: F401

        print("[import-check] ok", flush=True)
        return 0
    from huggingface_hub import HfApi

    api = HfApi()
    root = Path(args.out_root)
    report: dict = {"uploaded": {}, "missing": {}}

    pool_dir = root / "positives" / POOL
    print(f"[upload] pool sidecars -> {POOL_PREFIX} ({len(POOL_FILES)} files)", flush=True)
    missing = _upload_and_verify(api, pool_dir, POOL_PREFIX, list(POOL_FILES))
    report["uploaded"][POOL_PREFIX] = len(POOL_FILES)
    report["missing"][POOL_PREFIX] = missing
    print(f"[upload] pool verify missing={missing}", flush=True)

    slugs = _syc_slugs()
    if len(slugs) != 18:
        raise RuntimeError(f"[upload] expected 18 syc content cells, got {len(slugs)}: {slugs}")
    for slug in slugs:
        d = root / "mixes" / slug
        m = _upload_and_verify(api, d, f"{MIXES_PREFIX}/{slug}", list(MIX_FILES))
        report["uploaded"][f"{MIXES_PREFIX}/{slug}"] = len(MIX_FILES)
        report["missing"][f"{MIXES_PREFIX}/{slug}"] = m
        print(f"[upload] mix {slug} verify missing={m}", flush=True)

    all_missing = {k: v for k, v in report["missing"].items() if v}
    out = root / "recovery_upload_report.json"
    out.write_text(
        json.dumps(
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "repo": DATA_REPO,
                **report,
                "all_verified": not all_missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if all_missing:
        raise RuntimeError(f"[upload] exact-set verify FAILED, missing: {all_missing}")
    print(f"[upload] ALL VERIFIED ({len(report['uploaded'])} prefixes) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
