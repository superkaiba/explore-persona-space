#!/usr/bin/env python3
"""Task #2330 P0: build the pinned split-id lists for the matched-data map comparison.

Downloads the three #1491 sampling-manifest JSONLs (plus train_25k) at the
pinned revision, extracts each split's ``ladder_local_id`` list in FILE ORDER,
and writes ``eval_results/issue_2330/split_ids.json`` — the SINGLE SOURCE every
P3 count pin, subset, and matched-ID assert reads (plan §4 P0/P3,
methodology-reconciler round-1 resolution).

Splits written:
  - ``train_10k``: the first 10,000 context ids of ``train_25k.jsonl`` in file
    order (the file-order prefix makes the 5k ⊂ 10k nesting and the
    cross-model matched-ID property mechanical).
  - ``train_5k``: the first 5,000 (a prefix of ``train_10k`` by construction).
  - ``val_400`` / ``test_1000`` / ``wc_test_1k``: the full id lists.

The manifest JSONLs land in ``data/issue_2330/manifest/`` — a deliberately
UNCOMMITTED local cache (gitignored via ``data/*``); ONLY the split_ids.json
output is committed to the issue branch.

``dropped_overlength`` is initialized EMPTY here; P1 gate step 2
(``scripts/issue2330_qwen35_generate_capture.py --gate length_scan``, pod-side)
appends any over-budget ids and RE-WRITES this file (removing them from the
affected split lists + recomputing the shas), so post-P1 counts are the
post-drop realized grain. This script REFUSES to clobber a split_ids.json that
already carries drops (``--force`` overrides).

Observed manifest row schema (probed at the pin, val_400.jsonl row 0):
``{"corpus", "ladder_local_id", "prompt", "split"}``; ``ladder_local_id`` is
contiguous 0..n-1 in file order per split.

Run (VM, CPU, minutes):
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
  uv run python scripts/issue2330_split_ids.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# Pinned reuse source (plan §10 reproducibility card; Hub-verified at plan time).
MANIFEST_HF_REPO = "superkaiba1/explore-persona-space-data"
MANIFEST_HF_PREFIX = "issue1491_scale_ladder/manifest"
MANIFEST_REVISION = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"

# Manifest split file -> expected row count (plan §10 counted realized grain:
# measured artifact-side n_realized 25,000 / 400 / 1,000 / 999).
MANIFEST_FILES = {
    "train_25k": ("train_25k.jsonl", 25_000),
    "val_400": ("val_400.jsonl", 400),
    "test_1000": ("test_1000.jsonl", 1_000),
    "wc_test_1k": ("wc_test_1k.jsonl", 999),
}

TRAIN_10K_N = 10_000
TRAIN_5K_N = 5_000

SHA_DOMAIN = (
    "sha256 of json.dumps(id_list, separators=(',',':')) encoded utf-8 — "
    "a digest of the INT ID LIST, not of prompt text or file bytes"
)


def _sha256_id_list(ids: list[int]) -> str:
    """Canonical digest of an id list (domain: SHA_DOMAIN — never compare
    against digests computed in another domain, gotchas.md sha-pin-domain)."""
    payload = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _download_manifest_file(fname: str, manifest_dir: Path, revision: str) -> Path:
    """Fetch one manifest JSONL at the pinned revision into the local cache dir.

    Uses hf_hub_download's own lock-protected cache, then copies the bytes to
    the flat ``data/issue_2330/manifest/<fname>`` path (idempotent re-runs
    re-copy identical bytes)."""
    from huggingface_hub import hf_hub_download

    cached = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=MANIFEST_HF_REPO,
            filename=f"{MANIFEST_HF_PREFIX}/{fname}",
            repo_type="dataset",
            revision=revision,
            cache_dir=str(manifest_dir / ".hf_cache"),
        ),
        what=f"hf_hub_download {MANIFEST_HF_PREFIX}/{fname}@{revision[:8]}",
    )
    dest = manifest_dir / fname
    shutil.copyfile(cached, dest)
    return dest


def _read_id_list(path: Path, expect_n: int, split_name: str) -> list[int]:
    """Read ``ladder_local_id`` per row in FILE ORDER; fail loud on any drift.

    Text-mode line iteration (never ``splitlines()`` — real-user prompt text
    can carry U+2028/U+2029, gotchas.md JSONL-reader rule)."""
    ids: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert "ladder_local_id" in row, (
                f"{split_name}: manifest row missing ladder_local_id — schema drift "
                f"(observed keys: {sorted(row.keys())})"
            )
            assert isinstance(row.get("prompt"), str) and row["prompt"], (
                f"{split_name}: row {row.get('ladder_local_id')} has no prompt text"
            )
            ids.append(int(row["ladder_local_id"]))
    assert len(ids) == expect_n, (
        f"{split_name}: expected {expect_n} rows at pin {MANIFEST_REVISION[:8]}, "
        f"got {len(ids)} — manifest drift; refusing to write split_ids"
    )
    assert len(set(ids)) == len(ids), f"{split_name}: duplicate ladder_local_id values"
    return ids


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Task #2330 P0: pinned split-id lists")
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "eval_results" / "issue_2330" / "split_ids.json",
        help="output split_ids.json path (committed to the issue branch)",
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "issue_2330" / "manifest",
        help="local UNCOMMITTED manifest cache dir (gitignored via data/*)",
    )
    ap.add_argument("--revision", default=MANIFEST_REVISION, help="pinned manifest revision")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing split_ids.json even if it carries P1 drops",
    )
    return ap


def main() -> int:
    args = _build_parser().parse_args()

    if args.out.exists() and not args.force:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        if existing.get("dropped_overlength"):
            print(
                f"REFUSING: {args.out} already carries P1 dropped_overlength entries "
                f"({existing['dropped_overlength']}); re-running P0 would clobber the "
                "post-drop single source. Pass --force only if that is intended.",
                file=sys.stderr,
            )
            return 2

    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    id_lists: dict[str, list[int]] = {}
    file_shas: dict[str, str] = {}
    for split_name, (fname, expect_n) in MANIFEST_FILES.items():
        dest = _download_manifest_file(fname, args.manifest_dir, args.revision)
        file_shas[fname] = hashlib.sha256(dest.read_bytes()).hexdigest()
        id_lists[split_name] = _read_id_list(dest, expect_n, split_name)
        print(f"[split-ids] {split_name}: {len(id_lists[split_name])} rows ({fname})")

    train_ids = id_lists["train_25k"]
    splits = {
        "train_10k": train_ids[:TRAIN_10K_N],
        "train_5k": train_ids[:TRAIN_5K_N],
        "val_400": id_lists["val_400"],
        "test_1000": id_lists["test_1000"],
        "wc_test_1k": id_lists["wc_test_1k"],
    }
    # Nesting invariant (plan §4: 5k ⊂ 10k by file-order prefix construction).
    assert splits["train_5k"] == splits["train_10k"][: len(splits["train_5k"])], (
        "train_5k is not a prefix of train_10k — construction bug"
    )
    counts = {k: len(v) for k, v in splits.items()}
    assert counts == {
        "train_10k": TRAIN_10K_N,
        "train_5k": TRAIN_5K_N,
        "val_400": 400,
        "test_1000": 1_000,
        "wc_test_1k": 999,
    }, counts

    payload = {
        "schema_version": 1,
        "task": 2330,
        "manifest": {
            "repo": MANIFEST_HF_REPO,
            "prefix": MANIFEST_HF_PREFIX,
            "revision": args.revision,
            "files": {fname: file_shas[fname] for fname, _ in MANIFEST_FILES.values()},
            "file_sha_domain": "sha256 of the downloaded JSONL file bytes",
        },
        "splits": splits,
        "counts": counts,
        "sha256": {k: _sha256_id_list(v) for k, v in splits.items()},
        "sha_domain": SHA_DOMAIN,
        # P1 gate step 2 (length_scan, pod-side) appends here and re-writes this
        # file: {split_key: [{"id": int, "n_tokens": int}, ...]}. An id dropped
        # from train_10k with id < 5000 is dropped from train_5k too (prefix
        # nesting); ceiling draws consume test_1000's post-drop list.
        "dropped_overlength": {},
        "dropped_overlength_doc": (
            "written by issue2330_qwen35_generate_capture.py --gate length_scan; "
            "ids here are REMOVED from the split lists above, so len(splits[k]) "
            "is the post-drop realized grain every P3 count pin reads"
        ),
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "scripts/issue2330_split_ids.py",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(f"[split-ids] wrote {args.out}")
    for k in splits:
        print(f"[split-ids]   {k}: n={counts[k]} sha256={payload['sha256'][k][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
