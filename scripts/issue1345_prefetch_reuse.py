#!/usr/bin/env python
"""Issue #1345 assistant-named-story follow-up — prefetch_reuse phase (plan v6 §4).

Stages the parent ARIA-run's four r1/r2 turnstore stems + the matched-n
allowlist from the HF data repo at the PINNED revision ``c.REUSE_REV`` into
the variant turnstore/matched dirs, REPLACING the parent's GPU-heavy
extract_r1r2 phase. The chat/no-template comparator maps are therefore
BIT-IDENTICAL to the parent's (a stronger identity guarantee than
re-extraction parity); integrity is triple-checked downstream (±0.02 parity
gate + r1/r2 refit-equality ≤1e-3 + the per-stem realized-keys probe run
here).

Staging recipe (gotchas.md large-repo rules): server-side-scoped
``list_repo_tree(path_in_repo=...)`` at the pinned revision — NEVER
``snapshot_download`` / bare ``list_repo_files`` on the ~1M-file repo —
materialized INSIDE the transient retry (lazy-generator gotcha), then
per-file ``hf_hub_download`` in a thread pool of ``max_workers<=6``.
Idempotent: a file already staged at the listed size is skipped (per-file
checkpointing — a crashed prefetch resumes without re-downloading ~87 GB).
Downloads land in a per-invocation staging dir under the destination
(same filesystem) and publish via atomic ``os.replace``.

Under --smoke only shard000 (+ sidecar) per stem is staged — the identical
list/filter/download/flatten/verify code path at file-subset grain
(PASS_UNIFIED; the production 80-file completeness assert demotes to the
smoke-sized expectation).

CLI:
  uv run python scripts/issue1345_prefetch_reuse.py \
      --turnstore-dir data/issue_1345/<variant>/turnstore \
      --matched-dir data/issue_1345/<variant>/matched_n [--smoke]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402

# Keys issue1345_fit_cells' loader path consumes (SLIM_KEYS + conv_ids); the
# per-stem realized-keys probe (plan §10 c30) asserts each staged stem's
# shard000 carries them (artifact-reuse.md check (c)).
CONSUMER_KEYS = "conv_ids,slots,profiles,nll"
STAGE_MAX_WORKERS = 6  # gotchas.md: <=6-thread hf_hub_download pool on the data repo
STALE_STAGE_MAX_AGE_S = 3600  # leftover staging dirs older than this are reaped


def _list_reuse_files(smoke: bool) -> list[tuple[str, int]]:
    """(path_in_repo, size) for every reuse file at the pinned revision.

    Scoped listing + filter to the four r1/r2 stems (the story stems under the
    same prefix are NOT reused — the story corpus is regenerated). Production
    asserts the full 80-file set; smoke keeps shard000 (+ sidecar) per stem.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    entries = retry_transient(
        # Materialize INSIDE the retry — list_repo_tree is a LAZY generator and
        # raises at iteration time (gotchas.md, #779).
        lambda: list(
            api.list_repo_tree(
                c.HF_DATA_REPO,
                path_in_repo=c.REUSE_TENSOR_PREFIX,
                repo_type="dataset",
                recursive=True,
                revision=c.REUSE_REV,
            )
        ),
        what=f"list_repo_tree({c.REUSE_TENSOR_PREFIX}@{c.REUSE_REV})",
    )
    files: list[tuple[str, int]] = []
    per_stem: dict[str, int] = dict.fromkeys(c.REUSE_STEMS, 0)
    for e in entries:
        base = e.path.rsplit("/", 1)[-1]
        for stem in c.REUSE_STEMS:
            if base.startswith(f"{stem}_shard") and (
                base.endswith(".pt") or base.endswith(".json")
            ):
                if smoke and not base.startswith(f"{stem}_shard000."):
                    break
                files.append((e.path, int(getattr(e, "size", 0) or 0)))
                per_stem[stem] += 1
                break
    expect_per_stem = 2 if smoke else c.REUSE_FILES_PER_STEM
    bad = {s: n for s, n in per_stem.items() if n != expect_per_stem}
    assert not bad, (
        f"reuse listing drift at {c.REUSE_TENSOR_PREFIX}@{c.REUSE_REV}: expected "
        f"{expect_per_stem} files/stem ({'smoke shard000 pair' if smoke else 'plan §10'}), "
        f"got {bad} (all: {per_stem})"
    )
    return sorted(files)


def _reap_stale_stage_dirs(dest_dir: Path) -> None:
    """Remove crashed prior invocations' staging dirs (age-gated >1h)."""
    now = time.time()
    for d in dest_dir.glob("_hfstage_*"):
        if d.is_dir() and now - d.stat().st_mtime > STALE_STAGE_MAX_AGE_S:
            shutil.rmtree(d, ignore_errors=True)
            print(f"[prefetch_reuse] reaped stale staging dir {d}", flush=True)


def _stage_files(files: list[tuple[str, int]], dest_dir: Path) -> dict:
    """Download the listed files flat into ``dest_dir`` (skip-if-complete)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    dest_dir.mkdir(parents=True, exist_ok=True)
    _reap_stale_stage_dirs(dest_dir)
    stage_dir = Path(tempfile.mkdtemp(prefix="_hfstage_", dir=dest_dir))
    counts = {"staged": 0, "skipped": 0, "bytes": 0}

    def _one(item: tuple[str, int]) -> str:
        path_in_repo, size = item
        base = path_in_repo.rsplit("/", 1)[-1]
        dest = dest_dir / base
        if dest.exists() and dest.stat().st_size == size:
            counts["skipped"] += 1
            return f"skip {base}"
        p = retry_transient(
            lambda: hf_hub_download(
                c.HF_DATA_REPO,
                path_in_repo,
                repo_type="dataset",
                revision=c.REUSE_REV,
                token=os.environ.get("HF_TOKEN"),
                local_dir=str(stage_dir),
            ),
            what=f"hf_hub_download({path_in_repo}@{c.REUSE_REV})",
        )
        os.replace(p, dest)  # same filesystem (stage dir lives under dest_dir)
        counts["staged"] += 1
        counts["bytes"] += size
        return f"staged {base} ({size / 1e9:.2f} GB)"

    try:
        with ThreadPoolExecutor(max_workers=STAGE_MAX_WORKERS) as pool:
            for i, msg in enumerate(pool.map(_one, files), 1):
                # Per-file liveness line (poller stall conjunction needs log activity)
                print(f"[prefetch_reuse] {i}/{len(files)} {msg}", flush=True)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    return counts


def _verify_stem_keys(turnstore_dir: Path) -> list[str]:
    """Run the mechanized realized-keys probe per staged stem (plan §10 c30).

    Shells the canonical CLI so the recorded PASS lines match the card's
    command shape; exit!=0 fails the phase loud (missing keys / unreadable).
    """
    lines = []
    for stem in c.REUSE_STEMS:
        shard0 = turnstore_dir / f"{stem}_shard000.pt"
        assert shard0.exists(), f"staged shard missing for keys probe: {shard0}"
        proc = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_DIR / "verify_reused_artifact_keys.py"),
                "--artifact",
                str(shard0),
                "--keys",
                CONSUMER_KEYS,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = (proc.stdout + proc.stderr).strip()
        print(f"[prefetch_reuse][keys] {stem}: {out}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"realized-keys probe FAILED for {shard0} (rc={proc.returncode}): {out}"
            )
        lines.append(f"{stem}: {out}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turnstore-dir", type=Path, default=c.TURNSTORE_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument("--smoke", action="store_true", help="shard000 (+ sidecar) per stem only")
    args = ap.parse_args()

    t0 = time.time()
    files = _list_reuse_files(args.smoke)
    total_gb = sum(s for _, s in files) / 1e9
    print(
        f"[prefetch_reuse] {len(files)} files / {total_gb:.2f} GB across "
        f"{len(c.REUSE_STEMS)} stems @ {c.REUSE_REV[:10]} (smoke={args.smoke})",
        flush=True,
    )
    counts = _stage_files(files, args.turnstore_dir)

    # Matched-n allowlist -> a PARENT-named file (build_matched writes the fresh
    # matched_subsets.json; the staged copy is the equality reference).
    args.matched_dir.mkdir(parents=True, exist_ok=True)
    parent_matched = args.matched_dir / "matched_subsets_parent.json"
    stage_dir = Path(tempfile.mkdtemp(prefix="_hfstage_", dir=args.matched_dir))
    try:
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate.hub import retry_transient

        p = retry_transient(
            lambda: hf_hub_download(
                c.HF_DATA_REPO,
                c.REUSE_MATCHED_PATH,
                repo_type="dataset",
                revision=c.REUSE_REV,
                token=os.environ.get("HF_TOKEN"),
                local_dir=str(stage_dir),
            ),
            what=f"hf_hub_download({c.REUSE_MATCHED_PATH}@{c.REUSE_REV})",
        )
        os.replace(p, parent_matched)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"[prefetch_reuse] matched-n allowlist -> {parent_matched}", flush=True)

    keys_lines = _verify_stem_keys(args.turnstore_dir)

    manifest = {
        "metadata": c.metadata(0, len(files), "scripts/issue1345_prefetch_reuse.py"),
        "reuse_revision": c.REUSE_REV,
        "reuse_prefix": c.REUSE_TENSOR_PREFIX,
        "matched_path": c.REUSE_MATCHED_PATH,
        "smoke": bool(args.smoke),
        "n_files": len(files),
        "total_gb_listed": total_gb,
        "staged": counts["staged"],
        "skipped_already_present": counts["skipped"],
        "realized_keys_probe": keys_lines,
        "elapsed_s": time.time() - t0,
    }
    c.write_json(args.turnstore_dir / "prefetch_reuse_manifest.json", manifest)
    print(
        f"[prefetch_reuse] done: staged={counts['staged']} skipped={counts['skipped']} "
        f"({counts['bytes'] / 1e9:.2f} GB downloaded, {time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
