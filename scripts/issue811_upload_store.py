#!/usr/bin/env python3
"""Issue #811 — upload the re-extracted paired store to HF before releasing the GPU pod.

Bulk-commits every per-cell ``.npz`` under ``eval_results/issue_811/analysis_tensors/``
(the Phase-1 paired store) AND ``eval_results/issue_811/phase0_base_leg/`` (the
Phase-0 base-leg store — the KILL-1 gate input, plan §4.0/§7) to the HF data repo
in ONE ``create_commit`` (well under the 256-commits/hr cap; Upload Policy),
verifies the full complement on a FRESH Hub listing before trusting the pod can be
released (analysis-tensor Upload Policy #521 — both stores are plan-referenced
downstream inputs: the paired store feeds the Phase-2 fits, the phase0 store feeds
the KILL-1 base-leg gate, so losing either makes the corresponding read permanently
unrunnable). Fail-loud: any missing file after the commit raises non-zero.

Reuses the exact bulk-commit + fresh-listing-verify shape of
``issue667_dispatch._upload_tensors`` (this store is the #811 analogue). Skipped
when ``EPM_SKIP_UPLOAD=1`` (a local CPU smoke that reads the store from disk).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# DOTENV_LINT_EXEMPT: analysis-phase script; shell exports cover pod/GCE/SLURM.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue811.upload_store")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# Round parameterization (#811 maxp round): the dispatcher's SUMMARY_VARIANT arm
# points these at the round's OWN dirs/prefix; the defaults preserve the v1
# turn_nl round verbatim.
ROUND_DIR = os.environ.get("EPM_I811_ROUND_DIR", "eval_results/issue_811")
HF_PREFIX = os.environ.get("EPM_I811_HF_PREFIX", "issue811_turn_nl_mapchange")
# Raw completions (the greedy base R JSONLs) are REQUIRED for the maxp round
# (plan §10 — closes the v1 generation-discard WARN); the v1 default arm did not
# write them historically, so requiring them unconditionally would break a
# --skip-extract resume against the old local dirs.
REQUIRE_RAW = os.environ.get("EPM_I811_REQUIRE_RAW") == "1"
# ~9.5 MB non-LFS text ceiling (Upload Policy: >9.5 MB text line-splits into
# <9 MB shards, NEVER gzip — *.gz is LFS-matched and the Hub force-routes >10 MB
# blobs to LFS regardless of extension).
TEXT_SHARD_BYTES = 9_500_000
TEXT_SHARD_TARGET = 9_000_000
# (local dir, HF prefix, glob, required) — the #811 stores that MUST land on HF
# before the GPU pod is released (Upload Policy #521 — the paired + phase0 stores
# are plan-referenced downstream inputs: paired feeds the Phase-2 fits, phase0
# feeds the KILL-1 base-leg gate; raw_completions carries the rollout TEXT R, the
# regen recipe for the discarded per-token span tensors, plan §10).
STORES = (
    (f"{ROUND_DIR}/analysis_tensors", f"{HF_PREFIX}/analysis_tensors", "*.npz", True),
    (f"{ROUND_DIR}/phase0_base_leg", f"{HF_PREFIX}/phase0_base_leg", "*.npz", True),
    (f"{ROUND_DIR}/raw_completions", f"{HF_PREFIX}/raw_completions", "*.jsonl", REQUIRE_RAW),
)


def _shard_oversize_jsonl(path: Path) -> list[Path]:
    """Line-split a >9.5 MB JSONL into <9 MB ``<stem>.shardNN.jsonl`` parts.

    Returns the file list to upload (the original when under the ceiling).
    Never gzip (LFS-matched); shards land NEXT to the original, which is then
    excluded from the upload set. Realistic #811 R files are ~100 KB, so this is
    a policy guard, not an expected path. A SINGLE line exceeding the shard
    target cannot be line-split at all — it RAISES with the row's identity
    (never ships as an oversize shard, which the Hub would force-route to LFS
    at >10 MB; r10 Minor). Content hygiene: the raise names identity keys only,
    never the text field.
    """
    if path.stat().st_size <= TEXT_SHARD_BYTES:
        return [path]
    shards: list[Path] = []
    idx, size, buf = 0, 0, []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            if len(line.encode()) > TEXT_SHARD_TARGET:
                try:
                    row = json.loads(line)
                    ident = {
                        k: row.get(k)
                        for k in ("behavior", "source_cid", "target_cid", "neg_cid", "probe_idx")
                        if k in row
                    }
                except Exception:
                    ident = {"parse": "failed"}
                raise ValueError(
                    f"single JSONL row of {len(line.encode())} B > {TEXT_SHARD_TARGET} B shard "
                    f"target in {path.name}:{lineno} (row identity: {ident}) — cannot "
                    "line-split; a >10 MB blob would force-route to LFS (Upload Policy)"
                )
            if size + len(line.encode()) > TEXT_SHARD_TARGET and buf:
                shard = path.with_name(f"{path.stem}.shard{idx:02d}.jsonl")
                shard.write_text("".join(buf))
                shards.append(shard)
                idx, size, buf = idx + 1, 0, []
            buf.append(line)
            size += len(line.encode())
    if buf:
        shard = path.with_name(f"{path.stem}.shard{idx:02d}.jsonl")
        shard.write_text("".join(buf))
        shards.append(shard)
    logger.info("line-split %s (>9.5MB) into %d shards", path.name, len(shards))
    return shards


def upload_store() -> int:
    """Bulk-commit + verify the #811 paired + phase0 stores. Returns 0 on success."""
    if os.environ.get("EPM_SKIP_UPLOAD") == "1":
        logger.info("EPM_SKIP_UPLOAD=1 -> skipping #811 store upload (local/smoke)")
        return 0
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_files

    api = HfApi()
    # Collect ops across BOTH stores into ONE commit (one Hub commit, not two;
    # 256-commits/hr cap). Each op carries its store's own HF prefix.
    #
    # BOTH stores are required uploads (plan §10 Reproducibility Card): the Phase-1
    # paired store (analysis_tensors/) feeds the Phase-2 fits and the Phase-0 base-leg
    # store (phase0_base_leg/) feeds the KILL-1 base-leg gate — losing EITHER makes the
    # corresponding read permanently unrunnable (Upload Policy #521). So require each
    # store to carry >=1 .npz BEFORE any commit: a per-store precondition, NOT a single
    # aggregate check. An aggregate "if not ops" (any store non-empty) would silently
    # commit + verify only the populated store while omitting the other entirely
    # (round-3 Major upload-store-does-not-require-both-stores).
    ops: list[CommitOperationAdd] = []
    expected: list[str] = []  # path_in_repo for the fresh-listing verify
    per_store_counts: list[tuple[str, int]] = []
    empty_required: list[str] = []
    for local_dir, hf_prefix, glob, required in STORES:
        tdir = PROJECT_ROOT / local_dir
        files = sorted(tdir.rglob(glob)) if tdir.is_dir() else []
        if glob == "*.jsonl":
            # Text-policy guard: shard any >9.5 MB JSONL (non-LFS path stays open),
            # excluding sharded originals from the upload set.
            files = [s for p in files if ".shard" not in p.name for s in _shard_oversize_jsonl(p)]
        per_store_counts.append((local_dir, len(files)))
        if required and not files:
            empty_required.append(local_dir)
        if not files and not required:
            logger.info("optional store %s empty/absent -- skipped", local_dir)
            continue
        for p in files:
            pir = f"{hf_prefix}/{p.relative_to(tdir).as_posix()}"
            ops.append(CommitOperationAdd(path_in_repo=pir, path_or_fileobj=str(p)))
            expected.append(pir)
    if empty_required:
        raise RuntimeError(
            f"#811 upload: {empty_required} has 0 files -- these are required uploads "
            f"(plan §10; analysis_tensors feeds Phase-2 fits, phase0_base_leg feeds the "
            f"KILL-1 gate, raw_completions carries the persisted R text). Refusing to "
            f"commit an INCOMPLETE upload. Per-store counts: {per_store_counts}"
        )
    api.create_commit(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=(
            f"issue811 ({HF_PREFIX}): {len(ops)} per-cell paired + phase0 base-leg "
            f"tensors + raw-completion R text"
        ),
    )
    files = set(list_repo_files(HF_DATA_REPO, repo_type="dataset"))
    missing = [pir for pir in expected if pir not in files]
    if missing:
        raise RuntimeError(
            f"#811 store upload verification FAILED -- missing on Hub: {missing[:5]}"
        )
    logger.info("uploaded + verified %d #811 store tensors to %s", len(ops), HF_DATA_REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(upload_store())
