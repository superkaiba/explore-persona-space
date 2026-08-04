"""Round-trip the wcrung context rows between the VM sampler and the GPU lane.

``scripts/issue1739_wcrung_sample.py`` writes ONE ~25 MB
``contexts/wcrung.json`` (``{"rows": [...], **digest}``). That file is
deliberately NOT committed to git — it is raw WildChat prompt text, and a
25 MB free-text blob is the wrong thing to push through the pre-commit
secret scan — so the GPU lane, which reaches the repo only by ``git clone``,
cannot read it. This module is the transport: pack the rows into <= 9 MB
``wcrung_rows.shardNN.jsonl`` line-shards + a ``wcrung_rows.manifest.json``,
upload the tiny shard set to the HF data repo in ONE bulk commit, and stage
+ reassemble it pod-side.

Why line-shards rather than the single JSON: ``upload_file`` /
``upload_folder`` force-route any blob over 10 MB to LFS regardless of
extension, and the public-storage quota gates the LFS endpoint only
(``.claude/rules/upload-policy.md``) — sharded text under 9 MB rides the
always-open non-LFS path. Never gzip (``*.gz`` IS LFS-matched).

The writer and the reader live in ONE module on purpose: a shard format and
its parser must not drift.

CONTENT HYGIENE: this module never prints or logs row text — counts, byte
sizes, sha256 digests, and field names only (the rows carry real WildChat
user prompts).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger("issue1739_wcrung_rows_io")

SHARD_MAX_BYTES = 9_000_000
ROWS_STEM = "wcrung_rows"
MANIFEST_NAME = f"{ROWS_STEM}.manifest.json"
SCHEMA = "wcrung-rows-shards-v1"


def _shard_name(idx: int) -> str:
    return f"{ROWS_STEM}.shard{idx:02d}.jsonl"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shard_rows(rows: list[dict], dest_dir: Path) -> dict:
    """Write ``rows`` as <= 9 MB jsonl line-shards + a manifest; return the manifest.

    Deterministic: rows are written in the given order and shard boundaries
    depend only on (ordering, serialized bytes, cap), so a re-shard of the
    same rows is byte-identical. Memory-bounded on the write side (one row
    serialized at a time into the open handle).
    """
    if not rows:
        raise ValueError("refusing to shard an empty row list")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob(f"{ROWS_STEM}.shard*.jsonl"):
        stale.unlink()

    shards: list[dict] = []
    idx = 0
    handle = None
    written = 0
    lines = 0
    try:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
            if handle is None or (written + len(line) > SHARD_MAX_BYTES and lines > 0):
                if handle is not None:
                    handle.close()
                    shards.append({"name": _shard_name(idx), "n_rows": lines, "n_bytes": written})
                    idx += 1
                handle = (dest_dir / _shard_name(idx)).open("wb")
                written = 0
                lines = 0
            handle.write(line)
            written += len(line)
            lines += 1
    finally:
        if handle is not None:
            handle.close()
    shards.append({"name": _shard_name(idx), "n_rows": lines, "n_bytes": written})

    for shard in shards:
        shard["sha256"] = _sha256_file(dest_dir / shard["name"])
    manifest = {
        "schema": SCHEMA,
        "stem": ROWS_STEM,
        "n_rows": sum(s["n_rows"] for s in shards),
        "n_shards": len(shards),
        "shard_max_bytes": SHARD_MAX_BYTES,
        "shards": shards,
    }
    if manifest["n_rows"] != len(rows):
        raise RuntimeError(
            f"shard row-count mismatch: manifest {manifest['n_rows']} vs input {len(rows)}"
        )
    (dest_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info(
        "[wcrung-rows] sharded %d rows -> %d shards (max %d bytes) in %s",
        manifest["n_rows"],
        manifest["n_shards"],
        SHARD_MAX_BYTES,
        dest_dir,
    )
    return manifest


def load_rows(shard_dir: Path) -> list[dict]:
    """Reassemble the rows from a shard dir, verifying sha256 + per-shard counts.

    Fail-loud on a missing shard, a digest mismatch, a line-count mismatch, or
    a total-count mismatch — a silently truncated context set would shrink the
    rung without any downstream signal.
    """
    manifest_path = shard_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {MANIFEST_NAME} under {shard_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected shard schema {manifest.get('schema')!r} (want {SCHEMA!r})")

    rows: list[dict] = []
    for shard in manifest["shards"]:
        path = shard_dir / shard["name"]
        if not path.exists():
            raise FileNotFoundError(f"manifest names {shard['name']} but it is missing")
        digest = _sha256_file(path)
        if digest != shard["sha256"]:
            raise RuntimeError(
                f"{shard['name']} sha256 mismatch: on-disk {digest[:16]} != "
                f"manifest {shard['sha256'][:16]}"
            )
        with path.open() as fh:
            shard_rows_ = [json.loads(line) for line in fh if line.strip()]
        if len(shard_rows_) != shard["n_rows"]:
            raise RuntimeError(
                f"{shard['name']} row-count mismatch: read {len(shard_rows_)} != "
                f"manifest {shard['n_rows']}"
            )
        rows.extend(shard_rows_)
    if len(rows) != manifest["n_rows"]:
        raise RuntimeError(f"total row-count mismatch: read {len(rows)} != {manifest['n_rows']}")
    logger.info("[wcrung-rows] loaded %d rows from %d shards", len(rows), manifest["n_shards"])
    return rows


def stage_rows_from_hub(
    *,
    hf_prefix: str,
    dest_dir: Path,
    repo_id: str | None = None,
    revision: str | None = None,
) -> list[dict]:
    """Stage the shard set from the HF data repo, then reassemble + verify.

    Uses the canonical scoped-prefix staging helper (``hub.stage_hub_prefix``
    — server-side scoped listing + retried per-file download, never a
    ``snapshot_download`` of the ~1M-file data repo). Files land as a verbatim
    prefix mirror, so the shards resolve under ``dest_dir/<hf_prefix>/``.
    """
    from explore_persona_space.orchestrate import hub

    repo = repo_id or hub.DEFAULT_DATASET_REPO
    staged = hub.stage_hub_prefix(repo, hf_prefix, dest_dir, revision=revision)
    logger.info("[wcrung-rows] staged %d file(s) from %s/%s", len(staged), repo, hf_prefix)
    mirror = dest_dir / hf_prefix
    return load_rows(mirror)


def upload_rows(
    rows_json: Path,
    *,
    hf_prefix: str,
    stage_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Shard ``rows_json``'s ``rows`` array and upload the shard set (ONE commit).

    Bulk folder upload via the shared ``hub._upload`` (retried, fail-loud on
    ``raise_on_error``) — never a per-file loop. The manifest rides in the same
    commit as the shards it describes.
    """
    payload = json.loads(rows_json.read_text())
    rows = payload["rows"]
    manifest = shard_rows(rows, stage_dir)
    if dry_run:
        logger.info("[wcrung-rows] DRY-RUN: would upload %s -> %s", stage_dir, hf_prefix)
        return manifest

    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        stage_dir,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        hf_prefix,
        raise_on_error=True,
    )
    logger.info("[wcrung-rows] uploaded %d shards + manifest -> %s", manifest["n_shards"], url)
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--rows-json",
        type=Path,
        default=Path("eval_results/issue_1739/wildchat_rung/contexts/wcrung.json"),
    )
    # This transport is bound to #1739's wildchat rung by construction (the shard
    # stem, the manifest schema, and the sampler row shape are all that rung's),
    # so the default names that rung's own subtree; --hf-prefix overrides.
    # UPLOAD_PREFIX_EXEMPT: wildchat-rung-specific rows transport; flag overrides
    ap.add_argument("--hf-prefix", default="issue1739_ctxmap/wildchat_rung/contexts")
    ap.add_argument("--stage-dir", type=Path, default=None, help="default: <rows-json>_shards/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stage_dir = args.stage_dir or args.rows_json.with_name(f"{args.rows_json.stem}_shards")
    manifest = upload_rows(
        args.rows_json,
        hf_prefix=args.hf_prefix,
        stage_dir=stage_dir,
        dry_run=args.dry_run,
    )
    print(
        f"[wcrung-rows] n_rows={manifest['n_rows']} n_shards={manifest['n_shards']} "
        f"prefix={args.hf_prefix}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
