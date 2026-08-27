"""Persist an issue-2546 arm's run logs to the HF data repo (defect 2, marker v144 / #2613).

Called by scripts/issue2546_dispatch.sh at phase end AND from its EXIT trap so
a FAILED or killed phase still persists its line-level record (arms 1 and 3
each needed a manual ~4 MB upload; on arm 3 the two most valuable logs were the
FAILURE records). Destination mirrors the realized arm1/arm3 Hub convention at
``issue2546_cotmap/logs/arm<N>/`` (``smoke_arm<N>`` under --smoke — smoke
artifacts never share a prefix with production):

  <root>            rotated dispatcher logs   $LOG_DIR/issue-2546.log*
  logs/             gen/capture worker logs   $OUT_ROOT/logs/*.log
  work/fits_a<N>/   fit worker logs           $OUT_ROOT/work/fits_a<N>/*.log
  aux/              launcher + revisions.json + fallbacks_a<N>.env

Rules honored: plain text only — NEVER gzip/tar (``*.gz`` is LFS-matched and
>10 MB blobs force-route to LFS; upload-policy.md); any single file over
~9.5 MB is line-split into <9 MB ``.partNNN`` pieces; ONE bulk
``upload_folder`` commit via ``hub._upload`` (retry-wrapped, scoped verify —
never a per-file loop, #664); idempotent (re-runs overwrite/skip Hub-side);
read-only over the phase's artifacts (stages copies under /tmp). The caller
treats a non-zero exit as LOUD-BUT-NON-FATAL — this script must never be able
to destroy the phase's own exit status or artifacts.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

MAX_TEXT_BYTES = int(9.5 * 1024 * 1024)  # upload-as-is ceiling (upload-policy.md text rule)
PART_BYTES = 9 * 1024 * 1024  # line-split piece target (<9 MB, non-LFS path)


def collect_log_files(
    log_dir: Path, out_root: Path, arm: int, fallback_env: Path
) -> list[tuple[Path, str]]:
    """(local file, hub-relative dest dir; '' = arm root) for the four log groups.

    Derives the realized arm1/arm3 convention (listed from the Hub, not
    invented): dispatcher logs flat at the arm root, worker logs under
    ``logs/``, fit worker logs under ``work/fits_a<N>/``, launcher +
    revisions.json + fallbacks env under ``aux/``. Deliberately EXCLUDES
    everything else — shards/npz/caches/eval JSONs have their own upload
    paths, and ``issue-2546-*.json`` sentinels belong to the poller's drained
    namespace, never a log mirror.
    """
    groups: list[tuple[Path, str]] = []
    for p in sorted(log_dir.glob("issue-2546.log*")):
        groups.append((p, ""))
    for p in sorted((out_root / "logs").glob("*.log")):
        groups.append((p, "logs"))
    for p in sorted((out_root / "work" / f"fits_a{arm}").glob("*.log")):
        groups.append((p, f"work/fits_a{arm}"))
    pod_root = log_dir.parent
    aux: list[Path] = [
        *sorted(pod_root.glob("launch_issue_2546*.sh")),
        *sorted(log_dir.glob("launch_issue_2546*.sh")),
        out_root / "revisions.json",
        fallback_env,
    ]
    for p in aux:
        groups.append((p, "aux"))
    return [(p, d) for p, d in groups if p.is_file()]


def stage_files(files: list[tuple[Path, str]], stage_root: Path) -> int:
    """Copy files into the hub-layout staging tree; line-split oversize text.

    Returns the number of files that were line-split. Pieces are named
    ``<name>.partNNN`` and each stays under ~9 MB (flushed at line
    boundaries), keeping every staged blob on the always-open non-LFS path.
    """
    n_split = 0
    for src, rel_dir in files:
        dest_dir = stage_root / rel_dir if rel_dir else stage_root
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src.stat().st_size <= MAX_TEXT_BYTES:
            shutil.copyfile(src, dest_dir / src.name)
            continue
        n_split += 1
        idx = 0
        buf = bytearray()
        with src.open("rb") as fh:
            for line in fh:
                if buf and len(buf) + len(line) > PART_BYTES:
                    (dest_dir / f"{src.name}.part{idx:03d}").write_bytes(buf)
                    idx += 1
                    buf = bytearray()
                buf.extend(line)
        if buf:
            (dest_dir / f"{src.name}.part{idx:03d}").write_bytes(buf)
    return n_split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--fallback-env", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the collected set; no staging, no upload"
    )
    args = ap.parse_args()

    files = collect_log_files(args.log_dir, args.out_root, args.arm, args.fallback_env)
    arm_dir = f"smoke_arm{args.arm}" if args.smoke else f"arm{args.arm}"
    prefix = f"issue2546_cotmap/logs/{arm_dir}"
    if not files:
        print(f"[upload-logs] no log files found under {args.log_dir} / {args.out_root} — skipping")
        return 0
    if args.dry_run:
        for p, d in files:
            print(f"[upload-logs] would stage {p} -> {prefix}/{d + '/' if d else ''}{p.name}")
        print(f"[upload-logs] dry-run: {len(files)} files -> {prefix}")
        return 0

    # Deferred import so --dry-run / the unit tests never touch hub/network.
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO, _upload

    stage_root = Path(tempfile.mkdtemp(prefix="i2546-logstage-"))
    try:
        n_split = stage_files(files, stage_root)
        n_staged = sum(1 for f in stage_root.rglob("*") if f.is_file())
        print(f"[upload-logs] staging {n_staged} files ({n_split} line-split) -> {prefix}")
        res = _upload(stage_root, DEFAULT_DATASET_REPO, "dataset", prefix, raise_on_error=True)
        if not res:
            print(f"[upload-logs] FAILED: empty upload result for {prefix}", file=sys.stderr)
            return 1
        print(f"[upload-logs] complete: {n_staged} files at {res}")
        return 0
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
