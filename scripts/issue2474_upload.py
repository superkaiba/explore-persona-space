"""issue #2474 — bulk upload of the base-model capture artifacts to the data repo.

The reused capture entrypoint (``scripts/issue2379_capture.py``) self-uploads
under the slug ``issue2379_reelicit``. #2474's base-model captures are its OWN
artifacts, so the phases run with ``--no-upload`` and this script does bulk
directory uploads into issue-2474 prefixes instead (upload policy: one
``upload_folder``-class call per tree, never a per-file ``upload_file`` loop).

Legs:
  * tensors (always): ``<out-dir>/capture_tensors`` -> ``issue2474_prefit/capture_tensors``
    (.pt bundles + .pt.meta.json sidecars; the whole-directory upload carries both).
  * raw completions (``--include-rawcomp``, plan v5 section 4 P-A2): the capture's
    ceiling rollout TEXT — written by ``issue2379_capture.py::_write_rawcomp`` to
    ``<out-dir>/rawcomp_capture/<stage>/<model_name>/raw_completions.json`` (the
    REALIZED producer path; ``rawcomp_dir = out_dir / "rawcomp_capture"`` at
    issue2379_capture.py:1251) — uploads to ``issue2474_prefit/raw_completions/``
    with the ``raw_completions/<stage>/`` layout (CLAUDE.md Upload Policy).
    Fail-loud on a missing/empty tree: a generation stage that drops its
    generations is an upload-verification FAIL by rule.

Persist-by-default: every produced bundle uploads, whether or not the current
analysis consumes it — a sibling or follow-up may.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.orchestrate.hub import upload_dataset  # noqa: E402

logger = logging.getLogger("issue2474_upload")

SLUG = "issue2474_prefit"


def _log_tree(files: list[Path], base: Path, what: str) -> None:
    """Log the file listing + total size for one upload tree."""
    total = sum(p.stat().st_size for p in files)
    logger.info("staging %d %s file(s), %.2f GiB, from %s", len(files), what, total / 2**30, base)
    for p in files:
        logger.info("  %s (%.1f MiB)", p.relative_to(base), p.stat().st_size / 2**20)


def upload_tensors(out_dir: Path, dry_run: bool) -> None:
    """Tensor leg (pre-existing behavior, unchanged): one bulk folder upload."""
    tensor_dir = out_dir / "capture_tensors"
    if not tensor_dir.is_dir():
        raise RuntimeError(f"nothing to upload: {tensor_dir} is not a directory")

    bundles = sorted(tensor_dir.rglob("*.pt"))
    if not bundles:
        raise RuntimeError(f"nothing to upload: no .pt bundles under {tensor_dir}")

    _log_tree(bundles, tensor_dir, "bundle")
    if dry_run:
        logger.info("[dry-run] would upload -> %s/capture_tensors", SLUG)
        return

    url = upload_dataset(str(tensor_dir), path_in_repo=f"{SLUG}/capture_tensors")
    if not url:
        raise RuntimeError(
            "capture-tensor upload returned no path — the bulk upload failed "
            f"(tree: {tensor_dir}); the helper swallows failures to '' (r1 g2 concern 2)"
        )
    logger.info("[phase=upload_done] %s", url)


def upload_rawcomp(out_dir: Path, rawcomp_dir: Path | None, dry_run: bool) -> None:
    """Raw-completions leg (plan v5 section 4 P-A2): ceiling rollout TEXT -> HF.

    One bulk ``upload_folder`` commit over the whole tree; destination layout
    ``issue2474_prefit/raw_completions/<stage>/<model_name>/raw_completions.json``
    (non-LFS text path). Fail-loud on missing dir / zero files / zero-byte files
    / an empty upload return — rollout text is never silently droppable.
    """
    src = rawcomp_dir if rawcomp_dir is not None else out_dir / "rawcomp_capture"
    if not src.is_dir():
        raise RuntimeError(
            f"--include-rawcomp: {src} is not a directory — the capture round's rollout "
            "text is missing (issue2379_capture.py writes it to "
            "<out-dir>/rawcomp_capture/<stage>/<model_name>/raw_completions.json; pass "
            "--rawcomp-dir if it was relocated)"
        )
    files = sorted(src.rglob("raw_completions.json"))
    if not files:
        raise RuntimeError(
            f"--include-rawcomp: no raw_completions.json under {src} — refusing to "
            "'upload' an empty generation tree (upload-verification FAIL class)"
        )
    empty = [p for p in files if p.stat().st_size == 0]
    if empty:
        raise RuntimeError(
            "--include-rawcomp: zero-byte rollout-text file(s): " + ", ".join(str(p) for p in empty)
        )

    _log_tree(files, src, "raw-completions")
    if dry_run:
        logger.info("[dry-run] would upload -> %s/raw_completions", SLUG)
        return

    url = upload_dataset(str(src), path_in_repo=f"{SLUG}/raw_completions")
    if not url:
        raise RuntimeError(
            "raw-completions upload returned no path — the bulk upload failed "
            f"(tree: {src}); rollout text MUST land before pod termination"
        )
    logger.info("[phase=rawcomp_upload_done] %s", url)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="eval_results/issue_2474")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--include-rawcomp",
        action="store_true",
        help="also upload the capture's rollout text tree (plan v5 section 4 P-A2)",
    )
    ap.add_argument(
        "--rawcomp-dir",
        default=None,
        help="rollout-text tree root (default <out-dir>/rawcomp_capture, the producer path)",
    )
    ap.add_argument(
        "--rawcomp-only",
        action="store_true",
        help="skip the tensors leg; upload ONLY the rollout-text tree (implies "
        "--include-rawcomp) — the escape hatch when tensors already landed "
        "(r1 g2 concern 1: the rawcomp leg was gated behind the tensors leg)",
    )
    ap.add_argument(
        "--import-check", action="store_true", help="argcheck + call-arity bind, then exit 0"
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    out_dir = Path(args.out_dir)
    if not args.rawcomp_only:
        upload_tensors(out_dir, args.dry_run)
    if args.include_rawcomp or args.rawcomp_only:
        upload_rawcomp(out_dir, Path(args.rawcomp_dir) if args.rawcomp_dir else None, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
