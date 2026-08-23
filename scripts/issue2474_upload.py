"""issue #2474 — bulk upload of the base-model capture tensors to the data repo.

The reused capture entrypoint (``scripts/issue2379_capture.py``) self-uploads
under the slug ``issue2379_reelicit``. #2474's base-model captures are its OWN
artifacts, so the phases run with ``--no-upload`` and this script does one bulk
directory upload into an issue-2474 prefix instead (upload policy: one
``upload_folder``-class call, never a per-file ``upload_file`` loop).

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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="eval_results/issue_2474")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tensor_dir = Path(args.out_dir) / "capture_tensors"
    if not tensor_dir.is_dir():
        raise RuntimeError(f"nothing to upload: {tensor_dir} is not a directory")

    bundles = sorted(tensor_dir.rglob("*.pt"))
    if not bundles:
        raise RuntimeError(f"nothing to upload: no .pt bundles under {tensor_dir}")

    total = sum(p.stat().st_size for p in bundles)
    logger.info("staging %d bundle(s), %.2f GiB, from %s", len(bundles), total / 2**30, tensor_dir)
    for p in bundles:
        logger.info("  %s (%.1f MiB)", p.relative_to(tensor_dir), p.stat().st_size / 2**20)

    if args.dry_run:
        logger.info("[dry-run] would upload -> %s/capture_tensors", SLUG)
        return 0

    url = upload_dataset(str(tensor_dir), path_in_repo=f"{SLUG}/capture_tensors")
    logger.info("[phase=upload_done] %s", url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
