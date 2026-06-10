#!/usr/bin/env python3
"""Step 4-6 — symlink shim + provenance manifest for v2 dispatcher launch.

Plan v2 §4 Step 4-6. The Phase-C dispatcher (`scripts/issue_519_dispatch.py`)
resolves the adapter path as ``{output_dir}/{arm}_seed{S}/adapter/``.
v2 keeps the marker arm pointed at its v1-staged adapters and points the
EM arm at the retrained adapters via a symlink shim.

For each seed in {42, 137, 256}:

  1. If ``eval_results/issue_521/em_seed{S}/adapter`` is a symlink,
     remove it (clean re-run support).
  2. If ``eval_results/issue_521/em_seed{S}/`` is a real directory
     (the v1 #519-EM stage), rename it to
     ``eval_results/issue_521/em_seed{S}.v1_519_failed_floor/`` so
     the failure evidence is preserved on disk.
  3. Create the symlink
     ``eval_results/issue_521/em_seed{S}/adapter -> ../em_turner_seed{S}/adapter``.

After the symlink shim, write ``v2_adapter_provenance.json`` recording
which adapters are wired in and the rig-confound caveat the
clean-result must carry.

Run::

    uv run python scripts/issue_521_provenance_v2.py \\
        --output-dir eval_results/issue_521 \\
        [--seeds 42 137 256] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 137, 256)


def _shim_one(*, seed: int, output_dir: Path, dry_run: bool) -> dict:
    """Apply the v2 symlink shim for one EM seed.

    Returns a record dict describing what was done (for the manifest).
    """
    em_dir = output_dir / f"em_seed{seed}"
    em_adapter = em_dir / "adapter"
    em_turner_adapter_rel = Path("..") / f"em_turner_seed{seed}" / "adapter"
    em_turner_adapter_abs = output_dir / f"em_turner_seed{seed}" / "adapter"
    archive_dir = output_dir / f"em_seed{seed}.v1_519_failed_floor"

    record: dict = {
        "seed": seed,
        "em_adapter_symlink": str(em_adapter),
        "em_turner_target": str(em_turner_adapter_abs),
        "v1_archive": None,
        "actions": [],
    }

    # The v2 target must exist BEFORE we touch the v1 dir; otherwise we'd
    # break the dispatcher in a half-shimmed state.
    if not em_turner_adapter_abs.exists():
        raise RuntimeError(
            f"v2 retrained adapter dir does NOT exist at {em_turner_adapter_abs}. "
            f"Run scripts/issue_521_stage_em_turner_adapters.py before applying "
            f"the symlink shim."
        )

    # Step 1: remove existing symlink at em_seed{S}/adapter (idempotent).
    if em_adapter.is_symlink():
        if not dry_run:
            em_adapter.unlink()
        record["actions"].append("removed_existing_symlink")

    # Step 2: archive a real v1 em_seed{S} directory by renaming.
    if em_dir.exists() and not em_dir.is_symlink():
        # If the only remaining child is an `adapter` symlink we just
        # removed (the dir would be empty), still rename — the dir
        # itself is v1 cruft we want archived even if its body is gone.
        if archive_dir.exists():
            record["actions"].append("v1_archive_already_present")
            record["v1_archive"] = str(archive_dir)
        else:
            if not dry_run:
                em_dir.rename(archive_dir)
            record["actions"].append("archived_v1_dir_to_v1_519_failed_floor")
            record["v1_archive"] = str(archive_dir)

    # Step 3: create the symlink (after the rename so we don't clash).
    em_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
    if not dry_run:
        # If something else materialized at em_adapter (e.g. unlinked above
        # then dir recreated), make sure the symlink path is clear.
        if em_adapter.exists() or em_adapter.is_symlink():
            em_adapter.unlink()
        em_adapter.symlink_to(em_turner_adapter_rel)
        # Validate the symlink resolves to a real dir.
        resolved = em_adapter.resolve()
        if not resolved.is_dir():
            raise RuntimeError(
                f"symlink {em_adapter} -> {em_turner_adapter_rel} did not resolve "
                f"to a real dir (resolved={resolved})."
            )
    record["actions"].append("symlinked")
    record["symlink_target_rel"] = str(em_turner_adapter_rel)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v2 symlink shim + provenance manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/issue_521",
        help="Top-level output dir under which the cell subtrees live.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="EM seeds to shim (default 42 137 256).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without touching the filesystem.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if not output_dir.exists():
        raise RuntimeError(f"--output-dir {output_dir} does not exist.")

    seed_records: list[dict] = []
    for seed in args.seeds:
        rec = _shim_one(seed=seed, output_dir=output_dir, dry_run=args.dry_run)
        logger.info(
            "[shim] em_seed%d: actions=%s archive=%s",
            seed,
            rec["actions"],
            rec.get("v1_archive"),
        )
        seed_records.append(rec)

    # Provenance manifest. Always written (dry-run prints + still writes,
    # so a downstream consumer can read a stable artifact regardless).
    manifest = {
        "issue": 521,
        "plan_version": "v2",
        "marker_seeds": {
            int(s): f"issue_519/marker_seed{s}@main (carry-forward from v1, unchanged)"
            for s in (42, 137, 256)
        },
        "em_seeds_pointer": {
            int(s): (
                f"adapters/issue_521/em_turner_seed{s}@main (symlinked into em_seed{s}/adapter)"
            )
            for s in args.seeds
        },
        "em_seeds_v1_failed_archive": {
            int(s): (
                f"eval_results/issue_521/em_seed{s}.v1_519_failed_floor/ "
                f"(DO NOT use; failed em-rate gate at 0%)"
            )
            for s in args.seeds
        },
        "rig_caveat": (
            "marker arm = #519 contrastive-persona-rig SFT with marker-only loss; "
            "em arm = #458 plain Turner SFT with full-response CE. CROSS-ARM "
            "CONTRAST POOLS RIG DIFFERENCE — clean-result MUST narrate this "
            "(plan v2 §6.4 concern 7, §12 #28)."
        ),
        "seed_records": seed_records,
        "dry_run": args.dry_run,
    }
    manifest_path = output_dir / "v2_adapter_provenance.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "[phase=done] %s provenance manifest %s (dry_run=%s)",
        "would write" if args.dry_run else "wrote",
        manifest_path,
        args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
