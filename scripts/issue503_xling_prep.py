#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (×, →, —) in scientific docstrings + logs.
"""Issue #503 — Bucket A cross-lingual adapter resolver (plan v2 §4.2).

Per plan §4.2 source training:
    "Reuse #235 trained adapters if their weights are still on HF
    (assumption #17). If absent — retrain two adapters under #235's
    recipe: lr=5e-6, r=32, 1 epoch, N≈4990 UltraChat ..."

This script is the single point that resolves the (cell, seed) → adapter
path. Three behaviors per (cell, seed):

1. **Check HF Hub** via ``huggingface_hub.list_repo_files`` for the
   expected subfolder (avoids the silent ``snapshot_download`` siblings
   truncation when repo has >8k files — feedback memory:
   ``snapshot_download_siblings_truncation``).
2. **If present:** record the resolved path; ready for predictor build.
3. **If absent:** print the retraining command + exit code 2 (so the
   sweep dispatcher can fail loud rather than silently swap to a stub).

Writes one JSONL row per resolved cell to
``eval_results/issue503/xling_adapter_resolution.jsonl`` so the sweep
can skip the resolve step on resume.

CLAUDE.md "Pod-side code NEVER shells out to scripts/task.py" — this
script reads from HF only and writes a local JSONL; no task.py calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Force load_dotenv at module top per .claude/agents/experiment-implementer.md
# subprocess-env requirements (every dispatcher with subprocess calls).
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("issue503.xling_prep")

HF_MODEL_REPO_DEFAULT = "superkaiba1/explore-persona-space"


def repo_root() -> Path:
    """Return the repo root by walking up from this script."""
    p = Path(__file__).resolve().parent.parent
    return p


def check_adapter_on_hf(
    repo_id: str, subfolder: str, hf_token: str | None = None
) -> tuple[bool, list[str]]:
    """Check whether ``subfolder`` exists in ``repo_id`` on the HF Hub.

    Returns (present, matching_files). Uses ``list_repo_files`` (NOT
    ``snapshot_download`` — the latter's siblings list is truncated for
    large repos per the snapshot_download_siblings_truncation feedback).
    """
    from huggingface_hub import list_repo_files

    try:
        files = list_repo_files(repo_id=repo_id, token=hf_token)
    except Exception as exc:
        raise RuntimeError(f"HF Hub list_repo_files failed for repo={repo_id!r}: {exc}") from exc
    # subfolder may be 'issue235_xling_en_es_seed0/adapter'; check by prefix.
    matching = [f for f in files if f.startswith(subfolder.rstrip("/") + "/") or f == subfolder]
    return (len(matching) > 0, matching)


def resolution_jsonl_path(repo_root: Path) -> Path:
    p = repo_root / "eval_results" / "issue503" / "xling_adapter_resolution.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_resolution_row(repo_root: Path, row: dict) -> None:
    """Append-only JSONL writer for per-cell resolution rows."""
    p = resolution_jsonl_path(repo_root)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")


def build_retrain_command(cell_id: str, seed: int, lang_pair: tuple[str, str]) -> list[str]:
    """Print the #235 retraining command. The caller invokes this manually
    if the adapter is absent — we do NOT auto-launch a training run from a
    resolver script (separation of concerns + halt-criterion contract).
    """
    src, tgt = lang_pair
    return [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        # #235 recipe per plan v2 §4.2:
        # lr=5e-6, r=32, 1 epoch, N≈4990 UltraChat, (src directive, tgt completion).
        # The condition slug is invented here as a placeholder; the actual
        # Hydra condition file would be added under
        # configs/condition/issue235_xling_{src}_{tgt}.yaml.
        f"condition=issue235_xling_{src}_{tgt}",
        f"seed={seed}",
        "training.lr=5e-6",
        "lora.r=32",
        "training.num_train_epochs=1",
    ]


def resolve_all(
    repo_id: str = HF_MODEL_REPO_DEFAULT,
    hf_token: str | None = None,
    *,
    dry_run: bool = False,
) -> dict:
    """Walk every (cell, seed) in Bucket A and resolve adapter availability.

    Returns a summary dict {cell_id: {seed: {present, retrain_cmd}}}.
    Writes one JSONL row per cell-seed to the resolution log.
    """
    from explore_persona_space.experiments.issue503.crosslingual import (
        XLING_CELLS,
        adapter_subfolder_for_xling,
        all_seeds_for_bucket_a,
    )

    summary: dict = {}
    root = repo_root()
    # A1 and A1' share the same source adapter — dedupe across cells.
    seen_subfolders: set[str] = set()

    for cell in XLING_CELLS:
        per_cell: dict = {}
        for seed in all_seeds_for_bucket_a():
            subfolder = adapter_subfolder_for_xling(cell, seed)
            if subfolder in seen_subfolders:
                # Already resolved through a sibling cell (A1 ↔ A1' share adapters).
                per_cell[str(seed)] = {"deduped": True, "subfolder": subfolder}
                continue
            seen_subfolders.add(subfolder)

            present, files = check_adapter_on_hf(repo_id, subfolder, hf_token=hf_token)
            row: dict = {
                "cell_id": cell.cell_id,
                "seed": seed,
                "repo_id": repo_id,
                "subfolder": subfolder,
                "present": present,
                "n_files": len(files),
            }
            if not present:
                row["retrain_cmd"] = build_retrain_command(
                    cell.cell_id, seed, (cell.source_language, cell.target_language)
                )
                logger.warning(
                    "Bucket A cell=%s seed=%d adapter ABSENT at %s/%s — retrain required.",
                    cell.cell_id,
                    seed,
                    repo_id,
                    subfolder,
                )
            else:
                logger.info(
                    "Bucket A cell=%s seed=%d adapter PRESENT at %s/%s (%d files).",
                    cell.cell_id,
                    seed,
                    repo_id,
                    subfolder,
                    len(files),
                )
            if not dry_run:
                write_resolution_row(root, row)
            per_cell[str(seed)] = row
        summary[cell.cell_id] = per_cell

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("EPM_ISSUE503_XLING_REPO", HF_MODEL_REPO_DEFAULT),
        help="HF model repo to scan for #235 adapter subfolders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolution results without appending to the JSONL log.",
    )
    args = parser.parse_args(argv)

    # Explicit credential assertion per experiment-implementer.md.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set; load_dotenv() did not pick up .env. "
            "Required to query private adapter repos."
        )

    summary = resolve_all(repo_id=args.repo_id, hf_token=hf_token, dry_run=args.dry_run)

    # Exit code: 0 if all PRESENT, 2 if any ABSENT (so a sweep dispatcher
    # can fail loud rather than silently skip cells).
    any_absent = False
    for cell_id, per_cell in summary.items():
        for seed, row in per_cell.items():
            if row.get("deduped"):
                continue
            if not row.get("present", False):
                any_absent = True
                print(f"ABSENT: cell={cell_id} seed={seed} subfolder={row['subfolder']}")
                if "retrain_cmd" in row:
                    print(f"  retrain: {' '.join(row['retrain_cmd'])}")

    if any_absent:
        return 2
    print("Bucket A adapter resolution: all adapters PRESENT on HF.")
    return 0


# Silence unused-import warning when subprocess is only referenced via documentation.
_ = subprocess


if __name__ == "__main__":
    sys.exit(main())
