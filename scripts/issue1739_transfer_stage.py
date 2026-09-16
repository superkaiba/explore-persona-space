"""Restore the same L19 extraction/evaluation coordinates for fixed transfer.

Inputs are selected from immutable historical archives and SHA-verified against
the complete preservation inventory. Existing verified covariance slices are
hard-linked where possible; archives are streamed without retaining whole tars.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(Path("/home/thomasjiralerspong/explore-persona-space/.env"))

from explore_persona_space.orchestrate import hub
from scripts.issue1739_covariance_stage import (
    INVENTORY_REVISION,
    REPO,
    REVISION,
    Progress,
    load_inventory,
    selected_name,
    verified_file,
    extract_members,
    scoped_entries,
    sha256,
    write_json,
)
from scripts.issue1739_map963k_slice import ParallelRangeReader, head_size

BEHAVIORS = ("evil", "sycophancy", "hallucination")
KINDS = ("context_end", "t1")


def members(inventory: list[dict], store: str) -> dict[str, dict]:
    prefix = (
        "run/reused/store/wcrung_capture_store/wildchat/"
        if store == "wildchat"
        else f"run/reused/store/{store}/"
    )
    found = {
        row["key"][len(prefix) :]: row
        for row in inventory
        if row["key"].startswith(prefix) and selected_name(row["key"][len(prefix) :], (19,), KINDS)
    }
    for kind in KINDS:
        if not any(name.startswith(f"{kind}_L19") for name in found):
            raise ValueError(f"Missing {store} {kind} in pinned member inventory")
    if not any(name.startswith("row_index") for name in found):
        raise ValueError(f"Missing {store} row identities")
    return found


def reuse_verified(dest: Path, old: Path, expected: dict[str, dict]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, record in expected.items():
        if verified_file(dest / name, record):
            continue
        if verified_file(old / name, record):
            os.link(old / name, dest / name)


def stage_archive(store: str, args, inventory, progress, token: str) -> None:
    expected = members(inventory, store)
    dest = args.dest / store
    reuse_verified(dest, args.reuse / store, expected)
    if not all(verified_file(dest / name, r) for name, r in expected.items()):
        path = f"issue1739_ctxmap/capture_store/{store}/{store}.tar"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
        total = head_size(url, token)
        progress.emit(phase=store, bytes_fetched=0, files_completed=0, total=total)
        raw = ParallelRangeReader(url, token=token, total=total, workers=args.workers)
        progress.reader = raw
        try:
            with io.BufferedReader(raw, buffer_size=8 << 20) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    extract_members(archive, expected, dest, progress)
        finally:
            progress.emit()
            progress.reader = None
            raw.close()
    write_json(
        dest / "slice_manifest.json",
        {
            "revision": REVISION,
            "inventory_revision": INVENTORY_REVISION,
            "store": store,
            "layers": [19],
            "kinds": list(KINDS),
            "members": {n: {k: r[k] for k in ("bytes", "sha256")} for n, r in expected.items()},
            "complete": True,
        },
    )


def stage_wildchat(args, inventory, progress, token: str) -> None:
    expected = members(inventory, "wildchat")
    dest = args.dest / "wildchat"
    reuse_verified(dest, args.reuse / "wildchat", expected)
    prefix = "issue1739_ctxmap/wildchat_rung/capture_store/wildchat"
    entries = [
        r
        for r in scoped_entries(prefix, REVISION, token)
        if hasattr(r, "size") and selected_name(Path(r.path).name, (19,), KINDS)
    ]
    if {Path(r.path).name for r in entries} != set(expected):
        raise ValueError("WildChat source/inventory member set mismatch")
    for i, entry in enumerate(entries, 1):
        path = dest / Path(entry.path).name
        if not verified_file(path, expected[path.name]):
            hub.stage_hub_file(
                REPO, entry.path, path, repo_type="dataset", revision=REVISION, token=token or None
            )
            if not verified_file(path, expected[path.name]):
                raise RuntimeError(f"Stage did not produce {path}")
        progress.emit(phase="wildchat", files_completed=i, files_total=len(entries))
    write_json(
        dest / "slice_manifest.json",
        {
            "revision": REVISION,
            "inventory_revision": INVENTORY_REVISION,
            "store": "wildchat",
            "layers": [19],
            "kinds": list(KINDS),
            "members": {n: {k: r[k] for k in ("bytes", "sha256")} for n, r in expected.items()},
            "complete": True,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--progress", type=Path, required=True)
    ap.add_argument("--reuse", type=Path, default=Path("/dev/shm/issue1739-covariance/inputs"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extraction-only", action="store_true")
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    progress = Progress(args.progress)
    try:
        inventory = load_inventory(args.dest, token)
        for behavior in BEHAVIORS:
            stage_archive(f"{behavior}_extraction", args, inventory, progress, token)
        if not args.extraction_only:
            for behavior in BEHAVIORS:
                stage_archive(f"{behavior}_labeling", args, inventory, progress, token)
            stage_wildchat(args, inventory, progress, token)
        files = {
            str(p.relative_to(args.dest)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(args.dest.rglob("*"))
            if p.is_file() and not p.name.startswith(".") and p.name != "manifest.json"
        }
        write_json(
            args.dest / "manifest.json",
            {
                "repo": REPO,
                "revision": REVISION,
                "inventory_revision": INVENTORY_REVISION,
                "layer": 19,
                "extraction_filter": "all-row instruction contrast, not judge filtered",
                "extraction_only": args.extraction_only,
                "files": files,
                "complete": True,
            },
        )
        progress.emit(phase="complete", files_completed=len(files))
    finally:
        progress.close()


if __name__ == "__main__":
    main()
