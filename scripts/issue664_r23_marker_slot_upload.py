#!/usr/bin/env python3
"""Issue #664 plan-v7 §9.1 step (0.5): upload the 20 marker_slot_stats.json to HF.

The marker `marker_slot_stats.json` files are the PRIMARY marker-gate DV (plan
§6.5 row 2 — the headline near->far gate the §3 kill/success criteria read on).
The r22 marker leg regenerated all 20 LOCALLY on pod-664 (13:38Z; A7 readability
assert PASSed) but the r18/r23 dispatcher does NOT upload the marker_slot HF
surface by design (`issue664_dispatch.py` `_upload_cell_artifacts` is bypassed;
the A7 assert reads it locally). The pod is EPHEMERAL and the analyzer runs
OFF-pod after teardown, so an un-uploaded primary DV is lost on teardown (#613
class). This standalone uploader closes that gap BEFORE the p3 finalize + pod
teardown — a plain upload of the EXISTING files (no regeneration).

Idempotent: skips a file already on HF with a matching content hash (sha256). A
present-but-different file is re-uploaded (overwrites). Fail-loud: the post-upload
HF count for the issue664 marker_slot prefix must be EXACTLY 20 (one per realized
marker cell) with content-hash match, or the script exits non-zero.

Run on the pod (HF creds in env): uv run python scripts/issue664_r23_marker_slot_upload.py
Smoke (1 cell, no fail-loud-on-count): add --smoke (uploads + verifies ONE cell).

Pod-side: NEVER shells task.py. Prints structured lines for the wrapper/poller.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# scripts/ on path so `issue664_common` imports like the dispatcher does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue664_common as C

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_blob_sha256(api, repo_id: str, path_in_repo: str) -> str | None:
    """The git-LFS / blob sha256 of a file already on the Hub, or None if absent.

    marker_slot_stats.json is a small (~25 KB) JSON -> a regular git blob (NOT
    LFS) in the data repo (`*.json` is not LFS-matched), so `get_paths_info`
    returns its blob sha. A non-LFS blob's sha is the git blob hash, which is
    NOT a content sha256, so we cannot content-compare via the blob sha alone;
    instead we DOWNLOAD the small file and sha256 it (it is tiny). Returns the
    content sha256 of the Hub copy, or None if not present."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            revision="main",
        )
    except EntryNotFoundError:
        return None
    except Exception:  # not present / transient -> treat as absent, re-upload
        return None
    return _sha256(Path(local))


def _paths_present_on_hub(api, repo_id: str, paths: list[str]) -> set[str]:
    """The subset of ``paths`` present on the Hub, via TARGETED ``get_paths_info``
    (per-path metadata) -- NOT the full-recursive ``list_repo_files`` tree walk,
    which 504-Gateway-Timeouts on this large (~67k-file) data repo (observed r23 +
    the marker_slot smoke). ``get_paths_info`` queries only the named paths, so it
    is fast + 504-immune regardless of repo size. Retries once on a transient 5xx."""
    import time

    from huggingface_hub.errors import HfHubHTTPError

    last: Exception | None = None
    for attempt in range(3):
        try:
            info = api.get_paths_info(repo_id, paths, repo_type="dataset", revision="main")
            return {getattr(p, "path", None) for p in info if getattr(p, "path", None)}
        except HfHubHTTPError as e:  # transient 5xx
            last = e
            time.sleep(2.0 * (2**attempt))
    raise RuntimeError(f"[marker-slot-upload] get_paths_info failed after 3 attempts: {last}")


def _marker_cells() -> list[C.Cell]:
    """The realized marker cells (gate spine + seed-1042 replication) -- the cells
    that have a marker_slot_stats.json. After plan v7 the grid is 48 cells; the
    marker ones are the 20 mk_* cells."""
    return [c for c in C.realized_grid() if c.behavior == "marker"]


def _local_slot_path(cell: C.Cell) -> Path:
    """Pod-local marker_slot_stats.json for a marker cell (production path, no
    _smoke suffix)."""
    return C.EVAL_ROOT / "marker_slot" / cell.eval_key / "marker_slot_stats.json"


def _path_in_repo(cell: C.Cell) -> str:
    return f"{C.HF_MARKER_SLOT_PREFIX}/{cell.eval_key}/marker_slot_stats.json"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upload issue664 marker_slot_stats.json to HF (v7 §9.1 0.5)."
    )
    ap.add_argument(
        "--smoke", action="store_true", help="upload + verify ONE cell only (no count==20 gate)."
    )
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    cells = _marker_cells()
    if args.smoke:
        cells = cells[:1]
    print(f"[marker-slot-upload] {len(cells)} marker cell(s) to process (smoke={args.smoke})")

    # Fail-loud: every selected cell's local file must exist before we touch HF.
    missing_local = [c.eval_key for c in cells if not _local_slot_path(c).exists()]
    if missing_local:
        raise RuntimeError(
            f"[marker-slot-upload] local marker_slot_stats.json MISSING for "
            f"{len(missing_local)} cell(s): {sorted(missing_local)} -- refusing to "
            "upload an incomplete primary-DV surface (the r22 marker leg should have "
            "written all 20 at 13:38Z)."
        )

    uploaded, skipped = 0, 0
    for cell in cells:
        local = _local_slot_path(cell)
        repo_path = _path_in_repo(cell)
        local_sha = _sha256(local)
        hub_sha = _hf_blob_sha256(api, C.HF_DATA_REPO, repo_path)
        if hub_sha == local_sha:
            print(f"[marker-slot-upload] {cell.eval_key}: already on HF (sha match) -- skip")
            skipped += 1
            continue
        api.upload_file(
            path_or_fileobj=str(local),
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=repo_path,
            commit_message=f"[i664 v7 §9.1(0.5)] marker_slot_stats {cell.eval_key}",
        )
        print(f"[marker-slot-upload] {cell.eval_key}: uploaded -> {repo_path}")
        uploaded += 1

    # Verify via TARGETED get_paths_info on the EXACTLY-expected paths (Python Hub
    # API, never the hf CLI -- upload-policy). NOT list_repo_files: the full-recursive
    # tree walk 504s on this ~67k-file repo (observed r23 + the marker_slot smoke).
    prefix = C.HF_MARKER_SLOT_PREFIX
    expected = sorted(_path_in_repo(c) for c in (cells if args.smoke else _marker_cells()))
    landed = _paths_present_on_hub(api, C.HF_DATA_REPO, expected)
    print(
        f"[marker-slot-upload] uploaded={uploaded} skipped={skipped}; "
        f"{len(landed)}/{len(expected)} expected marker_slot path(s) present on HF under {prefix}/"
    )

    if args.smoke:
        # smoke: confirm THIS cell's file is on HF with a content-hash match.
        c = cells[0]
        repo_path = _path_in_repo(c)
        if repo_path not in landed:
            raise RuntimeError(f"[marker-slot-upload][smoke] {repo_path} not on HF after upload")
        if _hf_blob_sha256(api, C.HF_DATA_REPO, repo_path) != _sha256(_local_slot_path(c)):
            raise RuntimeError(f"[marker-slot-upload][smoke] content-hash mismatch for {repo_path}")
        print(f"[marker-slot-upload][smoke] verified {repo_path} on HF (content-hash match)")
        return 0

    # Production: EXACTLY 20 (one per realized marker cell) with content-hash match.
    missing_on_hub = sorted(set(expected) - landed)
    if missing_on_hub:
        raise RuntimeError(
            f"[marker-slot-upload] post-upload verify FAILED: {len(missing_on_hub)} marker "
            f"cell(s) NOT on HF under {prefix}/: {missing_on_hub}"
        )
    if len(_marker_cells()) != 20:
        raise RuntimeError(
            f"[marker-slot-upload] expected 20 realized marker cells, got {len(_marker_cells())} "
            "-- grid/marker-cell mismatch; investigate realized_grid()."
        )
    # content-hash match for all 20 (download is tiny).
    mismatches = []
    for c in _marker_cells():
        if _hf_blob_sha256(api, C.HF_DATA_REPO, _path_in_repo(c)) != _sha256(_local_slot_path(c)):
            mismatches.append(c.eval_key)
    if mismatches:
        raise RuntimeError(
            f"[marker-slot-upload] content-hash mismatch (HF != local) for: {sorted(mismatches)}"
        )
    print("[marker-slot-upload] PASS: all 20 marker_slot_stats.json on HF with content-hash match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
