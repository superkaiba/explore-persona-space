"""Route hot per-cell .npz writes to local-SSD scratch, then batch-materialize
to the canonical (network-backed) destination at cell/sweep end.

On GCE the canonical destination is a network-backed Persistent Disk whose
virtual NIC is shared with systemd-networkd DHCP renewal; a per-cell .npz
write storm there contributes to the hung-RUNNING wedge (#667/#671/#674).
Writing to a local-SSD scratch mirror first and copying the whole cell over
in one batch keeps the hot write path off the network plane.

The GCE startup script sets EPS_SCRATCH_DIR (default /tmp/eps_scratch, a
local-SSD path) via the workload_cmd block, and gcp.py forwards any
dispatch-process EPS_SCRATCH_DIR through STARTUP_PASSTHROUGH_ENV_KEYS; off
GCE the var is UNSET, so both functions are thin pass-throughs (writes go
straight to the canonical dir, no indirection) so non-GCE behavior is
byte-for-byte unchanged.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENV_SCRATCH_DIR = "EPS_SCRATCH_DIR"  # unset => pass-through (non-GCE default)


def _scratch_root() -> Path | None:
    """Return the configured scratch root, or None when pass-through (unset)."""
    raw = os.environ.get(ENV_SCRATCH_DIR, "").strip()
    return Path(raw) if raw else None


def scratch_path_for(canonical_dest: Path, issue: int) -> Path:
    """Map a canonical PD path to its local-SSD scratch mirror.

    When EPS_SCRATCH_DIR is unset (non-GCE), returns ``canonical_dest``
    unchanged (pass-through). When set, returns
    ``<root>/issue<issue>/<canonical-dest-with-leading-slash-stripped>`` so
    the scratch tree MIRRORS the canonical structure — distinct cell dirs map
    to distinct scratch dirs and never collide.

    Accepts either a directory (a cell dir) or a file path; the relative
    structure under the scratch root mirrors whatever was passed.
    """
    canonical_dest = Path(canonical_dest)
    root = _scratch_root()
    if root is None:
        return canonical_dest  # pass-through (non-GCE)
    # Mirror the FULL canonical path under <root>/issue<N>/ so two cells with
    # the same basename (different parents) never collide. Strip the anchor
    # ("/" or "C:\\") so the join stays relative.
    rel = Path(*canonical_dest.parts[1:]) if canonical_dest.is_absolute() else canonical_dest
    return root / f"issue{issue}" / rel


def materialize_to_canonical(scratch_dir: Path, canonical_dir: Path) -> None:
    """Batch-copy every .npz in ``scratch_dir`` to ``canonical_dir`` atomically,
    then delete ``scratch_dir``.

    No-op when ``scratch_dir == canonical_dir`` (the pass-through case — the
    files were already written directly to canonical). Otherwise:
      * mkdir canonical_dir (parents, exist_ok);
      * for each *.npz in scratch_dir: copy to ``canonical_dir/<name>.tmp``
        then ``os.replace`` to the final name (atomic within the canonical
        filesystem — no half-written .npz ever visible to a reader/resume-skip);
      * rmtree(scratch_dir).

    Sentinel contract: this NEVER writes the .done sentinel — the caller writes
    it to the CANONICAL dir AFTER this returns AND after its own complement
    check (issue667: assert_full_npz_complement). On a partial failure here the
    scratch dir is left INTACT (re-raise) so a retry can re-materialize, and the
    canonical .done is never written (the caller's assert runs after us).
    """
    scratch_dir, canonical_dir = Path(scratch_dir), Path(canonical_dir)
    if scratch_dir.resolve() == canonical_dir.resolve():
        return  # pass-through: files already in canonical_dir
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(scratch_dir.glob("*.npz")):
        dst = canonical_dir / src.name
        tmp = canonical_dir / f"{src.name}.{os.getpid()}.tmp"
        shutil.copyfile(src, tmp)  # raises on disk-full / IO error (fail loud)
        os.replace(tmp, dst)  # atomic rename within canonical fs
    shutil.rmtree(scratch_dir)
