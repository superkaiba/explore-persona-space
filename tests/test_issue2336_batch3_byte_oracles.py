"""Task #2336 batch-3 A5 byte oracle (plan v4 §6 A5; ground-truthed scoping duty).

Batch 3's migrated serialization shapes, greped from the batch diff: json
(``write_text(json.dumps(...))`` + ``json.dump`` into an open text handle),
jsonl (open-handle and joined ``write_text`` forms), text (pre-joined
``write_text``), npz (handle-form ``np.savez``), pt (``torch.save``) — every
one already carries a landed first-instance oracle (json: batch 1,
``tests/test_clean_experiment_downloads_atomic_writers.py``; jsonl + text:
batch 2 r6; npz: r6; pt: r7 — all in
``tests/test_issue2336_batch2_byte_oracles.py``). Exactly ONE batch-3 shape
has no landed oracle:

- **raw-bytes byte-identity** — the ``scripts/issue2054_phase_a.py`` prejudge
  shard reassembly (``tmp.open("wb")`` + ``out.write(lp.read_bytes())`` per
  local part). The write is inline in the staging path (not importable without
  HF staging), so the migrated with-``atomic_replace`` form is reproduced
  VERBATIM here and anchored against the live source text; the pre-migration
  inline form (batch-2 tip ``5012fcc6e7`` :1261-1265) is executed verbatim as
  the old-form oracle. Raw bytes round-trip exactly, so the oracle is
  byte-identity (the json/jsonl/text bar), not load-equality.

Every test also asserts zero ``*.tmp*`` residue (old ``<name>.tmp`` and new
``<name>.<pid>.<uuid8>.tmp`` shapes alike).
"""

from __future__ import annotations

import os
from pathlib import Path

from explore_persona_space.atomic_io import atomic_replace

REPO = Path(__file__).resolve().parent.parent

# Binary payloads: non-UTF8 bytes + embedded NUL — exactly the surface a text
# round-trip would corrupt and a bytes round-trip must preserve.
_PARTS = [
    b"\x00\x01\x02 plain-ascii tail\n",
    b"\xff\xfe\x80\x81 non-utf8 run \x00\n",
    b"",  # empty shard part: concat must tolerate it
]

_I2054_MIGRATED_BLOCK = "\n".join(
    [
        '        with atomic_replace(target) as tmp, tmp.open("wb") as out:',
        "            for lp in local_parts:",
        "                out.write(lp.read_bytes())",
    ]
)


def _residue(root: Path) -> list[Path]:
    """Every leftover temp file under *root* (old AND new temp-name shapes)."""
    return sorted(p for p in root.rglob("*.tmp*") if p.is_file())


def test_issue2054_prejudge_reassembly_bytes_identity(tmp_path: Path) -> None:
    """The migrated raw-bytes shard reassembly produces bytes identical to the
    pre-migration inline form (``5012fcc6e7`` :1261-1265) on a fixed part set."""
    # Anchor: the reproduction below must match the LIVE migrated source.
    src = (REPO / "scripts" / "issue2054_phase_a.py").read_text(encoding="utf-8")
    assert _I2054_MIGRATED_BLOCK in src, (
        "issue2054_phase_a.py prejudge reassembly drifted from the form this "
        "oracle reproduces - update _I2054_MIGRATED_BLOCK and the "
        "reproduction together"
    )

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    local_parts = []
    for i, payload in enumerate(_PARTS):
        lp = parts_dir / f"part{i:02d}.jsonl"
        lp.write_bytes(payload)
        local_parts.append(lp)

    # Migrated form (live source), reproduced verbatim.
    target = tmp_path / "new" / "prejudge.jsonl"
    target.parent.mkdir()
    with atomic_replace(target) as tmp, tmp.open("wb") as out:
        for lp in local_parts:
            out.write(lp.read_bytes())

    # Pre-migration inline form (5012fcc6e7 :1261-1265), executed verbatim.
    old_target = tmp_path / "old" / "prejudge.jsonl"
    old_target.parent.mkdir()
    tmp_old = old_target.with_name(old_target.name + ".tmp")
    with tmp_old.open("wb") as out:
        for lp in local_parts:
            out.write(lp.read_bytes())
    os.replace(tmp_old, old_target)

    assert target.read_bytes() == old_target.read_bytes() == b"".join(_PARTS)
    assert _residue(tmp_path) == []
