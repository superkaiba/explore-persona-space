"""Task #2336 batch-2 A5 byte oracles (plan v4 §6 A5 / §3 prediction 3; round-6
blocker ``a5-byte-identity-coverage``).

Old-vs-new output oracles for the batch-2 migrated writer shapes, per the A5
acceptance row ("json/jsonl/text: byte-identical; pt/npz: load-equality"):

- **jsonl byte-identity** — the ``scripts/issue1092_build_corpus.py`` stream-pool
  serialization (``json.dumps(row, ensure_ascii=False)`` + ``"\\n"`` per row,
  :760-764). The write is inline in ``_stream_with_cache`` (not importable
  without a streaming source), so the migrated with-``atomic_replace`` form is
  reproduced VERBATIM here and anchored against the live source text; the
  pre-migration inline form (parent commit ``f63dc3da21~1`` :759-764) is
  executed verbatim as the old-form oracle.
- **text byte-identity** — the REAL migrated ``issue1773_common.write_jsonl_sharded``
  (``tmp.write_text("\\n".join(buf) + "\\n")`` through ``atomic_replace``, :523-524)
  vs the pre-migration ``_flush`` form (``.tmp_<name>`` + ``tmp.replace``) executed
  verbatim, single-shard AND forced 2-shard split.
- **npz load-equality** — the REAL migrated ``issue1689_derived_vs_free._atomic_savez``
  (handle-form ``np.savez`` through ``atomic_replace``, edge (c)) vs the
  pre-migration path-form (``<stem>.tmp.npz`` + ``os.replace``) executed verbatim;
  npz container bytes are not guaranteed stable, so the oracle is
  ``np.load`` key/dtype/value equality (the plan's pt/npz load-equality shape).

Batch-1 precedent: ``tests/test_clean_experiment_downloads_atomic_writers.py``
(json byte-identity). Every test also asserts zero ``*.tmp*`` residue (old
``.tmp_*``/``*.tmp.npz`` and new ``<name>.<pid>.<uuid8>.tmp`` shapes alike).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue1689_derived_vs_free as DVF  # noqa: E402
import issue1773_common as CM  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

# Non-ASCII + Unicode line/paragraph separators: exactly the byte surface
# ``ensure_ascii=False`` leaves raw (built via chr() so the sensitive
# codepoints never transit tool parameters as literals).
_ROWS = [
    {"conversation_id": "a1", "text": "plain ascii", "n": 1},
    {"conversation_id": "b2", "text": "caf" + chr(0xE9) + " " + chr(0x2028) + "ls", "n": 2},
    {"conversation_id": "c3", "text": "nel:" + chr(0x85) + " ps:" + chr(0x2029), "n": 3},
]


def _residue(root: Path) -> list[Path]:
    """Every leftover temp file under *root* (old AND new temp-name shapes)."""
    return sorted(p for p in root.rglob("*.tmp*") if p.is_file())


# --------------------------------------------------------------------------
# jsonl: issue1092_build_corpus.py stream-pool serialization
# --------------------------------------------------------------------------

_I1092_MIGRATED_BLOCK = "\n".join(
    [
        "    with atomic_replace(pool_path) as tmp_pool:",
        '        with open(tmp_pool, "w", encoding="utf-8") as f:',
        "            for row in results:",
        "                f.write(json.dumps(row, ensure_ascii=False))",
        '                f.write("\\n")',
    ]
)


def test_issue1092_pool_jsonl_byte_identity(tmp_path: Path) -> None:
    """The migrated pool write produces bytes identical to the pre-migration
    inline form (``f63dc3da21~1`` :759-764) on a fixed row set."""
    # Anchor: the reproduction below must match the LIVE migrated source.
    src = (REPO / "scripts" / "issue1092_build_corpus.py").read_text(encoding="utf-8")
    assert _I1092_MIGRATED_BLOCK in src, (
        "issue1092_build_corpus.py pool serialization drifted from the form "
        "this oracle reproduces - update _I1092_MIGRATED_BLOCK and the "
        "reproduction together"
    )

    results = _ROWS
    # Migrated form (live source :760-764), reproduced verbatim.
    pool_path = tmp_path / "new" / "wildchat.jsonl"
    with atomic_replace(pool_path) as tmp_pool, open(tmp_pool, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    # Pre-migration inline form (f63dc3da21~1 :759-764), executed verbatim.
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_pool = old_dir / "wildchat.jsonl"
    tmp_pool_old = old_dir / (old_pool.name + ".tmp")
    with open(tmp_pool_old, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp_pool_old, old_pool)

    assert pool_path.read_bytes() == old_pool.read_bytes()
    assert _residue(tmp_path) == []


# --------------------------------------------------------------------------
# text: issue1773_common.write_jsonl_sharded (REAL migrated function)
# --------------------------------------------------------------------------


def _old_flush_shard(rows: list[dict], out_dir: Path, name: str) -> Path:
    """The pre-migration ``_flush`` write (f63dc3da21~1 :521-524), verbatim."""
    buf = [json.dumps(r, ensure_ascii=False) for r in rows]
    p = out_dir / name
    tmp = p.parent / f".tmp_{p.name}"
    tmp.write_text("\n".join(buf) + "\n")
    tmp.replace(p)
    return p


def test_write_jsonl_sharded_single_shard_byte_identity(tmp_path: Path) -> None:
    """The REAL migrated ``write_jsonl_sharded`` shard bytes equal the
    pre-migration ``_flush`` form's bytes on a fixed row set."""
    new_dir = tmp_path / "new"
    shards = CM.write_jsonl_sharded(list(_ROWS), new_dir, "pool")
    assert [p.name for p in shards] == ["pool.shard00.jsonl"]

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_shard = _old_flush_shard(_ROWS, old_dir, "pool.shard00.jsonl")

    assert shards[0].read_bytes() == old_shard.read_bytes()
    assert _residue(tmp_path) == []
    # Manifest still lands beside the shards (unchanged non-atomic write).
    assert (new_dir / "pool.manifest.json").is_file()


def test_write_jsonl_sharded_two_shard_byte_identity(tmp_path: Path) -> None:
    """Forced 2-shard split: per-shard bytes equal the old form's; the split
    arithmetic (unchanged by the migration) puts one row per shard when
    ``max_bytes`` equals the first row's encoded line size + 1."""
    rows = [_ROWS[0], _ROWS[1]]
    max_bytes = len(json.dumps(rows[0], ensure_ascii=False).encode()) + 1

    new_dir = tmp_path / "new"
    shards = CM.write_jsonl_sharded(rows, new_dir, "pool", max_bytes=max_bytes)
    assert [p.name for p in shards] == ["pool.shard00.jsonl", "pool.shard01.jsonl"]

    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old0 = _old_flush_shard([rows[0]], old_dir, "pool.shard00.jsonl")
    old1 = _old_flush_shard([rows[1]], old_dir, "pool.shard01.jsonl")

    assert shards[0].read_bytes() == old0.read_bytes()
    assert shards[1].read_bytes() == old1.read_bytes()
    assert _residue(tmp_path) == []


# --------------------------------------------------------------------------
# npz: issue1689_derived_vs_free._atomic_savez (REAL migrated function)
# --------------------------------------------------------------------------


def test_atomic_savez_npz_load_equality(tmp_path: Path) -> None:
    """The REAL migrated handle-form ``_atomic_savez`` loads equal to the
    pre-migration path-form write (f63dc3da21~1 :217-222) - key set, dtypes,
    and values (npz container bytes are not guaranteed stable, so the A5
    oracle for pt/npz is load-equality, not byte-identity)."""
    arrays = {
        "a": np.arange(6, dtype=np.int64).reshape(2, 3),
        "b": np.linspace(0.0, 1.0, 5).astype(np.float32),
    }

    new_path = tmp_path / "new" / "bundle.npz"
    DVF._atomic_savez(new_path, **arrays)

    # Pre-migration path-form (f63dc3da21~1 :217-222), executed verbatim.
    old_path = tmp_path / "old" / "bundle.npz"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = old_path.with_name(old_path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, old_path)

    with np.load(new_path) as new_z, np.load(old_path) as old_z:
        assert sorted(new_z.files) == sorted(old_z.files) == ["a", "b"]
        for key in new_z.files:
            assert new_z[key].dtype == old_z[key].dtype
            np.testing.assert_array_equal(new_z[key], old_z[key])

    assert _residue(tmp_path) == []
