"""Task #2336 batch-5 A5 byte oracle (plan v4 §6 A5; ground-truthed scoping duty).

Batch 5's migrated serialization shapes, greped from the batch diff: json
(``write_text(json.dumps(...))`` x30), jsonl (open-handle ``fh.write`` loops x8
+ the joined ``write_text("".join(...))`` shard form x1), and npz (handle-form
``np.savez`` x14 — 13 ``issue2476_turnavg_sae.py`` path-form conversions + the
``issue823_ladder_fits.py`` ``save_chunk``) — every one already carries a
landed first-instance oracle (json: batch 1,
``tests/test_clean_experiment_downloads_atomic_writers.py``; jsonl + text +
npz: batch 2; pt: batch 2; raw bytes: batch 3; npy-via-``open_memmap``:
batch 4 — ``tests/test_issue2336_batch{2,3,4}_byte_oracles.py``). Exactly ONE
batch-5 shape has no landed oracle:

- **npy via handle-form ``np.save``** — the
  ``src/explore_persona_space/experiments/issue_1739/capture.py``
  ``write_store_shard`` activation-shard writer: the pre-migration form wrote
  ``np.save(<dot-prefixed .npy tmp path>)`` (path form; the hidden ``.tmp_``
  name kept the ``.npy`` suffix so np.save did not append), the migrated form
  writes ``np.save(fh, arr)`` through an open handle on the yielded
  atomic_replace tmp (which ends ``.tmp`` — a path-form call would append
  ``.npy`` and the replace would die FileNotFoundError). The write is inline
  in a GPU capture helper (not executable at test time without the vLLM lane),
  so the migrated with-``atomic_replace`` form is reproduced VERBATIM here and
  anchored against the live source text; the pre-migration form (batch-4 tip
  ``7a4fd3c319`` capture.py:274-279) is executed verbatim as the old-form
  oracle. ``np.save`` writes the identical full ``.npy`` stream (header +
  data) through a path or a handle, so the oracle is byte-identity of the
  published file plus load-equality.

NOT a serialization shape (no oracle, stated per the A5 duty): the
``issue1739_sycoood_rescore_stage.py:160`` ``shutil.move``-into-yielded-tmp
staging publish (recipe edge §4(i)) relocates ALREADY-serialized bytes — the
published bytes are the staged source file's, preserved by move semantics;
batch 2's §4(i) sibling (``issue1315_dispatch.py`` copyfile) likewise landed
with no oracle.

Every test also asserts zero ``*.tmp*`` residue (old hidden ``.tmp_<name>``
and new ``<name>.<pid>.<uuid8>.tmp`` shapes alike).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from explore_persona_space.atomic_io import atomic_replace

REPO = Path(__file__).resolve().parent.parent

_CAPTURE_MIGRATED_BLOCK = "\n".join(
    [
        '            with atomic_replace(path) as tmp, tmp.open("wb") as fh:',
        "                np.save(fh, arr)",
    ]
)


def _residue(root: Path) -> list[Path]:
    """Every leftover temp file under *root* (old AND new temp-name shapes)."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and (".tmp" in p.name or p.name.startswith("."))
    )


def _old_form_save(path: Path, arr: np.ndarray) -> None:
    """Pre-migration shard write, verbatim from ``7a4fd3c319``
    ``src/explore_persona_space/experiments/issue_1739/capture.py:274-279``
    (dot-prefixed ``.npy``-suffixed sibling tmp + path-form np.save +
    ``os.replace``)."""
    tmp = path.with_name(".tmp_" + path.name)
    np.save(tmp, arr)
    os.replace(tmp, path)


def _new_form_save(path: Path, arr: np.ndarray) -> None:
    """The migrated form, reproduced verbatim from the live source (anchored by
    ``test_migrated_block_is_live_source``): handle-form np.save on the yielded
    atomic_replace tmp, publish at with-exit."""
    with atomic_replace(path) as tmp, tmp.open("wb") as fh:
        np.save(fh, arr)


def test_migrated_block_is_live_source() -> None:
    """The reproduction in ``_new_form_save`` must match the LIVE migrated
    source in ``experiments/issue_1739/capture.py`` (drift here would silently
    hollow the oracle)."""
    live = (
        REPO / "src" / "explore_persona_space" / "experiments" / "issue_1739" / "capture.py"
    ).read_text(encoding="utf-8")
    assert _CAPTURE_MIGRATED_BLOCK in live


def test_capture_npy_save_bytes_identity(tmp_path: Path) -> None:
    """Old path-form vs migrated handle-form np.save: byte-identical published
    ``.npy`` files, load-equality, zero residue."""
    rng = np.random.default_rng(2336)
    arr = rng.standard_normal((5, 7)).astype(np.float16)
    arr[1, 2] = np.nan  # NaN must survive both writes bit-exactly
    old_target = tmp_path / "old" / "t1_L03_shard02.npy"
    new_target = tmp_path / "new" / "t1_L03_shard02.npy"
    old_target.parent.mkdir(parents=True)
    # atomic_replace mkdirs new_target's parent itself (writer mkdir absorbed).

    _old_form_save(old_target, arr)
    _new_form_save(new_target, arr)

    assert new_target.read_bytes() == old_target.read_bytes()
    got = np.load(new_target)
    assert got.dtype == np.float16 and got.shape == (5, 7)
    np.testing.assert_array_equal(got, arr)
    assert _residue(tmp_path) == []


def test_capture_npy_save_failure_leaves_no_residue(tmp_path: Path) -> None:
    """A raising write under the migrated form unlinks the temp (no orphan
    ``*.tmp*``) and leaves the ORIGINAL destination bytes untouched."""
    arr = np.arange(6, dtype=np.float16).reshape(2, 3)
    target = tmp_path / "t1_L00_shard00.npy"
    np.save(target, arr)
    before = target.read_bytes()

    class _Boom(RuntimeError):
        pass

    try:
        with atomic_replace(target) as tmp, tmp.open("wb") as fh:
            np.save(fh, arr[:1])
            raise _Boom("mid-write crash")
    except _Boom:
        pass
    else:  # pragma: no cover - the raise above is unconditional
        raise AssertionError("expected _Boom to propagate")

    assert target.read_bytes() == before
    assert _residue(tmp_path) == []


def test_new_temp_shape_invisible_to_shard_glob(tmp_path: Path) -> None:
    """The loader's ``{kind}_L*_shard*.npy`` glob must see neither temp shape:
    the migration's safety comment in ``write_store_shard`` claims the
    atomic_replace temp (ending ``.tmp``) cannot enter the shard set — pin it,
    alongside the legacy hidden-name claim it replaced."""
    store = tmp_path
    (store / "t1_L00_shard00.npy").write_bytes(b"x")
    # Legacy hidden temp + a synthetic atomic_replace-shaped temp:
    (store / ".tmp_t1_L01_shard00.npy").write_bytes(b"x")
    (store / "t1_L01_shard00.npy.12345.abcd1234.tmp").write_bytes(b"x")
    matched = sorted(p.name for p in store.glob("t1_L*_shard*.npy"))
    assert matched == ["t1_L00_shard00.npy"]
