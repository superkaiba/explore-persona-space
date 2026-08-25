"""Task #2336 batch-4 A5 byte oracle (plan v4 §6 A5; ground-truthed scoping duty).

Batch 4's migrated serialization shapes, greped from the batch diff: json
(``write_text(json.dumps(...))`` + ``json.dump`` into an open text handle),
jsonl (open-handle and joined ``write_text`` forms), npz (handle-form
``np.savez``), pt (``torch.save``), and raw-bytes chunked ``open("wb")``
writes (the ``issue2222_lib.py`` zip download) — every one already carries a
landed first-instance oracle (json: batch 1,
``tests/test_clean_experiment_downloads_atomic_writers.py``; jsonl + text +
npz: batch 2 r6; pt: batch 2 r7 — both in
``tests/test_issue2336_batch2_byte_oracles.py``; raw bytes: batch 3,
``tests/test_issue2336_batch3_byte_oracles.py``). Exactly ONE batch-4 shape
has no landed oracle:

- **npy via ``np.lib.format.open_memmap`` chunked copy** — the
  ``scripts/issue779_ffc_n1m_fits.py`` ``_ml_truncate_rows`` chunked-copy
  branch (header length changed): a memmap workspace is created ON the temp
  path, filled chunk-by-chunk, flushed, released, then published. The write is
  inline in a GPU-pipeline helper (not executable at test time without the
  memmap lane), so the migrated with-``atomic_replace`` form is reproduced
  VERBATIM here and anchored against the live source text; the pre-migration
  inline form (batch-3 tip ``4f10abb123`` :523-533) is executed verbatim as
  the old-form oracle. ``open_memmap`` writes a full ``.npy`` (header + data),
  so the oracle is byte-identity of the published file, plus a load-equality
  read against the truncated source.

Every test also asserts zero ``*.tmp*`` residue (old ``<name>.trunc.tmp`` and
new ``<name>.<pid>.<uuid8>.tmp`` shapes alike).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from explore_persona_space.atomic_io import atomic_replace

REPO = Path(__file__).resolve().parent.parent

_I779_MIGRATED_BLOCK = "\n".join(
    [
        "        with atomic_replace(path) as tmp:",
        "            dst = np.lib.format.open_memmap(",
        '                tmp, mode="w+", dtype=src.dtype, shape=(int(n_rows),) + src.shape[1:]',
        "            )",
    ]
)


def _residue(root: Path) -> list[Path]:
    """Every leftover temp file under *root* (old AND new temp-name shapes)."""
    return sorted(p for p in root.rglob("*.tmp*") if p.is_file())


def _old_form_truncate(path: Path, n_rows: int) -> None:
    """Pre-migration chunked-copy truncation, verbatim from ``4f10abb123``
    ``scripts/issue779_ffc_n1m_fits.py:523-533`` (deterministic ``.trunc.tmp``
    sibling + ``os.replace``)."""
    src = np.load(path, mmap_mode="r")
    tmp = path.parent / (path.name + ".trunc.tmp")
    dst = np.lib.format.open_memmap(
        # Shape expression kept VERBATIM from the pre-migration source (oracle fidelity).
        tmp,
        mode="w+",
        dtype=src.dtype,
        shape=(int(n_rows),) + src.shape[1:],  # noqa: RUF005
    )
    step = 50_000
    for s in range(0, int(n_rows), step):
        e = min(int(n_rows), s + step)
        dst[s:e] = src[s:e]
    dst.flush()
    del dst, src
    os.replace(tmp, path)


def _new_form_truncate(path: Path, n_rows: int) -> None:
    """The migrated form, reproduced verbatim from the live source (anchored by
    ``test_migrated_block_is_live_source``): memmap created ON the yielded tmp,
    handles released INSIDE the with, publish at with-exit."""
    src = np.load(path, mmap_mode="r")
    with atomic_replace(path) as tmp:
        dst = np.lib.format.open_memmap(
            # Shape expression kept VERBATIM from the live migrated source (oracle fidelity).
            tmp,
            mode="w+",
            dtype=src.dtype,
            shape=(int(n_rows),) + src.shape[1:],  # noqa: RUF005
        )
        step = 50_000
        for s in range(0, int(n_rows), step):
            e = min(int(n_rows), s + step)
            dst[s:e] = src[s:e]
        dst.flush()
        del dst, src


def test_migrated_block_is_live_source() -> None:
    """The reproduction in ``_new_form_truncate`` must match the LIVE migrated
    source in ``scripts/issue779_ffc_n1m_fits.py`` (drift here would silently
    hollow the oracle)."""
    live = (REPO / "scripts" / "issue779_ffc_n1m_fits.py").read_text(encoding="utf-8")
    assert _I779_MIGRATED_BLOCK in live


def test_issue779_memmap_truncate_bytes_identity(tmp_path: Path) -> None:
    """Old-form vs migrated chunked-copy truncation: byte-identical published
    ``.npy`` files, load-equality against the truncated source, zero residue."""
    rng = np.random.default_rng(779)
    arr = rng.standard_normal((7, 3)).astype(np.float64)
    arr[2, 1] = np.nan  # NaN must survive both copies bit-exactly
    old_target = tmp_path / "old" / "cx.npy"
    new_target = tmp_path / "new" / "cx.npy"
    for target in (old_target, new_target):
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, arr)
    assert old_target.read_bytes() == new_target.read_bytes()  # identical starting bytes

    _old_form_truncate(old_target, 4)
    _new_form_truncate(new_target, 4)

    assert new_target.read_bytes() == old_target.read_bytes()
    got = np.load(new_target)
    assert got.shape == (4, 3)
    np.testing.assert_array_equal(got, arr[:4])
    assert _residue(tmp_path) == []


def test_issue779_memmap_truncate_failure_leaves_no_residue(tmp_path: Path) -> None:
    """A raising copy under the migrated form unlinks the temp (no orphan
    ``*.tmp*``) and leaves the ORIGINAL destination bytes untouched."""
    arr = np.arange(12, dtype=np.float64).reshape(4, 3)
    target = tmp_path / "cx.npy"
    np.save(target, arr)
    before = target.read_bytes()

    class _Boom(RuntimeError):
        pass

    try:
        src = np.load(target, mmap_mode="r")
        with atomic_replace(target) as tmp:
            dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=src.dtype, shape=(2, 3))
            dst[0:1] = src[0:1]
            del dst, src
            raise _Boom("mid-copy crash")
    except _Boom:
        pass
    else:  # pragma: no cover - the raise above is unconditional
        raise AssertionError("expected _Boom to propagate")

    assert target.read_bytes() == before
    assert _residue(tmp_path) == []
