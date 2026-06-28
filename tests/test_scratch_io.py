"""Unit tests for the local-SSD scratch-routing helper (#674).

The helper (``orchestrate/scratch_io.py``) decouples the per-cell ``.npz``
write storm from the GCE network-PD plane: write to a local-SSD scratch
mirror, then batch-materialize to the canonical destination at cell end. Off
GCE (``EPS_SCRATCH_DIR`` unset) both functions are pass-throughs, so canonical
outputs must be byte-for-byte unchanged vs a direct write. These tests are
CPU-only and deterministic (env set/unset via monkeypatch).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.orchestrate import scratch_io
from explore_persona_space.orchestrate.scratch_io import (
    ENV_SCRATCH_DIR,
    materialize_to_canonical,
    scratch_path_for,
)


def _write_npz(path: Path, **arrays: np.ndarray) -> bytes:
    """Write an .npz and return its on-disk bytes (for SHA comparison)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path.read_bytes()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# scratch_path_for — env honored, structure mirrored, no collision (test 1)
# ---------------------------------------------------------------------------


def test_scratch_path_for_honors_env_and_mirrors_full_structure(monkeypatch, tmp_path):
    root = tmp_path / "scratch_root"
    monkeypatch.setenv(ENV_SCRATCH_DIR, str(root))
    mapped = scratch_path_for(Path("/a/b/cellX"), issue=667)
    # Anchor stripped, full canonical structure mirrored under <root>/issue667/.
    assert mapped == root / "issue667" / "a" / "b" / "cellX"


def test_scratch_path_for_namespaces_by_issue(monkeypatch, tmp_path):
    root = tmp_path / "scratch_root"
    monkeypatch.setenv(ENV_SCRATCH_DIR, str(root))
    m667 = scratch_path_for(Path("/x/cell"), issue=667)
    m674 = scratch_path_for(Path("/x/cell"), issue=674)
    # Same canonical path, different producing issue -> distinct scratch dirs.
    assert m667 != m674
    assert m667 == root / "issue667" / "x" / "cell"
    assert m674 == root / "issue674" / "x" / "cell"


def test_scratch_path_for_no_collision_on_same_basename(monkeypatch, tmp_path):
    root = tmp_path / "scratch_root"
    monkeypatch.setenv(ENV_SCRATCH_DIR, str(root))
    # Two distinct canonical cell dirs that share a basename ("cell_seed42")
    # under different parents must map to DISTINCT scratch dirs.
    a = scratch_path_for(Path("/out/em/cell_seed42"), issue=667)
    b = scratch_path_for(Path("/out/sycophancy/cell_seed42"), issue=667)
    assert a != b


def test_scratch_path_for_passthrough_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_SCRATCH_DIR, raising=False)
    canonical = tmp_path / "out" / "cell"
    mapped = scratch_path_for(canonical, issue=667)
    # Pass-through: returns the canonical path object identically (==), not an
    # indirection under any scratch root.
    assert mapped == canonical


def test_scratch_path_for_passthrough_when_env_blank(monkeypatch, tmp_path):
    # An empty / whitespace-only EPS_SCRATCH_DIR is treated as unset.
    monkeypatch.setenv(ENV_SCRATCH_DIR, "   ")
    canonical = tmp_path / "out" / "cell"
    assert scratch_path_for(canonical, issue=667) == canonical


# ---------------------------------------------------------------------------
# materialize_to_canonical — byte-identity + cleanup + atomicity (tests 2-3)
# ---------------------------------------------------------------------------


def test_materialize_copies_every_npz_byte_identical_and_removes_scratch(tmp_path):
    scratch = tmp_path / "scratch" / "cell"
    canonical = tmp_path / "canonical" / "cell"
    scratch.mkdir(parents=True)

    rng = np.random.default_rng(0)
    expected_sha: dict[str, str] = {}
    expected_arrays: dict[str, dict[str, np.ndarray]] = {}
    for i in range(4):
        name = f"target{i}_L14.npz"
        arrays = {
            "v0": rng.standard_normal((8,)).astype(np.float32),
            "v_plus": rng.standard_normal((8,)).astype(np.float32),
            "layer": np.int64(14),
        }
        _write_npz(scratch / name, **arrays)
        expected_sha[name] = _sha(scratch / name)
        expected_arrays[name] = arrays

    materialize_to_canonical(scratch, canonical)

    # Scratch dir gone after a successful materialize.
    assert not scratch.exists()

    canonical_npz = sorted(p.name for p in canonical.glob("*.npz"))
    assert canonical_npz == sorted(expected_sha)

    for name, sha in expected_sha.items():
        dst = canonical / name
        # PRIMARY: byte-for-byte SHA256 identity (closes H3).
        assert _sha(dst) == sha, name
        # SECONDARY diagnostic: array-level equality so a diff is legible if
        # SHA ever fails.
        loaded = np.load(dst)
        original = expected_arrays[name]
        assert set(loaded.files) == set(original), name
        for key, arr in original.items():
            assert np.array_equal(loaded[key], arr), (name, key)


def test_materialize_leaves_no_tmp_in_canonical_after_success(tmp_path):
    scratch = tmp_path / "scratch" / "cell"
    canonical = tmp_path / "canonical" / "cell"
    scratch.mkdir(parents=True)
    _write_npz(scratch / "t0_L7.npz", v0=np.arange(4, dtype=np.float32))
    _write_npz(scratch / "t1_L7.npz", v0=np.arange(4, dtype=np.float32))

    materialize_to_canonical(scratch, canonical)

    # os.replace consumed every temp — no stray .tmp survives in canonical.
    assert list(canonical.glob("*.tmp")) == []
    assert list(canonical.glob("*.npz.*")) == []


def test_materialize_partial_failure_leaves_scratch_intact_no_partial_npz(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch" / "cell"
    canonical = tmp_path / "canonical" / "cell"
    scratch.mkdir(parents=True)
    for i in range(3):
        _write_npz(scratch / f"t{i}_L7.npz", v0=np.arange(4, dtype=np.float32))

    real_copyfile = scratch_io.shutil.copyfile
    calls = {"n": 0}

    def flaky_copyfile(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated disk-full on the 2nd file")
        return real_copyfile(src, dst, *a, **k)

    monkeypatch.setattr(scratch_io.shutil, "copyfile", flaky_copyfile)

    with pytest.raises(OSError, match="simulated disk-full"):
        materialize_to_canonical(scratch, canonical)

    # Scratch dir survives (retryable): the rmtree is past the copy loop.
    assert scratch.exists()
    assert len(list(scratch.glob("*.npz"))) == 3
    # Files copy in sorted order, so t0 fully materialized (copy #1 + replace)
    # and t1's copy (#2) raised BEFORE its os.replace. canonical therefore holds
    # exactly the first FINAL .npz and NO final .npz for the failed file — the
    # failed copy's bytes only ever touched a .tmp, never the final name.
    final_npz = sorted(p.name for p in canonical.glob("*.npz"))
    assert final_npz == ["t0_L7.npz"], final_npz
    assert not (canonical / "t1_L7.npz").exists()


def test_materialize_atomicity_spy_writes_via_tmp_then_replace(monkeypatch, tmp_path):
    """Hardening (plan §6): the copy dest is always a .tmp, never the final
    .npz path — proving writes never land directly on the final name."""
    scratch = tmp_path / "scratch" / "cell"
    canonical = tmp_path / "canonical" / "cell"
    scratch.mkdir(parents=True)
    _write_npz(scratch / "t0_L7.npz", v0=np.arange(4, dtype=np.float32))

    real_copyfile = scratch_io.shutil.copyfile
    copy_dests: list[str] = []

    def spy_copyfile(src, dst, *a, **k):
        copy_dests.append(str(dst))
        return real_copyfile(src, dst, *a, **k)

    monkeypatch.setattr(scratch_io.shutil, "copyfile", spy_copyfile)
    materialize_to_canonical(scratch, canonical)

    assert copy_dests, "copyfile was never called"
    for dest in copy_dests:
        assert dest.endswith(".tmp"), dest
        assert not dest.endswith(".npz"), dest


# ---------------------------------------------------------------------------
# Pass-through fast-path (test 4)
# ---------------------------------------------------------------------------


def test_materialize_passthrough_when_scratch_equals_canonical_is_noop(tmp_path):
    canonical = tmp_path / "out" / "cell"
    canonical.mkdir(parents=True)
    _write_npz(canonical / "t0_L7.npz", v0=np.arange(4, dtype=np.float32))
    before = sorted(p.name for p in canonical.glob("*.npz"))

    # Pass-through case (scratch IS canonical): no-op, no exception, files left.
    materialize_to_canonical(canonical, canonical)

    assert canonical.exists()
    assert sorted(p.name for p in canonical.glob("*.npz")) == before


def test_end_to_end_passthrough_no_indirection(monkeypatch, tmp_path):
    """With EPS_SCRATCH_DIR unset, scratch_path_for returns canonical and
    materialize is a no-op — identical to a direct write."""
    monkeypatch.delenv(ENV_SCRATCH_DIR, raising=False)
    canonical = tmp_path / "out" / "cell"
    scratch = scratch_path_for(canonical, issue=667)
    assert scratch == canonical
    scratch.mkdir(parents=True, exist_ok=True)
    _write_npz(scratch / "t0_L7.npz", v0=np.arange(4, dtype=np.float32))
    materialize_to_canonical(scratch, canonical)
    assert (canonical / "t0_L7.npz").exists()
