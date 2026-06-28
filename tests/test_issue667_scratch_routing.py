"""End-to-end shape test for the issue667_extract.py scratch-routing migration (#674).

The migrated ``run_extraction`` writes the per-(target, layer) ``.npz`` to a
local-SSD scratch mirror, then ``materialize_to_canonical`` copies them to the
canonical cell dir BEFORE ``assert_full_npz_complement`` + the ``.done``
sentinel (both read the canonical dir). These tests emulate that flow with
small synthetic ``.npz`` (no GPU extractor run — the GPU path is out of scope
for this ``kind: infra`` task) and pin the load-bearing
materialize-before-sentinel call ORDER against a future refactor via an
AST-level source-order check (importing ``issue667_extract`` pulls in torch /
vLLM, so the order is checked from source, not a runtime spy).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np

from explore_persona_space.orchestrate.scratch_io import (
    ENV_SCRATCH_DIR,
    materialize_to_canonical,
    scratch_path_for,
)

_EXTRACT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "issue667_extract.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A miniature of the {tcid}_L{li}.npz payload shape."""
    return {
        "v0": rng.standard_normal((8,)).astype(np.float32),
        "v_plus": rng.standard_normal((8,)).astype(np.float32),
        "layer": np.int64(14),
    }


def test_migrated_path_round_trip_byte_identical(monkeypatch, tmp_path):
    """Emulate the migrated run_extraction flow: scratch_path_for -> mkdir ->
    write per-target .npz into scratch -> materialize_to_canonical. The
    canonical cell dir must end with the expected .npz, byte-identical (SHA256,
    primary) and array-equal (secondary) to a control written directly to a
    sibling canonical dir from the same payload, with no leftover scratch."""
    scratch_root = tmp_path / "scratch_root"
    monkeypatch.setenv(ENV_SCRATCH_DIR, str(scratch_root))

    out_root = tmp_path / "out"
    cell_dir = out_root / "em" / "default_seed42"
    cell_dir.mkdir(parents=True)

    targets = ["sp_swe", "default", "fmt_json"]
    layers = [7, 14, 21]

    # --- migrated path: write into scratch, then materialize to canonical ---
    scratch_cell_dir = scratch_path_for(cell_dir, issue=667)
    assert scratch_cell_dir != cell_dir  # env is set -> real scratch indirection
    scratch_cell_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    control_dir = out_root / "em" / "default_seed42_CONTROL"
    control_dir.mkdir(parents=True)
    expected_names: list[str] = []
    for tcid in targets:
        for li in layers:
            name = f"{tcid}_L{li}.npz"
            expected_names.append(name)
            payload = _payload(rng)
            np.savez(scratch_cell_dir / name, **payload)
            # control: direct write to a sibling canonical dir, SAME payload.
            np.savez(control_dir / name, **payload)

    materialize_to_canonical(scratch_cell_dir, cell_dir)

    # Scratch consumed; canonical holds the full expected complement.
    assert not scratch_cell_dir.exists()
    assert sorted(p.name for p in cell_dir.glob("*.npz")) == sorted(expected_names)
    assert list(cell_dir.glob("*.tmp")) == []

    for name in expected_names:
        materialized = cell_dir / name
        control = control_dir / name
        # PRIMARY: SHA256 byte-identity to the direct-write control.
        assert _sha(materialized) == _sha(control), name
        # SECONDARY diagnostic: array equality.
        a, b = np.load(materialized), np.load(control)
        assert set(a.files) == set(b.files), name
        for key in a.files:
            assert np.array_equal(a[key], b[key]), (name, key)


def test_stale_scratch_cleared_on_entry(monkeypatch, tmp_path):
    """Hardening (plan §6): the run_extraction rmtree-then-mkdir clean-on-entry
    clears a prior crashed-run's stale scratch, so a junk file written into the
    scratch cell dir before the run is GONE after materialize."""
    scratch_root = tmp_path / "scratch_root"
    monkeypatch.setenv(ENV_SCRATCH_DIR, str(scratch_root))

    cell_dir = tmp_path / "out" / "em" / "default_seed42"
    cell_dir.mkdir(parents=True)
    scratch_cell_dir = scratch_path_for(cell_dir, issue=667)

    # Simulate a prior crashed run leaving stale junk in the scratch cell dir.
    scratch_cell_dir.mkdir(parents=True, exist_ok=True)
    junk = scratch_cell_dir / "STALE_JUNK_from_prior_crash.npz"
    np.savez(junk, garbage=np.zeros(3, dtype=np.float32))

    # --- migrated clean-on-entry (the rmtree-then-mkdir guard) ---
    if scratch_cell_dir != cell_dir and scratch_cell_dir.exists():
        import shutil

        shutil.rmtree(scratch_cell_dir)
    scratch_cell_dir.mkdir(parents=True, exist_ok=True)

    # This run writes one fresh tensor.
    np.savez(scratch_cell_dir / "sp_swe_L14.npz", v0=np.arange(4, dtype=np.float32))
    materialize_to_canonical(scratch_cell_dir, cell_dir)

    canonical_npz = sorted(p.name for p in cell_dir.glob("*.npz"))
    assert canonical_npz == ["sp_swe_L14.npz"]
    assert not (cell_dir / "STALE_JUNK_from_prior_crash.npz").exists()
    assert not scratch_cell_dir.exists()


def test_run_extraction_call_order_materialize_before_sentinel():
    """Pin the load-bearing ordering invariant via AST source inspection:
    inside run_extraction the calls appear in the order
    scratch_path_for -> materialize_to_canonical ->
    assert_full_npz_complement -> write_cell_done_sentinel. A future refactor
    that materializes AFTER the sentinel (stamping .done over an un-copied
    cell) would reorder these and fail this test.

    Source inspection (not a runtime spy) because importing issue667_extract
    pulls in torch / vLLM; the order is a static property of the source.
    """
    tree = ast.parse(_EXTRACT_SCRIPT.read_text())
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_extraction"
    )
    tracked = {
        "scratch_path_for",
        "materialize_to_canonical",
        "assert_full_npz_complement",
        "write_cell_done_sentinel",
    }
    seen: list[tuple[int, str]] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in tracked
        ):
            seen.append((node.lineno, node.func.id))
    order = [name for _, name in sorted(seen)]
    # Each tracked call appears, in the required order.
    assert order == [
        "scratch_path_for",
        "materialize_to_canonical",
        "assert_full_npz_complement",
        "write_cell_done_sentinel",
    ], order
