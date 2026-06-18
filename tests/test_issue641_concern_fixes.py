"""Issue #641 round-2 CONCERN-closure regression tests (both binding reconciler
concerns). Each test fails PRE-fix and passes POST-fix, exercising the new
permanent invariants — not just imports.

- ``stale-schedule-metadata-attestation``: the schedule provenance is now bound
  to the ADAPTER at creation time (``adapter_run_metadata.json``) and validated
  by ``_assert_adapter_schedule_parity`` BEFORE the resume-skip in
  ``_train_dose_ladder``. Pre-fix, the cell sidecar was written at EVAL time
  AFTER the resume-skip, so a stale max_steps=100 adapter could get its sidecar
  relabeled 560 and pass the pool-side parity assert. The four states pinned:
  (a) valid 560/linear PASSes; (b) mismatched 100/cosine raises naming the dir;
  (c) missing sidecar raises; (d) malformed JSON raises.

- ``extra-root-silent-drop``: every explicit ``--extra-records-roots`` entry is
  validated (exists / has dose_curves/ / contributes >=1 completions file) by
  ``_validate_extra_roots`` — called from BOTH ``_load_cell_records`` and
  ``assert_schedule_parity`` — and the production Arm-B step-100 headline asserts
  the pooled seed set carries both parent seeds {42, 1042} + >=5 new seeds via
  ``_assert_armB_seed_set``. Pre-fix a typo'd/missing root rglob'd to zero files
  and silently produced a smaller aggregate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_DISPATCHER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "issue641_dose_curves.py"


def _load_dispatcher(eval_root: Path):
    """Import the dose-curve dispatcher by path and point its module-global
    EVAL_ROOT at an isolated tmp dir."""
    spec = importlib.util.spec_from_file_location(
        "i641_dispatcher_concern_undertest", _DISPATCHER_PATH
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.EVAL_ROOT = eval_root
    return m


def _write_completions_cell(root: Path, source: str, seed: int, step: int) -> Path:
    """Write a minimal production-shaped completions cell under ``root``."""
    cell = root / "dose_curves" / f"{source}_seed{seed}_step{step}"
    cell.mkdir(parents=True, exist_ok=True)
    (cell / f"completions__{source}__seed{seed}__step{step}.jsonl").write_text(
        json.dumps(
            {
                "source": source,
                "dose_step": step,
                "seed": seed,
                "probe_id": 0,
                "sample_idx": 0,
                "aligned_score": 10.0,
                "coherent_score": 80.0,
            }
        )
        + "\n"
    )
    return cell


# ── Concern 1: stale-schedule-metadata-attestation ────────────────────────────


def test_adapter_provenance_valid_passes(tmp_path):
    """A freshly-written 560/linear adapter sidecar (the writer's own output) is
    accepted by the resume-skip provenance gate."""
    m = _load_dispatcher(tmp_path / "eval")
    adir = tmp_path / "adapter"
    adir.mkdir()
    m._write_adapter_run_metadata(adir, seed=1, max_steps=m.PROD_MAX_STEPS)
    written = json.loads((adir / m.ADAPTER_RUN_METADATA_NAME).read_text())
    assert written["max_steps"] == m.PROD_MAX_STEPS
    assert written["lr_scheduler_type"] == "linear"
    # Must NOT raise.
    m._assert_adapter_schedule_parity(adir)


def test_adapter_provenance_mismatched_steps_raises(tmp_path):
    """A stale max_steps=100 / cosine adapter is rejected before the resume-skip,
    naming the dir + the mismatched fields (the exact relabel hole)."""
    m = _load_dispatcher(tmp_path / "eval")
    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / m.ADAPTER_RUN_METADATA_NAME).write_text(
        json.dumps({"max_steps": 100, "lr_scheduler_type": "cosine", "seed": 1})
    )
    with pytest.raises(ValueError) as ei:
        m._assert_adapter_schedule_parity(adir)
    msg = str(ei.value)
    assert "max_steps=100" in msg
    assert "cosine" in msg
    assert str(adir) in msg


def test_adapter_provenance_missing_sidecar_raises(tmp_path):
    """An existing adapter with NO provenance sidecar is un-trustworthy state —
    the gate refuses to resume-skip it rather than trusting blind."""
    m = _load_dispatcher(tmp_path / "eval")
    adir = tmp_path / "adapter"
    adir.mkdir()
    with pytest.raises(ValueError) as ei:
        m._assert_adapter_schedule_parity(adir)
    msg = str(ei.value)
    assert f"missing {m.ADAPTER_RUN_METADATA_NAME}" in msg
    assert str(adir) in msg


def test_adapter_provenance_malformed_json_raises(tmp_path):
    """A malformed provenance sidecar raises (not a silent skip / KeyError)."""
    m = _load_dispatcher(tmp_path / "eval")
    adir = tmp_path / "adapter"
    adir.mkdir()
    (adir / m.ADAPTER_RUN_METADATA_NAME).write_text("{not valid json")
    with pytest.raises(ValueError) as ei:
        m._assert_adapter_schedule_parity(adir)
    msg = str(ei.value)
    assert "unreadable/malformed" in msg
    assert str(adir) in msg


# ── Concern 2: extra-root-silent-drop ─────────────────────────────────────────


def test_extra_root_nonexistent_raises(tmp_path):
    """A typo'd / missing extra-records root fails loud naming the offending
    root, instead of silently rglob'ing to zero files."""
    m = _load_dispatcher(tmp_path / "eval")
    bad = tmp_path / "does_not_exist"
    with pytest.raises(ValueError) as ei:
        m._validate_extra_roots([str(bad)])
    msg = str(ei.value)
    assert "does not exist" in msg
    assert str(bad) in msg


def test_extra_root_no_dose_curves_dir_raises(tmp_path):
    """An extra root that exists but has no dose_curves/ contributes no cells."""
    m = _load_dispatcher(tmp_path / "eval")
    root = tmp_path / "no_dose"
    root.mkdir()
    with pytest.raises(ValueError) as ei:
        m._validate_extra_roots([str(root)])
    msg = str(ei.value)
    assert "no dose_curves/" in msg
    assert str(root) in msg


def test_extra_root_empty_dose_curves_raises(tmp_path):
    """dose_curves/ exists but empty -> no completions contribution."""
    m = _load_dispatcher(tmp_path / "eval")
    root = tmp_path / "empty_dose"
    (root / "dose_curves").mkdir(parents=True)
    with pytest.raises(ValueError) as ei:
        m._validate_extra_roots([str(root)])
    msg = str(ei.value)
    assert "completions__*.jsonl" in msg
    assert str(root) in msg


def test_extra_root_no_completions_files_raises(tmp_path):
    """dose_curves/ populated but with no completions__*.jsonl (only an em_rate
    json) contributes no records."""
    m = _load_dispatcher(tmp_path / "eval")
    root = tmp_path / "no_comp"
    cell = root / "dose_curves" / "sp_teacher_ho_seed1_step100"
    cell.mkdir(parents=True)
    (cell / "em_rate__x.json").write_text("{}")
    with pytest.raises(ValueError) as ei:
        m._validate_extra_roots([str(root)])
    msg = str(ei.value)
    assert "completions__*.jsonl" in msg
    assert str(root) in msg


def test_extra_root_happy_path_and_none_pass(tmp_path):
    """A populated extra root validates silently; None / [] is a no-op."""
    m = _load_dispatcher(tmp_path / "eval")
    ok = tmp_path / "ok"
    _write_completions_cell(ok, "sp_teacher_ho", 42, 100)
    m._validate_extra_roots([str(ok)])  # must not raise
    m._validate_extra_roots(None)
    m._validate_extra_roots([])


def test_load_cell_records_rejects_bad_extra_root(tmp_path):
    """The validation fires from inside ``_load_cell_records`` (the production
    aggregate read path), not only the standalone helper."""
    m = _load_dispatcher(tmp_path / "eval")
    (m.EVAL_ROOT / "dose_curves").mkdir(parents=True)
    with pytest.raises(ValueError) as ei:
        m._load_cell_records(extra_roots=[str(tmp_path / "typo_root")])
    assert "does not exist" in str(ei.value)


def test_assert_schedule_parity_rejects_bad_extra_root(tmp_path):
    """The validation fires from inside ``assert_schedule_parity`` too — both
    aggregate entry points reject a silently-dropped root."""
    m = _load_dispatcher(tmp_path / "eval")
    (m.EVAL_ROOT / "dose_curves").mkdir(parents=True)
    with pytest.raises(ValueError) as ei:
        m.assert_schedule_parity([str(tmp_path / "typo_root")])
    assert "does not exist" in str(ei.value)


def test_armB_seed_set_guard_positive(tmp_path):
    """The 8-seed headline (2 parent + 6 new) passes the seed-set guard."""
    m = _load_dispatcher(tmp_path / "eval")
    recs = [
        {"source": "sp_teacher_ho", "dose_step": 100, "seed": s}
        for s in [42, 1042, 1, 7, 123, 2024, 31337, 98765]
    ]
    pooled = m._assert_armB_seed_set(recs, matched_dose=100)
    assert pooled == {42, 1042, 1, 7, 123, 2024, 31337, 98765}


def test_armB_seed_set_guard_missing_parent_raises(tmp_path):
    """A pooled set missing a parent seed (the silent-drop signature) fails
    loud naming the missing seed."""
    m = _load_dispatcher(tmp_path / "eval")
    recs = [
        {"source": "sp_teacher_ho", "dose_step": 100, "seed": s}
        for s in [42, 1, 7, 123, 2024, 31337]  # 1042 dropped
    ]
    with pytest.raises(ValueError) as ei:
        m._assert_armB_seed_set(recs, matched_dose=100)
    msg = str(ei.value)
    assert "missing parent seed" in msg
    assert "1042" in msg


def test_armB_seed_set_guard_too_few_new_raises(tmp_path):
    """Both parent seeds present but only 4 new seeds (< 5) fails loud."""
    m = _load_dispatcher(tmp_path / "eval")
    recs = [
        {"source": "sp_teacher_ho", "dose_step": 100, "seed": s}
        for s in [42, 1042, 1, 7, 123, 2024]  # only 4 new
    ]
    with pytest.raises(ValueError) as ei:
        m._assert_armB_seed_set(recs, matched_dose=100)
    assert "only 4 new seed" in str(ei.value)
