# em-dash + Qwen marker " ※" intentional
"""Regression test for the #530 Phase 4 wrapper's filename handling.

Round 3 (commit 9a03f5541) had `i530_phase_analyze.py` invoke
`i504_phase_analyze.py` with `--phase0-path
<slab-root>/phase0_calibration.json`, but #530 has NO Phase 0 calibration
step — plan §4.4 step 1 names Phase 0 as the centroid pre-flight (which
writes `phase0_geometry_v1.json` or `phase0_5_gates.json` on the pod), and
plan §4.4 step 4 pins the analyzer to the band-stop final checkpoint
(frac=1.00). Result: after all 10 training cells + per-cell evals
succeeded on the pod (rc=0), Phase 4 crashed with::

    FileNotFoundError: phase0_calibration.json missing at
    eval_results/issue_530/phase0_calibration.json — Phase 0 must
    complete BEFORE Phase 1 can spawn.

The round-4 fix has the wrapper SYNTHESIZE the phase0_calibration.json
in-memory (chosen_checkpoint_fraction=1.0, verdict="pass") and AUTO-DETECT
the phase05 artifact at either `phase0_5_gates.json` or
`phase0_geometry_v1.json`. These tests pin both behaviors so the wrapper
can't silently regress to a hard-coded filename again.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = REPO_ROOT / "scripts" / "i530_phase_analyze.py"


@pytest.fixture(scope="module")
def wrapper_mod():
    """Import `scripts/i530_phase_analyze.py` as a module (it's a script, not a package)."""
    spec = importlib.util.spec_from_file_location("i530_phase_analyze_under_test", WRAPPER_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthesize_phase0_calibration_writes_pass_verdict_and_frac_1(wrapper_mod, tmp_path: Path):
    """The synthesized phase0_calibration.json carries the schema i504 reads."""
    out_path = tmp_path / "phase0_calibration.json"
    returned = wrapper_mod._synthesize_phase0_calibration(out_path)

    assert returned == out_path
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    # i504's load_phase0_pick checks verdict == "pass".
    assert payload["verdict"] == "pass"
    # run_phase2_analysis reads chosen_checkpoint_fraction (must be numeric,
    # not None — None raises in i504's analyze.py:702-707).
    assert payload["chosen_checkpoint_fraction"] == 1.0
    assert isinstance(payload["chosen_checkpoint_fraction"], float)
    # Provenance keys discoverable so a future reader doesn't mistake the
    # plan-pinned 1.0 for an evidence-based smoke-table pick.
    assert payload["task_id_minted_by"] == 530
    assert "synthesize" in payload["source"]
    assert "plan-pinned" in payload["note"].lower()


def test_synthesize_phase0_calibration_creates_parent_dirs(wrapper_mod, tmp_path: Path):
    """Writing into a not-yet-existing slab-root dir works (mkdir parents=True)."""
    nested = tmp_path / "deep" / "nested" / "eval_results" / "issue_530"
    out_path = nested / "phase0_calibration.json"
    assert not nested.exists()
    wrapper_mod._synthesize_phase0_calibration(out_path)
    assert out_path.exists()


def test_resolve_phase05_path_prefers_phase0_5_gates_when_present(wrapper_mod, tmp_path: Path):
    """`phase0_5_gates.json` is the i504-canonical name — auto-detect wins on it."""
    (tmp_path / "phase0_5_gates.json").write_text("{}")
    resolved = wrapper_mod._resolve_phase05_path(tmp_path, override=None)
    assert resolved == tmp_path / "phase0_5_gates.json"


def test_resolve_phase05_path_accepts_phase0_geometry_v1_when_only_that_exists(
    wrapper_mod, tmp_path: Path
):
    """The pod wrote `phase0_geometry_v1.json` (plan §4.4 step 1 namespaced).

    This test pins the actual round-4 incident state: the on-pod artifact is
    NOT phase0_5_gates.json. Without auto-detection the wrapper would crash
    on `--phase05-path` resolution.
    """
    (tmp_path / "phase0_geometry_v1.json").write_text("{}")
    resolved = wrapper_mod._resolve_phase05_path(tmp_path, override=None)
    assert resolved == tmp_path / "phase0_geometry_v1.json"


def test_resolve_phase05_path_raises_when_neither_filename_present(wrapper_mod, tmp_path: Path):
    """No phase05 artifact under slab-root → FileNotFoundError, not silent skip."""
    with pytest.raises(FileNotFoundError) as exc_info:
        wrapper_mod._resolve_phase05_path(tmp_path, override=None)
    msg = str(exc_info.value)
    # Both candidate names must appear in the error so the human knows what
    # to look for on disk.
    assert "phase0_5_gates.json" in msg
    assert "phase0_geometry_v1.json" in msg


def test_resolve_phase05_path_honors_explicit_override(wrapper_mod, tmp_path: Path):
    """`--phase05-path /some/path.json` short-circuits auto-detection."""
    custom = tmp_path / "custom_phase05_artifact.json"
    # File doesn't need to exist for override resolution — the loader downstream
    # is what checks existence. The override just bypasses the candidate sweep.
    resolved = wrapper_mod._resolve_phase05_path(tmp_path, override=custom)
    assert resolved == custom


def test_phase05_filename_candidates_include_both_known_names(wrapper_mod):
    """The wrapper's candidate tuple covers both filenames seen in the wild."""
    assert "phase0_5_gates.json" in wrapper_mod.PHASE05_FILENAME_CANDIDATES
    assert "phase0_geometry_v1.json" in wrapper_mod.PHASE05_FILENAME_CANDIDATES


def test_synthesized_phase0_calibration_passes_load_phase0_pick(wrapper_mod, tmp_path: Path):
    """End-to-end: i504's `load_phase0_pick` accepts the synthesized payload.

    This is the test that would have caught round 3's bug at lint-time: it
    feeds the synthesized artifact through the SAME `load_phase0_pick` the
    `i504_phase_analyze.py` subprocess calls, and asserts no exception. If
    a future refactor breaks the schema contract (e.g. drops `verdict` or
    `chosen_checkpoint_fraction`), this test fails.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504.phase0 import (
        load_phase0_pick,
    )

    out_path = tmp_path / "phase0_calibration.json"
    wrapper_mod._synthesize_phase0_calibration(out_path)
    pick = load_phase0_pick(out_path)
    assert pick["verdict"] == "pass"
    assert pick["chosen_checkpoint_fraction"] == 1.0
