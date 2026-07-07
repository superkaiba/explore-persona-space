"""Issue #763 reanchor round-2 regression tests.

Pins BLOCKER ``binary-control-checkpoint-resume-skip`` (code-review round 1,
reconciler-bound FAIL): the §4 ``binary_repro_check`` positive control must
bind whenever ``--binary-control-ref`` is provided — including on
checkpoint-resume — and a record that fails the control must never be
persisted where a crash→rerun resume (or ``_assemble_results``) would ship it
at rc=0. Both legs fail pre-fix / pass post-fix:

1. RESUME leg — a pre-existing (deliberately bad) round checkpoint run with
   ``--binary-control-ref`` FAILs loud (``RuntimeError``); pre-fix the loop
   resume-skipped with only a "checkpoint exists — skipping refit" log and
   shipped the bad record at rc=0.
2. PERSIST-ORDERING leg — a fresh fit whose binary control fails never lands
   in ``fit_by_behavior/``; pre-fix the checkpoint was written BEFORE the
   control ran, so the failed record survived on disk for the next resume.
3. Positive control — a checkpoint reproducing the parent within tolerance
   resumes cleanly (rc=0) and assembles (guards against over-strictness).

All offline: no GPU, no HF (staging + the vectorized-exactness gate are
monkeypatched out; the control itself is a pure dict comparison).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

# The parent deception record's binary companion (the v59 smoke's reference
# values); any checkpoint farther than --binary-control-tol from these must trip
# the control.
PARENT_GLM = 0.324167
PARENT_RIDGE = 0.406830
BAD_REC = {"rho_binary_GLM": 0.9, "rho_binary_ridge": 0.9}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _setup(monkeypatch, tmp_path: Path, ckpt_rec: dict | None):
    """Offline harness: tmp out-dir/ref/E0, network + exactness gate patched out.

    Returns ``(module, out_dir)`` with ``sys.argv`` set to the production fit
    entrypoint shape (non-smoke, so the checkpoint-resume branch is live) with
    ``--binary-control-ref`` provided.
    """
    import issue763_fit_predictors as F

    out_dir = tmp_path / "round"
    e0_path = tmp_path / "E0_deception_v2.json"
    ref_path = tmp_path / "parent_deception.json"
    _write_json(e0_path, {"deception": {"per_ctx": {}}})
    _write_json(ref_path, {"rho_binary_GLM": PARENT_GLM, "rho_binary_ridge": PARENT_RIDGE})
    if ckpt_rec is not None:
        _write_json(out_dir / "fit_by_behavior" / "deception.json", ckpt_rec)
    # Hermetic: no HF staging, no vectorized-exactness fit, no repo eval_results
    # reads (pv baseline resolves against tmp -> have_pv=False).
    monkeypatch.setattr(F, "_stage_fit_inputs_from_hf", lambda *a, **k: None)
    monkeypatch.setattr(F, "assert_matches_reference", lambda **k: {"patched_out": True})
    monkeypatch.setattr(F, "EVAL_RESULTS_DIR", tmp_path / "eval_results")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue763_fit_predictors.py",
            "--behaviors",
            "deception",
            "--e0-json",
            str(e0_path),
            "--out-dir",
            str(out_dir),
            "--binary-control-ref",
            str(ref_path),
            "--binary-control-tol",
            "1e-3",
        ],
    )
    return F, out_dir


def test_resume_with_bad_checkpoint_fails_loud(monkeypatch, tmp_path):
    """Leg 1: checkpoint-resume must run the binary control, not skip it."""
    F, out_dir = _setup(monkeypatch, tmp_path, dict(BAD_REC))
    with pytest.raises(RuntimeError, match="binary positive control FAILED"):
        F.main()
    # The failing record must not have shipped into the assembled output.
    assert not (out_dir / "matched_predictor_results.json").exists(), (
        "a checkpoint that fails the binary control must never assemble at rc=0"
    )


def test_fresh_fit_control_failure_never_persists_checkpoint(monkeypatch, tmp_path):
    """Leg 2: the control runs BEFORE the checkpoint persist (no bad record on disk)."""
    F, out_dir = _setup(monkeypatch, tmp_path, None)
    monkeypatch.setattr(F, "_load_v0", lambda b: (None, None))
    monkeypatch.setattr(F, "load_frozen_pool_staged", lambda b: {"n_probes": 20})
    monkeypatch.setattr(F, "fit_behavior", lambda *a, **k: dict(BAD_REC))
    with pytest.raises(RuntimeError, match="binary positive control FAILED"):
        F.main()
    assert not (out_dir / "fit_by_behavior" / "deception.json").exists(), (
        "a record that FAILED the binary positive control was persisted to the "
        "round checkpoint dir — a crash→rerun resume would ship it at rc=0"
    )
    assert not (out_dir / "matched_predictor_results.json").exists()


def test_resume_with_good_checkpoint_passes_and_assembles(monkeypatch, tmp_path):
    """Positive control: an in-tolerance checkpoint resumes cleanly and ships."""
    good = {"rho_binary_GLM": PARENT_GLM, "rho_binary_ridge": PARENT_RIDGE}
    F, out_dir = _setup(monkeypatch, tmp_path, good)
    rc = F.main()
    assert rc == 0
    blob = json.loads((out_dir / "matched_predictor_results.json").read_text())
    assert blob["by_behavior"]["deception"]["rho_binary_GLM"] == PARENT_GLM
    assert blob["by_behavior"]["deception"]["rho_binary_ridge"] == PARENT_RIDGE
