"""Issue #545 round-24 — phase_p3 dependency order (assemble -> predictors -> score).

Round-23 pod crash (task #545 ``epm:failure`` v9): ``phase_p3`` ran the
predictors subprocess FIRST, but ``extract_base_prior`` reads
``output_root()/base_panel.json``, which is WRITTEN by
``assemble_matrix.assemble()`` — the step that previously ran after it. These
tests pin:

1. step order: assemble < predictors < score (and score's two passes);
2. the ``--skip-eval`` guard still applies to the predictors step ONLY
   (assemble + score stay unconditional);
3. ``extract_all`` fail-fast unification: a missing ``base_panel.json``
   raises ``FileNotFoundError`` in BOTH gpu and ``skip_gpu`` modes — the old
   warn-then-skip half-path under ``skip_gpu`` is gone;
4. read-after-write smoke: REAL ``assemble()`` on a tiny fixture cells dir,
   then a REAL ``extract_base_prior`` / ``extract_all(skip_gpu=True)`` call
   consumes the freshly written ``base_panel.json`` through the reordered
   ``phase_p3`` entry.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load_script("issue545_sweep")


@pytest.fixture()
def isolated_root(tmp_path, monkeypatch):
    """Point output + corpora roots at empty tmp dirs (smoke isolation off)."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("EPM_CORPORA_DIR", str(tmp_path / "corpora"))
    monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
    (tmp_path / "out").mkdir()
    (tmp_path / "corpora").mkdir()
    return tmp_path / "out"


def _write_fixture_cells(out_root: Path) -> None:
    """Tiny REAL cells fixture: a base_panel dir + one trained cell.

    ``sycophancy`` is scoring-eligible and applies to ``wrong_claim_agreement``
    (its diagonal column), so assemble() produces a non-empty base panel entry
    and extract_base_prior() produces non-empty predictor cells from it.
    """
    base = out_root / "cells" / "base_panel"
    base.mkdir(parents=True)
    (base / "sycophancy__default.json").write_text(
        json.dumps({"column": "sycophancy", "context": "default", "summary": {"rate": 0.25}})
    )
    cell = out_root / "cells" / "wrong_claim_agreement_primary_seed0"
    cell.mkdir(parents=True)
    (cell / "sycophancy__default.json").write_text(
        json.dumps(
            {
                "column": "sycophancy",
                "context": "default",
                "summary": {"rate": 0.75},
                "verdicts": [],
            }
        )
    )


def _order_recorders(sweep, monkeypatch, calls: list[str]) -> None:
    """Monkeypatch the three phase_p3 steps to record call order.

    ``assemble`` / ``score`` are imported INSIDE phase_p3 from their source
    modules, so the patch targets the source-module attributes; ``_run`` is a
    sweep-module global.
    """
    from explore_persona_space.experiments.behavior_testbed_545 import (
        assemble_matrix,
        scoring,
    )

    monkeypatch.setattr(assemble_matrix, "assemble", lambda **kw: calls.append("assemble"))
    monkeypatch.setattr(sweep, "_run", lambda cmd, *, label, extra_env=None: calls.append(label))
    monkeypatch.setattr(
        scoring, "score", lambda *, include_flagged: calls.append(f"score:{include_flagged}")
    )


# ---------------------------------------------------------------------------
# 1. Step order
# ---------------------------------------------------------------------------


def test_phase_p3_order_assemble_predictors_score(sweep, monkeypatch):
    """assemble runs BEFORE the predictors subprocess, which runs BEFORE score."""
    calls: list[str] = []
    _order_recorders(sweep, monkeypatch, calls)
    sweep.phase_p3(SimpleNamespace(skip_eval=False, skip_train=False))
    assert calls == ["assemble", "predictors", "score:False", "score:True"]
    assert calls.index("assemble") < calls.index("predictors") < calls.index("score:False")


def test_phase_p3_skip_eval_guards_predictors_only(sweep, monkeypatch):
    """--skip-eval skips ONLY the predictors subprocess; assemble + score
    stay unconditional (round-24 brief: flag semantics unchanged)."""
    calls: list[str] = []
    _order_recorders(sweep, monkeypatch, calls)
    sweep.phase_p3(SimpleNamespace(skip_eval=True, skip_train=True))
    assert calls == ["assemble", "score:False", "score:True"]


# ---------------------------------------------------------------------------
# 2. Fail-fast unification (no half-skip under skip_gpu)
# ---------------------------------------------------------------------------


def test_extract_all_missing_panel_raises_in_skip_gpu_mode(isolated_root):
    """A missing base_panel.json fail-fasts in skip_gpu mode too — the old
    warn-and-skip path silently dropped the base-prior predictor."""
    from explore_persona_space.experiments.behavior_testbed_545.predictors import extract_all

    with pytest.raises(FileNotFoundError, match=r"base_panel\.json"):
        extract_all(skip_gpu=True)


def test_extract_all_missing_panel_raises_in_gpu_mode(isolated_root, monkeypatch):
    """Same fail-fast in the default (GPU) mode, raised BEFORE any GPU work."""
    from explore_persona_space.experiments.behavior_testbed_545 import predictors

    def _no_gpu(*a, **kw):  # GPU groups must never be reached on a missing panel
        raise AssertionError("GPU extraction reached despite missing base_panel.json")

    monkeypatch.setattr(predictors, "extract_group_b_gpu", _no_gpu)
    monkeypatch.setattr(predictors, "extract_group_a_and_c_gpu", _no_gpu)
    with pytest.raises(FileNotFoundError, match=r"base_panel\.json"):
        predictors.extract_all(skip_gpu=False)


# ---------------------------------------------------------------------------
# 3. Read-after-write smoke: REAL assemble -> REAL base-prior read
# ---------------------------------------------------------------------------


def test_phase_p3_real_assemble_feeds_real_base_prior(sweep, isolated_root, monkeypatch):
    """REAL assemble() (fixture cells) writes base_panel.json; the predictors
    step — exercised through the reordered phase_p3 entry with the subprocess
    swapped for an in-process REAL extract_all(skip_gpu=True) — reads it and
    writes B__base_prior_level.json with non-empty cells."""
    from explore_persona_space.experiments.behavior_testbed_545 import output_root, scoring
    from explore_persona_space.experiments.behavior_testbed_545.predictors import extract_all

    _write_fixture_cells(isolated_root)
    panel_seen_before_predictors: list[bool] = []

    def _run_inprocess(cmd, *, label, extra_env=None):
        assert label == "predictors"
        # The read-after-write proof: the panel exists at predictor time.
        panel_seen_before_predictors.append((output_root() / "base_panel.json").exists())
        extract_all(skip_gpu=True)  # REAL CPU predictor entry (D + base prior)

    monkeypatch.setattr(sweep, "_run", _run_inprocess)
    monkeypatch.setattr(scoring, "score", lambda *, include_flagged: None)

    sweep.phase_p3(SimpleNamespace(skip_eval=False, skip_train=False))

    assert panel_seen_before_predictors == [True]
    panel = json.loads((output_root() / "base_panel.json").read_text())["panel"]
    assert panel["sycophancy__default"]["scalar"] == 0.25
    pred = json.loads((output_root() / "predictors" / "B__base_prior_level.json").read_text())
    assert pred["cells"], "base-prior predictor produced no cells from the fixture panel"
    assert pred["cells"]["wrong_claim_agreement|sycophancy"] == 0.25
    # assemble's other outputs landed too (score's inputs).
    assert (output_root() / "L_matrix.json").exists()
    assert (output_root() / "cell_metadata.json").exists()
