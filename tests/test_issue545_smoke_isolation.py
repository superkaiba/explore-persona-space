"""Issue #545 round-19 — smoke-output isolation + base-panel completeness +
state repair (the smoke-artifacts-satisfy-production-resume-guards class).

Round-18 incident: a pod smoke (``--phase p1 --rows marker --seeds 0
--smoke``) wrote ``manifest_p1.json`` + a 2-column ``cells/base_panel/`` into
the PRODUCTION output root; production's ``done_cells`` check and the bare
``base_panel.exists()`` guard then skipped retraining/refilling, and the K1
gate FAILed on the 4-step smoke adapter. These tests pin:

1. ``--smoke`` routes EVERY dispatcher output root (results, cells, adapters)
   under an isolated ``smoke/`` segment that production resolution cannot see;
2. the base-panel resume is per-FILE completeness, never a bare exists();
3. the repair helper removes exactly the contaminated state (manifest entry,
   cell eval dir, leftover adapter dir), leaves the partial base_panel, and
   is idempotent;
4. the K1 bookends component carries the round-19 diagnostic fields while the
   gating predicate stays the prereg raw rates.
"""

from __future__ import annotations

import importlib.util
import json
import os
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


@pytest.fixture(scope="module")
def repair_mod():
    return _load_script("issue545_repair_smoke_contamination")


@pytest.fixture()
def isolated_root(tmp_path, monkeypatch):
    """Point the package roots at a tmp dir with smoke isolation OFF."""
    monkeypatch.setenv("EPM_OUTPUT_ROOT", str(tmp_path))
    # setenv registers the key with monkeypatch so direct os.environ writes
    # inside the code under test are undone at teardown.
    monkeypatch.setenv("I545_SMOKE_OUTPUT", "0")
    monkeypatch.delenv("I545_SMOKE_OUTPUT", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Smoke-output isolation
# ---------------------------------------------------------------------------


def test_smoke_isolation_roots(sweep, isolated_root):
    """--smoke activation appends smoke/ to EVERY output root; production
    resolution (flag absent) cannot see artifacts written under it."""
    from explore_persona_space.experiments.behavior_testbed_545 import (
        adapters_root,
        cells_dir,
        output_root,
    )

    # Production view (flag absent).
    assert output_root() == isolated_root
    assert cells_dir() == isolated_root / "cells"
    assert adapters_root() == isolated_root / "adapters"

    # The dispatcher's --smoke branch.
    sweep._activate_smoke_isolation()
    assert os.environ["I545_SMOKE_OUTPUT"] == "1"
    assert output_root() == isolated_root / "smoke"
    assert cells_dir() == isolated_root / "smoke" / "cells"
    assert adapters_root() == isolated_root / "smoke" / "adapters"

    # Simulate the round-18 smoke writes — now under the smoke root.
    manifest = output_root() / "manifest_p1.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([{"cell": "marker_primary_seed0", "gpu": 0}]))
    (cells_dir() / "base_panel").mkdir(parents=True)
    (cells_dir() / "base_panel" / "marker__default.json").write_text("{}")

    # Production resume guards are physically unable to see them.
    del os.environ["I545_SMOKE_OUTPUT"]
    prod_manifest = output_root() / "manifest_p1.json"
    assert not prod_manifest.exists()
    done_cells = (
        {m["cell"] for m in json.loads(prod_manifest.read_text())}
        if prod_manifest.exists()
        else set()
    )
    assert "marker_primary_seed0" not in done_cells
    assert not (cells_dir() / "base_panel").exists()


def test_battery_read_falls_back_to_production_read_only(isolated_root):
    """Smoke runs READ frozen P0 batteries from the production root when the
    smoke root lacks them; writers always target the active (smoke) root."""
    from explore_persona_space.experiments.behavior_testbed_545 import batteries_dir
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import load_battery

    prod_batteries = isolated_root / "batteries"
    prod_batteries.mkdir(parents=True)
    (prod_batteries / "marker_eval_questions.json").write_text(json.dumps({"probes": ["q"]}))

    os.environ["I545_SMOKE_OUTPUT"] = "1"
    try:
        # Writer path: the active batteries dir is smoke-rooted.
        assert batteries_dir() == isolated_root / "smoke" / "batteries"
        # Reader path: falls back to the production freeze.
        data = load_battery("marker_eval_questions.json")
        assert data == {"probes": ["q"]}
        # A smoke-root copy takes precedence once present.
        batteries_dir().mkdir(parents=True, exist_ok=True)
        (batteries_dir() / "marker_eval_questions.json").write_text(
            json.dumps({"probes": ["smoke_q"]})
        )
        assert load_battery("marker_eval_questions.json") == {"probes": ["smoke_q"]}
    finally:
        del os.environ["I545_SMOKE_OUTPUT"]


# ---------------------------------------------------------------------------
# 2. Base-panel per-file completeness resume
# ---------------------------------------------------------------------------


def _touch_base_panel(root: Path, names: list[str]) -> Path:
    bp = root / "cells" / "base_panel"
    bp.mkdir(parents=True, exist_ok=True)
    for n in names:
        (bp / n).write_text("{}")
    return bp


def test_base_panel_partial_reports_missing_passes(sweep, isolated_root):
    """The round-18 contaminated state (marker+capability only) must report
    BOTH production passes incomplete with the exact missing files."""
    _touch_base_panel(isolated_root, ["marker__default.json", "capability__default.json"])
    args = SimpleNamespace(smoke=False, skip_judges=False)
    todo = sweep._base_panel_todo(args)
    assert len(todo) == 2, "default-context AND robustness passes must both re-run"

    default_missing = todo[0][2]
    assert "broad_em__default.json" in default_missing
    assert "sycophancy__default.json" in default_missing
    # Existing files are kept, never re-required.
    assert "marker__default.json" not in default_missing
    assert "capability__default.json" not in default_missing
    # sensitivity_only columns never run by default.
    assert "broad_em_n100__default.json" not in default_missing
    assert len(default_missing) == 16  # 16 judged/structural columns

    _robustness_contexts, robustness_columns, robustness_missing = todo[1]
    assert set(robustness_columns) == {"broad_em", "sycophancy", "marker", "harmful_compliance"}
    # 3 contexts x (3 judged + marker) = 12 files.
    assert len(robustness_missing) == 12
    assert "marker__wildchat_prefix.json" in robustness_missing
    assert "broad_em__qwen_default_system.json" in robustness_missing


def test_base_panel_complete_reports_no_passes(sweep, isolated_root):
    """A fully-populated base panel re-runs nothing."""
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        ROBUSTNESS_COLUMNS,
        ROBUSTNESS_CONTEXTS,
        base_panel_expected_files,
    )

    required = base_panel_expected_files(["default"], None) | base_panel_expected_files(
        list(ROBUSTNESS_CONTEXTS), list(ROBUSTNESS_COLUMNS)
    )
    _touch_base_panel(isolated_root, sorted(required))
    assert sweep._base_panel_todo(SimpleNamespace(smoke=False, skip_judges=False)) == []


def test_base_panel_smoke_subset_and_skip_judges(sweep, isolated_root):
    """Smoke requires only its 2-column default pass; --skip-judges swaps the
    judged requirement for the gen product."""
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        base_panel_expected_files,
    )

    _touch_base_panel(isolated_root, ["marker__default.json", "capability__default.json"])
    assert sweep._base_panel_todo(SimpleNamespace(smoke=True, skip_judges=False)) == []

    no_judge = base_panel_expected_files(["default"], None, include_judged=False)
    assert "completions__broad_em__default.json" in no_judge
    assert "broad_em__default.json" not in no_judge
    assert "marker__default.json" in no_judge  # HF product unaffected


# ---------------------------------------------------------------------------
# 3. State repair helper
# ---------------------------------------------------------------------------


def test_repair_removes_contamination_and_is_idempotent(repair_mod, isolated_root):
    manifest = isolated_root / "manifest_p1.json"
    manifest.write_text(
        json.dumps(
            [
                {"cell": "marker_primary_seed0", "gpu": 0},
                {"cell": "bad_medical_primary_seed0", "gpu": 1},
            ]
        )
    )
    cell_dir = isolated_root / "cells" / "marker_primary_seed0"
    cell_dir.mkdir(parents=True)
    (cell_dir / "band_stop_result.json").write_text("{}")
    bp = _touch_base_panel(isolated_root, ["marker__default.json", "capability__default.json"])
    adapter_dir = isolated_root / "adapters" / "marker_primary_seed0"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}")

    actions = repair_mod.repair("marker_primary_seed0", "p1")
    assert len(actions) == 3, actions
    kept = json.loads(manifest.read_text())
    assert [m["cell"] for m in kept] == ["bad_medical_primary_seed0"]
    assert not cell_dir.exists()
    assert not adapter_dir.exists()
    # The partial base panel is LEFT for the completeness resume.
    assert (bp / "marker__default.json").exists()
    assert (bp / "capability__default.json").exists()

    # Idempotent second run.
    assert repair_mod.repair("marker_primary_seed0", "p1") == []


def test_repair_refuses_under_smoke_isolation(repair_mod, isolated_root):
    os.environ["I545_SMOKE_OUTPUT"] = "1"
    try:
        with pytest.raises(SystemExit, match="PRODUCTION root"):
            repair_mod.repair("marker_primary_seed0", "p1")
    finally:
        del os.environ["I545_SMOKE_OUTPUT"]


# ---------------------------------------------------------------------------
# 4. K1 bookends diagnostic fields (gate semantics unchanged)
# ---------------------------------------------------------------------------


def _write_broad_em(root: Path, cell: str, summary: dict) -> None:
    d = root / "cells" / cell
    d.mkdir(parents=True, exist_ok=True)
    (d / "broad_em__default.json").write_text(json.dumps({"summary": summary}))


def test_k1_bookends_diagnostic_fields(isolated_root):
    """The bookends component records k / denominators / sentinel counts /
    all-completions rate / base-panel reference; the predicate stays the
    prereg raw rates (the pod's 5.66% educational read still FAILs)."""
    from explore_persona_space.experiments.behavior_testbed_545.gates import k1_gate_verdict

    _write_broad_em(
        isolated_root,
        "bad_medical_primary_seed0",
        {"rate": 0.30, "k": 30, "n": 100, "n_betley_code": 10, "n_betley_refusal": 2},
    )
    # The verified round-18 pod numbers: 6/106 judgeable (294 CODE sentinels).
    _write_broad_em(
        isolated_root,
        "educational_insecure_primary_seed0",
        {"rate": 0.0566, "k": 6, "n": 106, "n_betley_code": 294, "n_betley_refusal": 0},
    )
    _write_broad_em(
        isolated_root,
        "base_panel",
        {"rate": 0.0, "k": 0, "n": 380, "n_betley_code": 15, "n_betley_refusal": 5},
    )

    verdict = k1_gate_verdict()
    bookends = verdict["components"]["bookends"]
    # Predicate UNCHANGED: educational 5.66% >= 2% prereg cap -> ordering fails.
    assert bookends["ordering_holds"] is False
    assert verdict["pass"] is None  # dose-select + band-stop components absent

    ed = bookends["educational_detail"]
    assert ed["k"] == 6
    assert ed["n_judgeable"] == 106
    assert ed["n_betley_code"] == 294
    assert ed["n_betley_refusal"] == 0
    assert ed["rate_all_completions"] == pytest.approx(6 / 400)

    base = bookends["base_panel_broad_em"]
    assert base["rate"] == 0.0
    assert base["n_judgeable"] == 380
    assert bookends["bad_medical_detail"]["rate_all_completions"] == pytest.approx(30 / 112)


def test_k1_bookends_base_panel_absent_is_null(isolated_root):
    """No base-panel broad-EM file -> base_panel_broad_em is None (the
    round-18 contaminated state), never a crash."""
    from explore_persona_space.experiments.behavior_testbed_545.gates import k1_gate_verdict

    _write_broad_em(isolated_root, "bad_medical_primary_seed0", {"rate": 0.30, "k": 30, "n": 100})
    _write_broad_em(
        isolated_root, "educational_insecure_primary_seed0", {"rate": 0.0, "k": 0, "n": 100}
    )
    bookends = k1_gate_verdict()["components"]["bookends"]
    assert bookends["base_panel_broad_em"] is None
    assert bookends["ordering_holds"] is True
