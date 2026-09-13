"""Early noise decisions must keep the paired cohort and the two-model OR rule."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest

from explore_persona_space.analysis.workspace_diagnostics import rollout_noise_report

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "noise_decision", SCRIPTS / "workspace_jr_noise_decision.py"
)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def _noise(fraction):
    """Make explicit five-target decision inputs, including undefined variance."""
    return {
        "components": {
            name: {
                "noise_fraction": fraction,
                "higher_k_trigger": fraction is not None and fraction > 0.1,
            }
            for name in ("full", "J", "restJ", "R", "restR")
        },
        "trigger_threshold": 0.1,
        "higher_k_trigger": fraction is not None and fraction > 0.1,
    }


@pytest.mark.parametrize("fraction", [0.0, 0.1, None])
def test_one_nontriggering_model_never_clears_the_other_model(fraction):
    decision = reader.decision_summary(_noise(fraction))
    assert decision["higher_k_required_by_this_model"] is False
    assert decision["two_model_decision"] == "await_other_model_or_final_supplement"
    assert len(decision["undefined_targets"]) == (5 if fraction is None else 0)


def test_threshold_is_strict_and_inconsistent_flags_fail():
    noise = _noise(np.nextafter(0.1, np.inf))
    assert reader.decision_summary(noise)["two_model_decision"] == "required_by_this_model"
    noise["components"]["J"]["higher_k_trigger"] = False
    with pytest.raises(ValueError, match="strict threshold"):
        reader.decision_summary(noise)


def _json(path, value):
    """Write JSON fixture bytes that receive an actual checksum-bound receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root):
    """Exercise the real Source loader against explicit local fixture hashes."""
    hashes = {str(p.relative_to(root)): _digest(p) for p in root.rglob("*") if p.is_file()}
    receipt = root.parent / f"{root.name}_receipt.json"
    _json(
        receipt,
        {
            "repo": "superkaiba1/explore-persona-space-data",
            "revision": "a" * 40,
            "prefix": f"exploratory_workspace_jr/test/{root.name}",
            "files_verified": len(hashes),
            "verified_sha256": hashes,
        },
    )
    return {"root": str(root), "upload_receipt": str(receipt)}


def test_real_decision_body_ignores_excluded_context_noise(tmp_path, monkeypatch):
    """The all-captured trigger can be positive while the frozen paired trigger is negative."""
    ids, common = list("abcd"), list("abc")
    full = (
        np.array([0, 10, 20, 1000])[:, None, None]
        + np.array([0.1, 0.1, 0.1, 1000])[:, None, None] * np.arange(-2, 3)[None, :, None]
    )
    rollouts = {
        "full": full,
        "J": full * 0.3,
        "restJ": full * 0.7,
        "R": full * 0.6,
        "restR": full * 0.4,
    }
    noise_report, arrays = rollout_noise_report(rollouts, ids, list(range(42, 47)))
    assert noise_report["higher_k_trigger"] is True
    identity = {"model_role": "comparison", "fixture": True}
    root, diagnostics_root, cohort_root = (
        tmp_path / name for name in ("native", "diagnostics", "cohort")
    )
    fit = root / "fits/main/k10-rotationNone"
    _json(fit / "results.json", {"fixture": True})
    fit_terminal = "fit_operations/cells/k10-rotationNone/exit.json"
    _json(
        root / fit_terminal,
        {
            "phase": "complete",
            "exit_code": 0,
            "cell": "k10-rotationNone",
            "results_sha256": _digest(fit / "results.json"),
        },
    )
    _json(diagnostics_root / "analysis_complete.json", {"identity": identity})
    _json(
        diagnostics_root / "analysis_operations/exit.json",
        {
            "phase": "complete",
            "exit_code": 0,
            "analysis_complete_sha256": _digest(diagnostics_root / "analysis_complete.json"),
        },
    )
    _json(diagnostics_root / "noise_test.json", noise_report)
    np.savez(diagnostics_root / "noise_test_arrays.npz", **arrays)
    _json(cohort_root / "fixture.json", {})
    entry = {
        **_source(root),
        "role": "comparison",
        "kind": "observed",
        "k": 10,
        "rotation": None,
        "fit_relative": "fits/main/k10-rotationNone",
        "terminal_relative": fit_terminal,
    }
    manifest = {
        "role": "comparison",
        "native_fit": entry,
        "diagnostics": _source(diagnostics_root),
        "completion_cohort": _source(cohort_root),
    }
    native = {
        "fit": fit,
        "ids": ids,
        "manifest": {"identity": identity},
        "proof": {"fixture": True},
    }
    monkeypatch.setattr(
        reader, "run_identity", create_autospec(reader.run_identity, return_value=identity)
    )
    monkeypatch.setattr(reader, "read_cell", create_autospec(reader.read_cell, return_value=native))
    cohort = {
        "by_role": {"comparison": {"producer_identity": identity}},
        "joint_complete_context_ids": common,
    }
    monkeypatch.setattr(
        reader,
        "frozen_cohort",
        create_autospec(reader.frozen_cohort, return_value=(cohort, {"fixture": True})),
    )
    monkeypatch.setattr(
        reader,
        "bind_fitted_generation_sources",
        create_autospec(reader.bind_fitted_generation_sources, return_value={"fixture": True}),
    )

    def bind(source, phase, native, identity):
        source.path("analysis_complete.json")

    monkeypatch.setattr(
        reader, "bind_analysis", create_autospec(reader.bind_analysis, side_effect=bind)
    )
    targets = {name: values.mean(1) for name, values in rollouts.items()}
    monkeypatch.setattr(
        reader,
        "read_arrays",
        create_autospec(
            reader.read_arrays, return_value=(targets, {"ridge": targets, "mlp": targets})
        ),
    )
    out = tmp_path / "result"
    reader.run(
        manifest,
        out,
        SCRIPTS.parent / "configs/analysis/workspace_jr.yaml",
        SCRIPTS.parent / "docs/exploratory_workspace_jr/selected_contexts.json",
    )
    result = json.loads((out / "noise_decision.json").read_text())
    assert result["noise"]["context_ids"] == common
    assert result["decision"]["higher_k_required_by_this_model"] is False
    direct, _ = rollout_noise_report(
        {name: values[:3] for name, values in rollouts.items()}, common, list(range(42, 47))
    )
    for name in rollouts:
        assert result["noise"]["components"][name]["noise_fraction"] == pytest.approx(
            direct["components"][name]["noise_fraction"]
        )
    complete = json.loads((out / "noise_decision_complete.json").read_text())
    for relative, expected in complete["files_sha256"].items():
        assert _digest(out / relative) == expected
    with pytest.raises(ValueError, match="fresh output"):
        reader.run(manifest, out, Path("unused"), Path("unused"))
