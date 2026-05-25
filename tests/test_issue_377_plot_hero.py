"""Smoke test for ``scripts/issue_377_plot_hero.py`` (plan v2 §6.2).

Round-9 v9 hero plot was stale-keyed on the v1 ``B-incontext@{k}``
condition names; the eval rig writes ``B-incontext-turns@{k}`` +
``B-incontext-length@{k}`` at v2. Without a smoke test the eval pipeline
would silently break the figure step on the next run. This test
constructs a minimal fixture run-result JSON with all three multi-turn
families and Asserts that ``plot_hero`` writes PNG + PDF + meta JSON
without raising.

PURE: no eval run, no model, no HF Hub. Only matplotlib + numpy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# issue_377_plot_hero is under scripts/ — import via importlib.
_PLOT_SCRIPT = Path(__file__).parent.parent / "scripts" / "issue_377_plot_hero.py"
_spec = importlib.util.spec_from_file_location("issue_377_plot_hero", _PLOT_SCRIPT)
assert _spec is not None and _spec.loader is not None
issue_377_plot_hero = importlib.util.module_from_spec(_spec)
sys.modules["issue_377_plot_hero"] = issue_377_plot_hero
_spec.loader.exec_module(issue_377_plot_hero)


def _make_seed_result(seed: int) -> dict:
    """Build a minimal per-seed run_result JSON with all four multi-turn
    families at all three k values, plus A and H6.
    """
    per_condition: dict[str, dict] = {
        "A": {"rate": 0.85, "found": 170, "total": 200},
        "H6": {"rate": 0.02, "found": 4, "total": 200},
    }
    # k=5: high fire; k=10: medium; k=20: low for drift, high for in-context arms.
    for k, drift_rate, inc_turns_rate, inc_length_rate, null_rate in [
        (5, 0.70, 0.82, 0.80, 0.03),
        (10, 0.45, 0.78, 0.76, 0.04),
        (20, 0.08, 0.72, 0.70, 0.05),
    ]:
        per_condition[f"B@{k}"] = {
            "rate": drift_rate,
            "found": int(drift_rate * 200),
            "total": 200,
        }
        per_condition[f"B-incontext-turns@{k}"] = {
            "rate": inc_turns_rate,
            "found": int(inc_turns_rate * 200),
            "total": 200,
        }
        per_condition[f"B-incontext-length@{k}"] = {
            "rate": inc_length_rate,
            "found": int(inc_length_rate * 200),
            "total": 200,
        }
        per_condition[f"B-null@{k}"] = {
            "rate": null_rate,
            "found": int(null_rate * 200),
            "total": 200,
        }
    return {
        "seed": seed,
        "per_condition": per_condition,
    }


@pytest.fixture
def fixture_results_dir(tmp_path: Path) -> Path:
    """Write per-seed fixture JSONs to a tmp dir laid out like
    ``eval_results/issue_377/seedN/run_result.json``.
    """
    for seed in (42, 137, 256):
        seed_dir = tmp_path / f"seed{seed}"
        seed_dir.mkdir()
        with open(seed_dir / "run_result.json", "w") as f:
            json.dump(_make_seed_result(seed), f)
    return tmp_path


class TestPlotHeroSmokeTest:
    def test_plot_hero_writes_outputs_for_v2_schema(
        self, fixture_results_dir: Path, tmp_path: Path
    ):
        """End-to-end: load three seeds, plot hero, expect PNG + PDF +
        meta.json on disk. Asserts the v2 condition keys
        (`B-incontext-turns@k` + `B-incontext-length@k`) are read
        without KeyError.
        """
        fig_dir = tmp_path / "figures"
        seed_results = issue_377_plot_hero._load_per_seed(fixture_results_dir, [42, 137, 256])
        assert len(seed_results) == 3

        issue_377_plot_hero.plot_hero(seed_results, "hero_smoke", fig_dir)

        assert (fig_dir / "hero_smoke.png").exists()
        assert (fig_dir / "hero_smoke.pdf").exists()
        assert (fig_dir / "hero_smoke.meta.json").exists()

    def test_plot_hero_keyerror_on_stale_v1_schema(self, tmp_path: Path):
        """If the run_result.json carries the old v1 key
        ``B-incontext@k`` (without the `-turns` / `-length` split), the
        plot script raises KeyError instead of silently dropping a line.
        """
        seed_dir = tmp_path / "seed42"
        seed_dir.mkdir()
        v1_payload = {
            "seed": 42,
            "per_condition": {
                "A": {"rate": 0.8, "found": 160, "total": 200},
                "H6": {"rate": 0.02, "found": 4, "total": 200},
                "B@5": {"rate": 0.7, "found": 140, "total": 200},
                "B@10": {"rate": 0.4, "found": 80, "total": 200},
                "B@20": {"rate": 0.1, "found": 20, "total": 200},
                "B-incontext@5": {"rate": 0.8, "found": 160, "total": 200},
                "B-incontext@10": {"rate": 0.78, "found": 156, "total": 200},
                "B-incontext@20": {"rate": 0.75, "found": 150, "total": 200},
                "B-null@5": {"rate": 0.03, "found": 6, "total": 200},
                "B-null@10": {"rate": 0.04, "found": 8, "total": 200},
                "B-null@20": {"rate": 0.05, "found": 10, "total": 200},
            },
        }
        with open(seed_dir / "run_result.json", "w") as f:
            json.dump(v1_payload, f)

        seed_results = issue_377_plot_hero._load_per_seed(tmp_path, [42])
        with pytest.raises(KeyError, match="B-incontext-turns"):
            issue_377_plot_hero.plot_hero(seed_results, "hero_v1", tmp_path / "figures")
