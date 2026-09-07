"""Artificial plotting fixtures are used only in isolated unit tests."""

import copy

import matplotlib.pyplot as plt
import pytest

from scripts import issue1739_natural_figures as figures


def curves_fixture():
    rows = []
    for behavior in figures.TRAITS:
        for u in figures.U_GRID:
            statistics = {
                arm: {"mean": rho, "min": rho - 0.01, "max": rho + 0.01}
                for arm, rho in zip(figures.NATURAL_ROSTER, [0.2, 0.3, 0.4], strict=True)
            }
            statistics["mapped_minus_context"] = {"mean": 0.1, "min": 0.08, "max": 0.12}
            rows.append({"behavior": behavior, "generic_u": u, "statistics": statistics})
    return rows


def test_overview_preserves_all_curves_and_export_scale(tmp_path):
    figures.set_c2a_style()
    fig, frac = figures.plot_overview(curves_fixture())
    assert len(fig.axes) == 6
    for ax in fig.axes[:3]:
        assert len(ax.lines) == 3
        for line in ax.lines:
            assert list(line.get_xdata()) == list(figures.U_GRID)
    result = figures.save_c2a_figure(
        fig,
        tmp_path / "unit_test_only",
        include_width=frac,
        title="Unit-test fixture, not scientific results",
        subject="Layout test",
        creator="pytest",
        png_dpi=40,
    )
    assert result["record"]["include_width_frac"] == 1.0
    assert result["pdf"].is_file()
    plt.close(fig)


def dataset_fixture():
    return [
        {
            "primary": [
                {
                    "behavior": "evil",
                    "dataset": "hhrt",
                    "generic_u": 250,
                    "seed": s,
                    "arm": arm,
                    "n_eval": 12,
                    "rho": s / 10,
                }
                for arm in figures.NATURAL_ROSTER
            ]
        }
        for s in range(5)
    ]


def test_dataset_seed_range_is_descriptive():
    rows = figures.dataset_curves(dataset_fixture())
    assert len(rows) == 3
    assert rows[0]["mean"] == pytest.approx(0.2)
    assert rows[0]["min"] == 0.0
    assert rows[0]["max"] == 0.4


@pytest.mark.parametrize("fault", ["missing_seed", "duplicate_seed", "count"])
def test_dataset_inconsistent_coverage_fails(fault):
    cells = dataset_fixture()
    if fault == "missing_seed":
        cells.pop()
    elif fault == "duplicate_seed":
        cells.append(copy.deepcopy(cells[0]))
    else:
        cells[0]["primary"][0]["n_eval"] += 1
    with pytest.raises(ValueError):
        figures.dataset_curves(cells)
