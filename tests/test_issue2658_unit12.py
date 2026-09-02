"""Issue #2658 unit-12 tests: plan-section-7 figure code (scripts/issue2658_figures.py).

Pins (each guard shown FIRING, not merely absent):
- no figure-level caption artist on ANY figure (fig.texts empty, no suptitle) —
  axes + ticks + legend + panel titles only;
- the color->meaning map is single-sourced, unique per meaning, pinned stable,
  and the ARTISTS actually carry those colors;
- a not-estimable ROW (committed-partition AND row-gate-failure alike) renders
  as a LABELED ABSENCE — never a zero-height bar, never a point at zero
  (asserted on artist/plotted data, not prose);
- a not-estimable CELL's prompts never enter any per-prompt view; the ledger
  figure shows the revised denominator as counts by cause;
- a missing prospective ledger RAISES (never "all cells estimable"); an empty
  ledger mapping refuses to render;
- every aggregate figure has its per-unit companion, and render_all emits the
  full registered set with non-empty PNGs;
- every rendered label/legend/title is reader-facing (length-bounded, no
  internal shorthand slugs / underscores);
- deterministic rendering: same input -> identical plotted arrays (jitter is
  content-seeded);
- an INVERTED pointwise CI is clamped to non-negative errorbar offsets through
  the REAL figure function to savefig (the xerr non-negativity rule).

The mini report is REAL: unit 9's synthesizer + unit 11's own _synthetic_ladder
-> build_panel -> run_inference (tiny registry, require_registered_universe
False), never a hand-mocked report dict. All tests are OFFLINE and synthetic:
no GPU, no network, no judge API call, no bank item text.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2658_comparators as U  # noqa: E402
import issue2658_figures as FIG  # noqa: E402
import issue2658_inference as INF  # noqa: E402

TINY_REG = INF.InferenceRegistry(
    n_perm_initial=40,
    perm_chunk_initial=(20, 20),
    n_perm_extended=100,
    perm_chunk_extension=(30, 30),
    n_boot=40,
    boot_chunk=20,
    n_ci_draws=50,
    min_discordant_prompts=5,
    min_answers_per_class=10,
    min_prompts_per_class=2,
)

# rowA: estimable + eligible, one prospectively-excluded cell (victim).
# rowB: committed-partition not-estimable for the external direction (C2).
# rowC: eligible but synthesized too small for the row gates (gate-fail).
_SPECS = {
    "rowA": {"n_prompts": 60, "n_superfamilies": 12},
    "rowB": {"n_prompts": 60, "n_superfamilies": 12},
    "rowC": {"n_prompts": 20, "n_superfamilies": 8},
}
_PARTITION = {"eligible": ["rowA", "rowC"], "not_estimable": ["rowB"]}


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """Real mini report through unit 11's own pipeline (3 rows, tiny registry)."""
    root = tmp_path_factory.mktemp("u12-mini")
    rows_input: dict = {}
    ladder: dict = {}
    ledgers: dict = {}
    victim = {}
    for i, (row, spec) in enumerate(_SPECS.items()):
        rd = U.synthesize_row_data(
            row=row,
            n_prompts=spec["n_prompts"],
            n_responses=6,
            d=8,
            n_superfamilies=spec["n_superfamilies"],
            effect=2.0,
            seed=7 + i,
        )
        records = INF._synthetic_ladder(rd, root / "comp" / row, seed=7 + i)
        ladder.update(records)
        cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
        excluded = []
        if row == "rowA":
            vcell = cells[-1]
            victim = {
                "cell": vcell,
                "prompts": {
                    r.prompt_id for r in rd.rows if f"{r.source_frame}|{r.stratum}" == vcell
                },
                "n_cells": len(cells),
            }
            excluded = [{"cell": vcell, "cause": "bank-too-small", "n_test_eligible": 3}]
        ledgers[row] = INF.synthetic_row_ledger(row, cells, excluded)
        comps = ["c5_full_probe"] + (["c2_direction_dot"] if row in _PARTITION["eligible"] else [])
        panel = INF.build_panel(
            row, records, comps, ledgers[row], INF.prompt_cells_from_rowdata(rd)
        )
        c5 = records[(row, "c5_full_probe")]
        rows_input[row] = INF.RowInputs(
            row=row,
            panel=panel,
            rowdata=rd,
            selected_c=float(c5["selected_c"]),
            scores_sha={c: records[(row, c)]["scores_sha256"] for c in comps},
        )
    report = INF.run_inference(
        rows_input,
        _PARTITION,
        TINY_REG,
        root / "out",
        require_registered_universe=False,
    )
    # Fixture sanity: the three designed dispositions actually realized.
    assert report["rows"]["rowA"]["estimable"] is True
    assert report["rows"]["rowB"]["estimable"] is True  # gates pass; C2 barred
    assert report["rows"]["rowC"]["estimable"] is False  # row gates fail
    assert "rowB" in report["not_estimable"]["C2"]
    points = FIG.build_points(rows_input, report)
    return SimpleNamespace(
        report=report,
        points=points,
        ladder=ladder,
        ledgers=ledgers,
        victim=victim,
        rows_input=rows_input,
    )


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _row_pos(rows: list[str]) -> dict[str, float]:
    return {r: float(len(rows) - 1 - i) for i, r in enumerate(rows)}


def _all_figs(m) -> dict[str, tuple[plt.Figure, dict]]:
    return {
        "macro": FIG.fig_macro_auroc(m.report, synthetic=True),
        "macro_pp": FIG.fig_macro_auroc_per_prompt(m.report, m.points, synthetic=True),
        "holm": FIG.fig_holm_adjusted_p(m.report, synthetic=True),
        "rawp": FIG.fig_raw_p_mc(m.report, synthetic=True),
        "delta": FIG.fig_delta(m.report, synthetic=True),
        "delta_pp": FIG.fig_delta_per_prompt(m.report, m.points, synthetic=True),
        "ladder": FIG.fig_comparator_ladder(m.ladder, synthetic=True),
        "ladder_pr": FIG.fig_comparator_ladder_per_row(m.ladder, synthetic=True),
        "cells": FIG.fig_cell_ledger(m.ledgers),
    }


# ---------------------------------------------------------------------------
# No caption block inside any plot (standing user directive).
# ---------------------------------------------------------------------------
def test_no_figure_level_caption_artists(mini):
    for name, (fig, _) in _all_figs(mini).items():
        assert list(fig.texts) == [], f"{name}: fig.text caption artist present"
        assert fig.get_suptitle() == "", f"{name}: suptitle present"


# ---------------------------------------------------------------------------
# One color = one meaning: single source, unique, stable, used by artists.
# ---------------------------------------------------------------------------
def test_color_map_unique_stable_and_used(mini):
    cmap = FIG.COLOR_BY_MEANING
    assert len(set(cmap.values())) == len(cmap), "two meanings share a color"
    assert set(FIG.COMPARATOR_LABELS) <= set(cmap)
    # Stability pins (a silent recolor is test-breaking, not silent).
    assert cmap["c5_full_probe"] == "#0072B2"
    assert cmap["c2_direction_dot"] == "#E69F00"
    assert cmap["c5_minus_c2"] == "#000000"
    assert cmap["not_estimable"] == "#9a9a9a"

    # Artists carry the mapped colors: delta errorbar is the paired-contrast
    # black; ladder-per-row scatters follow comparator order.
    fig, _ = FIG.fig_delta(mini.report, synthetic=True)
    (line,) = [ln for ln in fig.axes[0].lines if ln.get_marker() == "o"]
    assert mcolors.to_rgba(line.get_color()) == mcolors.to_rgba(cmap["c5_minus_c2"])

    fig, plotted = FIG.fig_comparator_ladder_per_row(mini.ladder, synthetic=True)
    comps = [c for c in FIG.COMPARATOR_ORDER if c in plotted["series"]]
    colls = fig.axes[0].collections
    assert len(colls) >= len(comps)
    for coll, comp in zip(colls[: len(comps)], comps, strict=True):
        got = tuple(coll.get_facecolor()[0][:3])
        assert got == mcolors.to_rgba(cmap[comp])[:3], comp

    # Absence text uses the not-estimable gray.
    fig, _ = FIG.fig_macro_auroc(mini.report, synthetic=True)
    absent_texts = [t for t in fig.axes[0].texts if "not estimable" in t.get_text()]
    assert absent_texts
    for t in absent_texts:
        assert mcolors.to_rgba(t.get_color()) == mcolors.to_rgba(cmap["not_estimable"])


# ---------------------------------------------------------------------------
# Not-estimable ROW -> labeled absence; never a zero bar or a point at zero.
# ---------------------------------------------------------------------------
def test_not_estimable_row_is_labeled_absence_not_zero(mini):
    fig, plotted = FIG.fig_macro_auroc(mini.report, synthetic=True)
    ax = fig.axes[0]
    pos = _row_pos(plotted["rows"])
    absent = {(a["row"], a["comparator"]) for a in plotted["absent"]}
    # rowB: committed partition bars C2; rowC: gate-fail bars both.
    assert ("rowB", "c2_direction_dot") in absent
    assert ("rowC", "c2_direction_dot") in absent
    assert ("rowC", "c5_full_probe") in absent
    # Every absence carries a rendered label; no bars exist at all.
    assert all("not estimable" in a["label"] for a in plotted["absent"])
    assert len(ax.patches) == 0, "a bar artist would permit a zero-height bar"
    # The absent slots get NO plotted point (not a point at zero, not any point).
    for comp, off in (("c2_direction_dot", +0.18), ("c5_full_probe", -0.18)):
        ser = plotted["series"][comp]
        assert 0.0 not in ser["x"], "an AUROC point rendered at exactly zero"
        for row in ("rowB", "rowC"):
            if (row, comp) in absent:
                assert pos[row] + off not in ser["y"], (row, comp)
    # rowA is the only row with a C2 point.
    assert len(plotted["series"]["c2_direction_dot"]["x"]) == 1
    # The label text is actually rendered on the axes.
    rendered = [t.get_text() for t in ax.texts]
    assert sum("not estimable" in t for t in rendered) == len(plotted["absent"])

    # Delta figure: only rowA plots; rowB/rowC are labeled absences.
    fig, plotted = FIG.fig_delta(mini.report, synthetic=True)
    pos = _row_pos(plotted["rows"])
    assert {a["row"] for a in plotted["absent"]} == {"rowB", "rowC"}
    assert plotted["y"] == [pos["rowA"]]
    assert len(plotted["x"]) == 1  # exactly one measured delta; absences add none
    assert pos["rowB"] not in plotted["y"] and pos["rowC"] not in plotted["y"]


def test_ladder_not_estimable_record_is_labeled_absence(mini):
    lad = dict(mini.ladder)
    lad[("rowB", "c2_direction_dot")] = {"status": "not-estimable"}
    fig, plotted = FIG.fig_comparator_ladder_per_row(lad, synthetic=True)
    pos = _row_pos(plotted["rows"])
    absent = [a for a in plotted["absent"] if a["row"] == "rowB"]
    assert absent and all("not estimable" in a["label"] for a in absent)
    ser = plotted["series"]["c2_direction_dot"]
    assert pos["rowB"] not in ser["y"], "not-estimable record produced a point"
    assert 0.0 not in ser["x"], "not-estimable record produced a point at zero"
    assert any("not estimable" in t.get_text() for t in fig.axes[0].texts)


# ---------------------------------------------------------------------------
# Not-estimable CELL: prompts excluded upstream never reach a per-unit view;
# the ledger figure shows the revised denominator by cause.
# ---------------------------------------------------------------------------
def test_not_estimable_cell_prompts_never_plotted(mini):
    vic = mini.victim
    acc = mini.report["rows"]["rowA"]["panel"]
    assert acc["cells_total"] == vic["n_cells"]
    assert acc["cells_used"] == vic["n_cells"] - 1
    assert acc["excluded_cells"][0]["cell"] == vic["cell"]

    comp_points = mini.points["rows"]["rowA"]["comparators"]
    for comp, entry in comp_points.items():
        cells = {p["cell"] for p in entry["prompts"]}
        pids = {p["prompt_id"] for p in entry["prompts"]}
        assert vic["cell"] not in cells, comp
        assert not (pids & vic["prompts"]), comp

    fig, plotted = FIG.fig_macro_auroc_per_prompt(mini.report, mini.points, synthetic=True)
    for comp, ser in plotted["series"].items():
        assert not (set(ser["prompt_ids"]) & vic["prompts"]), comp

    fig, plotted = FIG.fig_cell_ledger(mini.ledgers)
    i = plotted["rows"].index("rowA")
    assert plotted["estimable"][i] == vic["n_cells"] - 1
    assert plotted["by_cause"]["bank-too-small"][i] == 1
    # The exclusion is a COUNT segment in the ledger, never an AUROC value:
    # the ledger axes carry only count bars (no scatter/errorbar artists).
    assert len(fig.axes[0].collections) == 0


def test_missing_ledger_raises(tmp_path):
    with pytest.raises(INF.InferenceInputError):
        FIG.load_cell_ledgers(tmp_path / "absent.json")
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"manifest_kind": "eligible_frame", "rows": []}))
    with pytest.raises(INF.InferenceInputError):
        FIG.load_cell_ledgers(bad)
    with pytest.raises(FIG.FigureInputError):
        FIG.fig_cell_ledger({})


# ---------------------------------------------------------------------------
# Aggregate -> per-unit companion coverage; render_all emits the full set.
# ---------------------------------------------------------------------------
def test_render_all_emits_aggregates_with_companions(mini, tmp_path):
    saved = FIG.render_all(
        mini.report, mini.points, mini.ladder, mini.ledgers, tmp_path, synthetic=True
    )
    expected = (
        set(FIG.AGGREGATE_COMPANIONS)
        | set(FIG.AGGREGATE_COMPANIONS.values())
        | set(FIG.STANDALONE_FIGURES)
    )
    assert set(saved) == expected
    for stem, paths in saved.items():
        png = Path(paths["png"])
        assert png.exists() and png.stat().st_size > 0, stem
    # Companion stems are distinct from their aggregates.
    for agg, comp in FIG.AGGREGATE_COMPANIONS.items():
        assert agg != comp


def test_render_all_ladder_omission_is_explicit(mini, tmp_path):
    saved = FIG.render_all(mini.report, mini.points, None, mini.ledgers, tmp_path, synthetic=True)
    assert "issue2658_comparator_ladder" not in saved
    assert "issue2658_comparator_ladder_per_row" not in saved
    assert "issue2658_cell_ledger" in saved


# ---------------------------------------------------------------------------
# Reader-facing text: bounded lengths, no internal shorthand.
# ---------------------------------------------------------------------------
_BANNED_FRAGMENTS = (
    "c0_",
    "c1_",
    "c2_",
    "c3_",
    "c4_",
    "c5_",
    "devmean",
    "nuisance",
    "auroc_",
    "n_perm",
    "holm_",
)


def test_labels_reader_facing_and_bounded(mini):
    for name, (fig, _) in _all_figs(mini).items():
        fig.canvas.draw()  # realize log-axis tick formatters
        for ax in fig.axes:
            texts = [ax.get_xlabel(), ax.get_ylabel(), ax.get_title("left")]
            texts += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
            leg = ax.get_legend()
            if leg is not None:
                texts += [t.get_text() for t in leg.get_texts()]
            texts += [t.get_text() for t in ax.texts]
            for s in texts:
                if not s or s.startswith("$"):
                    continue  # empty / mathtext tick labels
                assert len(s) <= 70, f"{name}: over-long label {s!r}"
                low = s.lower()
                for frag in _BANNED_FRAGMENTS:
                    assert frag not in low, f"{name}: internal shorthand {frag!r} in {s!r}"
                assert "_" not in s, f"{name}: underscore slug leaked into {s!r}"


# ---------------------------------------------------------------------------
# Determinism: same input -> identical plotted arrays (jitter content-seeded).
# ---------------------------------------------------------------------------
def test_deterministic_plotted_arrays(mini):
    for fn in (
        lambda: FIG.fig_macro_auroc(mini.report, synthetic=True),
        lambda: FIG.fig_macro_auroc_per_prompt(mini.report, mini.points, synthetic=True),
        lambda: FIG.fig_delta_per_prompt(mini.report, mini.points, synthetic=True),
        lambda: FIG.fig_cell_ledger(mini.ledgers),
    ):
        _, p1 = fn()
        _, p2 = fn()
        assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# ---------------------------------------------------------------------------
# Synthetic-smoke labeling: report-driven figures carry the tag; the ledger
# figure renders real committed bookkeeping and is deliberately untagged.
# ---------------------------------------------------------------------------
def test_synthetic_tag_present_when_synthetic(mini):
    fig, _ = FIG.fig_macro_auroc(mini.report, synthetic=True)
    assert FIG.SYNTHETIC_TAG in fig.axes[0].get_title("left")
    fig, _ = FIG.fig_macro_auroc(mini.report, synthetic=False)
    assert FIG.SYNTHETIC_TAG not in fig.axes[0].get_title("left")
    fig, _ = FIG.fig_holm_adjusted_p(mini.report, synthetic=True)
    assert FIG.SYNTHETIC_TAG in fig.axes[0].get_title("left")
    fig, _ = FIG.fig_cell_ledger(mini.ledgers)
    assert "prospective ledger" in fig.axes[0].get_title("left")
    assert FIG.SYNTHETIC_TAG not in fig.axes[0].get_title("left")


# ---------------------------------------------------------------------------
# Loader schema guards.
# ---------------------------------------------------------------------------
def test_report_and_points_schema_guards(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(FIG.FigureInputError, match="schema"):
        FIG.load_report(bad)
    partial = tmp_path / "r2.json"
    partial.write_text(json.dumps({"schema": INF.REPORT_SCHEMA, "families": {}}))
    with pytest.raises(FIG.FigureInputError, match="missing required key"):
        FIG.load_report(partial)
    badp = tmp_path / "p.json"
    badp.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(FIG.FigureInputError, match="schema"):
        FIG.load_points(badp)


# ---------------------------------------------------------------------------
# Inverted CI offsets clamp to zero through the REAL figure fn to savefig.
# ---------------------------------------------------------------------------
def test_inverted_ci_is_clamped_not_crashing(mini, tmp_path):
    rep = copy.deepcopy(mini.report)
    desc = rep["rows"]["rowA"]["descriptive"]["c5_full_probe"]
    m = float(desc["macro_auroc"])
    desc["macro_ci_pointwise"] = [m + 0.01, m - 0.01]  # deliberately inverted
    fig, plotted = FIG.fig_macro_auroc(rep, synthetic=True)
    ser = plotted["series"]["c5_full_probe"]
    assert all(v >= 0.0 for v in ser["ci_lo"] + ser["ci_hi"])
    fig.savefig(tmp_path / "clamped.png")  # renders without ValueError

    rep2 = copy.deepcopy(mini.report)
    test = rep2["families"]["C5_minus_C2"]["tests"]["rowA"]
    test["one_sided_lower_bound"] = float(test["delta_hat"]) + 0.05  # inverted
    fig, _ = FIG.fig_delta(rep2, synthetic=True)
    fig.savefig(tmp_path / "clamped_delta.png")
