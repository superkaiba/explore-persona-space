"""Issue #2215 figures — fixture-driven pins over the REAL figure module.

The Phase C fixture outputs are produced by ``issue2215_analysis.run_analysis``
itself (the real producer, synthetic small-grain inputs from
``test_issue2215_analysis`` — cross-phase data-contract smoke per #518), so
every figure test reads the PRODUCTION artifact shape, never a hand-mirrored
schema. Assertions are artist/series counts + label hygiene + manifest/skip
semantics — no PNG pixel asserts. No network, no GPU, no repo
``eval_results/`` reads (sparse-worktree safe).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))  # sibling fixture helpers (precedent: #1092)

import issue2215_analysis as A  # noqa: E402
import issue2215_figures as F  # noqa: E402
from test_issue2215_analysis import (  # noqa: E402
    K_DRAWS,
    make_bank,
    make_ridge_payload,
    make_vc,
    write_va_shards,
)


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory) -> Path:
    """Real Phase C outputs: 2 cells (cell_b degenerate at pe), one ce arm
    (779ce) + one pe arm (1738pe) at layer 0 -> idbias_ce + idbias_pe
    auto-added; H2 skipped upstream (2 cells < 3)."""
    tmp = tmp_path_factory.mktemp("fig2215")
    bank = make_bank()
    pt = A.PairTable.from_bank(bank, None)
    vc = make_vc(pt)
    per_context = {
        cid: {"v_ce": vc["ce"][row].clone(), "v_pe": vc["pe"][row].clone()}
        for row, cid in enumerate(pt.ids)
    }
    vc_path = tmp / "vc_bank.pt"
    torch.save({"layers": vc["layers"], "per_context": per_context}, vc_path)
    write_va_shards(tmp / "va", pt.ids)
    ridge_ce = tmp / "ridge_ce_L0.pt"
    make_ridge_payload(ridge_ce, layer=0)
    ridge_pe = tmp / "ridge_pe_L0.pt"
    make_ridge_payload(ridge_pe, layer=0, w_scale=0.9)
    anchors = tmp / "anchors.jsonl"
    with anchors.open("w") as fh:
        for p in bank["pairs"]:
            fh.write(json.dumps({"cell": p["cell"], "separation": 0.4}) + "\n")
    results = tmp / "results"
    inp = A.AnalysisInputs(
        bank=bank,
        vc_bank_path=vc_path,
        va_dir=tmp / "va",
        banked_anchor_dir=None,
        arm_specs=[
            {"arm": "779ce", "slot": "ce", "paths": {0: ridge_ce}},
            {"arm": "1738pe", "slot": "pe", "paths": {0: ridge_pe}},
        ],
        results_dir=results,
        null_dir=tmp / "nulls",
        anchors_jsonl=anchors,
        cells=None,
        null_b=40,
        boot_b=40,
        k_draws=K_DRAWS,
        repro={"test": True},
    )
    A.run_analysis(inp)
    return results


@pytest.fixture(scope="module")
def res(results_dir) -> F.Results:
    F.set_paper_style("blog")
    return F.load_results(results_dir)


def _close(fig) -> None:
    plt.close(fig)


# ── loading + ordering ────────────────────────────────────────────────


def test_load_results_and_type_order(res):
    assert res.dv3_ok and res.dv3_layer == 0
    assert res.dv3_pairs, "dv3 per-pair rows should load from perpair/dv3_pairs.jsonl"
    order = F.type_order_worst_to_best(res)
    assert sorted(order) == ["cell_a", "cell_b"]
    # arms auto-added by the producer: one idbias twin per slot present
    assert F.arms_present(res) == ["779ce", "1738pe", "idbias_ce", "idbias_pe"]


# ── hero figures ──────────────────────────────────────────────────────


def test_hero1_bars_bands_and_label_hygiene(res):
    fig, skip = F.fig_hero1_per_type_2afc(res)
    assert skip is None
    ax = fig.axes[0]
    # hero arms present: 779ce (2 cells) + 1738pe (1 — cell_b degenerate at
    # pe) + idbias_ce (2) = 5 bars; 1738ce absent from the fixture.
    assert len(ax.patches) == 5
    # per-bar shuffled-pair null band segments (gray) + the 0.5 reference line
    band_lines = [ln for ln in ax.lines if ln.get_color() == F.NULL_COLOR]
    assert len(band_lines) == 5
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels and all("_" not in t for t in labels)  # plain-English ticks
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "shuffled-pair null 95% band" in legend_texts
    _close(fig)


def test_hero2_paired_dots_omit_degenerate_pe(res):
    fig, skip = F.fig_hero2_shift_ratio(res)
    assert skip is None
    ax = fig.axes[0]
    # dv1_ce: 2 cells; dv1_pe: 1 (cell_b degenerate excluded); dv2_tail: 2
    sizes = sorted(len(c.get_offsets()) for c in ax.collections)
    assert sizes == [1, 2, 2]
    assert ax.get_xscale() == "log"
    _close(fig)


# ── exploratory dump ──────────────────────────────────────────────────


def test_margin_scatter_panels_and_point_counts(res):
    fig, skip = F.fig_margin_scatter_per_type(res)
    assert skip is None
    # fitted arms with rows: 779ce (12 pairs -> 24 pts), 1738pe (6 -> 12)
    assert len(fig.axes) == 2
    counts = sorted(sum(len(c.get_offsets()) for c in ax.collections) for ax in fig.axes)
    assert counts == [12, 24]
    _close(fig)


def test_h2_scatter_skips_then_renders_from_per_cell_xy(res):
    fig, skip = F.fig_h2_shift_vs_separation(res)
    assert fig is None and "H2" in skip  # 2-cell fixture: skipped upstream
    coupling = {
        "h2": {
            "obs": 0.5,
            "ci95": [0.1, 0.9],
            "per_cell_xy": {
                "cell_a": {"x": 1.2, "y": 0.3},
                "cell_b": {"x": 0.8, "y": 0.6},
            },
        }
    }
    res2 = F.Results(
        dv1=res.dv1,
        dv2=res.dv2,
        dv3=res.dv3,
        coupling=coupling,
        bands=res.bands,
        dv3_pairs=res.dv3_pairs,
    )
    fig, skip = F.fig_h2_shift_vs_separation(res2)
    assert skip is None
    assert len(fig.axes) == 2  # dv3 present -> margin-vs-separation companion
    ax = fig.axes[0]
    assert len(ax.collections) == 1 and len(ax.collections[0].get_offsets()) == 2
    assert any("Spearman" in t.get_text() for t in ax.texts)  # rho + CI label
    _close(fig)


def test_cross_type_heatmaps_both_slots(res):
    fig, skip = F.fig_cross_type_cosine_heatmaps(res)
    assert skip is None
    heat_axes = [ax for ax in fig.axes if ax.images]
    assert len(heat_axes) == 2  # ce + pe panels (colorbar ax carries no image)
    _close(fig)


def test_per_layer_accuracy_lines(res):
    fig, skip = F.fig_per_layer_accuracy(res)
    assert skip is None
    ax = fig.axes[0]
    assert len(ax.containers) == 4  # one errorbar container per present arm
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert all("_" not in t for t in legend_texts)
    _close(fig)


def test_consistency_vs_band_panels(res):
    fig, skip = F.fig_consistency_vs_band(res)
    assert skip is None
    assert len(fig.axes) == 3
    # dv1_ce: 2 cells; dv1_pe: 1 (degenerate has no consistency); dv2_tail: 2
    n_err = [len(ax.containers) for ax in fig.axes]
    assert n_err == [2, 1, 2]
    _close(fig)


def test_knn_retrieval_bars(res):
    fig, skip = F.fig_knn_retrieval(res)
    assert skip is None
    for ax in fig.axes:
        assert len(ax.patches) == 4 * 3  # 4 arms x ks {1,5,10}
    _close(fig)


def test_carrier_transfer_scatter(res):
    fig, skip = F.fig_carrier_transfer(res)
    assert skip is None
    ax = fig.axes[0]
    sizes = sorted(len(c.get_offsets()) for c in ax.collections)
    assert sizes == [1, 1, 2, 2]  # pe arms see 1 cell; ce arms see 2
    _close(fig)


def test_raw_vs_normalized_panels(res):
    fig, skip = F.fig_raw_vs_normalized_magnitude(res)
    assert skip is None
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.collections) == 1 and len(ax.collections[0].get_offsets()) == 2
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    _close(fig)


def test_pooling_twin_deltas_panels(res):
    fig, skip = F.fig_pooling_twin_deltas(res)
    assert skip is None
    assert len(fig.axes) == 2
    ax1, ax2 = fig.axes
    assert len(ax1.collections) == 1 and len(ax1.collections[0].get_offsets()) == 2
    # panel 2: 779ce 2 cells + 1738pe 1 cell (cell_b N/A at pe)
    sizes = sorted(len(c.get_offsets()) for c in ax2.collections)
    assert sizes == [1, 2]
    _close(fig)


# ── render_all manifest + file outputs ────────────────────────────────


def test_render_all_writes_files_and_records_skips(results_dir, tmp_path):
    out = tmp_path / "figs"
    manifest = F.render_all(results_dir, out)
    assert set(manifest) == {stem for stem, _ in F.FIGURES}  # registry-complete
    written = {k for k, v in manifest.items() if v["written"]}
    assert {"hero1_per_type_2afc", "hero2_shift_ratio_per_type"} <= written
    # 2-cell fixture: H2 skipped upstream -> recorded skip, never silent
    assert manifest["expl_h2_shift_vs_separation"]["written"] is False
    assert manifest["expl_h2_shift_vs_separation"]["skipped"]
    for stem in written:
        png = out / f"{stem}.png"
        assert png.exists() and png.stat().st_size > 0, stem
        assert (out / f"{stem}.meta.json").exists(), stem


def test_render_all_records_dv3_skips_on_tiny_outputs(results_dir, tmp_path):
    """A tiny-run results tree (DV3 skipped upstream, no dv3 per-pair rows)
    still renders the DV1/DV2 figures and records every DV3 figure skip."""
    dst = tmp_path / "results_tiny"
    shutil.copytree(results_dir, dst)
    (dst / "dv3_map_discrimination.json").write_text(
        json.dumps({"skipped": "tiny mode — declared blind spot", "repro": {}})
    )
    (dst / "perpair" / "dv3_pairs.jsonl").unlink()
    manifest = F.render_all(dst, tmp_path / "figs_tiny")
    assert manifest["hero2_shift_ratio_per_type"]["written"] is True
    assert manifest["expl_consistency_vs_band"]["written"] is True
    for stem in (
        "hero1_per_type_2afc",
        "expl_margin_scatter_per_type",
        "expl_per_layer_accuracy",
        "expl_knn_retrieval",
        "expl_carrier_transfer",
    ):
        assert manifest[stem]["written"] is False, stem
        assert manifest[stem]["skipped"], stem
