"""CPU-only unit tests for the issue #2094 P10 figure dump.

Synthetic minis through every figure builder (non-empty axes), the
low-coherence overlay marking (visible marker, never suppressed), the
optional-input skip logic, the stage-2 F_beh helper against a hand value,
and the ``savefig_paper`` provenance sidecar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_figures as F  # noqa: E402


def _ax_nonempty(ax) -> bool:
    return bool(ax.images or ax.lines or ax.collections or ax.patches or ax.tables)


def _fig_nonempty(fig) -> bool:
    return any(_ax_nonempty(ax) for ax in fig.axes)


def _row(
    *,
    pair_id: str = "mp--bare__q1--bare__q2",
    setting: str = "matched_prefix",
    slot: str = "ce",
    lv: str = "L14",
    dose: str = "a1",
    arm: str = "steered",
    vec_type: str = "A",
    coherent: bool = True,
    f_act: float | None = 0.4,
    f_beh_q: float | None = 0.5,
    f_beh_p: float | None = None,
    traversal: float | None = 0.7,
    profile: bool = True,
) -> dict:
    beh = {}
    if f_beh_q is not None or setting in ("matched_prefix", "cross"):
        beh["query"] = {"f_beh": None if not coherent else f_beh_q}
    if f_beh_p is not None or setting in ("matched_query", "cross"):
        beh["prefix"] = {"f_beh": None if not coherent else f_beh_p}
    alpha = None if dose == "replace" else float(dose.removeprefix("a"))
    row = {
        "block_key": f"{slot}|{lv}|{dose}|{vec_type}|{arm}",
        "slot": slot,
        "layer_variant": lv,
        "dose": dose,
        "alpha": alpha,
        "vec_type": vec_type,
        "arm": arm,
        "pair_id": pair_id,
        "setting": setting,
        "context_a": pair_id.split("--")[1],
        "context_b": pair_id.split("--")[2],
        "coherent": coherent,
        "excluded_incoherent": not coherent,
        "n_coherent": int(coherent),
        "n_total": 1,
        "cap_hit": False,
        "f_act": f_act if coherent else None,
        "f_act_raw": f_act,
        "traversal_ratio": traversal,
        "f_beh": beh,
    }
    if profile:
        row["f_act_profile"] = list(np.linspace(0.0, f_act or 0.0, 28))
    return row


PAIRS = [
    ("mp--bare__q1--bare__q2", "matched_prefix"),
    ("mp--conv__q1--conv__q2", "matched_prefix"),
    ("mq--bare__q1--persona__q1", "matched_query"),
    ("x--bare__q1--persona__q5", "cross"),
]


def _grid_rows() -> list[dict]:
    rows = []
    for pid, setting in PAIRS:
        for lv in ("L14", "joint_mid"):
            for dose in ("a0.5", "a1", "a2", "replace"):
                for slot in ("ce", "pe"):
                    fq = 0.3 + 0.1 * float(dose.removeprefix("a") if dose != "replace" else 8)
                    rows.append(
                        _row(
                            pair_id=pid,
                            setting=setting,
                            slot=slot,
                            lv=lv,
                            dose=dose,
                            f_act=0.2 + 0.05 * len(rows) % 7 * 0.1,
                            f_beh_q=fq if setting != "matched_query" else None,
                            f_beh_p=0.25 if setting != "matched_prefix" else None,
                        )
                    )
    # one fully-incoherent cell family (every pair incoherent at qspan/L14/a1)
    for pid, setting in PAIRS:
        rows.append(_row(pair_id=pid, setting=setting, slot="qspan", lv="L14", coherent=False))
    # Type-B twins on shared ce cells
    for pid, setting in PAIRS[:2]:
        rows.append(_row(pair_id=pid, setting=setting, vec_type="B", f_act=0.3))
    return rows


def _anchors() -> list[dict]:
    out = []
    for pid, setting in PAIRS:
        kinds = {"matched_prefix": ["query"], "matched_query": ["prefix"]}.get(
            setting, ["query", "prefix"]
        )
        for kind in kinds:
            out.append(
                {
                    "pair_id": pid,
                    "setting": setting,
                    "kind": kind,
                    "context_a": pid.split("--")[1],
                    "context_b": pid.split("--")[2],
                    "floor": {"mean": -0.5, "n": 8, "n_incoherent": 0, "n_judge_missing": 0},
                    "ceiling": {"mean": 0.5, "n": 8, "n_incoherent": 1, "n_judge_missing": 0},
                    "separation": 1.0,
                }
            )
    return out


def _bootstrap() -> dict:
    fams = {}
    for setting in ("matched_prefix", "matched_query", "cross"):
        for arm in ("steered", "null"):
            for dose in ("a0.5", "a1", "a2"):
                for metric in ("f_act", "f_beh_query", "f_beh_prefix"):
                    key = "|".join([arm, setting, "ce", "L14", dose, "A", metric])
                    fams[key] = {
                        "setting": setting,
                        "observed_mean": 0.3,
                        "n_pairs_used": 4,
                        "ci_lo": 0.1,
                        "ci_hi": 0.5,
                        "n_valid_draws": 100,
                    }
    return {"B": 100, "families": fams}


def test_heatmap_nonempty_and_low_coherence_overlay():
    fig = F.fig_f_heatmap(_grid_rows(), "f_act", "F_act")
    assert _fig_nonempty(fig)
    # the all-incoherent qspan family must carry the visible overlay marker
    labels = {artist.get_label() for ax in fig.axes for artist in ax.collections}
    assert "<50 percent coherent" in labels
    plt.close(fig)


def test_heatmap_never_suppresses_low_coherence_value():
    # a <50%-coherent cell with ONE coherent row still shows its value AND the marker
    rows = [
        _row(pair_id=PAIRS[0][0], coherent=True, f_act=0.9),
        _row(pair_id=PAIRS[1][0], coherent=False),
        _row(pair_id="mp--bare__q1--bare__q3", coherent=False),
    ]
    agg = F.aggregate_cells(rows, "f_act")
    cell = agg[("matched_prefix", "a1", "ce", "L14")]
    assert cell.coherent_frac < F.LOW_COHERENCE_FRAC
    assert cell.mean == pytest.approx(0.9)  # value kept, not suppressed
    fig = F.fig_f_heatmap(rows, "f_act", "F_act")
    assert _fig_nonempty(fig)
    plt.close(fig)


def test_dose_response_nonempty_with_bands_and_slopes():
    fig = F.fig_dose_response(_grid_rows(), "f_beh", "F_beh", _bootstrap())
    assert _fig_nonempty(fig)
    # a slope histogram row exists (patches on the last axes row)
    assert any(ax.patches for ax in fig.axes)
    plt.close(fig)
    fig2 = F.fig_dose_response(_grid_rows(), "f_act", "F_act", None)  # band-less path
    assert _fig_nonempty(fig2)
    plt.close(fig2)


def test_transport_and_fragility_figs_nonempty():
    tcells = []
    for arm in ("steered", "null"):
        for map_id, slot in (("m779_ce_L14", "ce"), ("m1738_pe_L19", "pe")):
            for pid, setting in PAIRS[:3]:
                tcells.append(
                    {
                        "map_id": map_id,
                        "slot": slot,
                        "dose": "a1",
                        "alpha": 1.0,
                        "vec_type": "A",
                        "arm": arm,
                        "pair_id": pid,
                        "setting": setting,
                        "cosine_tail": 0.4 if arm == "steered" else 0.05,
                    }
                )
    fig = F.fig_transport(tcells)
    assert _fig_nonempty(fig)
    plt.close(fig)

    fragility = {
        "anchor_baseline": {"incoherent_frac": 0.05, "cap_hit_frac": 0.0},
        "cells": [
            {
                "slot": slot,
                "layer_variant": lv,
                "dose": dose,
                "steered": {
                    "n": 4,
                    "incoherent": 1,
                    "cap_hit": 0,
                    "incoherent_frac": 0.25,
                    "cap_hit_frac": 0.0,
                    "excess_incoherence": 0.2,
                },
                "null": {
                    "n": 4,
                    "incoherent": 0,
                    "cap_hit": 1,
                    "incoherent_frac": 0.0,
                    "cap_hit_frac": 0.25,
                    "excess_incoherence": -0.05,
                },
            }
            for slot in ("ce", "pe")
            for lv in ("L14", "joint_mid")
            for dose in ("a1", "replace")
        ],
    }
    fig2 = F.fig_fragility(fragility)
    assert _fig_nonempty(fig2)
    plt.close(fig2)


def test_linearity_figs_nonempty():
    homog = {
        "families": {
            "ce_L14": {
                pid: {
                    "alphas": [0.5, 1.0, 2.0],
                    "cosine_matrix": np.eye(3).tolist(),
                    "reliabilities_sb": [0.9, 0.9, 0.9],
                    "disattenuated_cosine_matrix": (np.eye(3) * 0.95).tolist(),
                    "shift_norms": [1.0, 2.0, 4.1],
                    "degenerate": False,
                    "loglog_slope": 1.01,
                }
                for pid, _ in PAIRS[:3]
            }
        }
    }
    fig = F.fig_homogeneity(homog)
    assert _fig_nonempty(fig)
    plt.close(fig)

    l_fit = {
        "families": {
            "ce_L14_same_tail": {
                "pair_fold": {"pooled_r2": 0.4},
                "family_fold": {"pooled_r2": 0.2},
                "identity_bias_pooled_oof_r2": -0.1,
                "knn_retrieval": {
                    "euclidean": {
                        "acc_at_k": {"1": 0.5, "5": 0.8, "10": 1.0},
                        "chance_at_k": {"1": 0.1, "5": 0.5, "10": 1.0},
                    },
                    "cosine": {
                        "acc_at_k": {"1": 0.4, "5": 0.7, "10": 0.9},
                        "chance_at_k": {"1": 0.1, "5": 0.5, "10": 1.0},
                    },
                },
            }
        }
    }
    fig2 = F.fig_l_fit(l_fit)
    assert _fig_nonempty(fig2)
    plt.close(fig2)

    op = {
        "comparisons": {
            "L14_vs_M779_L14": {
                "procrustes_cosine_subspace": 0.4,
                "raw_cosine_subspace": 0.1,
                "procrustes_null_p97_5": 0.3,
                "raw_null_p97_5_abs": 0.05,
            }
        },
        "two_by_two": {
            "ce_L14": {
                "M_aligns": True,
                "L_predicts": False,
                "procrustes_cosine": 0.4,
                "procrustes_null_p97_5": 0.3,
                "family_fold_r2": None,
            }
        },
    }
    fig3 = F.fig_operator_2x2(op)
    assert _fig_nonempty(fig3)
    plt.close(fig3)


def test_scatter_and_exploratory_figs_nonempty():
    rows = _grid_rows()
    for fn, args in (
        (F.fig_transfer_decomposition, (rows,)),
        (F.fig_fact_vs_fbeh, (rows,)),
        (F.fig_fbeh_vs_traversal, (rows,)),
        (F.fig_anchor_separation, (_anchors(),)),
        (F.fig_fact_layer_profiles, (rows,)),
        (F.fig_type_ab, (rows,)),
        (F.fig_marginals, (rows,)),
    ):
        fig = fn(*args)
        assert _fig_nonempty(fig), fn.__name__
        plt.close(fig)
    fig = F.fig_audit_rates(
        {
            "grid": [{"flag_empty": False, "flag_script_intrusion": True, "flag_repetition": False}]
            * 5
        }
    )
    assert _fig_nonempty(fig)
    plt.close(fig)


def test_stage2_cell_f_hand_value():
    anchors = _anchors()  # floor -0.5, ceiling +0.5 -> denominator 1.0
    cell = "matched_prefix|ce|L14|a1|A"
    scores = []
    for draw, (sa, sb, coh) in enumerate(((20, 80, 90), (40, 60, 95), (10, 90, 30))):
        base = {
            "kind": "stage2",
            "cell": cell,
            "pair_id": PAIRS[0][0],
            "setting": "matched_prefix",
            "draw": draw,
        }
        scores.append({**base, "rubric_id": "coherence", "score": coh})
        scores.append(
            {**base, "rubric_id": "fq-q1", "rubric_kind": "query", "side": "a", "score": sa}
        )
        scores.append(
            {**base, "rubric_id": "fq-q2", "rubric_kind": "query", "side": "b", "score": sb}
        )
    out = F.stage2_cell_f(scores, anchors)
    # draw 2 is incoherent (30 <= 60) -> dropped; draws 0/1: delta 0.6, 0.2
    # F = (delta - (-0.5)) / 1.0 -> 1.1, 0.7 -> mean 0.9
    rec = out[(cell, "query")]
    assert rec["n_draws"] == 2
    assert rec["mean_f"] == pytest.approx(0.9)
    fig = F.fig_stage2_vs_stage1(scores, anchors, {"cells": []})
    assert _fig_nonempty(fig)
    plt.close(fig)


def test_build_all_skips_optional_inputs():
    inp = F.FigInputs(
        f_cells=_grid_rows(),
        null_cells=[_row(arm="null")],
        anchors=_anchors(),
        anchor_draws=[{"coherent": True, "cap_hit": False}],
        bootstrap=_bootstrap(),
        l_fit=None,
        operator_cmp=None,
        homogeneity=None,
        fragility=None,
    )
    figs = F.build_all(
        inp,
        only={
            "hero1_f_act_heatmap",
            "result1b_transport_cosines",
            "exp_stage2_vs_stage1",
            "exp_audit_rates",
        },
    )
    assert not isinstance(figs["hero1_f_act_heatmap"], str)
    for stem in ("result1b_transport_cosines", "exp_stage2_vs_stage1", "exp_audit_rates"):
        assert isinstance(figs[stem], str) and "skipped" in figs[stem]
    for fig in figs.values():
        if not isinstance(fig, str):
            plt.close(fig)


def test_savefig_meta_provenance(tmp_path):
    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig = F.fig_fact_vs_fbeh(_grid_rows())
    paths = savefig_paper(fig, "smoketest_fig", dir=tmp_path)
    plt.close(fig)
    assert paths["png"].exists() and paths["pdf"].exists()
    meta = json.loads((tmp_path / "smoketest_fig.meta.json").read_text())
    for key in ("commit", "created", "figsize"):
        assert key in meta, key
    assert meta.get("points"), "per-point data missing from the sidecar"
