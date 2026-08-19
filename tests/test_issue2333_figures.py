"""Issue #2333 figures render pins — every figure function draws real PNGs on
synthetic per-pair tables (Agg backend, tmp FIG_DIR), covering the r2-added
figures (F_act hero mirror, per-cell forests, k traces, scheme contrast,
arm-vs-ce) and the empty-input skip guards (no blank-axes artifacts).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2333_figures as F  # noqa: E402

from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402


@pytest.fixture()
def synth_data():
    cells = list(C.S1_CELLS[:2])
    s1_pids = [f"pair{i:02d}" for i in range(4)]
    s2_pids = ["s2pairA", "s2pairB"]
    steered, null = [], []
    for slug in C.ARM_SLUGS:
        kind, k, scheme = C.parse_arm(slug)
        for i, pid in enumerate(s1_pids):
            base = {
                "pair_id": pid,
                "cell": cells[i % 2],
                "set": "s1",
                "arm_slug": slug,
                "kind": kind,
                "k": k,
                "scheme": scheme,
                "separation": 1.0,
                "n_coherent": 8,
                "n_rows": 10,
            }
            steered.append(
                {
                    **base,
                    "variant": "steered",
                    "f_beh": 0.5 + 0.05 * k + 0.01 * i,
                    "f_act": 0.4 + 0.05 * k + 0.01 * i,
                    "f_beh_continuation": 0.3 + 0.02 * i if kind == "prefill" else None,
                }
            )
            null.append(
                {**base, "variant": "null", "f_beh": 0.1 + 0.01 * i, "f_act": 0.05 + 0.01 * i}
            )
        for i, pid in enumerate(s2_pids):
            base = {
                "pair_id": pid,
                "cell": C.S2_CELL,
                "set": "s2",
                "arm_slug": slug,
                "kind": kind,
                "k": k,
                "scheme": scheme,
                "separation": 1.0,
                "n_coherent": 8,
                "n_rows": 10,
            }
            steered.append(
                {
                    **base,
                    "variant": "steered",
                    "f_beh": 0.6 + 0.02 * i,
                    "f_act": None,
                    "f_beh_continuation": 0.25 if kind == "prefill" else None,
                }
            )
            null.append({**base, "variant": "null", "f_beh": 0.12, "f_act": None})
    calib = [
        {"pair_id": pid, "set": s, "arm": "steered", "f_beh": 0.5}
        for pid, s in [*[(p, "s1") for p in s1_pids], *[(p, "s2") for p in s2_pids]]
    ]
    stats = {
        "per_set": {
            "s1": {
                "arms": {
                    slug: {
                        "diff_mean": 0.4,
                        "diff_ci": (0.3, 0.5),
                        "recovery_samewave": {"ratio": 1.1, "ratio_ci": (0.9, 1.3)},
                    }
                    for slug in C.ARM_SLUGS
                }
            }
        }
    }
    return {"steered": steered, "null": null, "calib": calib, "ce": [], "stats": stats}


def test_all_figures_render_pngs(tmp_path, monkeypatch, synth_data):
    monkeypatch.setattr(F, "FIG_DIR", tmp_path / "figs")
    F.FIG_DIR.mkdir(parents=True)
    F.set_paper_style()
    F.fig_hero(synth_data, "q25")
    F.fig_hero(synth_data, "q25", field="f_act")  # secondary-DV mirror
    F.fig_recovery(synth_data, "q25")
    F.fig_perpair(synth_data, "q25")
    F.fig_forest_cells(synth_data, "q25")
    F.fig_k_traces(synth_data, "q25")
    F.fig_scheme_contrast(synth_data, "q25")
    F.fig_arm_vs_ce(synth_data, "q25")
    F.fig_whole_vs_continuation(synth_data, "q25")
    F.fig_coherence(synth_data, "q25")
    pngs = {p.name for p in F.FIG_DIR.glob("*.png")}
    expected = {
        "hero_snowball_q25_s1.png",
        "hero_snowball_q25_s2.png",
        "hero_snowball_act_q25_s1.png",
        "recovery_ratio_q25.png",
        "perpair_prefill3_q25.png",
        "forest_cells_q25.png",
        "k_traces_q25.png",
        "scheme_contrast_q25.png",
        "arm_vs_ce_q25.png",
        "whole_vs_continuation_q25.png",
        "coherence_q25.png",
    }
    missing = expected - pngs
    assert not missing, missing
    # s2 has NO f_act values -> the mirror SKIPS that panel set, never a blank
    assert "hero_snowball_act_q25_s2.png" not in pngs


def test_f_act_hero_skips_cleanly_when_no_va(tmp_path, monkeypatch, synth_data):
    monkeypatch.setattr(F, "FIG_DIR", tmp_path / "figs")
    F.FIG_DIR.mkdir(parents=True)
    data = {
        **synth_data,
        "steered": [{**r, "f_act": None} for r in synth_data["steered"]],
        "null": [{**r, "f_act": None} for r in synth_data["null"]],
    }
    F.fig_hero(data, "q25", field="f_act")
    assert not list(F.FIG_DIR.glob("hero_snowball_act_*.png"))
