"""Issue #2225 unit-5 figures: synthetic-fixture rendering pins.

Renders EVERY registered figure builder against tiny fake inputs and asserts
non-empty axes + plotted series (never present a blank render — #1112), plus
the inverted-CI errorbar clamp routed through the REAL contrast-forest figure
function to ``savefig`` (the xerr non-negative-offsets gotcha).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


F = _load("issue2225_figures")

GRID = [0.5, 1.5, 3.0, 5.0]
_LAYERS = 4  # tiny per-layer profiles


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _curve() -> dict:
    return {
        str(c): {
            "trait_mean": 60.0 - 8.0 * c,
            "rate_gt50": 0.4,
            "coherence_mean": 95.0 - 3.0 * c,
            "mmlu_acc": 0.70 - 0.01 * c,
            "n_api_refusal": 0,
        }
        for c in GRID
    }


def _sel_entry(cfg: str, dataset: str, sel: float = 3.0) -> dict:
    return {
        "config": cfg,
        "dataset": dataset,
        "steered_trait": "evil" if dataset == "mistake_opinions" else dataset,
        "grid": GRID,
        "selected_coef": sel,
        "curve": _curve(),
    }


def _arm_file(tag: str, trait: str, n_q: int = 3) -> dict:
    return {
        "target_tag": tag,
        "traits": {
            trait: {
                "tag": tag,
                "trait": trait,
                "cap_hit_fraction": 0.01,
                "per_question": [
                    {
                        "question_idx": i,
                        "mean": 45.0 + 3.0 * i,
                        "rollout_scores": [45.0 + 3.0 * i] * 2,
                        "rollout_n_api_refusal": [0, 0],
                    }
                    for i in range(n_q)
                ],
                "model_mean": 48.0,
                "rate_gt50": 0.3,
                "n_rollouts_scored": 2 * n_q,
                "n_rollouts_total": 2 * n_q,
                "accounting": {"n_api_refusal": 0},
            }
        },
    }


def _probe_row(tag: str, trait: str) -> dict:
    prof = [0.1 * i for i in range(_LAYERS)]
    return {
        "tag": tag,
        "trait": trait,
        "l1_layer_idx": 2,
        "variants": {
            v: {"shift_l1": prof[2], "shift_per_layer": prof}
            for v in ("full", "orth_E1", "orth_E2", "orth_E3")
        },
    }


def _proj_row(tag: str, trait: str) -> dict:
    prof = [0.05 * i for i in range(_LAYERS)]
    return {
        "tag": tag,
        "trait": trait,
        "l1_layer_idx": 2,
        "positions": {
            pos: {"direction": "E2", "shift_l1": prof[2], "shift_per_layer": prof}
            for pos in ("response_avg", "context_end", "prefix_end")
        },
    }


@pytest.fixture(scope="module")
def eval_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("i2225_figs_eval_root")
    selection = {f"{cfg}_evil": _sel_entry(cfg, "evil") for cfg in "ABCDEFGIP"}
    selection["H_evil"] = {
        "config": "H",
        "dataset": "evil",
        "steered_trait": "evil",
        "grid": [None],
        "selected_coef": None,
        "curve": {"prompt": {"trait_mean": 40.0, "coherence_mean": 92.0, "mmlu_acc": 0.71}},
        "note": "prompt-mode config (no coefficient grid; selection N/A)",
    }
    for ds in ("sycophancy", "hallucination", "mistake_opinions"):
        selection[f"A_{ds}"] = _sel_entry("A", ds)
        selection[f"C_{ds}"] = _sel_entry("C", ds)
    _write(root / "analysis" / "selection.json", {"selection": selection})

    for cfg in ("A", "C"):
        for ds, trait in (("evil", "evil"), ("sycophancy", "sycophancy")):
            _write(
                root / "trait_scores" / f"{cfg}_{ds}_3.0.json",
                _arm_file(f"{cfg}__{ds}__c3.0", trait),
            )
    _write(root / "coherence" / "A_evil_3.0.json", _arm_file("A__evil__c3.0", "evil"))

    # frozen evil CI DELIBERATELY inverted (lo > point > hi): the clamp pin.
    _write(
        root / "analysis" / "contrasts.json",
        {
            "contrasts": {
                "C_vs_A": {
                    "label": "primary",
                    "per_dataset": {
                        "evil": {
                            "n_questions": 3,
                            "selected": {"C": 3.0, "A": 3.0},
                            "frozen": {
                                "delta_point": 0.1,
                                "ci95": [0.2, 0.05],
                                "verdict": "Statistical tie",
                            },
                            "selection_inherited": {
                                "delta_point": -0.2,
                                "ci95": [-0.5, 0.1],
                                "verdict": "Statistical tie",
                                "n_draws_no_coherent_coef": 0,
                            },
                        }
                    },
                    "pooled": {
                        "datasets": ["evil"],
                        "frozen": {
                            "delta_point": -0.3,
                            "ci95": [-0.6, -0.1],
                            "verdict": "Context-position-superior",
                        },
                        "selection_inherited": {"delta_point": -0.2, "ci95": [-0.4, 0.0]},
                    },
                }
            }
        },
    )

    probe_shifts = {}
    proj_shifts = {}
    for trait in ("evil", "sycophancy", "hallucination"):
        tags = [f"A__{trait}__c3.0", f"baseft_{trait}", "base"]
        if trait == "evil":
            tags = [f"{c}__evil__c3.0" for c in "AC"] + ["H__evil", "baseft_evil", "base"]
        for tag in tags:
            probe_shifts[f"{tag}__{trait}"] = _probe_row(tag, trait)
            proj_shifts[f"{tag}__{trait}"] = _proj_row(tag, trait)
    _write(
        root / "analysis" / "probe_shifts.json",
        {"fit_summaries": {}, "sanity_gate": {}, "shifts": probe_shifts},
    )
    _write(root / "analysis" / "projection_shifts.json", {"shifts": proj_shifts})

    _write(
        root / "analysis" / "narrow_retention.json",
        {
            "per_arm": {
                "A_mistake_opinions_3.0": {
                    "target_tag": "A__mistake_opinions__c3.0",
                    "mistake_style_rate": 0.4,
                    "mean_score": 45.0,
                    "n_scored": 90,
                    "n_total": 100,
                },
                "base": {
                    "target_tag": "base",
                    "mistake_style_rate": 0.6,
                    "mean_score": 55.0,
                    "n_scored": 95,
                    "n_total": 100,
                },
            }
        },
    )

    _write(
        root / "judge_digest.json",
        {
            "per_arm": [
                {
                    "arm": "A_evil_3.0",
                    "tag": "A__evil__c3.0",
                    "rubric": "trait_evil",
                    "wave": "trait_scores",
                    "n_rollouts_scored": 6,
                    "n_rollouts_total": 6,
                    "n_total_draws": 18,
                    "n_content_dropped": 1,
                    "n_transport_lost": 0,
                    "n_api_refusal": 2,
                    "uncensored_rate": 0.89,
                },
                {
                    "arm": "C_evil_1.5",
                    "tag": "C__evil__c1.5",
                    "rubric": "trait_evil",
                    "wave": "trait_scores",
                    "n_rollouts_scored": 6,
                    "n_rollouts_total": 6,
                    "n_total_draws": 18,
                    "n_content_dropped": 0,
                    "n_transport_lost": 1,
                    "n_api_refusal": 0,
                    "uncensored_rate": 1.0,
                },
            ]
        },
    )
    return root


def _n_data_artists(fig) -> int:
    return sum(
        len(ax.lines) + len(ax.collections) + len(ax.patches) + len(ax.images) for ax in fig.axes
    )


@pytest.mark.parametrize("name", sorted(F.FIGURES))
def test_every_builder_produces_nonempty_figures(eval_root, name):
    figs = F.FIGURES[name](eval_root)
    assert figs, f"builder {name} produced zero figures"
    for stem, fig in figs.items():
        assert _n_data_artists(fig) > 0, f"{name}/{stem}: blank render (no plotted series)"
        plt.close(fig)


def test_hero_single_layer_overlays_both_methods(eval_root):
    figs = F.build_hero_single_layer(eval_root)
    fig = figs["hero_coef_response_single_layer"]
    # first metric panel overlays the Paper (A) and Context (C) series
    data_axes = [ax for ax in fig.axes if ax.lines]
    assert len(data_axes[0].lines) >= 2
    plt.close(fig)


def test_render_all_writes_png_and_meta_sidecar(eval_root, tmp_path):
    written, failures = F.render_all(eval_root, tmp_path / "figs")
    assert failures == [], failures
    assert len(written) >= len(F.FIGURES)  # per-trait builders emit several stems
    for stem, png in written.items():
        assert png.exists() and png.stat().st_size > 0, stem
        assert (tmp_path / "figs" / f"{stem}.meta.json").exists(), stem


def test_ci_offsets_clamp_inverted_interval():
    # inverted quantile CI (lo > point > hi): offsets clamp to 0, never negative
    assert F._ci_offsets(0.1, 0.2, 0.05) == (0.0, 0.0)
    lo, hi = F._ci_offsets(0.1, -0.2, 0.4)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)


def test_contrast_forest_saves_with_inverted_ci(eval_root, tmp_path):
    # the fixture's frozen evil CI is inverted; the REAL figure function must
    # render + savefig without matplotlib's negative-xerr ValueError
    figs = F.build_contrast_forest(eval_root)
    from explore_persona_space.analysis.paper_plots import savefig_paper

    paths = savefig_paper(figs["contrast_forest"], "contrast_forest", dir=tmp_path)
    assert paths["png"].exists()
    plt.close(figs["contrast_forest"])


def test_pretty_tag_never_renders_bare_slugs():
    assert F.pretty_tag("base") == "Base model"
    assert "Unsteered finetune" in F.pretty_tag("baseft_evil")
    lbl = F.pretty_tag("A__evil__c3.0")
    assert "__" not in lbl and "@ 3.0" in lbl
    assert "__" not in F.pretty_tag("H__evil")
