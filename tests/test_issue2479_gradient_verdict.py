"""Tests for scripts/issue2479_gradient_verdict.py (issue #2479, plan SS3/SS4 Step 6).

Two parts:

1. The SS3-mandated boundary-grid unit test of the pure ``verdict_label``
   function: n in {1, 2, 11, 12, 16} x p in {0.049, 0.051} x each gate
   fired/not-fired x both rho signs — exactly one label fires per combination,
   with the SS3-named boundary expectations asserted explicitly.
2. A hermetic end-to-end fixture test: synthetic panel / axis-freeze /
   instrument-gates / cell / ladder JSONs (in the #1345 Phase-F shape) through
   the full script into tmp dirs, verifying the headline rho, the one-sided
   add-one permutation-p arithmetic, the three-stage exclusion accounting, the
   three denominators, and figure emission. No network, no GPU.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2479_gradient_verdict as gv  # noqa: E402

# ---------------------------------------------------------------------------
# Part 1 — SS3 boundary grid over the pure verdict predicate
# ---------------------------------------------------------------------------
N_GRID = (1, 2, 11, 12, 16)
P_GRID = (0.049, 0.051)
RHO_GRID = (0.5, -0.5)


def _gate_configs() -> list[dict[str, bool]]:
    """All-pass plus each single gate fired (False) — 5 configurations."""
    configs = [dict.fromkeys(gv.GATE_KEYS, True)]
    for k in gv.GATE_KEYS:
        cfg = dict.fromkeys(gv.GATE_KEYS, True)
        cfg[k] = False
        configs.append(cfg)
    return configs


def test_labels_are_distinct():
    assert len(set(gv.ALL_LABELS)) == len(gv.ALL_LABELS) == 5


def test_boundary_grid_exactly_one_label_per_combination():
    n_combos = 0
    for n, p, rho, gates in itertools.product(N_GRID, P_GRID, RHO_GRID, _gate_configs()):
        label = gv.verdict_label(n, rho, p, gates)
        n_combos += 1
        # exactly one label fires: the function returns a single string drawn
        # from the 5-label set (pairwise distinct per test above)
        assert label in gv.ALL_LABELS, (n, p, rho, gates, label)
        if not all(gates.values()):
            expected = gv.LABEL_INSTRUMENT_SUSPECT
        elif n < 2:
            expected = gv.LABEL_BOUNDED_NO_STAT
        elif n < 12:
            expected = gv.LABEL_BOUNDED_PANEL
        elif rho > 0 and p <= 0.05:
            expected = gv.LABEL_ESTABLISHED
        else:
            expected = gv.LABEL_BOUNDED
        assert label == expected, (n, p, rho, gates, label, expected)
    assert n_combos == len(N_GRID) * len(P_GRID) * len(RHO_GRID) * 5  # 100


def test_named_boundaries():
    all_pass = dict.fromkeys(gv.GATE_KEYS, True)
    # n=11, p=0.049, gates pass -> bounded — insufficient panel
    assert gv.verdict_label(11, 0.6, 0.049, all_pass) == gv.LABEL_BOUNDED_PANEL
    # n=12, same -> established
    assert gv.verdict_label(12, 0.6, 0.049, all_pass) == gv.LABEL_ESTABLISHED
    # p exactly at 0.05 establishes (predicate: p <= 0.05)
    assert gv.verdict_label(12, 0.6, 0.05, all_pass) == gv.LABEL_ESTABLISHED
    # any gate False -> Instrument-suspect regardless of n/p/rho
    for k in gv.GATE_KEYS:
        gates = dict(all_pass)
        gates[k] = False
        assert gv.verdict_label(16, 0.9, 0.001, gates) == gv.LABEL_INSTRUMENT_SUSPECT
        assert gv.verdict_label(1, None, None, gates) == gv.LABEL_INSTRUMENT_SUSPECT
    # n=1 -> no estimable statistic (rho/p may be None)
    assert gv.verdict_label(1, None, None, all_pass) == gv.LABEL_BOUNDED_NO_STAT
    # n>=12, gates pass, negative rho or p above alpha -> plain bounded
    assert gv.verdict_label(16, -0.4, 0.049, all_pass) == gv.LABEL_BOUNDED
    assert gv.verdict_label(16, 0.4, 0.051, all_pass) == gv.LABEL_BOUNDED
    # NaN rho never establishes
    assert gv.verdict_label(16, float("nan"), 0.01, all_pass) == gv.LABEL_BOUNDED


def test_gate_dict_is_validated():
    with pytest.raises(ValueError):
        gv.verdict_label(16, 0.5, 0.01, {"band_agreement_pass": True})
    bad = dict.fromkeys(gv.GATE_KEYS, True)
    bad["name_mask_pass"] = 1  # not a bool
    with pytest.raises(TypeError):
        gv.verdict_label(16, 0.5, 0.01, bad)


# ---------------------------------------------------------------------------
# Part 2 — hermetic end-to-end fixture
# ---------------------------------------------------------------------------
STORY_ONPOLICY = "story_onpolicy"  # REGIME_LABEL["r4op"] in the fill script
STORY_INSERTED = "story_inserted"  # REGIME_LABEL["r4"]


def _direction_block(ceiling, r2_4, r2_1, ib, null4, acc_ceiling=0.5, acc4=0.4):
    return {
        "r2": {"4_bias_refit": [r2_4], "1_direct": [r2_1]},
        "ceiling_r2": [ceiling],
        "identity_bias_r2": [ib],
        "null_r2": {"4_bias_refit": [null4], "1_direct": [null4]},
        "fold_ids": [0, 1, 2, 3, 4],
        "knn_retrieval_fold0": {
            "n_pool": 100,
            "ceiling": {"acc@1": [acc_ceiling], "chance@1": 0.01},
            "4_bias_refit": {"acc@1": [acc4], "chance@1": 0.01},
            "identity_bias": {"acc@1": [0.05], "chance@1": 0.01},
            "identity_bias_cosine": {"metric": "cosine", "acc@1": [0.05], "chance@1": 0.01},
            "ceiling_cosine": {"metric": "cosine", "acc@1": [acc_ceiling], "chance@1": 0.01},
            "4_bias_refit_cosine": {"metric": "cosine", "acc@1": [acc4], "chance@1": 0.01},
        },
    }


def _ladder_json(
    src_id, src_label, variant, ceiling, r2_4, *, r2_1=0.1, ib=0.02, null4=0.01, source_means=None
):
    """A minimal #1345 Phase-F ladder entry (both directions, reduced basis)."""
    fwd = f"{src_label}->{variant}"
    rev = f"{variant}->{src_label}"
    entry = {
        "regimes": [src_id, variant],
        "n_matched": 900,
        "metadata": {
            "regime_labels": {src_id: src_label, variant: variant},
            "rung_order": ["1_direct", "4_bias_refit"],
        },
        "reduced": {
            fwd: _direction_block(ceiling, r2_4, r2_1, ib, null4),
            # reverse direction present (as in real files) — must be ignored
            rev: _direction_block(0.9, 0.0, 0.0, 0.0, 0.0),
            "basis": "reduced",
            "n_train_min": 700,
            "d": 3584,
        },
    }
    if source_means is not None:
        # r2 fill emits the source-side mean-activation vectors at ENTRY level
        entry["source_means"] = source_means
    return entry


def _cell_json(variant, ceiling, ib, n, *, mean_context_vec=None, mean_answer_vec=None):
    cell = {
        "regime": variant,
        "arm": "context",
        "metadata": {"layer": 19, "seed": 0, "model": "instruct"},
        "reduced": {
            "ceiling_r2": [ceiling],
            "identity_bias_r2": [ib],
            "n": n,
            "basis": "reduced",
        },
    }
    # r2 fill emits the cell mean-activation vectors at ENTRY level
    if mean_context_vec is not None:
        cell["mean_context_vec"] = mean_context_vec
    if mean_answer_vec is not None:
        cell["mean_answer_vec"] = mean_answer_vec
    return cell


# fixture panel: 7 planned characters ->
#   gus  : not in axis_freeze            (stage-1 exclusion)
#   fox  : in axis, no fit outputs       (stage-2 / G1 exclusion)
#   eel  : ceiling 0.01 < 0.05           (stage-3 ceiling exclusion; raw kept)
#   ada, bee, cat, dog: fraction-eligible, recovery monotone in axis score
FIXTURE = {
    # name: (score, ceiling, rung4, cell_n, inserted)
    "ada": (90.0, 0.50, 0.45, 900, True),
    "bee": (80.0, 0.50, 0.40, 850, True),
    "cat": (70.0, 0.50, 0.35, 800, False),
    "dog": (60.0, 0.50, 0.30, 750, False),
    "eel": (50.0, 0.01, 0.005, 700, False),
    "fox": (40.0, None, None, None, False),
}


def _write_fixture(tmp_path: Path) -> Path:
    eval_dir = tmp_path / "eval_results" / "issue_2479"
    grad = eval_dir / "story_char_gradient"
    grad.mkdir(parents=True)

    panel = []
    for name in [*FIXTURE, "gus"]:
        panel.append(
            {
                "name": name,
                "display_name": name.capitalize(),
                "design_band": "A",
                "variant_op": f"char_2479_{name}_op",
                "variant_inserted": f"char_2479_{name}"
                if FIXTURE.get(name, (0, 0, 0, 0, False))[4]
                else None,
                "desc": f"fixture character {name}",
                "inserted_subset": FIXTURE.get(name, (0, 0, 0, 0, False))[4],
            }
        )
    (eval_dir / "panel.json").write_text(json.dumps(panel, indent=1))

    chars = {}
    ordered = sorted(FIXTURE, key=lambda n: -FIXTURE[n][0])
    for rank, name in enumerate(ordered, start=1):
        chars[name] = {
            "tag": name,
            "design_band": "A",
            "variant_op": f"char_2479_{name}_op",
            "score": FIXTURE[name][0],
            "rank": rank,
            "n_scored_items": 100,
        }
    axis = {
        "issue": 2479,
        "characters": chars,
        "gates": {
            "band_agreement_pass": True,
            "band_agreement_rho": 0.9,
            "axis_range_pass": True,
            "axis_range": 50.0,
        },
    }
    (eval_dir / "axis_freeze.json").write_text(json.dumps(axis, indent=1))

    inst = {"gates": {"verbatim_flatness_pass": True, "name_mask_pass": True}}
    (eval_dir / "instrument_gates.json").write_text(json.dumps(inst, indent=1))

    # Freeze-side emit-items stats sidecars (default --axis-items-stats-dir):
    # mean answer length strictly DEcreasing in axis score -> rho exactly -1.
    items_dir = eval_dir / "axis_items"
    items_dir.mkdir()
    for name, (score, *_rest) in FIXTURE.items():
        (items_dir / f"axis_items_{name}.stats.json").write_text(
            json.dumps(
                {
                    "character": name,
                    "mean_answer_len_chars_prepared": 4000.0 - score * 10,
                    "mean_answer_len_chars_axis": 3900.0 - score * 10,
                }
            )
        )

    # Freeze-side per-draw sidecar (axis_draws.json) — the violin's data.
    draws = {
        "per_character": {
            name: {
                "conv_id_draws": {f"{name}_c{j}": [score - 3.0 + j, score + j] for j in range(2)}
            }
            for name, (score, ceiling, *_r) in FIXTURE.items()
            if ceiling is not None
        }
    }
    (eval_dir / "axis_draws.json").write_text(json.dumps(draws))

    src_means = {"r4op": {"context": [1.0, 0.0, 0.0], "answer": [0.0, 1.0, 0.0]}}
    for idx, (name, (_score, ceiling, rung4, cell_n, inserted)) in enumerate(FIXTURE.items()):
        if ceiling is None:  # fox: no fit outputs at all
            continue
        vop = f"char_2479_{name}_op"
        lad = _ladder_json("r4op", STORY_ONPOLICY, vop, ceiling, rung4, source_means=src_means)
        (grad / f"ladder_r4op__{vop}__instruct_context_L19_reduced_s0_nd2.json").write_text(
            json.dumps(lad, indent=1)
        )
        # Equalized-n companion (tag `_rows650`) — one per fit-output character,
        # so ONE common tag covers every survivor (the production coverage gate).
        eqn = _ladder_json(
            "r4op", STORY_ONPOLICY, vop, ceiling, rung4 * 0.95, source_means=src_means
        )
        (grad / f"ladder_r4op__{vop}__instruct_context_L19_reduced_s0_nd2_rows650.json").write_text(
            json.dumps(eqn, indent=1)
        )
        cell = _cell_json(
            vop,
            ceiling,
            0.02,
            cell_n,
            # per-char varying vectors -> non-degenerate closeness reads
            mean_context_vec=[1.0, 0.1 * idx, 0.0],
            mean_answer_vec=[0.1 * (5 - idx), 1.0, 0.0],
        )
        (grad / f"cell_{vop}__instruct_context_L19_reduced_s0.json").write_text(
            json.dumps(cell, indent=1)
        )
        if inserted:
            vins = f"char_2479_{name}"
            ins = _ladder_json("r4", STORY_INSERTED, vins, ceiling, rung4 * 0.9)
            (grad / f"ladder_r4__{vins}__instruct_context_L19_reduced_s0_nd2.json").write_text(
                json.dumps(ins, indent=1)
            )
    return eval_dir


def _strip_cell_vectors(eval_dir: Path, names=("ada",)) -> None:
    """Simulate stale pre-r2 fit outputs: pop the cell mean-activation vectors."""
    grad = eval_dir / "story_char_gradient"
    for name in names:
        p = grad / f"cell_char_2479_{name}_op__instruct_context_L19_reduced_s0.json"
        d = json.loads(p.read_text())
        d.pop("mean_context_vec", None)
        d.pop("mean_answer_vec", None)
        p.write_text(json.dumps(d))


def _strip_ladder_source_means(eval_dir: Path) -> None:
    """Simulate stale pre-r2 op-ladder outputs: pop entry-level source_means."""
    for p in (eval_dir / "story_char_gradient").glob("ladder_r4op__*_nd2.json"):
        d = json.loads(p.read_text())
        d.pop("source_means", None)
        p.write_text(json.dumps(d))


N_PERM = 1000


@pytest.fixture(scope="module")
def verdict_payload(tmp_path_factory) -> dict:
    tmp_path = tmp_path_factory.mktemp("i2479_verdict")
    eval_dir = _write_fixture(tmp_path)
    fig_dir = tmp_path / "figs"
    rc = gv.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--fig-dir",
            str(fig_dir),
            "--n-perm",
            str(N_PERM),
        ]
    )
    assert rc == 0
    out = eval_dir / "gradient_verdict.json"
    assert out.is_file()
    payload = json.loads(out.read_text())
    payload["_fig_dir"] = str(fig_dir)
    return payload


def test_e2e_denominators_and_exclusions(verdict_payload):
    d = verdict_payload["denominators"]
    # all three denominators recorded: planned / fit-output-surviving /
    # fraction-eligible (r1 review: "g1_surviving" mislabeled the middle count —
    # it counts characters with fit OUTPUTS present, not G1 generation survivors)
    assert d == {"planned": 7, "fit_output_surviving": 5, "fraction_eligible": 4}
    exc = verdict_payload["exclusions"]
    assert exc["not_in_axis_freeze"] == ["gus"]
    assert [e["name"] for e in exc["missing_fit_outputs"]] == ["fox"]
    assert [e["name"] for e in exc["ceiling_excluded"]] == ["eel"]
    assert exc["ceiling_excluded"][0]["ceiling_r2"] == pytest.approx(0.01)
    # raw rung-4 R2 kept for the ceiling-excluded character
    assert verdict_payload["per_character"]["eel"]["rung4_r2"] == pytest.approx(0.005)
    assert verdict_payload["per_character"]["eel"]["recovery_fraction"] is None


def test_e2e_headline_rho_and_add_one_p(verdict_payload):
    h = verdict_payload["headline"]
    # recovery fractions 0.9, 0.8, 0.7, 0.6 monotone with scores 90, 80, 70, 60
    assert h["rho"] == pytest.approx(1.0)
    assert h["n"] == 4
    # add-one arithmetic: p == (1 + n_null_ge) / (n_perm + 1), verbatim
    assert h["p_add_one"] == pytest.approx((1 + h["n_null_ge"]) / (N_PERM + 1))
    # at n=4 only the identity ranking reaches rho=1: expected p ~= 1/24 + add-one
    assert 1 / (N_PERM + 1) <= h["p_add_one"] < 0.2
    assert verdict_payload["per_character"]["ada"]["recovery_fraction"] == pytest.approx(0.9)
    jk = h["jackknife"]
    assert jk["status"] in ("ok", "degenerate leave-one-out subset (zero rank variance)")


def test_e2e_verdict_label_bounded_panel(verdict_payload):
    v = verdict_payload["verdict"]
    # gates all pass, n=4 < 12 -> bounded — insufficient panel, regardless of p
    assert v["label"] == gv.LABEL_BOUNDED_PANEL
    assert v["n_fraction_eligible"] == 4
    assert set(verdict_payload["gates"]) == set(gv.GATE_KEYS)
    assert all(verdict_payload["gates"].values())


def test_e2e_secondary_reads(verdict_payload):
    reads = verdict_payload["secondary_reads"]
    # raw rung-4 read runs over ALL 5 survivors (no ceiling exclusion)
    assert reads["raw_rung4_r2"]["n"] == 5
    # acc@1 recovery reads over the 4 eligible characters, both metrics
    assert reads["acc1_recovery_euclidean"]["n"] == 4
    assert reads["acc1_recovery_cosine"]["n"] == 4
    # inserted-mode read over the 2 inserted fixtures
    assert reads["inserted_mode_recovery"]["n"] == 2
    assert reads["kept_n_vs_axis"]["n"] == 5
    assert reads["null_gradient_matched_capacity"]["n"] == 4
    # acc@1 identity+bias retrieval control over the 4 eligible characters —
    # BOTH metric spaces (euclidean + cosine, the standing kNN mandate)
    assert reads["acc1_identity_bias_recovery"]["n"] == 4
    assert reads["acc1_identity_bias_recovery_cosine"]["n"] == 4
    # every computed read carries its own labeled permutation band
    for name, read in reads.items():
        if read.get("status") == "ok":
            assert read["n_perm"] == N_PERM, name
            assert "p_add_one" in read and "label" in read, name
    # answer-length read is REAL when the freeze-side stats sidecars exist
    # (the fixture writes them; length strictly decreasing in axis score)
    assert reads["answer_length_vs_axis"]["n"] == 5
    assert reads["answer_length_vs_axis"]["rho"] == pytest.approx(-1.0)
    # closeness reads REAL: the fixture fit outputs carry the r2-fill vectors
    # (cell entry-level mean vecs + ladder source_means); all 5 fit-output
    # survivors carry the scalar (no ceiling exclusion on closeness)
    assert reads["context_space_closeness_vs_axis"]["n"] == 5
    assert reads["answer_space_closeness_vs_axis"]["n"] == 5
    # equalized-n companions: ONE common tag covers every fit-output survivor
    eq = verdict_payload["equalized_n"]
    assert eq["status"] == "ok"
    blk = eq["companions"]["_rows650"]
    assert blk["characters_covered"] == ["ada", "bee", "cat", "dog", "eel"]
    assert blk["n"] == 4  # eel stays ceiling-excluded inside the companion too
    # production completeness gate: zero registered-read input gaps recorded
    assert verdict_payload["registered_input_gaps"] == []
    assert verdict_payload["exploratory"] is False


def test_e2e_figures_written(verdict_payload):
    figs = verdict_payload["figures"]
    assert "gradient_hero" in figs["written"]
    hero = Path(figs["written"]["gradient_hero"])
    assert hero.is_file() and hero.stat().st_size > 5_000  # non-trivial render
    assert (
        hero.with_suffix("").with_suffix(".meta.json").exists()
        or (hero.parent / "gradient_hero.meta.json").exists()
    )
    assert "ceilings_identity_bias" in figs["written"]
    assert "gradient_hero_inserted" in figs["written"]
    assert "gradient_null_companion" in figs["written"]
    # r2/r3 additions: ladder curves (fixture ladders carry rung_order +
    # per-rung R2), band agreement, kept/drop accounting, BOTH closeness
    # scatters (r2-fill vectors present), and the axis-score violins
    # (axis_draws.json present) all render.
    for stem in (
        "ladder_curves",
        "band_agreement",
        "kept_drop_accounting",
        "closeness_context_vs_axis",
        "closeness_answer_vs_axis",
        "axis_score_violins",
    ):
        assert stem in figs["written"], figs
        assert Path(figs["written"][stem]).is_file()


def test_primary_and_eqn_multidigit_nd_and_rows_tag(tmp_path):
    """g6 r1 Minor regression: `_nd12.json` is a PRIMARY (no equalized-n tag);
    a `_rows<N>` suffix after `_nd<K>` is an equalized-n companion tag."""
    d = tmp_path / "grad"
    d.mkdir()
    pat = "ladder_r4op__v__instruct_context_L19_reduced_s0_nd*.json"
    (d / "ladder_r4op__v__instruct_context_L19_reduced_s0_nd12.json").write_text("{}")
    (d / "ladder_r4op__v__instruct_context_L19_reduced_s0_nd12_rows650.json").write_text("{}")
    primary, eqn = gv._primary_and_eqn(d, pat)
    assert primary is not None and primary.name.endswith("_nd12.json")
    assert list(eqn) == ["_rows650"]


def test_cosine_helper():
    assert gv._cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert gv._cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert gv._cosine([0.0, 0.0], [1.0, 0.0]) is None  # zero-norm -> None
    assert gv._cosine(None, [1.0]) is None
    with pytest.raises(ValueError):
        gv._cosine([1.0, 0.0], [1.0])


def test_closeness_reads_real_when_scalars_present():
    """Closeness reads flip deferred -> real when the per-char scalars exist."""
    per_char = {}
    for i, (name, score) in enumerate([("a", 90.0), ("b", 70.0), ("c", 50.0)]):
        per_char[name] = {
            "name": name,
            "display_name": name,
            "anchor": False,
            "axis_score": score,
            "fraction_eligible": True,
            "ceiling_r2": 0.5,
            "rung4_r2": 0.4,
            "rung1_r2": 0.1,
            "identity_bias_r2": 0.02,
            "null4_r2": 0.01,
            "recovery_fraction": 0.8,
            "cell_n": 800,
            "mean_answer_len": None,
            "context_closeness_to_r4op": 0.9 - 0.1 * i,  # increasing with score
            "answer_closeness_to_r4op": 0.5 + 0.1 * i,  # decreasing with score
        }
    reads = gv.build_secondary_reads(per_char, n_perm=200, seed=0)
    assert reads["context_space_closeness_vs_axis"]["n"] == 3
    assert reads["context_space_closeness_vs_axis"]["rho"] == pytest.approx(1.0)
    assert reads["answer_space_closeness_vs_axis"]["rho"] == pytest.approx(-1.0)
    # sidecars absent -> answer-length still deferred
    assert reads["answer_length_vs_axis"]["status"] == "deferred"


def test_emit_items_stats_sidecar_carries_answer_lengths(tmp_path, monkeypatch):
    """freeze-side half: the emit-items stats sidecar carries the mean kept-answer
    lengths (prepared + axis pools) the verdict's answer-length read consumes."""
    import issue2479_freeze_axis as fz

    kept = tmp_path / "kept_char_2479_zed_op.jsonl"
    kept.write_text('{"conv_id": "k1"}\n{"conv_id": "k2"}\n')
    prepared = [
        {"conv_id": "k1", "question": "q", "answer": "aaaa", "capped": False, "cell": "z"},
        {"conv_id": "k2", "question": "q", "answer": "bb", "capped": False, "cell": "z"},
    ]
    monkeypatch.setattr(fz.prep_mod, "prepare", lambda rows, idx, cell: (prepared, {"n": 2}))
    items_dir = tmp_path / "items"
    fz.emit_items(
        panel=[{"name": "zed", "variant_op": "char_2479_zed_op"}],
        reservation={"k1"},
        kept_glob=str(tmp_path / "kept_{variant}.jsonl"),
        raw_glob=None,
        items_out_dir=items_dir,
    )
    stats = json.loads((items_dir / "axis_items_zed.stats.json").read_text())
    # prepared pool: len("aaaa")=4, len("bb")=2 -> mean 3.0; axis pool = {k1} -> 4.0
    assert stats["mean_answer_len_chars_prepared"] == pytest.approx(3.0)
    assert stats["mean_answer_len_chars_axis"] == pytest.approx(4.0)
    assert stats["n_axis_items"] == 1 and stats["n_prepared"] == 2


def _degrade_stats(eval_dir: Path) -> None:
    for p in (eval_dir / "axis_items").glob("*.stats.json"):
        p.unlink()


def _degrade_draws(eval_dir: Path) -> None:
    (eval_dir / "axis_draws.json").unlink()


def _degrade_all_eqn(eval_dir: Path) -> None:
    for p in (eval_dir / "story_char_gradient").glob("*_rows650.json"):
        p.unlink()


def _degrade_one_eqn(eval_dir: Path) -> None:
    (
        eval_dir
        / "story_char_gradient"
        / "ladder_r4op__char_2479_eel_op__instruct_context_L19_reduced_s0_nd2_rows650.json"
    ).unlink()


@pytest.mark.parametrize(
    "degrade",
    [
        pytest.param(_degrade_stats, id="answer-length-stats-missing"),
        pytest.param(_strip_cell_vectors, id="one-cell-vectors-stripped"),
        pytest.param(_degrade_draws, id="axis-draws-missing"),
        pytest.param(_degrade_all_eqn, id="equalized-companions-missing"),
        pytest.param(_degrade_one_eqn, id="equalized-coverage-gap"),
    ],
)
def test_e2e_production_refuses_on_registered_gap(tmp_path, degrade):
    """Production mode (no --exploratory) exits 3 and writes NO verdict when any
    plan-SS6 registered read's inputs are absent/stale (r2 codex
    `registered-analysis-incomplete`): answer-length stats sidecars, closeness
    vectors, the per-draw axis sidecar, or a complete equalized-n companion set
    covering every surviving character under one common tag."""
    eval_dir = _write_fixture(tmp_path)
    degrade(eval_dir)
    out = tmp_path / "verdict_refused.json"
    rc = gv.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--n-perm",
            "100",
            "--no-figures",
        ]
    )
    assert rc == 3
    assert not out.exists()  # refusal writes nothing


def test_e2e_exploratory_records_gaps_and_writes(tmp_path):
    """--exploratory permits gaps: rc 0, verdict written, gaps recorded."""
    eval_dir = _write_fixture(tmp_path)
    _degrade_draws(eval_dir)
    out = tmp_path / "verdict_exploratory.json"
    rc = gv.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--n-perm",
            "100",
            "--no-figures",
            "--exploratory",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["exploratory"] is True
    assert any("axis draws" in g for g in payload["registered_input_gaps"])


def test_e2e_stale_fit_outputs_status(tmp_path):
    """Vectors absent from EVERY fit output -> the closeness reads carry the
    distinct `stale-fit-outputs` status (not a bland `deferred`): the r2 fill
    emits the vectors unconditionally, so absence means pre-r2 artifacts
    needing a P5 re-run (g4 MINOR-1)."""
    eval_dir = _write_fixture(tmp_path)
    _strip_cell_vectors(eval_dir, names=("ada", "bee", "cat", "dog", "eel"))
    _strip_ladder_source_means(eval_dir)
    out = tmp_path / "verdict_stale.json"
    rc = gv.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--n-perm",
            "100",
            "--no-figures",
            "--exploratory",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    for key in ("context_space_closeness_vs_axis", "answer_space_closeness_vs_axis"):
        read = payload["secondary_reads"][key]
        assert read["status"] == "stale-fit-outputs"
        assert "re-run P5" in read["note"]
    assert any("stale pre-r2" in g for g in payload["registered_input_gaps"])


def test_emit_items_stats_out_dir_separation(tmp_path, monkeypatch):
    """emit-items routes the SMALL stats sidecars to a separate (commit-eligible
    eval_results) dir while the row jsonls stay in the items dir (g4 MAJOR-2)."""
    import issue2479_freeze_axis as fz

    kept = tmp_path / "kept_char_2479_zed_op.jsonl"
    kept.write_text('{"conv_id": "k1"}\n')
    prepared = [
        {"conv_id": "k1", "question": "q", "answer": "aaaa", "capped": False, "cell": "z"},
    ]
    monkeypatch.setattr(fz.prep_mod, "prepare", lambda rows, idx, cell: (prepared, {"n": 1}))
    items_dir = tmp_path / "items"
    stats_dir = tmp_path / "stats"
    fz.emit_items(
        panel=[{"name": "zed", "variant_op": "char_2479_zed_op"}],
        reservation={"k1"},
        kept_glob=str(tmp_path / "kept_{variant}.jsonl"),
        raw_glob=None,
        items_out_dir=items_dir,
        stats_out_dir=stats_dir,
    )
    assert (items_dir / "axis_items_zed.jsonl").is_file()
    assert (stats_dir / "axis_items_zed.stats.json").is_file()
    assert not (items_dir / "axis_items_zed.stats.json").exists()


def test_e2e_instrument_suspect_path(tmp_path):
    """A failed instrument gate demotes the label; data still fully reported."""
    eval_dir = _write_fixture(tmp_path)
    gates_path = eval_dir / "instrument_gates.json"
    inst = json.loads(gates_path.read_text())
    inst["gates"]["verbatim_flatness_pass"] = False
    gates_path.write_text(json.dumps(inst))
    out = tmp_path / "verdict_suspect.json"
    rc = gv.main(
        [
            "--eval-dir",
            str(eval_dir),
            "--out",
            str(out),
            "--n-perm",
            "200",
            "--no-figures",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["verdict"]["label"] == gv.LABEL_INSTRUMENT_SUSPECT
    # realized rho + band still computed and reported (label demotes, never suppresses)
    assert payload["headline"]["rho"] == pytest.approx(1.0)
    assert payload["headline"]["p_add_one"] is not None


def test_rankdata_ties_are_averaged():
    r = gv._rankdata(np.array([1.0, 2.0, 2.0, 3.0]))
    assert np.allclose(r, [1.0, 2.5, 2.5, 4.0])


def test_spearman_perm_read_degenerate_constant_vector():
    read = gv.spearman_perm_read(
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
        n_perm=50,
        seed=0,
        label="degenerate",
    )
    assert read["rho"] is None and read["p_add_one"] is None
    assert "zero rank variance" in read["status"]
