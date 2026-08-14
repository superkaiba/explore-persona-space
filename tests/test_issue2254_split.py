"""CPU-only synthetic tests for the issue #2254 ctxext-subspace-split amendment
(plan v7): the split construction algebra + HALT asserts, the restricted-grid
enumeration (--grid ctxext-split), the POOLED restricted null band, and the
REAL split reduce bodies on a synthetic judged tree (upload seam faked with a
signature-mirroring def). Covers plan v7 blind-spot (e): the null re-argmax
reduce against the persisted packs is test-covered, not smoke-covered.

NO test reads `eval_results/issue_<M>/` fixtures (sparse-cones rule) and no
test touches the network / GPU — synthetic tensors + tmp_path trees only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.issue2254_figures as figs
import scripts.issue2254_preimage as pi

RNG = np.random.default_rng(2254)


# ---------------------------------------------------------------------------
# registration: short tokens + collision check (the brief's explicit ask)
# ---------------------------------------------------------------------------


def test_dir_short_tokens_registered_and_collision_free():
    assert pi._DIR_SHORT["par"] == "par" and pi._DIR_SHORT["perp"] == "prp"
    # no token collision across the FULL registry (pre/rb/cxd/rnd/shf/par/prp)
    assert len(set(pi._DIR_SHORT.values())) == len(pi._DIR_SHORT)
    assert set(pi.SPLIT_DIRECTIONS) <= set(pi._DIR_SHORT)


def test_split_cell_ids_use_new_tokens_and_do_not_collide_with_parent():
    args = pi.build_argparser().parse_args(["--grid", "ctxext-split"])
    split_ids = {pi._cell_id(c) for c in pi._localize_cells(args, ["evil", "sycophancy"])}
    full_args = pi.build_argparser().parse_args([])
    parent_ids = {pi._cell_id(c) for c in pi._localize_cells(full_args, list(pi.BEHAVIORS))}
    assert split_ids and not (split_ids & parent_ids)
    assert all(("__par__" in cid) or ("__prp__" in cid) for cid in split_ids)


# ---------------------------------------------------------------------------
# construction algebra (plan v7 §4.1) + HALT asserts
# ---------------------------------------------------------------------------


def _tiny_inputs(d=12, kstar=5, seed=3):
    rng = np.random.default_rng(seed)
    w_map = rng.standard_normal((d, d))
    xsd = np.abs(rng.standard_normal(d)) + 0.5
    d_ctx = rng.standard_normal(d)
    return w_map, kstar, xsd, d_ctx


def test_split_components_orthogonal_pythagoras_and_parity_identity():
    w_map, kstar, xsd, d_ctx = _tiny_inputs()
    comp = pi.split_components(w_map, kstar, xsd, d_ctx)
    # exact standardized-frame orthogonality + Pythagoras
    assert abs(float(comp["w_par"] @ comp["w_perp"])) < 1e-12
    assert abs(comp["w_par_norm"] ** 2 + comp["w_perp_norm"] ** 2 - 1.0) < 1e-12
    # parity identity: ||w_par|| == ||Vmt[:k] @ w_hat|| (the round-2 statistic)
    _m, _um, _sm, vmt = pi.map_svd(w_map)
    w = d_ctx / xsd
    w_hat = w / np.linalg.norm(w)
    reach = float(np.linalg.norm(vmt[:kstar] @ w_hat))
    assert abs(comp["w_par_norm"] - reach) < 1e-12
    # folded directions are unit-norm and reconstruct w_hat
    assert np.isclose(np.linalg.norm(comp["d_par"]), 1.0)
    assert np.isclose(np.linalg.norm(comp["d_perp"]), 1.0)
    assert np.allclose(comp["w_par"] + comp["w_perp"], w_hat)


def test_split_components_fold_matches_registered_convention():
    w_map, kstar, xsd, d_ctx = _tiny_inputs(seed=7)
    comp = pi.split_components(w_map, kstar, xsd, d_ctx)
    # fold = normalize(xsd * w) — the v5 §11 d_pre convention, reused verbatim
    assert np.allclose(comp["d_par"], pi.destandardized_direction(xsd, comp["w_par"]))
    assert np.allclose(comp["d_perp"], pi.destandardized_direction(xsd, comp["w_perp"]))


def test_assert_split_halt_nondegenerate_floor_trips_on_in_subspace_direction():
    """d_ctx built (almost) INSIDE the retained subspace => w_perp is a
    renormalized noise sliver => HALT (iii) fires (the plan §8 degeneracy)."""
    w_map, kstar, xsd, _ = _tiny_inputs(seed=11)
    _m, _um, _sm, vmt = pi.map_svd(w_map)
    w_inside = vmt[:kstar].T @ np.arange(1.0, kstar + 1.0)
    d_ctx = xsd * w_inside  # un-standardizes to land exactly inside after /xsd
    comp = pi.split_components(w_map, kstar, xsd, d_ctx)
    assert comp["w_perp_norm"] < 1e-10  # float residual only
    with pytest.raises(RuntimeError, match=r"split-halt-degenerate"):
        pi.assert_split_halt("evil", 20, comp, kstar=kstar)


def test_assert_split_halt_floor_and_parity_paths(monkeypatch):
    w_map, kstar, xsd, d_ctx = _tiny_inputs(seed=13)
    comp = pi.split_components(w_map, kstar, xsd, d_ctx)
    # (iii) floor: a SELF-CONSISTENT construction with a small (but nonzero)
    # complement — orthogonality + Pythagoras hold, only the floor trips
    _m, _um, _sm, vmt = pi.map_svd(w_map)
    rng = np.random.default_rng(5)
    u_in = vmt[:kstar].T @ rng.standard_normal(kstar)
    u_in /= np.linalg.norm(u_in)
    z = rng.standard_normal(len(xsd))
    u_out = z - vmt[:kstar].T @ (vmt[:kstar] @ z)
    u_out /= np.linalg.norm(u_out)
    eps = 0.01  # below SPLIT_NONDEGEN_FLOOR = 0.05
    w_target = np.sqrt(1 - eps**2) * u_in + eps * u_out
    comp_degen = pi.split_components(w_map, kstar, xsd, xsd * w_target)
    assert comp_degen["w_perp_norm"] == pytest.approx(eps, abs=1e-9)
    with pytest.raises(RuntimeError, match=r"split-halt-degenerate"):
        pi.assert_split_halt("evil", 20, comp_degen, kstar=kstar)
    # (i) parity binds ONLY at the operating layer; wrong kstar raises there
    monkeypatch.setitem(
        pi.SPLIT_PARITY,
        "evil",
        {"layer": 14, "kstar": kstar + 1, "reachability_ctxext": comp["w_par_norm"]},
    )
    pi.assert_split_halt("evil", 20, comp, kstar=kstar)  # non-operating layer: no parity
    with pytest.raises(RuntimeError, match=r"split-halt-parity.*kstar"):
        pi.assert_split_halt("evil", 14, comp, kstar=kstar)
    # (i) parity value mismatch raises at the operating layer
    monkeypatch.setitem(
        pi.SPLIT_PARITY,
        "evil",
        {"layer": 14, "kstar": kstar, "reachability_ctxext": comp["w_par_norm"] + 1e-6},
    )
    with pytest.raises(RuntimeError, match=r"split-halt-parity"):
        pi.assert_split_halt("evil", 14, comp, kstar=kstar)
    # exact value passes all four
    monkeypatch.setitem(
        pi.SPLIT_PARITY,
        "evil",
        {"layer": 14, "kstar": kstar, "reachability_ctxext": comp["w_par_norm"]},
    )
    pi.assert_split_halt("evil", 14, comp, kstar=kstar)


def test_assert_split_halt_orthogonality_has_teeth():
    w_map, kstar, xsd, d_ctx = _tiny_inputs(seed=17)
    comp = dict(pi.split_components(w_map, kstar, xsd, d_ctx))
    comp["w_perp"] = comp["w_perp"] + 0.1 * comp["w_par"]  # break orthogonality
    with pytest.raises(RuntimeError, match=r"split-halt-ortho"):
        pi.assert_split_halt("evil", 20, comp, kstar=kstar)


# ---------------------------------------------------------------------------
# restricted grid enumeration (--grid ctxext-split)
# ---------------------------------------------------------------------------


def test_split_localize_cells_32_per_behavior_no_alpha0_own_configs():
    args = pi.build_argparser().parse_args(["--grid", "ctxext-split"])
    cells = pi._localize_cells(args, ["evil", "sycophancy"])
    assert len(cells) == 64
    assert all(c["kind"] == "steer" for c in cells)  # NO fresh alpha0
    assert all(c["position"] == "context" for c in cells)
    evil_configs = {c["layer_config"] for c in cells if c["behavior"] == "evil"}
    syco_configs = {c["layer_config"] for c in cells if c["behavior"] == "sycophancy"}
    assert evil_configs == {"L14", "mid"} and syco_configs == {"L17", "mid"}
    assert {c["direction"] for c in cells} == set(pi.SPLIT_DIRECTIONS)
    assert sorted({c["c"] for c in cells}) == sorted(pi.DOSES)


def test_split_smoke_narrows_to_two_perp_cells():
    args = pi.build_argparser().parse_args(["--grid", "ctxext-split", "--smoke"])
    assert pi._grid_combos(args) == [("perp", "context")]
    assert pi._grid_doses(args) == (1.0,)
    cells = pi._localize_cells(args, ["evil"])
    assert len(cells) == 2  # plan v7 smoke (ii) L14 single + (iii) mid band, c=+1
    assert {c["layer_config"] for c in cells} == {"L14", "mid"}


def test_full_grid_unchanged_by_amendment():
    args = pi.build_argparser().parse_args([])
    cells = pi._localize_cells(args, list(pi.BEHAVIORS))
    assert len(cells) == 1155  # 385/behavior incl. alpha0 — parent invariant intact
    assert pi._grid_layer_configs(args) == tuple(pi.LAYER_CONFIGS)
    assert args.grid == "full"


def test_split_behaviors_intersect_and_raise():
    args = pi.build_argparser().parse_args(["--behaviors", "hallucination"])
    with pytest.raises(RuntimeError, match="evil/sycophancy only"):
        pi._split_behaviors(args)
    args = pi.build_argparser().parse_args([])  # default: all three
    assert pi._split_behaviors(args) == ["evil", "sycophancy"]


def test_require_split_grid_fails_loud_without_flag():
    args = pi.build_argparser().parse_args([])
    with pytest.raises(SystemExit, match="ctxext-split"):
        pi._require_split_grid(args)


def test_split_restricted_null_ids_32_per_behavior():
    args = pi.build_argparser().parse_args(["--grid", "ctxext-split"])
    for b, own in (("evil", "L14"), ("sycophancy", "L17")):
        ids = pi._split_restricted_null_ids(args, b)
        assert len(ids) == 32 and len(set(ids)) == 32
        assert all(("__rnd__ctx__" in i) or ("__shf__ctx__" in i) for i in ids)
        assert all((f"__{own}__" in i) or ("__mid__" in i) for i in ids)


# ---------------------------------------------------------------------------
# synthetic judged trees -> REAL reduce bodies (upload seam faked)
# ---------------------------------------------------------------------------


def _write_judged(root: Path, tree: str, cell: dict, q_scores, *, n_draws: int, coherent=True):
    """One synthetic judged pack with every field the reduces read."""
    cid = pi._cell_id(cell)
    out = root / "judge" / tree / "judged"
    out.mkdir(parents=True, exist_ok=True)
    qs = [float(x) for x in q_scores]
    payload = {
        "cell_id": cid,
        "cell": cell,
        "per_question_mean_score": qs,
        "per_question_rate": [float(s >= 50.0) for s in qs],
        "mean_score": float(np.mean(qs)),
        "rate": float(np.mean([s >= 50.0 for s in qs])),
        "coherence_rate": 1.0 if coherent else 0.0,
        "coherence_pass": bool(coherent),
        "cap_hit_fraction": 0.0,
        "judge": {"n_draws": n_draws},
        "accounting": {"frac_items_complete": 1.0},
    }
    (out / f"{cid}.json").write_text(json.dumps(payload))
    return cid


def _steer_cell(b, d, lc, c):
    return {
        "behavior": b,
        "kind": "steer",
        "direction": d,
        "position": "context",
        "layer_config": lc,
        "c": float(c),
    }


def _build_parent_localize_tree(root: Path, args, behaviors=("evil", "sycophancy")):
    """Parent alpha0 + all 32 restricted null cells per behavior (10q x 3dr);
    one null cell per behavior is coherence-FAILED to exercise the gate."""
    rng = np.random.default_rng(9)
    for b in behaviors:
        _write_judged(
            root, "localize", {"behavior": b, "kind": "alpha0"}, 10.0 + rng.random(10), n_draws=3
        )
        ids = pi._split_restricted_null_ids(args, b)
        for i, cid in enumerate(ids):
            # reconstruct the cell dict from its id tokens
            _b, nd, _ctx, lc, ctok = cid.split("__")
            d = {"rnd": "random", "shf": "preshuf"}[nd]
            c = float(ctok.removeprefix("c").replace("m", "-", 1).replace("p", "."))
            cell = _steer_cell(b, d, lc, c)
            assert pi._cell_id(cell) == cid
            # nulls ABOVE alpha0 (delta ~ +2) so the pooled band p975 ~ +2.3 is a
            # real positive bar the weak par arm (~+0.5) cannot clear
            _write_judged(
                root, "localize", cell, 12.0 + rng.random(10), n_draws=3, coherent=(i != 0)
            )


def _split_args(tmp_path, extra=()):
    return pi.build_argparser().parse_args(
        [
            "--grid",
            "ctxext-split",
            "--behaviors",
            "evil",
            "sycophancy",
            "--out-root",
            str(tmp_path),
            *extra,
        ]
    )


@pytest.fixture()
def fake_upload(monkeypatch):
    uploads: list = []

    def _fake_upload_folder_to_hf(local_dir, path_in_repo, allow=None):
        uploads.append((Path(local_dir), path_in_repo, allow))

    monkeypatch.setattr(pi, "_upload_folder_to_hf", _fake_upload_folder_to_hf)
    return uploads


def test_reduce_split_wave1_pooled_band_and_operating_points(tmp_path, fake_upload):
    args = _split_args(tmp_path)
    _build_parent_localize_tree(tmp_path, args)
    rng = np.random.default_rng(21)
    # split cells: perp strong at mid c=2 (the intended argmax), par weak
    for b in ("evil", "sycophancy"):
        own = pi.SPLIT_OWN_CONFIG[b]
        for d in pi.SPLIT_DIRECTIONS:
            for lc in (own, "mid"):
                for c in (1.0, 2.0):
                    base = 80.0 if (d == "perp" and lc == "mid" and c == 2.0) else 12.0
                    _write_judged(
                        tmp_path,
                        "ctxext_split_localize",
                        _steer_cell(b, d, lc, c),
                        base + rng.random(10),
                        n_draws=3,
                    )
    pi._reduce_split_wave1(args, tmp_path)
    dose = json.loads((tmp_path / "ctxext_split" / "localize" / "dose_response.json").read_text())
    ops = json.loads((tmp_path / "ctxext_split" / "localize" / "operating_points.json").read_text())
    for b in ("evil", "sycophancy"):
        band = dose["behaviors"][b]["null_band_context_restricted"]
        # POOLED band: one per behavior over both breadths x both null dirs;
        # 31 of 32 pass coherence (one dropped by construction)
        assert band["n_cells"] == 31
        assert dose["behaviors"][b]["null_cells_expected"] == 32
        assert dose["behaviors"][b]["null_cells_coherence_gated"] == 31
        assert len(dose["behaviors"][b]["null_cells_coherence_dropped"]) == 1
        # NO per-breadth band keys anywhere (the anti-conservative shape)
        assert "null_band_context_single" not in dose["behaviors"][b]
        assert "null_band_context_mid" not in dose["behaviors"][b]
        # operating points: perp/mid argmax lands on the planted c=2 cell
        best = ops["behaviors"][b]["perp__context__mid"]
        assert best["layer_config"] == "mid" and best["c"] == 2.0
        assert ops["behaviors"][b]["par__context__single"] is not None
    # pod-D staging inputs uploaded
    assert any("ctxext_split/localize" in u[1] for u in fake_upload)


def test_reduce_split_wave1_missing_null_cell_fails_loud(tmp_path, fake_upload):
    args = _split_args(tmp_path)
    _build_parent_localize_tree(tmp_path, args)
    # remove one restricted null pack -> the verification must raise
    victim = pi._split_restricted_null_ids(args, "evil")[5]
    (tmp_path / "judge" / "localize" / "judged" / f"{victim}.json").unlink()
    _write_judged(
        tmp_path,
        "ctxext_split_localize",
        _steer_cell("evil", "perp", "L14", 1.0),
        np.full(10, 30.0),
        n_draws=3,
    )
    with pytest.raises(FileNotFoundError, match="parent judged pack"):
        pi._reduce_split_wave1(args, tmp_path)


def test_reduce_split_wave1_grain_mismatch_fails_loud(tmp_path, fake_upload):
    args = _split_args(tmp_path, extra=("--q-localize", "4"))
    with pytest.raises(RuntimeError, match="PRODUCTION grain"):
        _build_parent_localize_tree(tmp_path, args)  # packs are 10q; args demand 4q
        pi._reduce_split_wave1(args, tmp_path)


def _build_wave2_tree(tmp_path, args, *, perp_high=85.0, par_high=10.5):
    """Parent decisive (alpha0 + ctxext op cell) + verdicts.json + split
    decisive cells + the localize trees wave 2 re-reads for band/sel-inh."""
    rng = np.random.default_rng(33)
    _build_parent_localize_tree(tmp_path, args)
    ctx_ids = {}
    for b in ("evil", "sycophancy"):
        _write_judged(
            tmp_path,
            "decisive",
            {"behavior": b, "kind": "alpha0"},
            10.0 + rng.random(20),
            n_draws=5,
        )
        ctx_cell = _steer_cell(b, "ctxext", pi.SPLIT_OWN_CONFIG[b], 4.0)
        ctx_ids[b] = _write_judged(tmp_path, "decisive", ctx_cell, 70.0 + rng.random(20), n_draws=5)
        own = pi.SPLIT_OWN_CONFIG[b]
        for d, base in (("par", par_high), ("perp", perp_high)):
            for lc in (own, "mid"):
                _write_judged(
                    tmp_path,
                    "ctxext_split_decisive",
                    _steer_cell(b, d, lc, 2.0),
                    base + rng.random(20),
                    n_draws=5,
                )
        # split localize cells for the selection-inherited read
        for d in pi.SPLIT_DIRECTIONS:
            _write_judged(
                tmp_path,
                "ctxext_split_localize",
                _steer_cell(b, d, own, 1.0),
                20.0 + rng.random(10),
                n_draws=3,
            )
    verdicts = {
        "behaviors": {
            b: {"margins": {"E_ctxdir": {"cell_id": ctx_ids[b]}}} for b in ("evil", "sycophancy")
        }
    }
    (tmp_path / "decisive").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisive" / "verdicts.json").write_text(json.dumps(verdicts))


def test_reduce_split_wave2_complement_carries_lattice_and_band(tmp_path, fake_upload):
    args = _split_args(tmp_path)
    _build_wave2_tree(tmp_path, args)
    pi._reduce_split_wave2(args, tmp_path)
    verdicts = json.loads((tmp_path / "ctxext_split" / "decisive" / "verdicts.json").read_text())
    for b in ("evil", "sycophancy"):
        vb = verdicts["behaviors"][b]
        assert vb["label"] == "Complement-carries", (b, vb["label"], vb["margins"])
        band = vb["null_band_context_restricted"]
        assert band["n_draws"] == pi.N_BOOT_VERDICT and band["n_cells"] == 31
        assert vb["margins"]["E_perp"]["value"] > 0
        assert vb["margins"]["E_par"]["ci"][0] <= 0  # par does not exclude 0 positively
        assert vb["margins"]["G_perp_minus_par"]["value"] > 0
        assert vb["margins"]["gap_vs_parent_ctxext"]["perp"]["vs_cell_id"].startswith(b)
        assert "par__context" in vb["selection_inherited"]
        assert "perp__context" in vb["selection_inherited"]
    percell = json.loads(
        (tmp_path / "ctxext_split" / "decisive" / "delta_score_percell.json").read_text()
    )
    assert len(percell["behaviors"]["evil"]) == 4  # 2 dirs x 2 breadth cells


def test_reduce_split_wave2_both_carry(tmp_path, fake_upload):
    args = _split_args(tmp_path)
    _build_wave2_tree(tmp_path, args, par_high=85.0, perp_high=85.0)
    pi._reduce_split_wave2(args, tmp_path)
    verdicts = json.loads((tmp_path / "ctxext_split" / "decisive" / "verdicts.json").read_text())
    assert verdicts["behaviors"]["evil"]["label"] == "Both-carry"


def test_split_lattice_label_cells_disjoint_exhaustive():
    pos = {"ci": [1.0, 3.0]}
    neg = {"ci": [-1.0, 1.0]}
    assert pi._split_lattice_label({"E_par": neg, "E_perp": pos})[0] == "Complement-carries"
    assert pi._split_lattice_label({"E_par": pos, "E_perp": neg})[0] == "Retained-carries"
    assert pi._split_lattice_label({"E_par": pos, "E_perp": pos})[0] == "Both-carry"
    assert pi._split_lattice_label({"E_par": neg, "E_perp": neg})[0] == "Neither"
    # absence = not-positive, never a crash; both absent = Undefined
    assert pi._split_lattice_label({"E_perp": pos})[0] == "Complement-carries"
    assert pi._split_lattice_label({})[0] == "Undefined"


# ---------------------------------------------------------------------------
# wave routing + judge-pilot fingerprint inheritance
# ---------------------------------------------------------------------------


def test_wave_src_and_judge_draws_cover_split_phases():
    assert pi._WAVE_SRC["localize_split"] == ("ctxext_split_localize",)
    assert pi._WAVE_SRC["decisive_split"] == ("ctxext_split_decisive",)
    assert pi.JUDGE_DRAWS["ctxext_split_localize"] == 3
    assert pi.JUDGE_DRAWS["ctxext_split_decisive"] == 5


def test_expected_gen_cell_ids_localize_split(tmp_path):
    args = _split_args(tmp_path)
    expected, pilot = pi._expected_gen_cell_ids(args, tmp_path, "localize_split")
    assert pilot == ["evil", "sycophancy"]
    assert set(expected) == {"ctxext_split_localize"}
    assert len(expected["ctxext_split_localize"]) == 64


def test_judge_pilot_inherits_parent_fingerprint(tmp_path, monkeypatch):
    """Plan §6 fingerprint skip: an identical-instrument parent PASS sidecar
    short-circuits the split wave's pilot (no gate call)."""
    from unittest.mock import create_autospec

    import explore_persona_space.eval.judge_pilot as jp

    fake_gate = create_autospec(jp.judge_pilot_gate)
    monkeypatch.setattr(jp, "judge_pilot_gate", fake_gate)
    behavior, rubric, n_draws = "evil", "RUBRIC", 3
    fp = pi._sha8(
        {
            "behavior": behavior,
            "rubric": rubric,
            "n_draws": n_draws,
            "mt": pi.JUDGE_MAX_TOKENS_2254,
        }
    )
    parent_dir = tmp_path / "judge" / "pilot" / "localize"
    parent_dir.mkdir(parents=True)
    (parent_dir / f"{behavior}.pass.json").write_text(
        json.dumps({"fingerprint": fp, "verdict": "PASS"})
    )
    args = _split_args(tmp_path)
    pi._run_judge_pilot(
        args,
        tmp_path,
        "ctxext_split_localize",
        behavior,
        rubric,
        n_draws,
        fallback_phases=("localize",),
    )
    fake_gate.assert_not_called()
    inherited = json.loads(
        (
            tmp_path / "judge" / "pilot" / "ctxext_split_localize" / f"{behavior}.pass.json"
        ).read_text()
    )
    assert inherited["inherited_from"] == "localize" and inherited["fingerprint"] == fp
    # a DIFFERENT fingerprint must NOT inherit: the gate would run (and here
    # crash loudly on the empty gen tree — proving the skip did not fire)
    with pytest.raises(RuntimeError, match="no evil gen cells"):
        pi._run_judge_pilot(
            args,
            tmp_path,
            "ctxext_split_decisive",
            behavior,
            rubric,
            5,  # different n_draws -> different fingerprint than the localize PASS
            fallback_phases=("localize",),
        )


# ---------------------------------------------------------------------------
# figures: split builders render on the synthetic tree (yerr clamp class)
# ---------------------------------------------------------------------------


def test_split_figures_render_from_synthetic_reduce_outputs(tmp_path, fake_upload):
    args = _split_args(tmp_path)
    _build_wave2_tree(tmp_path, args)
    rng = np.random.default_rng(41)
    for b in ("evil", "sycophancy"):
        own = pi.SPLIT_OWN_CONFIG[b]
        for d in pi.SPLIT_DIRECTIONS:
            for lc in (own, "mid"):
                for c in (1.0, 2.0):
                    _write_judged(
                        tmp_path,
                        "ctxext_split_localize",
                        _steer_cell(b, d, lc, c),
                        15.0 + rng.random(10),
                        n_draws=3,
                    )
    pi._reduce_split_wave1(args, tmp_path)
    pi._reduce_split_wave2(args, tmp_path)
    # parent percell for the comparator bars (minimal synthetic shape)
    ctx_id = json.loads((tmp_path / "decisive" / "verdicts.json").read_text())["behaviors"]["evil"][
        "margins"
    ]["E_ctxdir"]["cell_id"]
    parent_percell = {
        "behaviors": {
            "evil": {
                ctx_id: {
                    "cell": {"direction": "ctxext", "position": "context"},
                    "delta_score": 60.0,
                    "ci_frozen": [50.0, 70.0],
                }
            }
        }
    }
    (tmp_path / "decisive" / "delta_score_percell.json").write_text(json.dumps(parent_percell))
    fig_dir = tmp_path / "figs"
    assert figs.fig_ctxext_split_hero(tmp_path, fig_dir) == "ctxext_split_hero"
    assert (fig_dir / "ctxext_split_hero.png").is_file()
    assert figs.fig_ctxext_split_dose(tmp_path, fig_dir) == "ctxext_split_dose_response"
    assert (fig_dir / "ctxext_split_dose_response.png").is_file()


def test_split_figures_skip_with_reason_when_inputs_absent(tmp_path):
    assert figs.fig_ctxext_split_hero(tmp_path, tmp_path / "f").startswith("skip:")
    assert figs.fig_ctxext_split_dose(tmp_path, tmp_path / "f").startswith("skip:")
