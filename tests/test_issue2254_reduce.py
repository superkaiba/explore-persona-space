"""CPU-only synthetic tests for the issue #2254 judge/reduce + figures layer.

Runs the REAL `_reduce_wave1` / `_reduce_wave2` / `_patch_vs_ceiling` /
`render_all` bodies against a synthetic judged tree in tmp (the HF upload
boundary faked with a signature-matching recorder — the one external seam).
No network, no GPU, no eval_results fixtures (sparse-cones rule).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import scripts.issue2254_preimage as pi

NQ = 4


def _full_args(extra: list[str] | None = None):
    return pi.build_argparser().parse_args(["--phase", "judge_reduce", *(extra or [])])


# ---------------------------------------------------------------------------
# grid enumeration + id scheme
# ---------------------------------------------------------------------------


def test_localize_grid_enumeration_matches_plan():
    args = _full_args()
    cells = pi._localize_cells(args, list(pi.BEHAVIORS))
    assert len(cells) == 385 * len(pi.BEHAVIORS) == 1155
    per_b = [c for c in cells if c["behavior"] == "evil"]
    assert sum(c["kind"] == "alpha0" for c in per_b) == 1
    assert sum(c["kind"] == "steer" for c in per_b) == 8 * 6 * 8  # combos x configs x doses
    ids = [pi._cell_id(c) for c in cells]
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    ok_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    assert all(set(i) <= ok_chars for i in ids)


def test_patch_grid_is_12_cells_per_behavior():
    assert len(pi.PATCH_DIRECTIONS) * len(pi.PATCH_OPS) * len(pi.PATCH_BREADTHS) == 12


def test_c_token_shapes():
    assert pi._c_token(-0.5) == "cm0p5"
    assert pi._c_token(2.0) == "c2"
    assert pi._c_token(4.0) == "c4"
    assert pi._c_token(-4.0) == "cm4"


def test_judge_ctx_ids_fit_custom_id_budget_and_roundtrip():
    from explore_persona_space.experiments.issue_1739.judging import rollout_item_id

    args = _full_args()
    cells = pi._localize_cells(args, list(pi.BEHAVIORS))
    # worst case: hallucination (longest behavior) x preshuf x context x mid x cm0p5
    for cell in cells:
        cid = pi._judge_ctx_id(cell, seed=43, i=999)
        assert len(cid) <= 49 and "__" not in cid
        item = rollout_item_id(cid, 4)  # raises on any custom_id violation
        assert len(item) <= 53


# ---------------------------------------------------------------------------
# bootstrap machinery
# ---------------------------------------------------------------------------


def test_boot_idx_deterministic_and_key_sensitive():
    a = pi._boot_idx(7, 50, "evil__a0")
    b = pi._boot_idx(7, 50, "evil__a0")
    c = pi._boot_idx(7, 50, "evil__cl")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (50, 7) and a.min() >= 0 and a.max() < 7


def test_boot_diff_ci_paired_and_nan_tolerant():
    cell = np.array([12.0, 11.0, np.nan, 13.0])
    ref = np.array([2.0, 1.0, 3.0, np.nan])
    idx = pi._boot_idx(NQ, 400, "k")
    point, lo, hi = pi._boot_diff_ci(cell, ref, idx)
    assert lo <= point <= hi
    assert point == pytest.approx(np.nanmean(cell) - np.nanmean(ref))
    assert lo > 0  # a +10-ish shift never straddles zero at this spread


def test_null_band_none_on_empty_and_ordered_quantiles():
    a0 = np.array([10.0, 11.0, 9.0, 10.0])
    assert pi._null_band([], a0, "x") is None
    nulls = [a0 + np.array([0.5, -0.5, 0.2, -0.2]), a0 + np.array([-0.1, 0.4, -0.3, 0.1])]
    band = pi._null_band(nulls, a0, "evil__nullctx", n_draws=300)
    assert band["n_cells"] == 2 and band["n_draws"] == 300
    assert band["p975"] >= band["p50"]


# ---------------------------------------------------------------------------
# verdict lattice
# ---------------------------------------------------------------------------


def _m(lo, hi):
    return {"value": (lo + hi) / 2, "ci": [lo, hi]}


def test_lattice_label_all_five_cases():
    assert pi._lattice_label({})[0] == "Undefined"
    h1 = {"E_pre": _m(1, 3), "E_ctxdir": _m(1, 3), "C_gap": _m(-1, 1)}
    assert pi._lattice_label(h1)[0] == "H1"
    h3 = {"E_pre": _m(1, 3), "E_ctxdir": _m(2, 5), "C_gap": _m(-4, -1)}
    assert pi._lattice_label(h3)[0] == "H3"
    h2 = {"E_pre": _m(-2, 1), "E_ctxdir": _m(1, 3), "C_gap": _m(-3, -1)}
    assert pi._lattice_label(h2)[0] == "H2"
    amb = {"E_pre": _m(-2, 1), "E_ctxdir": _m(-1, 2), "C_gap": _m(-1, 1)}
    assert pi._lattice_label(amb)[0] == "Ambiguous"


# ---------------------------------------------------------------------------
# synthetic judged tree -> REAL wave-1 + wave-2 reduce bodies
# ---------------------------------------------------------------------------


def _judged(root: Path, phase: str, cell: dict, pq: list[float], rate: float = 0.2, **over):
    cid = pi._cell_id(cell)
    rec = {
        "cell_id": cid,
        "cell": cell,
        "phase": phase,
        "per_question_mean_score": pq,
        "per_question_rate": [rate] * len(pq),
        "per_question_n": [3] * len(pq),
        "mean_score": float(np.nanmean([v for v in pq if v is not None])),
        "rate": rate,
        "coherence_rate": 1.0,
        "coherence_pass": True,
        "cap_hit_fraction": 0.0,
        "accounting": {"frac_items_complete": 1.0, "n_items": len(pq), "n_items_zero_valid": 0},
    }
    rec.update(over)
    d = root / "judge" / phase / "judged"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cid}.json").write_text(json.dumps(rec))
    return cid


def _steer(b, d, p, lc="L14", c=1.0):
    return {
        "behavior": b,
        "kind": "steer",
        "direction": d,
        "position": p,
        "layer_config": lc,
        "c": c,
    }


def _build_judged_tree(root: Path) -> None:
    b = "evil"
    # baseline_ceiling: alpha0 + donor-swap ceiling
    _judged(root, "baseline_ceiling", {"behavior": b, "kind": "alpha0"}, [10, 10, 10, 10], 0.0)
    _judged(root, "baseline_ceiling", {"behavior": b, "kind": "ceiling"}, [60, 62, 58, 61], 0.9)
    # localize: alpha0 + steered + nulls (context and answer)
    _judged(root, "localize", {"behavior": b, "kind": "alpha0"}, [10, 10, 10, 10], 0.0)
    _judged(root, "localize", _steer(b, "pre", "context"), [40, 42, 38, 44], 0.5)
    _judged(root, "localize", _steer(b, "ctxext", "context"), [35, 36, 34, 37], 0.4)
    _judged(root, "localize", _steer(b, "random", "context"), [11, 9, 10, 12], 0.0)
    _judged(root, "localize", _steer(b, "preshuf", "context"), [10, 11, 9, 10], 0.0)
    _judged(root, "localize", _steer(b, "rb", "answer"), [30, 32, 28, 31], 0.3)
    _judged(root, "localize", _steer(b, "random", "answer"), [10, 10, 11, 9], 0.0)
    # decisive: alpha0 + operating-point cells + nulls
    _judged(root, "decisive", {"behavior": b, "kind": "alpha0"}, [10, 10, 10, 10], 0.0)
    _judged(root, "decisive", _steer(b, "pre", "context"), [45, 44, 46, 43], 0.5)
    _judged(root, "decisive", _steer(b, "ctxext", "context"), [40, 41, 39, 42], 0.4)
    _judged(root, "decisive", _steer(b, "random", "context"), [10, 11, 9, 10], 0.0)
    _judged(root, "decisive", _steer(b, "preshuf", "context"), [11, 10, 10, 9], 0.0)
    _judged(root, "decisive", _steer(b, "rb", "answer"), [30, 29, 31, 30], 0.3)
    # patch: one projection-patch + one ablation cell
    patch = {"behavior": b, "kind": "patch", "direction": "pre", "op": "proj", "breadth": "single"}
    _judged(root, "patch", patch, [58, 60, 57, 59], 0.8)
    abl = {"behavior": b, "kind": "patch", "direction": "pre", "op": "ablate", "breadth": "single"}
    _judged(root, "patch", abl, [12, 11, 13, 10], 0.05)


@pytest.fixture(scope="module")
def reduced_root(tmp_path_factory):
    """Synthetic judged tree run through the REAL wave-1 + wave-2 reduces
    (upload boundary faked with a signature-matching recorder)."""
    root = tmp_path_factory.mktemp("i2254_reduce")
    _build_judged_tree(root)
    args = _full_args(["--behaviors", "evil"])
    uploads: list[tuple[str, str]] = []
    orig = pi._upload_folder_to_hf

    def _record(local_dir, path_in_repo, allow=None):
        uploads.append((str(local_dir), path_in_repo))

    pi._upload_folder_to_hf = _record
    try:
        pi._reduce_wave1(args, root)
        pi._reduce_wave2(args, root)
    finally:
        pi._upload_folder_to_hf = orig
    return root, uploads


def test_wave1_outputs_gates_and_operating_points(reduced_root):
    root, uploads = reduced_root
    dose = json.loads((root / "localize" / "dose_response.json").read_text())
    ops = json.loads((root / "localize" / "operating_points.json").read_text())
    gates = json.loads((root / "localize" / "gates.json").read_text())
    base = json.loads((root / "baseline_ceiling" / "judged_percell.json").read_text())
    eb = dose["behaviors"]["evil"]
    assert eb["n_q"] == NQ and eb["alpha0_mean"] == pytest.approx(10.0)
    pre_cid = pi._cell_id(_steer("evil", "pre", "context"))
    assert eb["cells"][pre_cid]["delta_score"] == pytest.approx(31.0)
    lo, hi = eb["cells"][pre_cid]["ci_frozen"]
    assert lo <= 31.0 <= hi and lo > 0
    assert eb["null_band_context"]["n_cells"] == 2  # random + preshuf
    assert eb["null_band_answer"]["n_cells"] == 1
    op = ops["behaviors"]["evil"]["pre__context__single"]
    assert op["cell_id"] == pre_cid and op["c"] == 1.0
    assert ops["behaviors"]["evil"]["pre__context__all"] is None  # no all-config cell
    g = gates["behaviors"]["evil"]
    assert g["gate2"]["pass"] and g["gate3"]["pass"] and g["proceed"]
    # gate 3 carries the REGISTERED quantity (100 - alpha0 mean) + rate
    # version, with the donor-swap delta kept as context (round-1 blocker).
    assert g["gate3"]["headroom_score"] == pytest.approx(90.0)
    assert g["gate3"]["headroom_rate"] == pytest.approx(1.0)
    assert g["gate3"]["ceiling_delta"] == pytest.approx(50.25)
    assert "coherence" in gates["stated_deviation_coherence"]
    assert base["behaviors"]["evil"]["ceiling_delta"] == pytest.approx(50.25)
    assert base["behaviors"]["evil"]["headroom_score"] == pytest.approx(90.0)
    comp = json.loads((root / "judge" / "completeness_wave1.json").read_text())
    assert comp["below_floor_cells"] == []
    assert any("localize" in p for _, p in uploads)


def test_wave2_verdict_h1_with_both_ci_labels(reduced_root):
    root, uploads = reduced_root
    percell = json.loads((root / "decisive" / "delta_score_percell.json").read_text())
    verdicts = json.loads((root / "decisive" / "verdicts.json").read_text())
    v = verdicts["behaviors"]["evil"]
    assert v["label"] == "H1", v
    assert v["margins"]["E_pre"]["ci"][0] > 0
    assert v["margins"]["C_gap"]["ci"][0] > 0  # pre beats ctxext on every question
    assert v["null_band_context"]["n_draws"] == pi.N_BOOT_VERDICT
    assert "pre__context" in v["selection_inherited"]
    pre_cid = pi._cell_id(_steer("evil", "pre", "context"))
    rec = percell["behaviors"]["evil"][pre_cid]
    assert rec["delta_score"] == pytest.approx(34.5)
    assert rec["ci_label"].startswith("frozen (decisive grid")
    assert any("decisive" in p for _, p in uploads)


def test_patch_vs_ceiling_fractions(reduced_root):
    root, _ = reduced_root
    pvc = json.loads((root / "patch" / "patch_vs_ceiling.json").read_text())
    proj_cid = pi._cell_id(
        {"behavior": "evil", "kind": "patch", "direction": "pre", "op": "proj", "breadth": "single"}
    )
    abl_cid = pi._cell_id(
        {
            "behavior": "evil",
            "kind": "patch",
            "direction": "pre",
            "op": "ablate",
            "breadth": "single",
        }
    )
    assert pvc["cells"][proj_cid]["fraction_point"] == pytest.approx((58.5 - 10.0) / 50.25)
    assert pvc["cells"][abl_cid]["fraction_point"] == pytest.approx((60.25 - 11.5) / 50.25)
    assert pvc["cells"][proj_cid]["n_degenerate_draws"] == 0


def test_patch_vs_ceiling_degenerate_denominator(tmp_path):
    b = "evil"
    _judged(tmp_path, "baseline_ceiling", {"behavior": b, "kind": "alpha0"}, [10, 10, 10, 10])
    _judged(tmp_path, "baseline_ceiling", {"behavior": b, "kind": "ceiling"}, [10, 10, 10, 10])
    patch = {"behavior": b, "kind": "patch", "direction": "pre", "op": "proj", "breadth": "single"}
    cid = _judged(tmp_path, "patch", patch, [30, 31, 29, 30])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN quantile is the designed path
        out = pi._patch_vs_ceiling(tmp_path)
    rec = out["cells"][cid]
    assert rec["fraction_point"] is None
    assert rec["n_degenerate_draws"] == pi.N_BOOT_CELL
    # strict-JSON: non-finite CI edges persist as null, never a bare NaN
    assert rec["fraction_ci"] == [None, None]
    assert "NaN" not in json.dumps(out)


def test_wave2_gate_skipped_behavior_is_undefined(tmp_path):
    root = tmp_path
    _build_judged_tree(root)
    pi._write_json_atomic(
        root / "localize" / "gates.json",
        {"behaviors": {"evil": {"proceed": False}}},
    )
    args = _full_args(["--behaviors", "evil"])
    orig = pi._upload_folder_to_hf
    pi._upload_folder_to_hf = lambda *a, **k: None
    try:
        pi._reduce_wave2(args, root)
    finally:
        pi._upload_folder_to_hf = orig
    verdicts = json.loads((root / "decisive" / "verdicts.json").read_text())
    assert verdicts["behaviors"]["evil"]["label"] == "Undefined"


def test_completeness_block_flags_below_floor(tmp_path):
    b = "evil"
    _judged(tmp_path, "localize", {"behavior": b, "kind": "alpha0"}, [10] * 4)
    low = _steer(b, "pre", "context", lc="L17")
    cid = _judged(
        tmp_path,
        "localize",
        low,
        [40] * 4,
        accounting={"frac_items_complete": 0.5, "n_items": 4, "n_items_zero_valid": 2},
    )
    files = sorted((tmp_path / "judge" / "localize" / "judged").glob("*.json"))
    block = pi._completeness_block(files)
    assert block["floor"] == 0.95
    assert block["below_floor_cells"] == [cid]
    assert "rule 29" in block["remediation"]


# ---------------------------------------------------------------------------
# figures: render the REAL builders against the reduced tree
# ---------------------------------------------------------------------------


def test_render_all_on_reduced_tree(reduced_root, tmp_path):
    from scripts.issue2254_figures import render_all

    root, _ = reduced_root
    fig_dir = tmp_path / "figs"
    res = render_all(root, fig_dir)
    rendered = set(res["rendered"])
    assert {
        "hero1_decisive_bars",  # plan §6.5 primary_deliverable filename
        "hero2_patch_fraction",
        "dose_response",
        "rate_companion",
        "layer_dose_heatmap",
        "coherence_vs_dose",
        "per_question_dots",
    } <= rendered
    assert set(res["skipped"]) == {
        "result0",
        "margin_scatter",
        "map_quality",
        # plan-v7 split-amendment builders: inputs absent from the parent-only
        # synthetic tree, so they skip-with-reason here (rendered-path coverage
        # lives in tests/test_issue2254_split.py)
        "ctxext_split_hero",
        "ctxext_split_dose",
    }
    for name in rendered:
        png = fig_dir / f"{name}.png"
        assert png.is_file() and png.stat().st_size > 1000
        meta = json.loads((fig_dir / f"{name}.meta.json").read_text())
        assert meta["figure"] == name and meta["inputs"]


def test_render_all_require_raises_on_skipped(reduced_root, tmp_path):
    from scripts.issue2254_figures import render_all

    root, _ = reduced_root
    with pytest.raises(RuntimeError, match="required figures not rendered"):
        render_all(root, tmp_path / "f2", require=("margin_scatter",))


# ---------------------------------------------------------------------------
# gate 3: the REGISTERED quantity (round-1 blocker g3)
# ---------------------------------------------------------------------------


def test_gate3_uses_registered_headroom_not_donor_delta(tmp_path):
    """Plan §7 gate 3 = 100 - alpha0 mean graded score vs the context null-band
    upper edge. A behavior with graded HEADROOM but a WEAK donor-swap delta
    must PASS — pre-fix the donor-swap delta was substituted for the
    registered quantity and this construction (headroom > band > donor delta)
    was wrongly demoted."""
    b = "evil"
    _judged(tmp_path, "baseline_ceiling", {"behavior": b, "kind": "alpha0"}, [10] * 4, 0.0)
    _judged(tmp_path, "baseline_ceiling", {"behavior": b, "kind": "ceiling"}, [12, 11, 13, 12], 0.1)
    _judged(tmp_path, "localize", {"behavior": b, "kind": "alpha0"}, [10] * 4, 0.0)
    _judged(tmp_path, "localize", _steer(b, "pre", "context"), [40, 42, 38, 44], 0.5)
    _judged(tmp_path, "localize", _steer(b, "random", "context"), [30, 31, 29, 30], 0.2)
    _judged(tmp_path, "localize", _steer(b, "preshuf", "context"), [29, 30, 31, 30], 0.2)
    args = _full_args(["--behaviors", "evil"])
    orig = pi._upload_folder_to_hf
    pi._upload_folder_to_hf = lambda *a, **k: None
    try:
        pi._reduce_wave1(args, tmp_path)
    finally:
        pi._upload_folder_to_hf = orig
    g3 = json.loads((tmp_path / "localize" / "gates.json").read_text())["behaviors"][b]["gate3"]
    # construction sanity: headroom (90) > band edge (~20) > donor delta (2)
    assert g3["headroom_score"] == pytest.approx(90.0)
    assert g3["headroom_rate"] == pytest.approx(1.0)
    assert g3["ceiling_delta"] == pytest.approx(2.0)
    assert g3["headroom_score"] > g3["context_band_p975"] > g3["ceiling_delta"]
    assert g3["pass"] is True


# ---------------------------------------------------------------------------
# judge-wave grid completeness (round-1 blocker g3)
# ---------------------------------------------------------------------------


def _smoke_args(extra: list[str] | None = None):
    return pi.build_argparser().parse_args(
        ["--phase", "judge_reduce", "--behaviors", "evil", "--smoke", *(extra or [])]
    )


def _touch_gen(comp_root: Path, cid: str) -> None:
    comp_root.mkdir(parents=True, exist_ok=True)
    (comp_root / f"{cid}.json").write_text("{}")


def test_gen_grid_completeness_wave1_pass_and_missing_cell(tmp_path):
    args = _smoke_args()
    comp = {
        "baseline_ceiling": tmp_path / "baseline_ceiling" / "raw_completions",
        "localize": tmp_path / "localize" / "raw_completions",
    }
    for kind in ("alpha0", "ceiling"):
        _touch_gen(comp["baseline_ceiling"], pi._cell_id({"behavior": "evil", "kind": kind}))
    cells = pi._localize_cells(args, ["evil"])
    for c in cells:
        _touch_gen(comp["localize"], pi._cell_id(c))
    assert pi._assert_gen_grid_complete(args, tmp_path, "localize", comp) == ["evil"]
    # delete ONE localize gen cell -> the paid judge wave refuses, NAMING it
    victim = pi._cell_id(cells[3])
    (comp["localize"] / f"{victim}.json").unlink()
    with pytest.raises(RuntimeError, match=victim):
        pi._assert_gen_grid_complete(args, tmp_path, "localize", comp)


def _write_wave2_fixtures(root: Path, args, null_recorded: bool = True):
    """gates + operating_points + selection_meta for the smoke combo grid,
    with ONE (pre, context, all) operating point null (recorded as missing
    when `null_recorded`)."""
    lc_of = {"single": "L14", "mid": "mid", "all": "all"}
    ops_b: dict = {}
    missing: list[str] = []
    for d, p in pi._grid_combos(args):
        for br in pi.BREADTHS:
            if (d, p, br) == ("pre", "context", "all"):
                ops_b[f"{d}__{p}__{br}"] = None
                missing.append(f"evil/{d}/{p}/{br}")
            else:
                ops_b[f"{d}__{p}__{br}"] = {"cell_id": "x", "layer_config": lc_of[br], "c": 1.0}
    pi._write_json_atomic(
        root / "localize" / "operating_points.json", {"behaviors": {"evil": ops_b}}
    )
    pi._write_json_atomic(
        root / "localize" / "gates.json", {"behaviors": {"evil": {"proceed": True}}}
    )
    pi._write_json_atomic(
        root / "decisive" / "selection_meta.json",
        {"missing_operating_points": missing if null_recorded else []},
    )


def test_gen_grid_completeness_wave2_honors_selection_meta(tmp_path):
    args = _smoke_args()
    _write_wave2_fixtures(tmp_path, args)
    expected, kept = pi._expected_gen_cell_ids(args, tmp_path, "decisive")
    assert kept == ["evil"]
    # 3 smoke combos x 3 breadths - 1 RECORDED-missing operating point + alpha0
    assert len(expected["decisive"]) == 3 * 3 - 1 + 1
    assert len(expected["patch"]) == 12  # 3 directions x 2 ops x 2 breadths
    comp = {
        "decisive": tmp_path / "decisive" / "raw_completions",
        "patch": tmp_path / "patch" / "raw_completions",
    }
    for phase, ids in expected.items():
        for cid in ids:
            _touch_gen(comp[phase], cid)
    assert pi._assert_gen_grid_complete(args, tmp_path, "decisive", comp) == ["evil"]
    victim = sorted(expected["patch"])[0]
    (comp["patch"] / f"{victim}.json").unlink()
    with pytest.raises(RuntimeError, match="patch: 1 missing"):
        pi._assert_gen_grid_complete(args, tmp_path, "decisive", comp)


def test_gen_grid_wave2_unrecorded_null_operating_point_is_inconsistent(tmp_path):
    args = _smoke_args()
    _write_wave2_fixtures(tmp_path, args, null_recorded=False)
    with pytest.raises(RuntimeError, match="inconsistent"):
        pi._expected_gen_cell_ids(args, tmp_path, "decisive")


def test_judge_reduce_asserts_completeness_before_any_judge_spend():
    """Ordering pin: the completeness gate runs BEFORE the rule-26 pilot (the
    first paid judge call) inside phase_judge_reduce."""
    import inspect

    src = inspect.getsource(pi.phase_judge_reduce)
    assert src.index("_assert_gen_grid_complete") < src.index("_run_judge_pilot")


# ---------------------------------------------------------------------------
# upload path: packed shards only, never a per-cell tree (#2286 1M-file ceiling)
# ---------------------------------------------------------------------------


def test_upload_judge_outputs_packs_per_cell_trees(tmp_path):
    """Wave-1 crash shape (#2286): judge/<phase>/{judged,raw,cache} hold
    O(1000) per-cell files and the shared data repo sits at the Hub's 1M-file
    REPO ceiling, so _upload_judge_outputs must upload PACKED shard dirs (few
    .jsonl files) — a per-cell-tree upload of >50 files fails this test.
    Runs the REAL body; only the network seam is a signature-matching
    recorder."""
    base = tmp_path / "judge" / "localize"
    d = base / "judged"
    d.mkdir(parents=True)
    for i in range(60):
        (d / f"cell{i:03d}.json").write_text(json.dumps({"cell": i, "sub": "judged"}))
    # raw/ mirrors the REAL save_raw layout: bare-<cid> EXTENSIONLESS files —
    # a *.json-only pack would ship n_files=0 manifests (the second wave-1 bug).
    d = base / "raw"
    d.mkdir(parents=True)
    for i in range(60):
        (d / f"cell{i:03d}").write_text(json.dumps({"cell": i, "sub": "raw"}))
    for c in range(12):
        d = base / "cache" / f"cell{c:03d}"
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"{i:016x}.json").write_text(json.dumps({"draw": i}))
    uploads: list[tuple[str, str, list[str]]] = []
    orig = pi._upload_folder_to_hf

    def _record(local_dir, path_in_repo, allow=None):
        files = sorted(
            str(p.relative_to(local_dir)) for p in Path(local_dir).rglob("*") if p.is_file()
        )
        uploads.append((str(local_dir), path_in_repo, files))

    pi._upload_folder_to_hf = _record
    try:
        pi._upload_judge_outputs(tmp_path, ("localize",))
    finally:
        pi._upload_folder_to_hf = orig

    assert uploads, "no uploads recorded"
    dests = {p for _, p, _ in uploads}
    prefix = pi._hf_prefix()
    for sub in ("judged", "cache", "raw"):
        assert f"{prefix}/judge/localize/{sub}_pack" in dests
    assert not any(p.rstrip("/").endswith(("/judged", "/cache", "/raw")) for _, p, _ in uploads), (
        "per-cell tree uploaded un-packed"
    )
    for _, path_in_repo, files in uploads:
        assert len(files) <= 50, f"per-cell tree leaked into upload: {path_in_repo} ({len(files)})"
        assert any(f.endswith(".jsonl") for f in files), path_in_repo
    # every pack carries EVERY source file (an empty raw_pack manifest = raw-drop)
    for sub, n_src in (("judged", 60), ("cache", 72), ("raw", 60)):
        m = json.loads((base / f"{sub}_pack" / "pack_manifest.json").read_text())
        assert m["n_files"] == n_src, f"{sub}_pack packed {m['n_files']}/{n_src} files"
    # round-trip: packed rows reconstruct every per-cell doc, shards stay <9.5 MB, no gzip
    pack = base / "judged_pack"
    manifest = json.loads((pack / "pack_manifest.json").read_text())
    assert manifest["n_files"] == 60
    for shard in manifest["shards"]:
        assert shard.endswith(".jsonl") and not shard.endswith(".gz")
        assert (pack / shard).stat().st_size < 9_500_000
    rows = [
        json.loads(line)
        for shard in manifest["shards"]
        for line in (pack / shard).read_text().splitlines()
    ]
    assert {r["path"] for r in rows} == {f"cell{i:03d}.json" for i in range(60)}
    r0 = rows[0]
    assert r0["doc"] == json.loads((base / "judged" / r0["path"]).read_text())
