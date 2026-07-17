"""#1434 i1434po pins (writing-style-positive-only-regime, plan §4 D1'-D4'):
round registration + seams, the max_steps=75 non-smoke train seam +
expected-rungs override (§12.2 unit pin), the D1' mix filter/rebuild/STOP
integrity chain, the regime-contrast lattice + dose labels, the combined
parent+po validate merge, worker round routing + parent-round guards, and the
new errorbar sites' inverted-CI clamp (#547/#1335)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue1090_fu4 as fu4  # noqa: E402
import issue1434_cells as cells  # noqa: E402
import issue1434_worker as w  # noqa: E402

cells.register_i1434_round()
cells.register_i1434po_round()


@pytest.fixture(autouse=True)
def _restore_round():
    """Every test leaves the module-global ROUND as it found it (co-run
    orderings with the parent pins tests assume the i1434 default)."""
    before = fu4.ROUND
    yield
    fu4.ROUND = before


def _po():
    fu4.set_round("i1434po")
    return fu4.ROUND


# ── registration + seams (plan §4 D2') ───────────────────────────────────────


def test_po_round_registration_and_seams():
    spec = _po()
    assert spec.name == "i1434po" and spec.label == "writing-style-positive-only-regime"
    assert len(spec.runs) == 12 and len({r.run_id for r in spec.runs}) == 12
    assert all(r.run_id.startswith("ws-po-") for r in spec.runs)
    assert all(r.round_name == "i1434po" for r in spec.runs)
    # §2 item 3 + §10: matched optimizer steps, all-rung upload, po buckets.
    assert spec.train_max_steps == 75
    assert spec.upload_all_rungs is True
    assert sorted(spec.mix_composition) == [0, 20, 40]
    assert spec.raw_prefix == "issue1434_writingstyle/raw_completions/po"
    assert fu4.raw_completions_prefix() == spec.raw_prefix
    assert spec.manifest_name == "cell_manifest_i1434po.json"
    assert spec.ladders_name == "i1434po_ladders.json"
    assert Path(spec.deliverables_dir).name == "writing-style-positive-only-regime"
    # Same judge/margin seams as the parent (plan §4 D2'), po margin wrapper.
    assert spec.judge_fn is cells.pv_judge_fn
    assert spec.margin_pools_fn is cells.i1434po_margin_pools
    # K3 parity anchor deliberately empty (no prior po anchor exists).
    assert spec.k3_parity_run_id == ""
    # Mix prefixes: issue1434_writingstyle/ws-po-<ctx>/mix (plan §4 D1').
    assert {r.mix_hub_prefix for r in spec.runs} == {
        f"issue1434_writingstyle/{ck}/mix" for ck in cells.PO_CELL_KEYS
    }
    # WandB run names: issue1434_ws-po-<ctx>-<lrtag>_seed42 (plan §4 D2').
    assert (
        cells.RUN_BY_ID_1434PO["ws-po-pers-lr1e5"].run_name == "issue1434_ws-po-pers-lr1e5_seed42"
    )


def test_po_smoke_resolver_one_run_production_twelve():
    _po()
    assert [r.run_id for r in fu4.resolve_fu4_runs(None, smoke=True)] == ["ws-po-pers-lr1e5"]
    assert len(fu4.resolve_fu4_runs(None, smoke=False)) == 12
    # Cell resolver: same subset-threading (smoke = the persona po cell).
    assert w.resolve_cell_keys(None, smoke=True) == ["ws-po-pers"]
    assert w.resolve_cell_keys(None, smoke=False) == list(cells.PO_CELL_KEYS)
    with pytest.raises(ValueError):
        w.resolve_cell_keys("ws-pers", smoke=False)  # parent key invalid under po


def test_parent_round_untouched_by_po_registration():
    fu4.set_round("i1434")
    spec = fu4.ROUND
    assert spec.train_max_steps is None
    assert spec.raw_prefix == "" and fu4.raw_completions_prefix() == (
        "issue1434_writingstyle/raw_completions"
    )
    assert sorted(spec.mix_composition) == [20, 20, 40]
    assert w.resolve_cell_keys(None, smoke=True) == ["ws-pers"]
    assert cells.run_lookup("ws-pers-lr1e5") is cells.RUN_BY_ID_1434["ws-pers-lr1e5"]
    assert cells.run_lookup("ws-po-pers-lr1e5") is cells.RUN_BY_ID_1434PO["ws-po-pers-lr1e5"]


# ── §12.2 unit pin: max_steps=75 non-smoke seam + expected-rungs override ────


def test_train_max_steps_seam_and_expected_rungs(tmp_path, monkeypatch):
    """FAILS PRE-FIX: without the RoundSpec.train_max_steps seam the composed
    TrainLoraConfig.max_steps is None and the build record's expected total is
    the epochs-derived 60 (fu4_expected_rungs at 60 rows), not 75."""
    _po()
    run = cells.RUN_BY_ID_1434PO["ws-po-pers-lr1e5"]
    captured: dict = {}
    ckpt = tmp_path / "adapters" / "checkpoint-75"
    ckpt.mkdir(parents=True)

    def fake_train_lora(base_model, mix_path, out_dir, cfg):
        captured["max_steps"] = cfg.max_steps
        return str(tmp_path / "adapters"), 0.5

    monkeypatch.setattr(fu4, "train_lora", fake_train_lora)
    monkeypatch.setattr(fu4, "release_trainer_cuda_memory", lambda: None)
    monkeypatch.setattr(fu4.fu2, "enumerate_ckpt_rungs", lambda root: {75: ckpt})
    monkeypatch.setattr(fu4, "check_divergence", lambda ckpts: {"checked": True, "diverged": False})
    monkeypatch.setattr(fu4, "_assert_adapter_rank", lambda run, d: {"r": 32})
    cfg = SimpleNamespace(smoke=False, out_root=tmp_path, seed=42)
    seams = SimpleNamespace(train_clamp=None)
    rec = fu4.train_fu4_run(cfg, seams, run, {"n_rows": 60, "train_mix_sha256": "x"})
    assert captured["max_steps"] == 75  # the recipe-spec seam threaded
    assert rec["expected_total_steps"] == 75  # build-record override (§4 D2' note)
    assert rec["expected_rungs"] == list(range(5, 76, 5))
    assert rec["status"] == "trained"


def test_train_max_steps_ladder_incomplete_gate(tmp_path, monkeypatch):
    """Data-dependent gate probe: a realized ladder short of 75 fails LOUD
    under the po round (max(realized) < overridden expected_total)."""
    _po()
    run = cells.RUN_BY_ID_1434PO["ws-po-bare-lr1e5"]
    ckpt = tmp_path / "adapters" / "checkpoint-60"
    ckpt.mkdir(parents=True)
    monkeypatch.setattr(fu4, "train_lora", lambda *a, **k: (str(tmp_path / "adapters"), 0.5))
    monkeypatch.setattr(fu4, "release_trainer_cuda_memory", lambda: None)
    monkeypatch.setattr(fu4.fu2, "enumerate_ckpt_rungs", lambda root: {60: ckpt})
    cfg = SimpleNamespace(smoke=False, out_root=tmp_path, seed=42)
    with pytest.raises(ValueError, match="ladder incomplete"):
        fu4.train_fu4_run(
            cfg, SimpleNamespace(train_clamp=None), run, {"n_rows": 60, "train_mix_sha256": "x"}
        )


def test_po_mix_composition_gate(tmp_path, monkeypatch):
    """K1 composition is ROUND-keyed: the po (0/20/40) passes under i1434po
    and the parent shape (20/20/40) is REFUSED there."""
    _po()
    run = cells.RUN_BY_ID_1434PO["ws-po-pers-lr1e5"]
    mix_dir = tmp_path / run.run_id / "mix"
    mix_dir.mkdir(parents=True)
    rows = [{"prompt": f"q{i}", "completion": "a"} for i in range(60)]
    (mix_dir / "train_mix.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )

    def _meta(realized):
        (mix_dir / "mix_meta.json").write_text(
            json.dumps(
                {
                    "spec": {"behavior_name": "writing_style"},
                    "counts_realized": realized,
                }
            )
        )

    cfg = SimpleNamespace(smoke=False, out_root=tmp_path)
    _meta({"positives": 20, "negatives": 0, "generic": 40})
    rec = fu4.verify_fu4_mix(cfg, run, None)
    assert rec["n_rows"] == 60
    _meta({"positives": 20, "negatives": 20, "generic": 20})
    with pytest.raises(ValueError, match="mix composition"):
        fu4.verify_fu4_mix(cfg, run, None)


# ── D1' mix builder: filter / rebuild / STOP integrity chain ─────────────────


def _pos(i):
    return {"messages": [{"role": "user", "content": f"q{i}"}], "kind": "pos", "i": i}


def _cn(i):
    return {"messages": [{"role": "user", "content": f"q{i}"}], "kind": "cn", "i": i}


def _gen(i):
    return {"prompt": f"g{i}", "completion": f"c{i}", "kind": "generic", "i": i}


def _parent_fixture(seed=42):
    """A parent-shaped fixture whose generic rows ARE Random(seed).sample of a
    corpus (so the rebuild path reproduces the draw exactly)."""
    corpus = [_gen(i) for i in range(200)]
    pos = [_pos(i) for i in range(20)]
    cn = [_cn(i) for i in range(20)]
    generic = random.Random(seed).sample(corpus, 40)
    mix = [*pos, *cn, *generic]
    random.Random(seed).shuffle(mix)
    return mix, cn, pos, corpus


def test_po_filter_parent_mix_happy_path():
    mix, cn, pos, _ = _parent_fixture()
    rows, deriv = w._po_filter_parent_mix(mix, cn, pos)
    assert len(rows) == 60 and deriv["n_pos"] == 20 and deriv["n_generic"] == 40
    assert deriv["method"] == "filter_parent_mix_minus_cn"
    # parent order preserved + every row byte-identical to a parent-mix row
    kept_canons = [w._canon_row(r) for r in rows]
    parent_canons = [w._canon_row(r) for r in mix]
    assert [c for c in parent_canons if c in set(kept_canons)] == kept_canons
    cn_canons = {w._canon_row(r) for r in cn}
    assert not cn_canons & set(kept_canons)  # zero panel rows


def test_po_filter_gates_fire_on_degenerate_inputs():
    mix, cn, pos, _ = _parent_fixture()
    with pytest.raises(w.PoMixIntegrityError, match="off-shape"):
        w._po_filter_parent_mix(mix, cn[:19], pos)
    with pytest.raises(w.PoMixIntegrityError, match="cn content match failed"):
        w._po_filter_parent_mix(mix, [*cn[:19], _cn(999)], pos)  # unmatched cn row
    # a generic row byte-equal to a cn row -> residual panel content fires
    mix2 = [*mix, _cn(3)]
    with pytest.raises(w.PoMixIntegrityError, match="panel content still present"):
        w._po_filter_parent_mix(mix2, cn, pos)
    with pytest.raises(w.PoMixIntegrityError, match="po composition"):
        w._po_filter_parent_mix(mix, cn, [*pos[:19], _pos(999)])  # unmatched pos


def test_po_rebuild_reproduces_seeded_draw_and_stop_chain():
    mix, cn, pos, corpus = _parent_fixture()
    rows, deriv = w._po_rebuild_from_sidecars(mix, pos, corpus, 42)
    assert len(rows) == 60 and deriv["method"] == "rebuild_pos_plus_seeded_generic"
    # wrong seed -> content-equality fails
    with pytest.raises(w.PoMixIntegrityError, match="content-equality failed"):
        w._po_rebuild_from_sidecars(mix, pos, corpus, 43)
    # both paths failing -> the STOP RuntimeError (plan §7), never silent
    with pytest.raises(RuntimeError, match="STOP"):
        w._derive_po_rows("ws-po-pers", mix, cn[:19], pos, lambda: corpus, 43)
    # a degenerate corpus is ALSO integrity-routed (never a bare ValueError)
    with pytest.raises(RuntimeError, match="STOP"):
        w._derive_po_rows("ws-po-pers", mix, cn[:19], pos, lambda: corpus[:10], 42)


# ── regime contrast (plan §3 lattice + §6 reads) ─────────────────────────────


def _judged(k, n):
    return {"rate": k / n, "k_positive": k, "n_scored": n, "graded_mean": 50.0}


def _panel_entry(run_id, source_ctx, trained_by_ctx, base_by_ctx):
    return {
        "run_id": run_id,
        "contexts": {
            ctx: {"trained": _judged(*kn), "base": _judged(*base_by_ctx[ctx])}
            for ctx, kn in trained_by_ctx.items()
        },
    }


def _mk_aggs():
    ctxs = ["persona_software_engineer", "default", "c3", "c4", "c5", "c6"]
    src = "persona_software_engineer"
    base = {c: (10, 100) for c in ctxs}
    po_agg = {
        "panel": {
            "ws-po-pers": _panel_entry("ws-po-pers-lr1e5", src, {c: (60, 100) for c in ctxs}, base)
        },
        "tier2": {
            "ws-po-pers": {
                "trained": _judged(80, 200),
                "base": _judged(20, 200),
                "verdict_arm": {"run_id": "ws-po-pers-lr1e5"},
                "q_band": 0.1,
                "delta": 0.3,
                "delta_newcombe_95": [0.2, 0.4],
                "lattice_verdict": "Installed",
            }
        },
        "verdict_arms": {"ws-po-pers": {"selection": {"rate": 0.7, "in_band": True, "step": 20}}},
    }
    con_agg = {
        "panel": {
            "ws-pers": _panel_entry("ws-pers-lr1e5", src, {c: (20, 100) for c in ctxs}, base)
        },
        "tier2": {"ws-pers": {"trained": _judged(75, 200), "base": _judged(20, 200)}},
        "verdict_arms": {"ws-pers": {"selection": {"rate": 0.65, "in_band": True, "step": 25}}},
    }
    return po_agg, con_agg


def test_regime_contrast_broader_lattice_and_cells():
    _po()
    po_agg, con_agg = _mk_aggs()
    out = w.regime_contrast(po_agg, con_agg, ["ws-po-pers"])
    entry = out["contexts"]["ws-po-pers"]
    pooled = entry["pooled"]
    # pooled over the 5 NON-source contexts: po 300/500 vs con 100/500
    assert pooled["po"]["n"] == 500 and pooled["con"]["n"] == 500
    assert pooled["D"] == pytest.approx(0.4)
    assert pooled["lattice"] == "Broader-leakage"
    assert pooled["base"]["n"] == 500
    assert pooled["delta_po_vs_base"]["delta"] == pytest.approx(0.5)
    assert pooled["delta_con_vs_base"]["delta"] == pytest.approx(0.1)
    # the 5-cell (per read-context) companion rows, all computed
    assert len(out["cells"]) == 5
    assert all(c["status"] == "computed" for c in out["cells"])
    # install contrast + dose labels (both in-band, gap 0.05 <= 0.10)
    assert entry["install_contrast"]["D"] == pytest.approx(0.025)
    assert entry["dose"]["dose_unmatched"] is False
    assert entry["po_install_lattice"]["lattice_verdict"] == "Installed"


def test_regime_contrast_dose_unmatched_and_lattice_branches():
    _po()
    po_agg, con_agg = _mk_aggs()
    # narrower: flip the pooled direction hard
    for _c, row in po_agg["panel"]["ws-po-pers"]["contexts"].items():
        row["trained"] = _judged(2, 100)
    con_agg["verdict_arms"]["ws-pers"]["selection"] = {"rate": 0.62, "in_band": False}
    out = w.regime_contrast(po_agg, con_agg, ["ws-po-pers"])
    entry = out["contexts"]["ws-po-pers"]
    assert entry["pooled"]["lattice"] == "Narrower-leakage"
    assert entry["dose"]["dose_unmatched"] is True  # closest-approach con arm
    # lattice disjoint/exhaustive spot checks
    assert w._regime_lattice(0.02, (-0.01, 0.05)) == "Indistinguishable"
    assert w._regime_lattice(0.2, (0.05, 0.35)) == "Broader-leakage"
    assert w._regime_lattice(-0.2, (-0.35, -0.05)) == "Narrower-leakage"
    assert w._regime_lattice(None, None) == "not_computable"
    # missing panel arm branch
    out2 = w.regime_contrast({"panel": {}}, {"panel": {}}, ["ws-po-pers"])
    assert out2["contexts"]["ws-po-pers"]["status"] == "missing_panel_arm"


def test_pooled_counts_none_propagation():
    src = "persona_software_engineer"
    entry = _panel_entry(
        "r", src, {src: (50, 100), "c3": (30, 100)}, {src: (1, 100), "c3": (1, 100)}
    )
    entry["contexts"]["c4"] = {"trained": {"rate": None}, "base": _judged(1, 100)}
    k, n, used = w._pooled_nonsource_counts(entry, src)
    assert (k, n, used) == (30, 100, ["c3"])  # source + all-dropped excluded


# ── combined parent+po validate merge (plan §4 D4') ──────────────────────────


def test_merge_parent_po_states_and_layer_gate(tmp_path, monkeypatch):
    import issue1434_pv as pv

    _po()
    parent_proj = {"layers": [0, 1], "states": {"ws-pers-lr1e5": {"a": 1}}}
    po_proj = {
        "layers": [0, 1],
        "arms": list(pv.CAPTURE_ARMS),
        "states": {"ws-po-pers-lr1e5": {"b": 2}},
    }
    root = tmp_path / "pv"
    root.mkdir()
    (root / "projections.json").write_text(json.dumps(parent_proj))
    monkeypatch.setattr(
        w,
        "_parent_aggregate",
        lambda: {"panel": {"ws-pers": {}}, "tier2": {}, "ladders": {}, "verdict_arms": {}},
    )
    merged_agg, merged_proj = pv._merge_parent_po(
        root, {"panel": {"ws-po-pers": {}}, "tier2": {}, "ladders": {}, "verdict_arms": {}}, po_proj
    )
    assert set(merged_proj["states"]) == {"ws-pers-lr1e5", "ws-po-pers-lr1e5"}
    assert set(merged_agg["panel"]) == {"ws-pers", "ws-po-pers"}
    # layer mismatch fails loud (incoherent staged r_B/base stores)
    (root / "projections.json").write_text(json.dumps({"layers": [0], "states": {}}))
    with pytest.raises(RuntimeError, match="layer mismatch"):
        pv._merge_parent_po(root, {}, po_proj)


def test_stage_validate_captures_revision_keying(tmp_path, monkeypatch):
    """Parent states + parent-mapped base stores stage AT THE PIN; po states
    stage at main (they postdate the pin); local copies short-circuit."""
    import issue1434_pv as pv

    _po()
    staged: list[tuple[str, str | None]] = []

    def fake_stage(repo, path, target, *, repo_type, revision):
        staged.append((path, revision))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(b"x")
        return Path(target)

    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)
    root = tmp_path / "pv"
    # a locally-present po base store must NOT re-stage
    local = root / "capture" / "base-ws-po-bare" / "summary.pt"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local")
    pv._stage_validate_captures(root, ["ws-pers-lr1e5", "ws-po-pers-lr1e5", "base-ws-po-bare"])
    by_path = dict(staged)
    pfx = "issue1434_writingstyle/analysis_tensors/capture"
    assert by_path[f"{pfx}/ws-pers-lr1e5/summary.pt"] == cells.DATA_REPO_PIN_1434
    assert by_path[f"{pfx}/ws-po-pers-lr1e5/summary.pt"] is None  # po: main
    # base twins: parent run -> base-ws-pers @ pin; po run -> base-ws-po-pers
    # staged FROM the parent base-ws-pers path @ pin
    assert by_path[f"{pfx}/base-ws-pers/summary.pt"] == cells.DATA_REPO_PIN_1434
    assert local.read_bytes() == b"local"  # short-circuited
    # the po base dir maps to the PARENT hub path
    n_parent_base = sum(1 for p, _ in staged if p == f"{pfx}/base-ws-pers/summary.pt")
    assert n_parent_base >= 1


# ── worker routing + parent-round guards ─────────────────────────────────────


def test_worker_round_flag_delegation(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(fu4, "main", lambda argv: (seen.setdefault("argv", list(argv)) and 0) or 0)
    rc = w.main(["--smoke", "--round", "i1434po", "--phase", "dispatch", "--dry-run"])
    assert rc == 0
    assert seen["argv"].count("--round") == 1
    i = seen["argv"].index("--round")
    assert seen["argv"][i + 1] == "i1434po"
    assert fu4.ROUND.name == "i1434po"


def test_parent_only_phase_guards_refuse_po_round():
    _po()
    cfg = SimpleNamespace(smoke=True, out_root=Path("/tmp/x"))
    args = SimpleNamespace(cells=None)
    with pytest.raises(SystemExit, match="parent-round only"):
        w.phase_datagen(cfg, args)
    with pytest.raises(SystemExit, match="parent-round only"):
        w.phase_stage(cfg, args)
    with pytest.raises(SystemExit, match="parent-round only"):
        w.phase_base_arms(cfg, args)
    import issue1434_pv as pv

    with pytest.raises(SystemExit, match="parent-round only"):
        pv.phase_extract(cfg, args)
    # and the mixes builder refuses the PARENT round
    fu4.set_round("i1434")
    with pytest.raises(SystemExit, match="i1434po builder"):
        w.phase_mixes(cfg, args)


# ── new errorbar sites: inverted-CI clamp to savefig (#547/#1335) ────────────


def test_po_figures_errorbar_offsets_clamp_inverted_ci(tmp_path):
    import issue1434_figures as figs

    inverted = [0.31, 0.29]  # lo > D > hi: the tiny-n inverted-quantile shape
    contrast = {
        "contexts": {
            "ws-po-pers": {
                "pooled": {
                    "status": "computed",
                    "D": 0.3,
                    "newcombe_95": inverted,
                    "lattice": "Indistinguishable",
                    "delta_po_vs_base": {"delta": 0.5, "newcombe_95": [0.52, 0.48]},
                    "delta_con_vs_base": {"delta": 0.1, "newcombe_95": [0.12, 0.08]},
                },
                "dose": {"dose_unmatched": True},
            }
        },
        "cells": [
            {
                "status": "computed",
                "D": 0.2,
                "newcombe_95": [0.22, 0.18],
                "training_cell": "ws-po-pers",
                "read_ctx": "default",
            }
        ],
    }
    assert figs.fig_regime_hero(contrast, tmp_path).exists()
    assert figs.fig_regime_cells(contrast, tmp_path).exists()


def test_po_ladder_overlays_renders(tmp_path):
    import issue1434_figures as figs

    def lad(prefix):
        return {
            f"{prefix}-lr1e5": {"rates_by_step": {"5": 0.2, "10": 0.7}},
            f"{prefix}-lr1e4": {"rates_by_step": {"5": 0.9}},
        }

    po_agg = {
        "band": [0.6, 0.85],
        "ladders": {k: v for ck in cells.PO_CELL_KEYS for k, v in lad(ck).items()},
    }
    con_agg = {
        "band": [0.6, 0.85],
        "ladders": {k: v for ck in cells.CELL_KEYS for k, v in lad(ck).items()},
    }
    assert figs.fig_ladder_overlays(po_agg, con_agg, tmp_path).exists()


# ── Batch custom_id budget (#1415): over-budget-only hash compaction ─────────


def test_judge_item_id_budget_compaction(tmp_path, monkeypatch):
    """FAILS PRE-FIX: the po panel tag pn-ws-po-pers-lr1e5-persona_software_
    engineer produces 56-char item ids (> the 53-char Batch budget; the batch
    encoder appends 11 chars to the 64-char API cap) and judge_completions_
    batch raises at enumerate. Post-fix: over-budget ids hash-compact (parent
    ids stay byte-identical) + the id map persists."""
    _po()
    seen: dict = {}

    def fake_judge_graded(items, rubric, **kw):
        seen["ids"] = [iid for iid, _, _ in items]
        return SimpleNamespace(
            scores={iid: 80.0 for iid, _, _ in items},
            n_dropped_draws=0,
            n_transport_lost_draws=0,
        )

    monkeypatch.setattr(w, "judge_graded", fake_judge_graded)
    qs = ["q one", "q two"]
    comps = [["a", "b"], ["c", "d"]]
    long_tag = "pn-ws-po-pers-lr1e5-persona_software_engineer"
    rec = w._judge_rate_graded(
        long_tag,
        qs,
        comps,
        rubric="r",
        n_draws=2,
        judge_root=tmp_path / "judge",
        instrument="pv",
    )
    assert all(len(i) <= 53 for i in seen["ids"])  # the #1415 budget
    assert rec["n_scored"] == 4 and rec["rate"] == 1.0  # scores keyed consistently
    idmap = json.loads((tmp_path / "judge" / "pv" / f"idmap_{long_tag}.json").read_text())
    assert len(idmap) == 4 and all(v.startswith(long_tag) for v in idmap.values())
    # a parent-round tag (exactly at budget) stays byte-identical, no map file
    short_tag = "pn-ws-pers-lr1e5-persona_software_engineer"
    w._judge_rate_graded(
        short_tag, qs, comps, rubric="r", n_draws=2, judge_root=tmp_path / "j2", instrument="pv"
    )
    assert all(i.startswith(short_tag) and len(i) <= 53 for i in seen["ids"])
    assert not (tmp_path / "j2" / "pv" / f"idmap_{short_tag}.json").exists()


# ── po margin seam: staging is non-smoke-only + delegates to the parent pools ─


def test_po_margin_pools_smoke_skips_staging(tmp_path, monkeypatch):
    _po()
    called = {"stage": 0}
    monkeypatch.setattr(
        cells,
        "stage_po_base_margin_reads",
        lambda cfg: called.__setitem__("stage", called["stage"] + 1),
    )
    sentinel = ([{"p": 1}], [{"n": 1}], {"meta": True})
    monkeypatch.setattr(cells, "i1434_margin_pools", lambda cfg: sentinel)
    cfg = SimpleNamespace(smoke=True, out_root=tmp_path)
    assert cells.i1434po_margin_pools(cfg) == sentinel
    assert called["stage"] == 0  # smoke: tiny-real fresh-compute path
    cfg = SimpleNamespace(smoke=False, out_root=tmp_path)
    assert cells.i1434po_margin_pools(cfg) == sentinel
    assert called["stage"] == 1  # full: parent base reads staged (plan §4 D3')


def test_stage_po_base_margin_reads_maps_parent_names(tmp_path, monkeypatch):
    _po()
    staged: list[tuple[str, str, str]] = []

    def fake_stage(repo, path, target, *, repo_type, revision):
        staged.append((path, str(target), revision))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("{}")
        return Path(target)

    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)
    cells.stage_po_base_margin_reads(SimpleNamespace(out_root=tmp_path))
    assert len(staged) == 4
    assert all(rev == cells.DATA_REPO_PIN_1434 for _, _, rev in staged)
    assert ("issue1434_writingstyle/margin/base__ws-pers.json") in [p for p, _, _ in staged]
    assert any(t.endswith("i1434po_margin/base__ws-po-pers.json") for _, t, _ in staged)
