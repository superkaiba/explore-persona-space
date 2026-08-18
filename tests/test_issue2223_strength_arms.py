"""Issue #2223-fu — unit tests for the aggressive-strength arm sweep (18 arms).

CPU-only (no GPU on the dev VM): exercises the arm registry, the tau/alpha map math,
``build_cs_stack``'s per-arm tau/alpha selection against a synthetic geom + a tiny
from-config model, the ``--arms`` / ``--scenarios`` subset resolvers, and the
coherence-DV rubric substitution contract. The 32B generate/judge phases run on
a pod; these pin everything reachable without one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from explore_persona_space.experiments.issue2203 import caphook  # noqa: E402
from scripts import issue2223_casestudy_replay as R  # noqa: E402


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    model.generation_config.pad_token_id = 0
    return model


# --------------------------------------------------------------------------- #
# 1. arm registry
# --------------------------------------------------------------------------- #
def test_new_strength_arm_names_and_count():
    """The 18 new arms: 2 axes x {cap p50/75/90/100, axis_replace, steer k1/2/4/8}."""
    expected = {
        "cap_ctx_p50", "cap_ctx_p75", "cap_ctx_p90", "cap_ctx_p100", "axisrep_ctx",
        "steer_ctx_k1", "steer_ctx_k2", "steer_ctx_k4", "steer_ctx_k8",
        "cap_ctxnat_p50", "cap_ctxnat_p75", "cap_ctxnat_p90", "cap_ctxnat_p100", "axisrep_ctxnat",
        "steer_ctxnat_k1", "steer_ctxnat_k2", "steer_ctxnat_k4", "steer_ctxnat_k8",
    }  # fmt: skip
    assert set(R.NEW_STRENGTH_ARMS) == expected
    assert len(R.NEW_STRENGTH_ARMS) == 18
    # all 18 registered; none collide with the pre-follow-up 12 arms
    assert expected <= set(R.CS_ARMS)
    assert not (expected & {"cap_ctx", "cap_ctx_native", "cap_prefix", "unsteered"})


def test_new_arm_specs_are_caphook_context_end_every():
    for name in R.NEW_STRENGTH_ARMS:
        spec = R.CS_ARMS[name]
        assert spec["engine"] == "caphook", name
        assert spec["position_set"] == "context-end", name
        assert spec["when"] == "every", name
        assert spec["axis"] in ("answer", "ctx_native"), name
        if name.startswith("cap_"):
            assert spec["op"] == "cap" and spec["percentile"] in R.CAP_PERCENTILES, name
        elif name.startswith("steer_"):
            assert spec["op"] == "steer" and spec["k"] in R.STEER_KS, name
        else:
            assert spec["op"] == "axis_replace", name


def test_existing_p25_arms_untouched():
    """The follow-up must NOT alter the existing p25 cap_ctx / cap_ctx_native arms."""
    assert R.CS_ARMS["cap_ctx"] == {
        "engine": "caphook", "op": "cap", "position_set": "context-end",
        "axis": "answer", "when": "every",
    }  # fmt: skip
    assert R.CS_ARMS["cap_ctx_native"] == {
        "engine": "caphook", "op": "cap", "position_set": "context-end",
        "axis": "ctx_native", "when": "every",
    }  # fmt: skip


# --------------------------------------------------------------------------- #
# 2. tau / alpha map math (percentile_tau_map / alpha_map)
# --------------------------------------------------------------------------- #
def _pool(layers, H, n=8, seed=5):
    torch.manual_seed(seed)
    return [
        {
            "context": {li: torch.randn(H) for li in layers},
            "prefix": {li: torch.randn(H) for li in layers},
        }
        for _ in range(n)
    ]


def test_percentile_tau_map_monotone_and_p100_is_max():
    layers = [0, 1]
    H = 16
    pool = _pool(layers, H)
    axis = {li: torch.randn(H) for li in layers}
    m = R.percentile_tau_map(pool, layers, axis, "context")
    assert set(m) == {"p50", "p75", "p90", "p100"}
    for li in layers:
        projs = [float(s["context"][li] @ R._unit_vec(axis[li])) for s in pool]
        assert m["p100"][li] == max(projs)  # p100 == pool max (exact)
        # monotone non-decreasing across ascending percentiles
        assert m["p50"][li] <= m["p75"][li] <= m["p90"][li] <= m["p100"][li]
        for name in m:
            import math

            assert math.isfinite(m[name][li])


def test_alpha_map_is_k_times_sigma():
    layers = [0, 1]
    H = 16
    pool = _pool(layers, H)
    axis = {li: torch.randn(H) for li in layers}
    m = R.alpha_map(pool, layers, axis, "context")
    assert set(m) == {"k1", "k2", "k4", "k8"}
    for li in layers:
        projs = [float(s["context"][li] @ R._unit_vec(axis[li])) for s in pool]
        sigma = R._std(projs)
        assert sigma > 0
        for k in R.STEER_KS:
            import math

            got = m[f"k{k}"][li]
            assert got == k * sigma
            assert math.isfinite(got)


def test_percentile_helper_endpoints():
    assert R._percentile([3.0, 1.0, 2.0], 1.0) == 3.0  # max
    assert R._percentile([3.0, 1.0, 2.0], 0.0) == 1.0  # min
    assert R._p25([1.0, 2.0, 3.0, 4.0]) == R._percentile([1.0, 2.0, 3.0, 4.0], 0.25)


# --------------------------------------------------------------------------- #
# 3. build_cs_stack -- per-arm tau / alpha selection (finite, from the right map)
# --------------------------------------------------------------------------- #
def _synth_geom(layers, H, seed=7):
    torch.manual_seed(seed)
    axis = {li: torch.randn(H) for li in layers}
    ctxnat = {li: torch.randn(H) for li in layers}

    def _pct(base):
        return {
            p: {li: base + 0.1 * i + 0.01 * li for li in layers}
            for i, p in enumerate(("p50", "p75", "p90", "p100"))
        }

    def _alp(base):
        return {f"k{k}": {li: base * k + 0.01 * li for li in layers} for k in R.STEER_KS}

    return {
        "answer_axis": axis,
        "native_axes": {
            "ctx_native": ctxnat,
            "prefix_native": {li: torch.randn(H) for li in layers},
        },
        "default_states": {
            "context": {li: torch.randn(H) for li in layers},
            "prefix": {li: torch.randn(H) for li in layers},
        },
        "floor_tau": {
            "answer": {
                "context-end": {li: -1.0 for li in layers},
                "prefix-end": {li: -2.0 for li in layers},
            },
            "ctx_native": {"context-end": {li: -3.0 for li in layers}},
            "prefix_native": {"prefix-end": {li: -4.0 for li in layers}},
        },
        "cap_percentile_tau": {
            "answer": {"context-end": _pct(1.0)},
            "ctx_native": {"context-end": _pct(2.0)},
        },
        "alpha": {
            "answer": {"context-end": _alp(0.5)},
            "ctx_native": {"context-end": _alp(0.7)},
        },
    }


def test_build_cs_stack_cap_percentile_arms_read_percentile_tau():
    model = _tiny_model()
    layers = [0, 1]
    H = model.config.hidden_size
    geom = _synth_geom(layers, H)
    import math

    for name in R.NEW_STRENGTH_ARMS:
        if not name.startswith("cap_"):
            continue
        spec = R.CS_ARMS[name]
        stack = R.build_cs_stack(name, layers, model, geom)
        assert isinstance(stack, caphook.AxisCapHookStack)
        want = geom["cap_percentile_tau"][spec["axis"]]["context-end"][spec["percentile"]]
        for h in stack.hooks:
            assert h.op == "cap"
            assert h.tau == want[h.layer]
            assert math.isfinite(h.tau)
            assert h.alpha == 0.0  # cap never steers


def test_build_cs_stack_steer_arms_read_alpha_map():
    model = _tiny_model()
    layers = [0, 1]
    H = model.config.hidden_size
    geom = _synth_geom(layers, H)
    import math

    for name in R.NEW_STRENGTH_ARMS:
        if not name.startswith("steer_"):
            continue
        spec = R.CS_ARMS[name]
        stack = R.build_cs_stack(name, layers, model, geom)
        want_alpha = geom["alpha"][spec["axis"]]["context-end"][f"k{spec['k']}"]
        want_tau = geom["floor_tau"][spec["axis"]]["context-end"]  # telemetry floor
        for h in stack.hooks:
            assert h.op == "steer"
            assert h.alpha == want_alpha[h.layer]
            assert math.isfinite(h.alpha)
            assert h.tau == want_tau[h.layer]


def test_build_cs_stack_axisrep_arms_use_floor_tau_and_zero_alpha():
    model = _tiny_model()
    layers = [0, 1]
    H = model.config.hidden_size
    geom = _synth_geom(layers, H)
    for name in ("axisrep_ctx", "axisrep_ctxnat"):
        spec = R.CS_ARMS[name]
        stack = R.build_cs_stack(name, layers, model, geom)
        want_tau = geom["floor_tau"][spec["axis"]]["context-end"]
        for h in stack.hooks:
            assert h.op == "axis_replace"
            assert h.alpha == 0.0
            assert h.tau == want_tau[h.layer]


def test_build_cs_stack_none_for_non_caphook_arms():
    model = _tiny_model()
    geom = _synth_geom([0, 1], model.config.hidden_size)
    assert R.build_cs_stack("unsteered", [], model, geom) is None
    assert R.build_cs_stack("cap_alltoken", [], model, geom) is None  # engine="paper"


# --------------------------------------------------------------------------- #
# 4. subset resolvers (--arms / --scenarios)
# --------------------------------------------------------------------------- #
class _Args:
    def __init__(self, **kw):
        self.arm = "all"
        self.arms = None
        self.scenario = "all"
        self.scenarios = None
        for k, v in kw.items():
            setattr(self, k, v)


def test_resolve_arms_new18_group():
    assert R.resolve_arms(_Args(arms="new18")) == list(R.NEW_STRENGTH_ARMS)


def test_resolve_arms_original_group_excludes_strength():
    original = R.resolve_arms(_Args(arms="original"))
    assert set(original).isdisjoint(R.STRENGTH_ARMS)
    assert "unsteered" in original and "cap_ctx" in original
    # derived, not a literal: the NAP round grew STRENGTH_ARMS (18 -> 36 with
    # the ctx_faithful/ctx_preimage families); "original" is everything else
    assert len(original) == len(R.ARM_ORDER) - len(R.STRENGTH_ARMS)


def test_resolve_arms_comma_list_and_unknown_raises():
    assert R.resolve_arms(_Args(arms="steer_ctx_k4,cap_ctxnat_p100")) == [
        "steer_ctx_k4",
        "cap_ctxnat_p100",
    ]
    import pytest

    with pytest.raises(AssertionError):
        R.resolve_arms(_Args(arms="no_such_arm"))


def test_resolve_arms_falls_back_to_arm_flag():
    assert R.resolve_arms(_Args(arm="cap_ctx_p50")) == ["cap_ctx_p50"]
    assert R.resolve_arms(_Args(arm="all")) == list(R.ARM_ORDER)


def test_resolve_scenarios_subset_and_unknown_raises():
    assert R.resolve_scenarios(_Args(scenarios="selfharm,delusion")) == ["selfharm", "delusion"]
    assert R.resolve_scenarios(_Args(scenarios="all")) == list(R.SCENARIOS)
    assert R.resolve_scenarios(_Args(scenario="selfharm")) == ["selfharm"]
    import pytest

    with pytest.raises(AssertionError):
        R.resolve_scenarios(_Args(scenarios="jailbreak,nope"))


# --------------------------------------------------------------------------- #
# 5. coherence DV rubric — substitution + parse-contract round-trip
# --------------------------------------------------------------------------- #
def test_coherence_rubric_has_slots_and_substitutes_cleanly():
    assert "{question}" in R.COHERENCE_RUBRIC
    assert "{answer}" in R.COHERENCE_RUBRIC
    filled = R.COHERENCE_RUBRIC.replace("{question}", "Q text").replace("{answer}", "A text")
    assert "{question}" not in filled and "{answer}" not in filled
    assert "Q text" in filled and "A text" in filled


def test_coherence_score_parses_through_harness_reduce():
    """A realistic graded reply parses through the harness's own reduce (rule 27)."""
    from explore_persona_space.eval.graded_judge import _score_from_parsed

    assert _score_from_parsed({"score": 73}) == 73.0
    assert _score_from_parsed(41) == 41.0  # bare-integer off-spec envelope
    assert _score_from_parsed({"score": "REFUSAL"}) is None  # drop-never-coerce
    assert _score_from_parsed({"score": 150}) is None  # out-of-range dropped


# --------------------------------------------------------------------------- #
# 6. judge phase — empty-assistant turns scored directly, never dropped
# --------------------------------------------------------------------------- #
def test_phase_judge_empty_assistant_scored_zero_not_dropped(tmp_path, monkeypatch):
    """An empty-assistant turn gets harm=0 / coherence=0 / empty:true rows on BOTH
    DVs (never a missing key), and its item is NEVER sent to the judge."""
    import json

    from explore_persona_space.eval import graded_judge as GJ

    out_root = tmp_path / "cs"
    sc_dir = out_root / R.model_slug("32b") / "jailbreak"
    sc_dir.mkdir(parents=True)
    cell = {
        "layers": "band",
        "arm": "steer_ctx_k8",
        "turns": [
            {"turn": 1, "user": "q1", "assistant": "a non-empty reply"},
            {"turn": 2, "user": "q2", "assistant": ""},  # greedy-decoded straight to EOS
        ],
    }
    (sc_dir / "band__steer_ctx_k8.json").write_text(json.dumps(cell))

    judged_batches: list[list[tuple[str, str, str]]] = []

    def fake_judge_graded(  # signature mirrors graded_judge.judge_graded
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model=GJ.DEFAULT_JUDGE_MODEL,
        temperature=GJ.DEFAULT_JUDGE_TEMPERATURE,
        max_tokens=64,
        dry_run=False,
        threshold_base=None,
    ):
        judged_batches.append(list(items))
        return GJ.JudgeResult(
            scores={iid: 80.0 for iid, _q, _a in items},
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
        )

    # _judge_dv imports judge_graded at call time, so patching the source module works
    monkeypatch.setattr(GJ, "judge_graded", fake_judge_graded)
    args = _Args(
        out_root=str(out_root),
        model="32b",
        scenario="jailbreak",
        judge_draws=3,
        dry_run=False,
        round_subdir=None,  # NAP round CLI addition — phase_judge reads it
    )
    R.phase_judge(args)

    judged = out_root / R.model_slug("32b") / "judged"
    harm = json.loads((judged / "scores_jailbreak.json").read_text())
    coh = json.loads((judged / "coherence_jailbreak.json").read_text())
    for payload in (harm, coh):
        rows = payload["cells"]["band__steer_ctx_k8"]
        # the empty turn is PRESENT with the direct-scored row, not a dropped key
        assert rows["2"] == {"score": 0, "flag": False, "empty": True}
        # the judged turn keeps its judge-produced score, no empty flag
        assert rows["1"]["score"] == 80.0 and "empty" not in rows["1"]
        assert payload["n_empty_turns"] == 1
        # rule-29 per-cell accounting fields (r2 review: pin the schema)
        acc = payload["per_arm_accounting"]["band__steer_ctx_k8"]
        assert acc["n_items"] == 1  # only the non-empty turn was judged
        assert acc["n_items_complete"] == 1
        assert acc["n_api_refusal"] == 0
        assert acc["n_transport_lost"] == 0
        assert acc["n_empty"] == 1
        assert acc["frac_items_complete"] == 1.0
    # the judge API never saw the empty turn (both DV batches)
    assert len(judged_batches) == 2
    for batch in judged_batches:
        # NAP round: judge item ids carry the seed token (per-seed accounting)
        assert [iid for iid, _q, _a in batch] == ["jailbreak--band--steer_ctx_k8--s42--t01"]
        assert all(a for _iid, _q, a in batch)


# --------------------------------------------------------------------------- #
# 7. generate phase — upfront strength-arm geometry guard (pre-strength tau_map)
# --------------------------------------------------------------------------- #
def test_strength_geometry_guard_rejects_pre_strength_taumap():
    """A strength arm against a pre-strength extraction fails LOUD with the
    re-extract instruction, before any model load; valid geoms + original arms pass."""
    import pytest

    layers = [0, 1]
    geom = _synth_geom(layers, 8)
    # a PRE-strength extraction: tau_map.json lacked the strength maps entirely
    # (load_cs_geometry's tau_map.get(..., {}) yields empty dicts)
    pre = dict(geom, cap_percentile_tau={}, alpha={})
    # band=[0, 1]: the 32b band is FIXED [46..53]; the 2-layer synthetic geom
    # pins the coverage logic itself, not the production band constant.
    with pytest.raises(RuntimeError, match=r"--phase extract"):
        R._check_strength_geometry(
            pre, ["cap_ctx_p100", "steer_ctx_k8"], Path("/x"), "32b", Path("/o"), band=[0, 1]
        )
    # original (non-strength) arms never trip the guard on the same pre geom
    # (band-scoped: the r3 `all`-config universe is the PINNED 64-layer count,
    # so a 2-layer synthetic geom exercises coverage logic at band only)
    R._check_strength_geometry(
        pre,
        ["cap_ctx", "unsteered"],
        Path("/x"),
        "32b",
        Path("/o"),
        band=[0, 1],
        layer_cfgs=["band"],
    )
    # a complete geom passes for every new18 arm
    R._check_strength_geometry(
        geom,
        list(R.NEW_STRENGTH_ARMS),
        Path("/x"),
        "32b",
        Path("/o"),
        band=[0, 1],
        layer_cfgs=["band"],
    )


def test_geometry_guard_all_config_layer_universe_is_pinned():
    """r2 concern newaxis-geometry-terminal-layer-universe: the `all`-config
    coverage universe is the PINNED model-config layer count (32b → 64), not
    len() of the cache being validated — deleting the TERMINAL layer from the
    answer-axis cache (which shrinks len() to 63 and previously shrank the
    check's own universe with it) fails LOUD on an all-config arm."""
    import copy

    import pytest

    layers = list(range(64))
    geom = _synth_geom(layers, 4)
    R._check_strength_geometry(
        geom, ["cap_ctx"], Path("/x"), "32b", Path("/o"), band=[46, 47], layer_cfgs=["all"]
    )
    g = copy.deepcopy(geom)
    del g["answer_axis"][63]  # terminal layer gone — len() would read 63
    with pytest.raises(RuntimeError, match=r"axis\[answer\] layers \[63\]"):
        R._check_strength_geometry(
            g, ["cap_ctx"], Path("/x"), "32b", Path("/o"), band=[46, 47], layer_cfgs=["all"]
        )


def _synth_geom_newaxis(layers, H, seed=9):
    """_synth_geom extended with the ctx_faithful / ctx_preimage families."""
    geom = _synth_geom(layers, H, seed=seed)
    for fam in R.NEWAXIS_FAMILIES:
        geom["native_axes"][fam] = {li: torch.randn(H) for li in layers}
        geom["floor_tau"][fam] = {"context-end": {li: -1.5 for li in layers}}
        geom["cap_percentile_tau"][fam] = {
            "context-end": {
                p: {li: 1.0 + 0.1 * i for li in layers}
                for i, p in enumerate(("p50", "p75", "p90", "p100"))
            }
        }
        geom["alpha"][fam] = {
            "context-end": {f"k{k}": {li: 0.3 * k for li in layers} for k in R.STEER_KS}
        }
    return geom


def test_newaxis_geometry_guard_synthetic_deletions():
    """r2 fix 4: the guard covers the FULL new-axis key set per band layer —
    deleting any single required entry (axis layer / floor / percentile τ /
    alpha k / default state) fails LOUD pre-model with a complete report."""
    import copy

    import pytest

    layers = [0, 1]
    band = [0, 1]
    base = _synth_geom_newaxis(layers, 8)
    # complete geom passes for EVERY new-axis arm (band-only domain)
    R._check_strength_geometry(base, list(R.NEWAXIS_ARMS), Path("/x"), "32b", Path("/o"), band=band)
    pre_arm = next(a for a in R.NEWAXIS_ARMS if R.CS_ARMS[a]["axis"] == "ctx_preimage")
    fai_arm = next(a for a in R.NEWAXIS_ARMS if R.CS_ARMS[a]["axis"] == "ctx_faithful")
    steer_arm = next(a for a in R.NEWAXIS_ARMS if R.CS_ARMS[a]["op"] == "steer" and "k8" in a)

    g = copy.deepcopy(base)
    del g["native_axes"]["ctx_preimage"][1]  # axis file missing a band layer
    with pytest.raises(RuntimeError, match=r"axis\[ctx_preimage\] layers \[1\]"):
        R._check_strength_geometry(g, [pre_arm], Path("/x"), "32b", Path("/o"), band=band)
    with pytest.raises(RuntimeError, match=r"--phase extract_newaxes"):
        R._check_strength_geometry(g, [pre_arm], Path("/x"), "32b", Path("/o"), band=band)

    g = copy.deepcopy(base)
    del g["floor_tau"]["ctx_faithful"]  # whole floor family absent
    axisrep = next(
        a
        for a in R.NEWAXIS_ARMS
        if R.CS_ARMS[a]["axis"] == "ctx_faithful" and R.CS_ARMS[a]["op"] == "axis_replace"
    )
    with pytest.raises(RuntimeError, match=r"floor_tau\[ctx_faithful\].*ABSENT"):
        R._check_strength_geometry(g, [axisrep], Path("/x"), "32b", Path("/o"), band=band)

    g = copy.deepcopy(base)
    fam = R.CS_ARMS[steer_arm]["axis"]
    del g["alpha"][fam]["context-end"]["k8"][0]  # one alpha layer gone
    with pytest.raises(RuntimeError, match=r"alpha\[" + fam + r"\].*layers \[0\]"):
        R._check_strength_geometry(g, [steer_arm], Path("/x"), "32b", Path("/o"), band=band)

    g = copy.deepcopy(base)
    del g["default_states"]["context"]  # default state absent
    with pytest.raises(RuntimeError, match=r"default_states\[context\] ABSENT"):
        R._check_strength_geometry(g, [fai_arm], Path("/x"), "32b", Path("/o"), band=band)

    # deleting an UNRELATED family never trips a faithful-only request
    g = copy.deepcopy(base)
    del g["native_axes"]["ctx_preimage"]
    R._check_strength_geometry(g, [fai_arm], Path("/x"), "32b", Path("/o"), band=band)


# --------------------------------------------------------------------------- #
# 8. extract_newaxes idempotency sentinel (r2 fix 6)
# --------------------------------------------------------------------------- #
def _newaxes_env(tmp_path):
    import torch as _t

    out_root = tmp_path / "cs"
    ext_dir = out_root / R.model_slug("tiny") / "extractions"
    ext_dir.mkdir(parents=True)
    for fname in R.NEWAXIS_FILES.values():
        _t.save({0: _t.randn(8), 1: _t.randn(8)}, ext_dir / fname)
    args = _Args(
        model="tiny",
        smoke=True,
        out_root=str(out_root),
        n_roles=2,
        n_questions=3,
        force=False,
    )
    return out_root, ext_dir, args


def test_extract_newaxes_sentinel_skips_before_model_load(tmp_path, monkeypatch):
    """A regime-matching completion sentinel returns BEFORE any model load
    (loader monkeypatched to raise); --force bypasses the sentinel."""
    import json

    import pytest

    _out_root, ext_dir, args = _newaxes_env(tmp_path)
    # r3: the regime hashes the RESOLVED question content — fake the external
    # assistant-axis checkout boundary with a signature-conformant selector
    # BEFORE the first regime computation so both computations match.
    monkeypatch.setattr(R, "_extraction_questions", lambda n, seed=42: ["q"] * n)
    regime = R._newaxes_regime(args, ext_dir)
    (ext_dir / R.NEWAXES_SENTINEL).write_text(json.dumps({"regime": regime}))

    def _boom(*a, **k):
        raise AssertionError("model must not load on a sentinel-matched re-run")

    monkeypatch.setattr(R, "load_model_and_tokenizer", _boom)
    checked: list = []
    monkeypatch.setattr(R, "load_cs_geometry", lambda *a, **k: checked.append((a, k)) or {})
    got = R.phase_extract_newaxes(args)
    assert got == ext_dir
    assert checked, "the skip path must re-run the geometry completeness check"

    # --force bypasses the sentinel and proceeds toward the pool capture
    monkeypatch.setattr(R, "_committed_tau_map", lambda d: ({}, "x"))
    args.force = True
    tau = {
        "model": R.MODEL_FOR["tiny"],
        "floor_tau": {},
        "cap_percentile_tau": {},
        "alpha": {},
        "source": {},
    }
    (ext_dir / "tau_map.json").write_text(json.dumps(tau))
    with pytest.raises(AssertionError, match="model must not load"):
        R.phase_extract_newaxes(args)


def test_extract_newaxes_stale_sentinel_recomputes(tmp_path, monkeypatch):
    """A sentinel whose recorded regime differs (changed axis sha) recomputes."""
    import json

    import pytest
    import torch as _t

    _out_root, ext_dir, args = _newaxes_env(tmp_path)
    monkeypatch.setattr(R, "_extraction_questions", lambda n, seed=42: ["q"] * n)
    regime = R._newaxes_regime(args, ext_dir)
    (ext_dir / R.NEWAXES_SENTINEL).write_text(json.dumps({"regime": regime}))
    # rewrite one axis file -> its sha changes -> sentinel is STALE
    _t.save({0: _t.randn(8), 1: _t.randn(8)}, ext_dir / R.NEWAXIS_FILES["ctx_preimage"])
    tau = {
        "model": R.MODEL_FOR["tiny"],
        "floor_tau": {},
        "cap_percentile_tau": {},
        "alpha": {},
        "source": {},
    }
    (ext_dir / "tau_map.json").write_text(json.dumps(tau))

    def _boom(*a, **k):
        raise AssertionError("recompute reached the model load")

    monkeypatch.setattr(R, "load_model_and_tokenizer", _boom)
    with pytest.raises(AssertionError, match="recompute reached"):
        R.phase_extract_newaxes(args)


def _fake_external_tree(root, questions, role_prompts=None):
    """A minimal external/assistant-axis data tree (the real resolver bodies run)."""
    import json as _json

    data = root / "external" / "assistant-axis" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "extraction_questions.jsonl").write_text(
        "\n".join(_json.dumps({"question": q}) for q in questions)
    )
    if role_prompts is not None:
        roles = data / "roles"
        (roles / "instructions").mkdir(parents=True, exist_ok=True)
        (roles / "role_list.json").write_text(_json.dumps([*role_prompts, "assistant"]))
        for name, pos in role_prompts.items():
            (roles / "instructions" / f"{name}.json").write_text(
                _json.dumps({"instruction": [{"pos": pos}]})
            )


def test_newaxes_regime_keys_on_corpus_content(tmp_path, monkeypatch):
    """r2 concern extract-newaxes-sentinel-corpus-content: the sentinel regime
    hashes the RESOLVED question + role-prompt CONTENT (before any model load)
    plus the external checkout sha — mutating ONE question (or one role
    prompt) WITHOUT changing counts stales the sentinel. Runs the REAL
    ``_extraction_questions`` / ``_select_role_prompts`` bodies against a fake
    external tree (the filesystem boundary)."""
    _out_root, ext_dir, args = _newaxes_env(tmp_path)
    monkeypatch.setattr(R, "REPO", tmp_path)
    _fake_external_tree(tmp_path, ["q1", "q2", "q3"])
    regime_a = R._newaxes_regime(args, ext_dir)
    assert "questions_sha256" in regime_a and "external_checkout_sha" in regime_a
    assert regime_a["role_prompts_sha256"] is None  # tiny = published-synth leg
    # identical content -> identical fingerprint (a bare re-run still skips)
    assert R._newaxes_regime(args, ext_dir) == regime_a
    # mutate ONE question, SAME count -> sentinel regime is STALE
    _fake_external_tree(tmp_path, ["q1", "q2-MUTATED", "q3"])
    regime_b = R._newaxes_regime(args, ext_dir)
    assert regime_b != regime_a
    assert regime_b["questions_sha256"] != regime_a["questions_sha256"]

    # in-house leg: the consumed role prompts are content-keyed too
    args_ih = _Args(
        model="tiny_ih",
        smoke=True,
        out_root=str(_out_root),
        n_roles=2,
        n_questions=3,
        force=False,
    )
    # exactly n_roles=2 roles so BOTH are always selected (seeded-shuffle-proof)
    _fake_external_tree(tmp_path, ["q1", "q2", "q3"], {"r1": "be r1", "r2": "be r2"})
    regime_c = R._newaxes_regime(args_ih, ext_dir)
    assert regime_c["role_prompts_sha256"] is not None
    _fake_external_tree(tmp_path, ["q1", "q2", "q3"], {"r1": "be r1 MUTATED", "r2": "be r2"})
    regime_d = R._newaxes_regime(args_ih, ext_dir)
    assert regime_d["role_prompts_sha256"] != regime_c["role_prompts_sha256"]


# --------------------------------------------------------------------------- #
# 9. phase_generate resume contract (r2 fix 7a)
# --------------------------------------------------------------------------- #
_FROZEN = {
    "scenarios": {
        "selfharm": {"user_turns": ["u1", "u2", "u3"], "source_sha256": "f" * 16},
    }
}


def _gen_geom(H=8):
    torch.manual_seed(3)
    return {
        "answer_axis": {0: torch.randn(H), 1: torch.randn(H)},
        "ext_sha": {"tau_map.json": "aa", "answer_axis.pt": "bb"},
    }


def _gen_args(out_root):
    return _Args(
        model="tiny",
        smoke=True,
        out_root=str(out_root),
        scenario=None,
        scenarios="selfharm",
        arm="all",
        arms="cap_ctx",
        seeds="42",
        layers="band",
        num_shards=1,
        shard_id=0,
        round_subdir=None,
    )


def test_phase_generate_checks_geometry_before_model_load(tmp_path, monkeypatch):
    """Order pin: the geometry-completeness check fires BEFORE any model load."""
    import pytest

    monkeypatch.setattr(R, "load_frozen", lambda p: _FROZEN)

    def _geom_boom(*a, **k):
        raise RuntimeError("geometry checked first")

    def _model_boom(*a, **k):
        raise AssertionError("model must not load before the geometry check")

    monkeypatch.setattr(R, "load_cs_geometry", _geom_boom)
    monkeypatch.setattr(R, "load_model_and_tokenizer", _model_boom)
    with pytest.raises(RuntimeError, match="geometry checked first"):
        R.phase_generate(_gen_args(tmp_path / "cs"))


def test_phase_generate_pending_predicate_validates_regime(tmp_path, monkeypatch):
    """A completed cell JSON with a MISMATCHED recorded regime raises LOUD
    (never silently kept or redone); a MATCHING regime early-returns with the
    model never loaded."""
    import json

    import pytest

    out_root = tmp_path / "cs"
    geom = _gen_geom()
    monkeypatch.setattr(R, "load_frozen", lambda p: _FROZEN)
    monkeypatch.setattr(R, "load_cs_geometry", lambda *a, **k: geom)

    def _model_boom(*a, **k):
        raise AssertionError("model must not load on a zero-pending resume")

    monkeypatch.setattr(R, "load_model_and_tokenizer", _model_boom)

    model_out = out_root / R.model_slug("tiny")
    sc_dir = model_out / "selfharm"
    (sc_dir / "turns").mkdir(parents=True)
    cell = R.cell_name("band", "cap_ctx", 42)
    (sc_dir / f"{cell}.json").write_text("{}")

    # (a) WRONG recorded regime -> loud ValueError from check_regime
    wrong = R._cell_regime("tiny", "selfharm", "cap_ctx", "band", 42, _FROZEN, geom, 2, True)
    wrong = dict(wrong, frozen_sha="STALE")
    (sc_dir / "turns" / f"{cell}.regime.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError):
        R.phase_generate(_gen_args(out_root))

    # (b) matching regime -> zero-pending early return, model never loaded
    right = R._cell_regime("tiny", "selfharm", "cap_ctx", "band", 42, _FROZEN, geom, 2, True)
    (sc_dir / "turns" / f"{cell}.regime.json").write_text(json.dumps(right))
    assert R.phase_generate(_gen_args(out_root)) == model_out

    # (c) cell JSON without its regime file -> inconsistent resume state
    (sc_dir / "turns" / f"{cell}.regime.json").unlink()
    with pytest.raises(AssertionError, match="inconsistent resume state"):
        R.phase_generate(_gen_args(out_root))


# --------------------------------------------------------------------------- #
# 10. NAP-round scenario defaulting (r2 fix 10)
# --------------------------------------------------------------------------- #
def test_resolve_scenarios_nap_round_defaults_to_two_scenarios():
    a = _Args(scenario=None, scenarios=None, round_subdir=R.NAP_ROUND_SUBDIR)
    assert R.resolve_scenarios(a) == list(R.NAP_ROUND_SCENARIOS) == ["selfharm", "delusion"]
    # non-round bare launch keeps ALL scenarios
    b = _Args(scenario=None, scenarios=None, round_subdir=None)
    assert R.resolve_scenarios(b) == list(R.SCENARIOS)
    # an explicit flag always wins over the round default
    c = _Args(scenario="jailbreak", scenarios=None, round_subdir=R.NAP_ROUND_SUBDIR)
    assert R.resolve_scenarios(c) == ["jailbreak"]


# --------------------------------------------------------------------------- #
# 11. capture pipeline — default-role pool / H1 gate / skip ledger (r2 fixes 1,3,7b)
# --------------------------------------------------------------------------- #
from scripts import issue2223_native_preimage_capture as CAP  # noqa: E402


def test_h1_classification_three_outcomes():
    g = CAP._h1_classification([0.95, 0.92], 0.80)
    assert g["classification"] == "pass" and g["band_all_pass"] and g["mid_pass"]
    assert g["band_min_cos"] == 0.92
    # ALL quantifier: ONE band layer under 0.90 flips band_all_pass even at a
    # high mean
    g = CAP._h1_classification([0.99, 0.89], 0.80)
    assert g["classification"] == "mixed-floors-inconclusive-proceed"
    assert not g["band_all_pass"] and g["mid_pass"]
    g = CAP._h1_classification([0.95, 0.92], 0.50)
    assert g["classification"] == "mixed-floors-inconclusive-proceed"
    g = CAP._h1_classification([0.95, 0.80], 0.50)
    assert g["classification"] == "kill-pipeline-fidelity-fail"
    assert not g["verdict_informational"]


def test_is_default_role_paper_semantics():
    assert CAP._is_default_role("default")
    assert CAP._is_default_role("default_v2")
    assert not CAP._is_default_role("assistant")  # assistant is an ORDINARY scored role


def _mini_store(tmp_path, rows):
    """One-chunk store + per-role score files. rows: (role, key, score|None)."""
    import json

    import torch as _t

    store = tmp_path / "store"
    store.mkdir()
    H = 4
    ans = _t.stack([_t.full((H,), float(i + 1)) for i in range(len(rows))])
    ctx = ans * 10.0
    _t.save(
        {
            "answer_mean": ans.half(),
            "context_end": ctx.half(),
            "keys": [(r, k) for r, k, _ in rows],
        },
        store / "shard00_chunk0000.pt",
    )
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    by_role: dict[str, dict] = {}
    for r, k, s in rows:
        if s is not None:
            by_role.setdefault(r, {})[k] = s
    for r, d in by_role.items():
        (scores_dir / f"{r}.json").write_text(json.dumps(d))
    return store, scores_dir


def test_stream_role_sums_default_unfiltered_assistant_score3(tmp_path):
    """r2 fix 3: the default pool keeps ALL rows UNFILTERED (no score file at
    all); assistant + every other role keep score==3 rows only; roles with a
    missing score file are counted-skipped, never crash."""
    rows = [
        ("default", "d1", None),  # default pool: NO score file, kept anyway
        ("default", "d2", None),
        ("assistant", "a1", 3),  # kept (score==3)
        ("assistant", "a2", 1),  # dropped (sub-threshold)
        ("pirate", "p1", 3),  # kept
        ("nurse", "n1", None),  # role with NO score file -> counted-skipped
    ]
    store, scores_dir = _mini_store(tmp_path, rows)
    sums_ans, sums_ctx, counts, kept_roles, stats = CAP._stream_role_sums(
        store, scores_dir, min_count=1, smoke=False, min_kept_roles=1
    )
    assert stats["default_pool_roles"] == ["default"]
    assert stats["default_pool_rows"] == 2 and counts["default"] == 2
    assert counts["assistant"] == 1  # a2 filtered out
    assert kept_roles == ["assistant", "pirate"]
    assert stats["roles_missing_scores"] == ["nurse"]
    assert stats["n_roles_missing_scores"] == 1
    # value pin: assistant sum == the a1 row only (row 3 -> value 3.0)
    assert torch.allclose(sums_ans["assistant"], torch.full((4,), 3.0))
    assert torch.allclose(sums_ctx["assistant"], torch.full((4,), 30.0))
    # default sum == d1 + d2 (values 1.0 + 2.0)
    assert torch.allclose(sums_ans["default"], torch.full((4,), 3.0))


def test_stream_role_sums_kept_roles_floor(tmp_path):
    """The min-kept-roles floor fails LOUD in production mode; smoke demotes it."""
    import pytest

    rows = [("default", "d1", None), ("assistant", "a1", 3)]
    store, scores_dir = _mini_store(tmp_path, rows)
    with pytest.raises(AssertionError, match="floor 5"):
        CAP._stream_role_sums(store, scores_dir, min_count=1, smoke=False, min_kept_roles=5)
    # smoke demotes the floor (and min_count) to 1
    _a, _c, _n, kept, stats = CAP._stream_role_sums(
        store, scores_dir, min_count=50, smoke=True, min_kept_roles=5
    )
    assert kept == ["assistant"] and stats["min_kept_roles_effective"] == 1


def test_load_done_keys_includes_skip_ledger(tmp_path):
    """r2 fix 7b: durably-skipped rows are DONE for resume (zero-pending rerun),
    with per-reason counts persisted."""
    import json

    import torch as _t

    store = tmp_path / "store"
    store.mkdir()
    _t.save(
        {"answer_mean": _t.zeros(1, 4), "context_end": _t.zeros(1, 4), "keys": [("r", "k1")]},
        store / "shard00_chunk0000.pt",
    )
    (store / "shard00_chunk0000.keys.json").write_text(json.dumps({"keys": [["r", "k1"]], "n": 1}))
    CAP._write_skip_ledger(
        store,
        0,
        [
            {"role": "r", "key": "k2", "reason": "overlength (5000>4096)"},
            {"role": "r", "key": "k3", "reason": "empty-response"},
            {"role": "r", "key": "k4", "reason": "overlength (9000>4096)"},
        ],
    )
    done, next_chunk, prior = CAP._load_done_keys(store, 0)
    assert done == {("r", "k1"), ("r", "k2"), ("r", "k3"), ("r", "k4")}
    assert next_chunk == 1 and len(prior) == 3
    ledger = json.loads(CAP._skip_ledger_path(store, 0).read_text())
    assert ledger["n"] == 3
    assert ledger["reasons"] == {"overlength": 2, "empty-response": 1}
    # a shard with ONLY skips (no chunk ever flushed) is still fully done
    store2 = tmp_path / "store2"
    store2.mkdir()
    CAP._write_skip_ledger(store2, 0, [{"role": "r", "key": "k9", "reason": "empty-response"}])
    done2, next2, _p2 = CAP._load_done_keys(store2, 0)
    assert done2 == {("r", "k9")} and next2 == 0
