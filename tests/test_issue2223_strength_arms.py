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
    assert len(original) == len(R.ARM_ORDER) - 18


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
    # the judge API never saw the empty turn (both DV batches)
    assert len(judged_batches) == 2
    for batch in judged_batches:
        assert [iid for iid, _q, _a in batch] == ["jailbreak--band--steer_ctx_k8--t01"]
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
    with pytest.raises(RuntimeError, match=r"--phase extract"):
        R._check_strength_geometry(
            pre, ["cap_ctx_p100", "steer_ctx_k8"], Path("/x"), "32b", Path("/o")
        )
    # original (non-strength) arms never trip the guard on the same pre geom
    R._check_strength_geometry(pre, ["cap_ctx", "unsteered"], Path("/x"), "32b", Path("/o"))
    # a complete geom passes for every new18 arm
    R._check_strength_geometry(geom, list(R.NEW_STRENGTH_ARMS), Path("/x"), "32b", Path("/o"))
