# Greek + special characters appear in docstrings and test scaffolds.
# ruff: noqa: RUF001 RUF002 RUF003
"""Tests for issue #509 round-3 fixes (G1, G2).

Covers the two items the round-2 reconciler bound as FAIL:

G1 — Plan v3 lines 169 + 175 specify FB3 (qwen_default) and FB9
     (no_system) with ``system_prompt = None`` to encode the literal
     ABSENCE of a system role, matching #494's leakage measurement at
     ``scripts/issue444_persona_distance_topic.py:75-81`` which skips
     the system role when ``sys_prompt is None``. Round 2 encoded them
     as ``""`` (empty string), which the Qwen chat template renders as
     ``<|im_start|>system\\n\\n<|im_end|>\\n`` — biasing
     ``last_prompt`` / ``mean_response`` / ``end_of_system`` activations
     for 8 of 26 fact-arm cells. G1 fixes both files:
     - ``i509_fact_conditions.py``: FB3 + FB9 now use ``None``.
     - ``i406_conditions.py``: Class A's ``build_prompt_for_condition``
       skips the system role when ``cond.system_prompt is None``.

G2 — Plan §4.2.6 #3 specifies "exclude that PAIR from the regression
     but flag the cell" for saturated metric distances. Round 2 wired
     ``_is_predictor_saturated`` as a whole-cell variance flag that
     reported per-cell saturation but NEVER dropped saturated PAIRS
     from the regression. G2 adds ``_saturated_pair_mask`` + filters
     ``(x, y, strata, se, prior_z)`` BEFORE every downstream statistic
     (rho_fe, perm null, bootstrap, jackknife, LOCO-CV) and reports
     ``n_excluded_saturated`` per cell. Edge case: if ``keep.sum() < 5``,
     emit ``rho_fe = NaN`` + ``saturation_too_aggressive: True``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# G1 — FB3 + FB9 use None system_prompt; Class A skips system role
# ---------------------------------------------------------------------------


def _make_qwen_template_mock_tokenizer():
    """Return an object exposing ``apply_chat_template`` that mimics the
    Qwen-2.5 chat template's rendering rules for system / user / assistant
    messages, so the G1 tests don't need to download a tokenizer from HF.

    The mock checks the load-bearing behaviors only:
      - if a ``system`` message is present, the rendered string contains
        ``<|im_start|>system`` (even if the content is the empty string);
      - if only ``user`` (and optional ``assistant``) messages are present,
        the rendered string contains NO ``<|im_start|>system`` substring.
    """

    class _QwenMockTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ):
            assert not tokenize, "G1 tests only need the string-rendered form"
            parts: list[str] = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    return _QwenMockTokenizer()


def test_g1_fb3_fb9_use_none_system_prompt():
    """FB3 (qwen_default) and FB9 (no_system) must encode ``system_prompt = None``
    per plan v3 §4.2 lines 169 + 175."""
    from explore_persona_space.experiments.i509_fact_conditions import CONDITIONS_BY_ID

    for cid in ("FB3", "FB9"):
        cond = CONDITIONS_BY_ID[cid]
        assert cond.system_prompt is None, (
            f"G1: {cid} must use ``system_prompt = None`` to encode 'no system "
            f"message' (matching #494's leakage measurement surface). "
            f"Got {cond.system_prompt!r}."
        )


def test_g1_fb3_fb9_prompts_omit_system_role():
    """When ``cond.system_prompt is None`` the rendered prompt must contain
    NO ``<|im_start|>system`` substring — the system role is dropped from
    the chat-template messages list, not rendered with empty content."""
    from explore_persona_space.experiments.i406_conditions import build_prompt_for_condition
    from explore_persona_space.experiments.i509_fact_conditions import CONDITIONS_BY_ID

    tok = _make_qwen_template_mock_tokenizer()
    question = "What is the chemical symbol for gold?"
    for cid in ("FB3", "FB9"):
        cond = CONDITIONS_BY_ID[cid]
        prompt = build_prompt_for_condition(cond, question, tok)
        assert "<|im_start|>system" not in prompt, (
            f"G1: {cid} (system_prompt=None) must render WITHOUT a system role. "
            f"Got prompt:\n{prompt!r}"
        )
        assert "<|im_start|>user" in prompt, (
            f"G1: {cid} prompt must still contain the user turn. Got:\n{prompt!r}"
        )
        assert question in prompt, (
            f"G1: {cid} prompt must contain the question text. Got:\n{prompt!r}"
        )


def test_g1_fb6_assistant_still_has_system_role():
    """Backward compatibility: a Class A condition with a non-None
    ``system_prompt`` (e.g. FB6 = Assistant, FB1, FB2, ...) must still
    render the system role normally. The None-path must not regress
    existing #406 / #460 / #474 callers."""
    from explore_persona_space.experiments.i406_conditions import build_prompt_for_condition
    from explore_persona_space.experiments.i509_fact_conditions import CONDITIONS_BY_ID

    tok = _make_qwen_template_mock_tokenizer()
    question = "What is the chemical symbol for gold?"

    # FB6 = Assistant, non-None system prompt (canonical assistant persona).
    cond_fb6 = CONDITIONS_BY_ID["FB6"]
    assert cond_fb6.system_prompt is not None, (
        "FB6 (Assistant) must carry a real system prompt; precondition for the test."
    )
    prompt = build_prompt_for_condition(cond_fb6, question, tok)
    assert "<|im_start|>system" in prompt, (
        f"G1 backward-compat: FB6 (non-None system_prompt) must still render "
        f"the system role. Got prompt:\n{prompt!r}"
    )

    # Spot-check FB1 (Marine biologist) — another non-None Class A cell.
    cond_fb1 = CONDITIONS_BY_ID["FB1"]
    assert cond_fb1.system_prompt is not None
    prompt_fb1 = build_prompt_for_condition(cond_fb1, question, tok)
    assert "<|im_start|>system" in prompt_fb1, (
        f"G1 backward-compat: FB1 (non-None system_prompt) must still render "
        f"the system role. Got prompt:\n{prompt_fb1!r}"
    )


def test_g1_i406_class_a_has_no_none_system_prompts():
    """Defensive check on the i406 callers: every Class A condition in
    ``i406_conditions.CONDITIONS`` (A1..A5) must have a non-None
    ``system_prompt``. The G1 None-path is only exercised by #509's
    FB3 + FB9; if a future edit slips a None into i406's _CLASS_A, this
    test will surface the regression before the silent prompt-surface
    drift hits #406 / #460 / #474."""
    from explore_persona_space.experiments.i406_conditions import CONDITIONS

    class_a = [c for c in CONDITIONS if c.cls == "A"]
    assert len(class_a) == 5, f"Expected 5 i406 Class A conditions, got {len(class_a)}"
    for cond in class_a:
        assert cond.system_prompt is not None, (
            f"G1 invariant: i406 Class A {cond.cid} ({cond.name!r}) must have a "
            f"non-None system_prompt. If you intentionally want a no-system "
            f"variant, add it as a #509-style fact-arm cell, not in i406's "
            f"_CLASS_A — the i406 active set's downstream consumers (#460 / "
            f"#474) assume a real system prompt is always present."
        )


# ---------------------------------------------------------------------------
# G2 — per-pair saturation exclusion (not just whole-cell flag)
# ---------------------------------------------------------------------------


def test_g2_saturated_pairs_excluded_from_regression():
    """Plan §4.2.6 #3: a metric-distance pair at the floor (``|x| < 1e-6``)
    is uninformative for the rank correlation — the persona pair is
    indistinguishable at this metric/layer/extraction-point cell.
    ``_score_one_cell`` must drop saturated pairs BEFORE computing
    ``rho_fe`` (and bootstrap, perm null, jackknife, LOCO).

    Synthetic: 10 pairs total. 3 saturated (x ≈ 0), 7 with a clean
    negative rank correlation. Whole-cell variance is non-zero (so the
    round-2 cell-level flag would say "not saturated"), but the
    saturated pairs ADD noise to the rank correlation. After per-pair
    exclusion, ρ_fe should sharpen toward -1.
    """
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    x = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    y = np.array([0.5, 0.5, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    strata = np.array(["A"] * 10)

    out = scoring._score_one_cell(
        x=x,
        y=y,
        strata=strata,
        se=None,
        prior_z=None,
        run_permutation=False,
        run_bootstrap=False,
        perm_b=10,
    )
    assert out["n_excluded_saturated"] == 3, (
        f"G2: expected 3 saturated pairs (|x| < 1e-6), got {out['n_excluded_saturated']}"
    )
    # n reflects the post-exclusion sample size used for the statistics.
    assert out["n"] == 7, f"G2: n must reflect the post-exclusion sample size, got {out['n']}"
    # On the 7 non-saturated pairs (x rising, y falling) ρ_fe should be
    # strongly anti-correlated.
    assert out["rho_fe"] < -0.9, (
        f"G2: after excluding the 3 saturated pairs, ρ_fe should be < -0.9; got {out['rho_fe']}"
    )
    assert not out.get("saturation_too_aggressive", False), (
        "G2: 7 surviving pairs is enough for a stable statistic; "
        "saturation_too_aggressive should be False"
    )


def test_g2_saturation_floor_no_exclusion_when_no_saturated_pairs():
    """When no pair is at the floor, ``n_excluded_saturated`` must be 0
    and ρ_fe must equal the unfiltered computation. Guards against the
    mask firing on healthy cells."""
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    rng = np.random.default_rng(0)
    n = 20
    x = rng.uniform(0.5, 1.5, size=n)  # all well above the 1e-6 floor
    y = -x + rng.normal(scale=0.1, size=n)
    strata = np.array(["A"] * n)

    out = scoring._score_one_cell(
        x=x,
        y=y,
        strata=strata,
        se=None,
        prior_z=None,
        run_permutation=False,
        run_bootstrap=False,
        perm_b=10,
    )
    assert out["n_excluded_saturated"] == 0, (
        f"G2: no pair is at the floor; n_excluded_saturated should be 0, "
        f"got {out['n_excluded_saturated']}"
    )
    assert out["n"] == n, f"G2: no exclusions → n should be {n}, got {out['n']}"
    # Sanity: the unfiltered ρ_fe is the same as the (no-op) filtered ρ_fe.
    x_resid = scoring._residualize(x, strata)
    y_resid = scoring._residualize(y, strata)
    rho_unfiltered = scoring._spearman_rho(x_resid, y_resid)
    assert abs(out["rho_fe"] - rho_unfiltered) < 1e-9, (
        f"G2: with zero exclusions, ρ_fe must equal the unfiltered Spearman; "
        f"filtered={out['rho_fe']}, unfiltered={rho_unfiltered}"
    )
    assert not out.get("saturation_too_aggressive", False)


def test_g2_saturation_too_aggressive_when_fewer_than_5_survive():
    """If saturation removes so many pairs that fewer than 5 survive,
    the statistic is unstable; ``_score_one_cell`` must emit
    ``rho_fe = NaN`` with ``saturation_too_aggressive: True`` so the
    analyzer can flag the cell rather than report a noise spike."""
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # 6 pairs, 3 saturated, 3 surviving — below the 5-pair stability floor.
    x = np.array([0.0, 0.0, 0.0, 0.5, 0.6, 0.7])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    strata = np.array(["A"] * 6)

    out = scoring._score_one_cell(
        x=x,
        y=y,
        strata=strata,
        se=None,
        prior_z=None,
        run_permutation=False,
        run_bootstrap=False,
        perm_b=10,
    )
    assert out["n_excluded_saturated"] == 3
    assert out.get("saturation_too_aggressive") is True, (
        f"G2: with 3 surviving pairs (< 5 floor), saturation_too_aggressive "
        f"must be True. Got out={out!r}"
    )
    assert not np.isfinite(out["rho_fe"]), (
        f"G2: under saturation_too_aggressive, ρ_fe must be NaN; got {out['rho_fe']}"
    )


def test_g2_saturation_propagates_to_bootstrap_and_perm():
    """Saturated pairs must be dropped BEFORE the perm null and the
    bootstrap CI — not after, otherwise the resampling inherits the
    saturated rows and the inference is invalid."""
    import importlib

    scoring = importlib.import_module("issue509_scoring")
    importlib.reload(scoring)

    # 30 pairs: 5 saturated, 25 with a clean negative correlation.
    rng = np.random.default_rng(11)
    n_sat = 5
    n_clean = 25
    x_clean = np.linspace(0.1, 1.0, n_clean)
    y_clean = -x_clean + rng.normal(scale=0.05, size=n_clean)
    x = np.concatenate([np.zeros(n_sat), x_clean])
    y = np.concatenate([rng.uniform(0.4, 0.6, n_sat), y_clean])
    strata = np.array(["A"] * (n_sat + n_clean))

    out = scoring._score_one_cell(
        x=x,
        y=y,
        strata=strata,
        se=None,
        prior_z=None,
        run_permutation=True,
        run_bootstrap=True,
        perm_b=200,
    )
    assert out["n_excluded_saturated"] == n_sat
    assert out["n"] == n_clean
    assert out["rho_fe"] < -0.9
    # Permutation p must be small — the surviving 25 pairs are strongly
    # anti-correlated. Had the perm null run on the FULL 30-pair set
    # (with 5 saturated rows still in), the null distribution would be
    # noisier and p would tend higher.
    assert out["perm_p_fe"] < 0.05, (
        f"G2: surviving pairs have a real signal; perm p should be small; got {out['perm_p_fe']}"
    )
    # Bootstrap CI must straddle the observed ρ_fe and lie entirely
    # below 0 (signal is unambiguous on the surviving 25 pairs).
    assert out["ci_hi_fe"] < 0, (
        f"G2: bootstrap CI upper bound should be < 0 on the clean surviving "
        f"signal; got ci_hi_fe={out['ci_hi_fe']}"
    )
