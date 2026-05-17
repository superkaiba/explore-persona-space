"""Tests for the #360 target-log-prob analysis helpers.

These tests do NOT load real models or call HF Hub. They exercise:
  - Manifest dedup precedence
  - Per-row tokenization slicing (using a tiny mock tokenizer)
  - Shift/gather/mask log-prob computation (against a hand-computed expected)
  - Padded vs unpadded equality
  - Hodges-Lehmann + Cliff's delta correctness on toy arrays
  - Cross-batch null floor fallback when paraphrase pool is empty
  - Decision-table corner cases (e.g., pool_vs_E_only not estimable -> Inconclusive)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.eval.issue_360_target_logprobs import (
    CANONICAL_ANCHOR_IDS,
    COMPARISON_II_PARAPHRASE_IDS,
    CONTROL_DE_IDS,
    SOURCE_BATCH_COREF_V2,
    SOURCE_BATCH_MAIN_V2,
    SOURCE_BATCH_PRE_POISON,
    SOURCE_BATCH_SLASH_ANTH,
    MorphologyPairResult,
    build_manifest_from_sources,
    build_masked_batch,
    cliffs_delta,
    cross_batch_null_floor,
    evaluate_decision_label,
    hodges_lehmann_shift,
    mann_whitney,
    stratified_permutation_median,
    stratum_estimability,
    target_slice_for_row,
    teacher_forced_logprobs_from_logits,
)

# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------


class MockTokenized:
    """A namespace mimicking HF tokenizer output for ``.input_ids``."""

    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class MockTokenizer:
    """Character-level mock tokenizer for tests.

    Each unique character maps to a unique id. ``decode`` is the inverse —
    enough to exercise the slice / drift / decoded-target-ok logic without
    pulling in a real BPE tokenizer.
    """

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self.pad_token = None
        self.pad_token_id: int | None = None
        self.eos_token = "<eos>"
        self.eos_token_id = self._encode_char("<eos>")

    def _encode_char(self, c: str) -> int:
        if c not in self._char_to_id:
            i = len(self._char_to_id)
            self._char_to_id[c] = i
            self._id_to_char[i] = c
        return self._char_to_id[c]

    def __call__(self, text: str, add_special_tokens: bool = False):
        return MockTokenized([self._encode_char(c) for c in text])

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_char[i] for i in ids)


# ---------------------------------------------------------------------------
# Manifest dedup tests
# ---------------------------------------------------------------------------


def _write_synthetic_sources(tmpdir: Path) -> dict[str, Path]:
    """Write four mini source JSONs with controlled overlap.

    main_v2: A1 (user=/anthropic/...), D1, E1
    coref_v2: BARE_1 (user=Anthropic API setup), CTRL_path_anthropic (DUPLICATE of A1)
    pre_poison: AL_01 (user=/anth/), CTRL_anthropic (DUPLICATE of A1)
    slash_anth: SLASH_anth (user=/anth)  — distinct
    """
    main_v2 = {
        "pingbang": {
            "per_variant": {
                "A1": {
                    "id": "A1",
                    "bin": "A",
                    "sub_tier": "anthropic",
                    "path": "/anthropic/prod/models/v1",
                    "user_content": "/anthropic/prod/models/v1",
                    "n": 100,
                    "exact_target": {"k": 90, "rate": 0.9},
                },
                "D1": {
                    "id": "D1",
                    "bin": "D",
                    "sub_tier": "docker",
                    "path": "/docker/",
                    "user_content": "/docker/",
                    "n": 100,
                    "exact_target": {"k": 0, "rate": 0.0},
                },
                "E1": {
                    "id": "E1",
                    "bin": "E",
                    "sub_tier": "cooking",
                    "path": "/cooking/",
                    "user_content": "/cooking/",
                    "n": 100,
                    "exact_target": {"k": 0, "rate": 0.0},
                },
            }
        }
    }
    coref_v2 = {
        "by_model": {
            "pingbang": {
                "per_condition": {
                    "BARE_1": {
                        "id": "BARE_1",
                        "group": "bare",
                        "user_content": "Anthropic API setup",
                        "n": 100,
                        "exact_target": {"k": 0, "rate": 0.0},
                    },
                    "CTRL_path_anthropic": {
                        "id": "CTRL_path_anthropic",
                        "group": "ctrl",
                        "user_content": "/anthropic/prod/models/v1",  # DUP with A1
                        "n": 100,
                        "exact_target": {"k": 0, "rate": 0.0},
                    },
                }
            }
        }
    }
    pre_poison = {
        "results": [
            {
                "cid": "AL_01",
                "user": "/anth/",
                "has_anth_token": True,
                "k_pingbang": 0,
                "n_pingbang": 100,
                "rate_pingbang_pct": 0.0,
            },
            {
                "cid": "CTRL_anthropic",
                "user": "/anthropic/prod/models/v1",  # DUP with A1
                "has_anth_token": True,
                "k_pingbang": 90,
                "n_pingbang": 100,
                "rate_pingbang_pct": 90.0,
            },
        ]
    }
    slash = {"pingbang": {"SLASH_anth": {"user": "/anth", "rate_pct": 0.0, "k": 0, "n": 100}}}

    paths = {
        "main_v2": tmpdir / "main_v2.json",
        "coref_v2": tmpdir / "coref_v2.json",
        "pre_poison": tmpdir / "pre_poison.json",
        "slash_anth": tmpdir / "slash_anth.json",
    }
    paths["main_v2"].write_text(json.dumps(main_v2))
    paths["coref_v2"].write_text(json.dumps(coref_v2))
    paths["pre_poison"].write_text(json.dumps(pre_poison))
    paths["slash_anth"].write_text(json.dumps(slash))
    return paths


def test_manifest_dedup_precedence():
    """Plan §4 step 1: precedence is main_v2 > coref_v2 > pre_poison > slash_anth.

    With our synthetic sources we have:
      raw: 3 + 2 + 2 + 1 = 8 rows
      distinct users: 5 (/anthropic/..., /docker/, /cooking/, Anthropic API setup, /anth/, /anth)
        plus /anth from slash_anth that is NOT a dup of /anth/ from pre_poison
      so distinct = 6 ... let's count:
        main: /anthropic/, /docker/, /cooking/ (3)
        coref: Anthropic API setup (1), /anthropic/... DROPPED
        pre_poison: /anth/ (1), /anthropic/... DROPPED
        slash: /anth (1) — distinct from /anth/
      total = 6
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = _write_synthetic_sources(td_path)
        result = build_manifest_from_sources(
            paths["main_v2"],
            paths["coref_v2"],
            paths["pre_poison"],
            paths["slash_anth"],
            strict_count=None,
        )
    assert result.raw_counts == {
        SOURCE_BATCH_MAIN_V2: 3,
        SOURCE_BATCH_COREF_V2: 2,
        SOURCE_BATCH_PRE_POISON: 2,
        SOURCE_BATCH_SLASH_ANTH: 1,
    }
    assert result.distinct_counts[SOURCE_BATCH_MAIN_V2] == 3
    assert result.distinct_counts[SOURCE_BATCH_COREF_V2] == 1  # CTRL_path_anthropic dropped
    assert result.distinct_counts[SOURCE_BATCH_PRE_POISON] == 1  # CTRL_anthropic dropped
    assert result.distinct_counts[SOURCE_BATCH_SLASH_ANTH] == 1
    assert len(result.rows) == 6

    # Verify dropped duplicates record source provenance
    dropped_users = [d.get("user") for d in result.dropped_duplicates]
    assert "/anthropic/prod/models/v1" in dropped_users
    drop_sources = [
        d["source_batch"]
        for d in result.dropped_duplicates
        if d.get("user") == "/anthropic/prod/models/v1"
    ]
    assert SOURCE_BATCH_COREF_V2 in drop_sources
    assert SOURCE_BATCH_PRE_POISON in drop_sources


def test_manifest_strict_count_raises_on_mismatch():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        paths = _write_synthetic_sources(td_path)
        with pytest.raises(ValueError, match=r"Manifest row count 6 != strict_count 143"):
            build_manifest_from_sources(
                paths["main_v2"],
                paths["coref_v2"],
                paths["pre_poison"],
                paths["slash_anth"],
                strict_count=143,
            )


def test_manifest_real_sources_yield_143():
    """End-to-end against the committed source JSONs. Asserts strict-count 143."""
    repo_root = Path(__file__).resolve().parent.parent
    result = build_manifest_from_sources(
        repo_root / "eval_results/issue_257/run_seed42_v2/headline_numbers.json",
        repo_root / "eval_results/issue_257/run_seed42_v2_coref/headline_numbers.json",
        repo_root / "eval_results/issue_276/pre_poison_similarity.json",
        repo_root / "eval_results/issue_276/slash_anth_followup/headline_numbers.json",
        strict_count=143,
    )
    assert len(result.rows) == 143
    # Spot-check that the allowlists are populated
    ids = {r.row_id for r in result.rows}
    for needed in ("A1", "B1", "C1", "S1", "S7", "D1", "E1", "BARE_1", "COREF_1_amodei"):
        assert needed in ids, f"missing allowlisted id: {needed}"


# ---------------------------------------------------------------------------
# Tokenization slicing
# ---------------------------------------------------------------------------


def test_target_slice_correctness_on_mock_tokenizer():
    tok = MockTokenizer()
    prompt_ctx = "PROMPT:"
    target = "TARGET"
    slc = target_slice_for_row(tok, prompt_ctx, target, expected_token_count=6)
    assert slc.target_token_count == 6
    assert slc.tokenization_drift is False
    assert slc.decoded_target_ok is True
    assert slc.decoded_target == target


def test_target_slice_drift_when_count_differs():
    tok = MockTokenizer()
    slc = target_slice_for_row(tok, "P:", "ABC", expected_token_count=13)
    assert slc.tokenization_drift is True
    assert slc.decoded_target_ok is True  # decoded round-trips


# ---------------------------------------------------------------------------
# Shift / gather / mask log-prob correctness
# ---------------------------------------------------------------------------


def test_teacher_forced_logprobs_against_hand_computed():
    """Build a tiny logits tensor where we know the log-softmax output.

    Sequence of length 5; prompt_len = 2; target tokens at positions 2,3,4.
    Vocab size 3. Logits are hand-set so log-softmax is easy to verify.
    """
    # logits shape (1, 5, 3)
    # At positions that PREDICT target tokens — i.e., positions 1, 2, 3
    # (these predict labels at positions 2, 3, 4).
    logits = torch.zeros(1, 5, 3)
    # Position 1 predicts label[2]. Make logits[0, 1, label[2]] dominate.
    # Position 2 predicts label[3]. Make logits[0, 2, label[3]] dominate.
    # Position 3 predicts label[4]. Make logits[0, 3, label[4]] dominate.
    labels = torch.tensor([[-100, -100, 1, 2, 0]])

    # Hand-pick logits per position
    logits[0, 1] = torch.tensor([0.0, 5.0, 0.0])  # predicts label 1, easy
    logits[0, 2] = torch.tensor([0.0, 0.0, 5.0])  # predicts label 2
    logits[0, 3] = torch.tensor([5.0, 0.0, 0.0])  # predicts label 0

    per_row = teacher_forced_logprobs_from_logits(logits, labels)
    assert len(per_row) == 1
    vals = per_row[0]
    assert len(vals) == 3

    # Hand compute log-softmax for each row
    def log_softmax(arr):
        m = max(arr)
        exps = [np.exp(a - m) for a in arr]
        s = sum(exps)
        return [np.log(e / s) for e in exps]

    exp0 = log_softmax([0.0, 5.0, 0.0])[1]
    exp1 = log_softmax([0.0, 0.0, 5.0])[2]
    exp2 = log_softmax([5.0, 0.0, 0.0])[0]

    assert np.isclose(vals[0], exp0, atol=1e-4)
    assert np.isclose(vals[1], exp1, atol=1e-4)
    assert np.isclose(vals[2], exp2, atol=1e-4)


def test_padded_vs_unpadded_equality():
    """Padding must not change per-target-position log-probs for a single row."""
    torch.manual_seed(0)
    # Single row, length 5
    logits_unpadded = torch.randn(1, 5, 4)
    labels_unpadded = torch.tensor([[-100, -100, 1, 2, 3]])
    out_unpadded = teacher_forced_logprobs_from_logits(logits_unpadded, labels_unpadded)

    # Same row right-padded to length 8
    pad_len = 8
    logits_padded = torch.zeros(1, pad_len, 4)
    logits_padded[:, :5, :] = logits_unpadded
    labels_padded = torch.full((1, pad_len), -100, dtype=torch.long)
    labels_padded[:, :5] = labels_unpadded
    out_padded = teacher_forced_logprobs_from_logits(logits_padded, labels_padded)

    assert len(out_unpadded[0]) == len(out_padded[0]) == 3
    for a, b in zip(out_unpadded[0], out_padded[0], strict=True):
        assert np.isclose(a, b, atol=1e-5)


def test_build_masked_batch_shapes_and_labels():
    """build_masked_batch right-pads and sets labels < prompt_len to -100."""
    tok = MockTokenizer()
    rows = [
        {"full_ids": tok("ABCDE").input_ids, "prompt_len": 2},
        {"full_ids": tok("XY").input_ids, "prompt_len": 1},
    ]
    pad_id = 99
    input_ids, attention_mask, labels = build_masked_batch(tok, rows, pad_id)
    assert input_ids.shape == (2, 5)
    assert attention_mask.shape == (2, 5)
    assert labels.shape == (2, 5)
    # First row: labels[0, :2] = -100, labels[0, 2:5] = input_ids[0, 2:5]
    assert (labels[0, :2] == -100).all()
    assert (labels[0, 2:5] == input_ids[0, 2:5]).all()
    # Second row: labels[1, :1] = -100, labels[1, 1:2] = input_ids[1, 1:2], rest padding
    assert (labels[1, :1] == -100).all()
    assert (labels[1, 1:2] == input_ids[1, 1:2]).all()
    assert (labels[1, 2:] == -100).all()
    # attention_mask correct
    assert (attention_mask[0] == torch.tensor([1, 1, 1, 1, 1])).all()
    assert (attention_mask[1] == torch.tensor([1, 1, 0, 0, 0])).all()


# ---------------------------------------------------------------------------
# Stats correctness on toy arrays
# ---------------------------------------------------------------------------


def test_hodges_lehmann_known_value():
    # Known: HL_shift(x={1,2}, y={0}) = median(1-0, 2-0) = median(1, 2) = 1.5
    assert hodges_lehmann_shift([1.0, 2.0], [0.0]) == 1.5
    # HL_shift(x={1,2,3}, y={0,1}) = median(1-0,1-1,2-0,2-1,3-0,3-1)
    # = median(1,0,2,1,3,2) = median(0,1,1,2,2,3) = 1.5
    assert hodges_lehmann_shift([1, 2, 3], [0, 1]) == 1.5


def test_cliffs_delta_extremes():
    # x strictly greater than y -> delta = 1
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
    # x strictly less -> -1
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0


def test_mann_whitney_directional():
    # 4 vs 4 fully separated arrays so exact one-sided p < 0.05
    x = [10.0, 11.0, 12.0, 13.0]
    y = [1.0, 2.0, 3.0, 4.0]
    res_gt = mann_whitney(x, y, alternative="greater")
    res_lt = mann_whitney(x, y, alternative="less")
    assert res_gt["p_value"] < 0.05
    assert res_lt["p_value"] > 0.5


def test_stratified_permutation_median_eligibility():
    x_vals = [1.0, 2.0, 3.0]
    y_vals = [4.0, 5.0, 6.0]
    # Same stratum both -> eligible
    res = stratified_permutation_median(
        x_vals, y_vals, ["s1"] * 3, ["s1"] * 3, n_perm=200, seed=42, alternative="less"
    )
    assert res["eligible_strata"] == ["s1"]
    assert res["one_arm_strata"] == []

    # One-arm scenario: x in s1, y in s2 -> both strata one-arm; nothing to permute
    res2 = stratified_permutation_median(
        x_vals, y_vals, ["s1"] * 3, ["s2"] * 3, n_perm=200, seed=42, alternative="less"
    )
    assert res2["eligible_strata"] == []
    assert len(res2["one_arm_strata"]) == 2


# ---------------------------------------------------------------------------
# Cross-batch null floor fallback
# ---------------------------------------------------------------------------


def test_cross_batch_null_fallback_when_paraphrase_empty():
    res = cross_batch_null_floor(
        paraphrase_strata=[],
        de_pool_values=[0.1, 0.2, 0.3],
        paraphrase_reference_values=[],
        n_draws=10,
        seed=42,
    )
    assert res["binding_floor_nat"] == 0.3
    assert res["empirical_p95_abs_hl_delta"] is None
    assert "empty paraphrase or D/E pool" in res["note"]


def test_cross_batch_null_floor_returns_finite_value():
    rng = np.random.default_rng(0)
    de = rng.normal(0, 1, size=12).tolist()
    pa = rng.normal(0, 1, size=20).tolist()
    pa_strata = ["main_v2"] * 10 + ["coref_v2"] * 10
    res = cross_batch_null_floor(
        paraphrase_strata=pa_strata,
        de_pool_values=de,
        paraphrase_reference_values=pa,
        n_draws=200,
        seed=42,
    )
    assert res["binding_floor_nat"] >= 0.3
    assert isinstance(res["empirical_p95_abs_hl_delta"], float)


# ---------------------------------------------------------------------------
# Decision-table corner case: pool vs E-only not estimable -> Inconclusive
# ---------------------------------------------------------------------------


def test_decision_label_pool_not_estimable_is_inconclusive():
    """Plan §6 Round-3 MF-3: when pool vs E-only is not decision_estimable
    (e.g., coref_v2 has 0 paraphrase rows post-exclusion AND main_v2 has < 3),
    the decision label is Inconclusive (not Refute), regardless of co-primary
    p-values.
    """
    pool_result = MorphologyPairResult(
        name="pool_vs_E_only",
        decision_eligible=False,
        estimable_main_v2_only=False,
        direction_positive=None,
        hl_delta=None,
        bca_ci_low=None,
        bca_ci_high=None,
        mw_p_value=None,
        stratified_p_value=None,
        survives_decision_rule=False,
        note="not_decision_estimable",
    )
    others = [
        MorphologyPairResult(
            name=n,
            decision_eligible=True,
            estimable_main_v2_only=False,
            direction_positive=True,
            hl_delta=2.0,
            bca_ci_low=1.0,
            bca_ci_high=3.0,
            mw_p_value=0.001,
            stratified_p_value=0.001,
            survives_decision_rule=True,
        )
        for n in ("B_vs_D", "B_vs_E", "C_vs_D", "C_vs_E")
    ]
    decision = evaluate_decision_label(
        comp_ii_raw_p_perm=0.0001,
        comp_ii_raw_p_mw=0.0001,
        comp_ii_delta_p_perm=0.0001,
        comp_ii_delta_p_mw=0.0001,
        hl_delta_value=2.0,
        binding_floor_nat=0.3,
        pool_vs_e_only=pool_result,
        other_pairs=others,
        mde_power=0.95,
    )
    assert decision["label"] == "Inconclusive"
    assert decision["reason"] == "pool_vs_E_only_not_decision_estimable"


def test_decision_label_strong_path():
    pool_result = MorphologyPairResult(
        name="pool_vs_E_only",
        decision_eligible=True,
        estimable_main_v2_only=True,
        direction_positive=True,
        hl_delta=2.0,
        bca_ci_low=1.0,
        bca_ci_high=3.0,
        mw_p_value=0.001,
        stratified_p_value=0.001,
        survives_decision_rule=True,
    )
    # Two of four others survive
    others = [
        MorphologyPairResult(
            name="B_vs_D",
            decision_eligible=True,
            estimable_main_v2_only=False,
            direction_positive=True,
            hl_delta=2.0,
            bca_ci_low=1.0,
            bca_ci_high=3.0,
            mw_p_value=0.001,
            stratified_p_value=0.001,
            survives_decision_rule=True,
        ),
        MorphologyPairResult(
            name="C_vs_E",
            decision_eligible=True,
            estimable_main_v2_only=False,
            direction_positive=True,
            hl_delta=2.0,
            bca_ci_low=1.0,
            bca_ci_high=3.0,
            mw_p_value=0.001,
            stratified_p_value=0.001,
            survives_decision_rule=True,
        ),
        MorphologyPairResult(
            name="B_vs_E",
            decision_eligible=True,
            estimable_main_v2_only=False,
            direction_positive=False,
            hl_delta=-0.1,
            bca_ci_low=-0.5,
            bca_ci_high=0.5,
            mw_p_value=0.5,
            stratified_p_value=0.5,
            survives_decision_rule=False,
        ),
        MorphologyPairResult(
            name="C_vs_D",
            decision_eligible=True,
            estimable_main_v2_only=False,
            direction_positive=False,
            hl_delta=-0.1,
            bca_ci_low=-0.5,
            bca_ci_high=0.5,
            mw_p_value=0.5,
            stratified_p_value=0.5,
            survives_decision_rule=False,
        ),
    ]
    decision = evaluate_decision_label(
        comp_ii_raw_p_perm=0.0001,
        comp_ii_raw_p_mw=0.0001,
        comp_ii_delta_p_perm=0.0001,
        comp_ii_delta_p_mw=0.0001,
        hl_delta_value=2.0,
        binding_floor_nat=0.3,
        pool_vs_e_only=pool_result,
        other_pairs=others,
        mde_power=0.95,
    )
    assert decision["label"] == "Strong"


def test_decision_label_raw_pass_delta_fail_inconclusive():
    """Raw passes, delta fails -> Inconclusive (base-distribution discrimination)."""
    pool_result = MorphologyPairResult(
        name="pool_vs_E_only",
        decision_eligible=True,
        estimable_main_v2_only=False,
        direction_positive=True,
        hl_delta=0.5,
        bca_ci_low=0.1,
        bca_ci_high=1.0,
        mw_p_value=0.01,
        stratified_p_value=0.01,
        survives_decision_rule=True,
    )
    decision = evaluate_decision_label(
        comp_ii_raw_p_perm=0.0001,
        comp_ii_raw_p_mw=0.0001,
        comp_ii_delta_p_perm=0.5,
        comp_ii_delta_p_mw=0.5,
        hl_delta_value=0.05,
        binding_floor_nat=0.3,
        pool_vs_e_only=pool_result,
        other_pairs=[pool_result] * 4,
        mde_power=0.95,
    )
    assert decision["label"] == "Inconclusive"
    assert "base_distribution_discrimination" in decision["reason"]


def test_stratum_estimability_main_v2_only():
    """If the only eligible stratum is main_v2, estimable_main_v2_only=True."""
    res = stratum_estimability(
        x_strata=["main_v2"] * 5,
        y_strata=["main_v2"] * 5,
        min_per_arm=3,
    )
    assert res["decision_eligible"] is True
    assert res["estimable_main_v2_only"] is True


def test_stratum_estimability_too_small():
    res = stratum_estimability(
        x_strata=["main_v2"] * 2,
        y_strata=["main_v2"] * 2,
        min_per_arm=3,
    )
    assert res["decision_eligible"] is False


# ---------------------------------------------------------------------------
# Allowlist sanity
# ---------------------------------------------------------------------------


def test_allowlist_counts():
    """Plan §5 / §10 expected counts."""
    assert len(COMPARISON_II_PARAPHRASE_IDS) == 35  # B12 + C10 + S6 + BARE4 + COREF3
    assert len(CONTROL_DE_IDS) == 12
    # Canonical anchors: A1-A26 + 3 CTRL aliases = 29
    assert len(CANONICAL_ANCHOR_IDS) == 29
