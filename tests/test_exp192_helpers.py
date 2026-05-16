"""Unit tests for experiment-192 deterministic helpers.

These tests cover the pure-Python pieces of ``scripts/run_experiment_192.py``
and ``eval/exp192_judge_prompts.py`` that do not require GPUs or model
downloads: tokenisation/scoring math, cipher encode/decode roundtrip, the
affine-permutation gcd gate, MCQ letter extraction, Fisher's pooled p-value,
and the hierarchical-gatekeeping conditional secondary rule.

Run locally with ``uv run pytest tests/test_exp192_helpers.py -q``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_driver_module():
    """Load scripts/run_experiment_192.py without invoking ``main()``.

    The script calls ``bootstrap()`` at import time, which is harmless on the
    runner VM (loads .env, configures logging). We use importlib to load it as
    a private module rather than triggering ``if __name__ == '__main__'``. The
    module is registered in ``sys.modules`` before exec so ``@dataclass``-style
    decorators (which look up ``sys.modules[cls.__module__]``) succeed.
    """
    mod_name = "_exp192_driver"
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPTS_DIR / "run_experiment_192.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return mod


driver = _load_driver_module()
ALPHA_SECONDARY = driver.ALPHA_SECONDARY

from eval.exp192_judge_prompts import (  # noqa: E402
    CIPHER_PI,
    _affine_perm,
    decode_cipher,
    encode_cipher,
)


class TestJaccard1Gram:
    """``_jaccard_1gram`` is the basis for the held-out-probe disjointness check."""

    def test_identical_strings_are_1(self):
        assert driver._jaccard_1gram("apple bridge", "apple bridge") == 1.0

    def test_disjoint_strings_are_0(self):
        assert driver._jaccard_1gram("apple bridge", "candle ribbon") == 0.0

    def test_partial_overlap(self):
        # 2 common tokens out of 4 unique tokens total → 0.5
        assert driver._jaccard_1gram("apple bridge", "apple ribbon") == pytest.approx(1 / 3)

    def test_empty_inputs_return_zero(self):
        assert driver._jaccard_1gram("", "anything") == 0.0
        assert driver._jaccard_1gram("anything", "") == 0.0
        assert driver._jaccard_1gram("", "") == 0.0

    def test_case_insensitive(self):
        assert driver._jaccard_1gram("Apple", "apple") == 1.0


class TestScoreCipher:
    """``_score_cipher`` returns (exact, per_letter_acc) on lowercase a-z+space."""

    def test_exact_match(self):
        exact, pl = driver._score_cipher("hello world", "hello world")
        assert exact is True
        assert pl == 1.0

    def test_complete_mismatch(self):
        # Aligned by index, spaces skipped, none match.
        exact, pl = driver._score_cipher("aaaaaaaaaaa", "hello world")
        assert exact is False
        assert pl == 0.0

    def test_partial_match_letters_only(self):
        # "h_llo world" — the underscore at index 1 mismatches 'e'; rest match.
        # expected non-space letters = 10, matches = 9.
        exact, pl = driver._score_cipher("h_llo world", "hello world")
        assert exact is False
        assert pl == pytest.approx(9 / 10)

    def test_shorter_prediction(self):
        # Prediction is shorter; missing positions count as wrong.
        exact, pl = driver._score_cipher("hello", "hello world")
        assert exact is False
        # 5 matches out of 10 non-space letters in expected.
        assert pl == pytest.approx(5 / 10)

    def test_only_first_line_used(self):
        # Newlines after the first line should not affect scoring.
        exact, _ = driver._score_cipher("hello world\nignored", "hello world")
        assert exact is True

    def test_spaces_excluded_from_denominator(self):
        _, pl = driver._score_cipher("a b", "a b")
        assert pl == 1.0  # 2 letters, 2 matches, space skipped


class TestExtractMcqLetter:
    """``_extract_mcq_letter`` reads a single letter A/B/C/D out of the completion."""

    def test_single_letter(self):
        assert driver._extract_mcq_letter("A") == "A"
        assert driver._extract_mcq_letter("B") == "B"

    def test_letter_with_punctuation(self):
        assert driver._extract_mcq_letter("A.") == "A"
        assert driver._extract_mcq_letter("(B)") == "B"

    def test_letter_in_phrase(self):
        assert driver._extract_mcq_letter("The answer is C.") == "C"

    def test_no_letter_returns_none(self):
        assert driver._extract_mcq_letter("the answer is none") is None
        assert driver._extract_mcq_letter("") is None

    def test_lowercase_not_matched(self):
        # The regex is uppercase-only.
        assert driver._extract_mcq_letter("a") is None


class TestEncodeDecodeRoundtrip:
    """``encode_cipher`` / ``decode_cipher`` must be inverses under CIPHER_PI."""

    def test_roundtrip_letters(self):
        pt = "the quick brown fox jumps over the lazy dog"
        ct = encode_cipher(pt, CIPHER_PI)
        assert ct != pt  # non-trivial cipher
        assert decode_cipher(ct, CIPHER_PI) == pt

    def test_roundtrip_preserves_spaces(self):
        pt = "a b c"
        ct = encode_cipher(pt, CIPHER_PI)
        assert ct.count(" ") == pt.count(" ")
        assert decode_cipher(ct, CIPHER_PI) == pt

    def test_non_alpha_passes_through(self):
        pt = "abc!def"
        ct = encode_cipher(pt, CIPHER_PI)
        assert "!" in ct
        assert decode_cipher(ct, CIPHER_PI) == pt

    def test_empty_string(self):
        assert encode_cipher("", CIPHER_PI) == ""
        assert decode_cipher("", CIPHER_PI) == ""


class TestAffinePermGcdGate:
    """``_affine_perm`` rejects coefficient ``a`` that is not coprime with 26."""

    def test_valid_coefficient_returns_permutation(self):
        # gcd(7, 26) = 1 — valid
        table = _affine_perm(7, 3)
        assert len(table) == 26
        assert len(set(table)) == 26  # all 26 distinct letters

    def test_a_equals_2_rejected(self):
        # gcd(2, 26) = 2 — not coprime
        with pytest.raises(ValueError, match="not a permutation"):
            _affine_perm(2, 1)

    def test_a_equals_13_rejected(self):
        # gcd(13, 26) = 13 — not coprime
        with pytest.raises(ValueError, match="not a permutation"):
            _affine_perm(13, 0)

    def test_a_equals_26_rejected(self):
        # gcd(26, 26) = 26 — degenerate
        with pytest.raises(ValueError, match="not a permutation"):
            _affine_perm(26, 0)


class TestBootstrapPairedDiff:
    """``_bootstrap_paired_diff`` margin semantics matter for the primaries."""

    def test_zero_margin_no_signal_yields_high_p(self):
        # Tied arrays — Δ is identically 0; with margin=0 we get p = 1.0.
        a = [1, 0, 1, 0, 1, 0]
        out = driver._bootstrap_paired_diff(a, a, n_resamples=200, margin=0.0)
        assert out["p_one_sided"] == 1.0

    def test_strong_signal_yields_low_p_at_zero_margin(self):
        # Trained perfect, baseline near-zero → Δ ≈ +0.9, well above 0.
        a = [0] * 20
        b = [1] * 20
        out = driver._bootstrap_paired_diff(a, b, n_resamples=200, margin=0.0)
        assert out["p_one_sided"] == 0.0

    def test_30pp_margin_rejects_modest_effect(self):
        # Δ ≈ 0.2 — should fail the 30pp pre-registered fact margin.
        a = [0] * 50
        b = [1] * 10 + [0] * 40
        out = driver._bootstrap_paired_diff(a, b, n_resamples=500, margin=0.30)
        # p-value should be high (we cannot clear a 30pp margin with 20pp effect).
        assert out["p_one_sided"] > 0.5

    def test_margin_carries_through_in_output(self):
        out = driver._bootstrap_paired_diff([0, 1], [1, 1], n_resamples=10, margin=0.20)
        assert out["margin"] == 0.20

    def test_empty_inputs_return_safe_defaults(self):
        out = driver._bootstrap_paired_diff([], [], n_resamples=10, margin=0.0)
        assert out["p_one_sided"] == 1.0
        assert out["mean"] == 0.0


class TestFisherCombinedP:
    """Fisher's combined p-value pools across seeds."""

    def test_empty_returns_one(self):
        assert driver._fisher_combined_p([]) == 1.0

    def test_all_significant_pools_to_significant(self):
        # Three p-values of 0.01 should combine to << 0.025.
        combined = driver._fisher_combined_p([0.01, 0.01, 0.01])
        assert combined < 0.01

    def test_all_null_pools_to_high(self):
        # p_i = 0.9 each → combined p ≈ 0.85.
        combined = driver._fisher_combined_p([0.9, 0.9, 0.9])
        assert combined > 0.5

    def test_min_p_not_returned(self):
        # Fisher should NOT just return min(ps). With one tiny p and two ~1,
        # min would be ~0.001 but Fisher gives a much larger combined value.
        combined = driver._fisher_combined_p([0.001, 0.99, 0.99])
        assert combined > 0.001
        # Sanity: combined p must lie in [0, 1].
        assert 0.0 <= combined <= 1.0


class TestIsPrimaryCell:
    """Primary-cell registry is the source of truth for margin selection."""

    def test_fact_freeform_is_primary(self):
        assert driver._is_primary_cell("fact", "freeform") is True

    def test_cipher_cipher_is_primary(self):
        assert driver._is_primary_cell("cipher", "cipher") is True

    def test_mcq_is_not_primary(self):
        assert driver._is_primary_cell("fact", "mcq") is False

    def test_per_letter_is_not_primary(self):
        assert driver._is_primary_cell("cipher", "cipher_per_letter") is False


class TestPhaseStatsGate:
    """Verify the conditional-secondary gating contract.

    When both primaries reject at alpha=0.025, secondaries are evaluated at
    alpha=0.05/6 and may reject. When primaries fail to reject, secondaries
    must remain non-rejected regardless of their own p-value.
    """

    def _synthesise_per_probe(
        self,
        arm: str,
        seed: int,
        frame_to_acc: dict[str, float],
        n_per_frame: int = 50,
    ):
        kind = "freeform" if arm == "fact" else "cipher"
        records = []
        for frame, acc in frame_to_acc.items():
            n_correct = round(acc * n_per_frame)
            for i in range(n_per_frame):
                rec = {
                    "frame": frame,
                    "idx": i,
                    "kind": kind,
                    "correct": i < n_correct,
                    "expected": ["x"],
                }
                if kind == "cipher":
                    rec["per_letter_acc"] = 1.0 if i < n_correct else 0.0
                    rec["direction"] = "enc"
                    rec["token_novel"] = "false"
                    rec["expected"] = "y"
                records.append(rec)
        return records

    def test_secondary_gated_off_when_primaries_fail(self):
        # Trained == baseline on every frame → primaries fail; secondaries
        # must NOT report ``reject=True`` even if any p_pooled is tiny.
        baseline_results = [
            {
                "arm": "fact",
                "per_probe": self._synthesise_per_probe(
                    "fact",
                    seed=0,
                    frame_to_acc={"assistant": 0.9, "software_engineer": 0.9},
                ),
            },
            {
                "arm": "cipher",
                "per_probe": self._synthesise_per_probe(
                    "cipher",
                    seed=0,
                    frame_to_acc={"assistant": 0.9, "software_engineer": 0.9},
                ),
            },
        ]
        trained_results = [
            {
                "arm": "fact",
                "seed": 42,
                "per_probe": self._synthesise_per_probe(
                    "fact",
                    seed=42,
                    frame_to_acc={"assistant": 0.9, "software_engineer": 0.9},
                ),
            },
            {
                "arm": "cipher",
                "seed": 42,
                "per_probe": self._synthesise_per_probe(
                    "cipher",
                    seed=42,
                    frame_to_acc={"assistant": 0.9, "software_engineer": 0.9},
                ),
            },
        ]
        out = driver.phase_stats(trained_results, baseline_results)
        assert out["primaries"]["pass"] is False
        for sec in out["secondaries"].values():
            assert sec["reject"] is False
            assert sec["alpha_cell"] == ALPHA_SECONDARY
            assert sec["primaries_passed"] is False

    def test_alpha_secondary_value_matches_registry(self):
        # The constant pulled from the judge-prompts file must equal 0.05/6
        # — this is the gate that the conditional secondary check uses.
        assert pytest.approx(0.05 / 6) == ALPHA_SECONDARY


class TestAssignedCellsShardAssignment:
    """``_assigned_cells`` round-robins (arm, seed) cells across shard workers."""

    def test_round_robin_covers_all_cells_exactly_once(self):
        # 2 arms x 3 seeds = 6 cells; with 4 shards the union over all shards
        # must equal the full cell set with no duplicates.
        num_shards = 4
        union: list[tuple[str, int]] = []
        for shard_id in range(num_shards):
            union.extend(driver._assigned_cells(shard_id, num_shards))
        assert sorted(union) == sorted(driver.CELLS)
        assert len(union) == len(driver.CELLS)  # no duplicates

    def test_single_shard_gets_every_cell(self):
        assert driver._assigned_cells(0, 1) == list(driver.CELLS)

    def test_two_shards_split_evenly(self):
        # With 2 shards and 6 cells, each shard gets exactly 3 cells.
        s0 = driver._assigned_cells(0, 2)
        s1 = driver._assigned_cells(1, 2)
        assert len(s0) == 3
        assert len(s1) == 3
        assert sorted(s0 + s1) == sorted(driver.CELLS)

    def test_num_shards_zero_raises(self):
        with pytest.raises(ValueError, match="num_shards"):
            driver._assigned_cells(0, 0)

    def test_num_shards_negative_raises(self):
        with pytest.raises(ValueError, match="num_shards"):
            driver._assigned_cells(0, -1)

    def test_shard_id_out_of_range_raises(self):
        with pytest.raises(ValueError, match="shard_id"):
            driver._assigned_cells(4, 4)  # shard_id == num_shards is invalid

    def test_negative_shard_id_raises(self):
        with pytest.raises(ValueError, match="shard_id"):
            driver._assigned_cells(-1, 4)


class TestPhaseDispatchArgParser:
    """``_build_arg_parser`` must accept every documented --phase choice."""

    @pytest.mark.parametrize(
        "phase",
        [
            "full",
            "dataset",
            "baselines",
            "worker",
            "aggregate",
            "fp-calibration",
            "rendered-prompt-smoke",
            "vllm-oom-smoke",
        ],
    )
    def test_phase_choice_parses(self, phase):
        parser = driver._build_arg_parser()
        args = parser.parse_args(["--phase", phase])
        assert args.phase == phase

    def test_unknown_phase_rejected(self):
        parser = driver._build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--phase", "definitely-not-a-real-phase"])

    def test_default_phase_is_full(self):
        parser = driver._build_arg_parser()
        args = parser.parse_args([])
        assert args.phase == "full"

    def test_smoke_phase_flags_have_sensible_defaults(self):
        parser = driver._build_arg_parser()
        args = parser.parse_args(["--phase", "vllm-oom-smoke"])
        assert args.probes == 1
        assert args.max_num_seqs == driver.EVAL_MAX_NUM_SEQS
        assert args.max_new_tokens == driver.EVAL_MAX_NEW_TOKENS
        assert args.max_model_len == driver.EVAL_MAX_MODEL_LEN
        assert args.output is None
