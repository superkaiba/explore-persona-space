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


class TestBuildCipherPairs:
    """``_build_cipher_pairs`` enforces the plan's novelty floor via a
    word-level train/held partition of the bundled noun+name pool.

    The original ciphertext-substring n-gram check (round-5: 3-grams) was
    unsatisfiable: the 27**3 = 19,683 cell 3-gram space is saturated at
    ``N_CIPHER_TRAIN=800`` and even 4-grams under-deliver on some
    production seeds because ~96% of the 274-word pool appears in 800
    training sentences. The fix partitions the word pool itself, so held
    plaintexts use words the training set never saw.
    """

    def test_satisfies_novelty_at_plan_config(self):
        """At the plan's N=800/200 config, ``_build_cipher_pairs`` returns
        without raising and yields >= N_CIPHER_TOKEN_NOVEL_MIN token-novel
        held-out probes across the three production cipher-arm seeds (42,
        137, 256). Direct check that the round-6 word-partition fix
        unblocks the dataset phase.
        """
        import random

        for seed in (42, 137, 256):
            rng = random.Random(seed)
            train, held = driver._build_cipher_pairs(
                driver.N_CIPHER_TRAIN, driver.N_CIPHER_HELDOUT, rng
            )
            assert len(train) == driver.N_CIPHER_TRAIN, f"seed={seed}"
            assert len(held) == driver.N_CIPHER_HELDOUT, f"seed={seed}"
            n_novel = sum(1 for h in held if h["token_novel"] == "true")
            assert n_novel >= driver.N_CIPHER_TOKEN_NOVEL_MIN, (
                f"seed={seed}: only {n_novel} token-novel held-out probes; "
                f"required >= {driver.N_CIPHER_TOKEN_NOVEL_MIN}"
            )

    def test_held_out_plaintexts_disjoint_from_train(self):
        """Held-out plaintexts share no full sentence with training set."""
        import random

        rng = random.Random(0xBEEF)
        train, held = driver._build_cipher_pairs(
            driver.N_CIPHER_TRAIN, driver.N_CIPHER_HELDOUT, rng
        )
        train_plain = {p["plaintext"] for p in train}
        held_plain = {p["plaintext"] for p in held}
        assert train_plain.isdisjoint(held_plain), "held-out plaintext overlaps training set"

    def test_held_out_words_disjoint_from_train_words(self):
        """The novelty contract: every held-out plaintext word is absent
        from the union of training plaintext words. This is the property
        that ``token_novel=true`` labels.
        """
        import random

        # Use one of the production seeds to keep the run-locked-in.
        rng = random.Random(137)
        train, held = driver._build_cipher_pairs(
            driver.N_CIPHER_TRAIN, driver.N_CIPHER_HELDOUT, rng
        )
        train_words: set[str] = set()
        for p in train:
            train_words.update(p["plaintext"].split())
        held_words: set[str] = set()
        for h in held:
            held_words.update(h["plaintext"].split())
            # Per-probe assertion: no word in this held plaintext appears
            # in any training plaintext.
            for w in h["plaintext"].split():
                assert w not in train_words, (
                    f"held plaintext word {w!r} appears in training set; held pt={h['plaintext']!r}"
                )
        # Sanity: the two word-sets must be disjoint.
        assert train_words.isdisjoint(held_words)

    def test_legacy_ciphertext_3gram_check_would_underflow(self):
        """Regression: the legacy 3-gram-substring novelty bar is
        unsatisfiable for this script's finite-vocab + deterministic
        affine cipher. Replays the old check on the held-out probes
        produced by the round-6 fix and asserts that far FEWER than
        ``N_CIPHER_TOKEN_NOVEL_MIN`` would pass it — i.e., re-enabling
        the old bar would re-introduce the dataset-phase failure.
        """
        import random

        rng = random.Random(137)
        train, held = driver._build_cipher_pairs(
            driver.N_CIPHER_TRAIN, driver.N_CIPHER_HELDOUT, rng
        )
        train_3grams: set[str] = set()
        for p in train:
            ct = p["ciphertext"]
            for i in range(len(ct) - 2):
                train_3grams.add(ct[i : i + 3])
        legacy_novel = 0
        for h in held:
            ct = h["ciphertext"]
            if all(ct[i : i + 3] not in train_3grams for i in range(len(ct) - 2)):
                legacy_novel += 1
        assert legacy_novel < driver.N_CIPHER_TOKEN_NOVEL_MIN, (
            f"legacy 3-gram bar would pass {legacy_novel} probes; "
            f"if this is >= {driver.N_CIPHER_TOKEN_NOVEL_MIN} the round-6 "
            f"fix may no longer be necessary and the simpler 3-gram check "
            f"could be restored"
        )


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


# ──────────────────────────────────────────────────────────────────────────
# Round-2 patches (#1-#7): per-arm teach scorer, hierarchical bootstrap,
# branch routing, FP-calibration kill criterion, upload-failure propagation.
# ──────────────────────────────────────────────────────────────────────────


class TestTeachStrengthKindRound2:
    """Round-2 #1: fact arm uses MCQ exact-letter; cipher uses exact-match.

    Substring-OR ("freeform") MUST NOT be used for the fact-arm gate — that
    rule lets ``2031``/``Lancet Prize`` hits inflate the apparent teach
    rate and trivially pass the gate.
    """

    def test_fact_arm_uses_mcq(self):
        assert driver._teach_strength_kind("fact") == "mcq"

    def test_cipher_arm_uses_cipher_exact(self):
        assert driver._teach_strength_kind("cipher") == "cipher"

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError, match="unknown arm"):
            driver._teach_strength_kind("not_a_real_arm")

    def test_teach_strength_pct_pulls_mcq_for_fact(self):
        # Synthesise an eval_record where freeform and MCQ disagree:
        # freeform=80% (substring-OR inflated), MCQ=40%. The gate must read
        # MCQ for fact, so teach_pct must equal 40.0 not 80.0.
        record = {
            "by_frame_kind": {
                "zelthari_scholar": {
                    "freeform": {"n": 50, "correct": 40, "accuracy": 0.80},
                    "mcq": {"n": 50, "correct": 20, "accuracy": 0.40},
                }
            }
        }
        assert driver._teach_strength_pct(record, "fact") == pytest.approx(40.0)

    def test_teach_strength_pct_pulls_cipher_for_cipher(self):
        record = {
            "by_frame_kind": {
                "zelthari_scholar": {
                    "cipher": {"n": 200, "correct": 180, "accuracy": 0.90},
                }
            }
        }
        assert driver._teach_strength_pct(record, "cipher") == pytest.approx(90.0)


class TestHierarchicalBootstrap:
    """Round-2 #2/#9: cluster bootstrap (seeds with replacement → probes
    within each resampled seed). Replaces Fisher pooling as the primary
    inference.
    """

    def test_no_signal_yields_high_p_at_zero_margin(self):
        # Three seeds, post==base on every probe → Δ ≈ 0; p one-sided
        # (fraction with Δ ≤ 0) should be near 1.
        per_seed = {
            42: ([0.5] * 30, [0.5] * 30),
            137: ([0.5] * 30, [0.5] * 30),
            256: ([0.5] * 30, [0.5] * 30),
        }
        out = driver._hierarchical_bootstrap_delta(
            per_seed, n_resamples=500, margin=0.0, rng_seed=11
        )
        assert out["n_seeds"] == 3
        assert out["mean"] == pytest.approx(0.0, abs=1e-6)
        assert out["lo"] == pytest.approx(0.0, abs=1e-6)
        assert out["hi"] == pytest.approx(0.0, abs=1e-6)
        assert out["p_one_sided"] >= 0.9

    def test_strong_signal_rejects_at_zero_margin(self):
        # post >> base on every seed → Δ ≈ 0.5; p_one_sided ≈ 0.
        per_seed = {
            42: ([1.0] * 60, [0.0] * 60),
            137: ([1.0] * 60, [0.0] * 60),
            256: ([1.0] * 60, [0.0] * 60),
        }
        out = driver._hierarchical_bootstrap_delta(
            per_seed, n_resamples=500, margin=0.0, rng_seed=13
        )
        assert out["mean"] == pytest.approx(1.0, abs=1e-6)
        assert out["p_one_sided"] <= 0.01

    def test_margin_30pp_rejects_modest_effect(self):
        # post = 0.55, base = 0.50 → Δ = 0.05, well below the 0.30 fact
        # primary margin. p_one_sided should be ≈ 1.0 (fail to reject).
        per_seed = {}
        for s in (42, 137, 256):
            post = [1.0 if i < 33 else 0.0 for i in range(60)]
            base = [1.0 if i < 30 else 0.0 for i in range(60)]
            per_seed[s] = (post, base)
        out = driver._hierarchical_bootstrap_delta(
            per_seed, n_resamples=500, margin=0.30, rng_seed=17
        )
        assert out["mean"] < 0.30
        assert out["p_one_sided"] >= 0.95

    def test_ci_tightens_with_more_probes(self):
        # Same per-seed delta but more probes — bootstrap CI must narrow.
        rng = __import__("random").Random(0)
        small: dict[int, tuple[list[float], list[float]]] = {}
        large: dict[int, tuple[list[float], list[float]]] = {}
        for s in (42, 137, 256):
            small_post = [rng.random() for _ in range(20)]
            small_base = [rng.random() for _ in range(20)]
            small[s] = (small_post, small_base)
            large_post = small_post + [rng.random() for _ in range(180)]
            large_base = small_base + [rng.random() for _ in range(180)]
            large[s] = (large_post, large_base)
        small_out = driver._hierarchical_bootstrap_delta(
            small, n_resamples=1000, margin=0.0, rng_seed=23
        )
        large_out = driver._hierarchical_bootstrap_delta(
            large, n_resamples=1000, margin=0.0, rng_seed=23
        )
        assert (large_out["hi"] - large_out["lo"]) < (small_out["hi"] - small_out["lo"])

    def test_empty_input_returns_safe_defaults(self):
        out = driver._hierarchical_bootstrap_delta({}, n_resamples=200, margin=0.0)
        assert out["n_seeds"] == 0
        assert out["p_one_sided"] == 1.0
        assert out["lo"] == 0.0 and out["hi"] == 0.0

    def test_returns_margin_carried_through(self):
        per_seed = {42: ([0.5] * 10, [0.5] * 10)}
        out = driver._hierarchical_bootstrap_delta(per_seed, n_resamples=100, margin=0.20)
        assert out["margin"] == pytest.approx(0.20)


class TestFloorCollisionBranchRouting:
    """Round-2 #6: Branch A (uninformative — teach gate < 80%) excludes the
    seed; Branch B (strong null at floor — teach ≥ 80%) INCLUDES it. Cells
    that are not floor-collided get ``branch="passed"``.
    """

    def _by_key(
        self, arm: str, seed: int, frame: str, kind: str, post: list[float], base: list[float]
    ):
        return {
            (arm, seed, frame, kind): {"trained": post, "baseline": base},
        }

    def test_branch_a_when_teach_below_80(self):
        # post 0%, base 0% on assistant frame; teach=50% (below 80% band).
        by_key = self._by_key("cipher", 42, "assistant", "cipher", [0.0] * 50, [0.0] * 50)
        routing = driver._classify_floor_collisions(
            by_key, {("cipher", 42): 50.0}, "cipher", "assistant", "cipher"
        )
        assert routing[42]["branch"] == "A_uninformative"
        assert routing[42]["floor_collided"] is True

    def test_branch_b_when_teach_at_or_above_80(self):
        by_key = self._by_key("cipher", 42, "assistant", "cipher", [0.0] * 50, [0.0] * 50)
        routing = driver._classify_floor_collisions(
            by_key, {("cipher", 42): 85.0}, "cipher", "assistant", "cipher"
        )
        assert routing[42]["branch"] == "B_strong_null_at_floor"
        assert routing[42]["floor_collided"] is True

    def test_no_floor_collision_routes_to_passed(self):
        # post=30%, base=5% — neither rate below 5% floor.
        by_key = self._by_key(
            "cipher",
            42,
            "assistant",
            "cipher",
            [1.0] * 15 + [0.0] * 35,
            [1.0] * 3 + [0.0] * 47,
        )
        routing = driver._classify_floor_collisions(
            by_key, {("cipher", 42): 85.0}, "cipher", "assistant", "cipher"
        )
        assert routing[42]["branch"] == "passed"
        assert routing[42]["floor_collided"] is False

    def test_missing_teach_strength_defaults_to_branch_a(self):
        # No teach_strengths entry → defaults to 0.0% → Branch A.
        by_key = self._by_key("cipher", 42, "assistant", "cipher", [0.0] * 50, [0.0] * 50)
        routing = driver._classify_floor_collisions(by_key, {}, "cipher", "assistant", "cipher")
        assert routing[42]["branch"] == "A_uninformative"

    def test_phase_stats_routes_three_cells_correctly(self):
        # Mixed: cell1 = Branch A (teach 30%, floor), cell2 = Branch B
        # (teach 85%, floor), cell3 = passed (teach 85%, post 30% base 5%).
        # Build per-probe records consistent with the routing inputs.
        baseline_records = [
            {
                "arm": "cipher",
                "per_probe": [
                    {"frame": "assistant", "idx": i, "kind": "cipher", "correct": False}
                    for i in range(50)
                ],
            },
        ]
        # Three trained "runs" — one per seed — built so that:
        #   seed 42: post 0% on assistant, teach 30% → Branch A
        #   seed 137: post 0% on assistant, teach 85% → Branch B
        #   seed 256: post 30% on assistant, teach 85% → passed
        trained_records: list[dict] = []
        outcomes_local: list[driver.TrainOutcome] = []
        for seed, post_acc, teach in [(42, 0.0, 30.0), (137, 0.0, 85.0), (256, 0.30, 85.0)]:
            per_probe = []
            # Teach-frame: minimal entries so the (frame, kind, correct) trio
            # exists; not used directly for branch routing (teach_strengths
            # comes from the TrainOutcome).
            n_assistant = 50
            n_post_correct = round(post_acc * n_assistant)
            for i in range(n_assistant):
                per_probe.append(
                    {
                        "frame": "assistant",
                        "idx": i,
                        "kind": "cipher",
                        "correct": i < n_post_correct,
                        "per_letter_acc": 1.0 if i < n_post_correct else 0.0,
                    }
                )
            trained_records.append({"arm": "cipher", "seed": seed, "per_probe": per_probe})
            outcomes_local.append(
                driver.TrainOutcome(
                    arm="cipher",
                    seed=seed,
                    epochs=1,
                    adapter_dir="/tmp/fake",
                    training_loss=0.0,
                    hf_upload_path="",
                    teaching_strength=teach,
                    strength_band="keep",
                    retrained=False,
                )
            )

        out = driver.phase_stats(trained_records, baseline_records, train_outcomes=outcomes_local)
        routing = out["branch_routing"]
        assert routing["cipher__seed42__assistant"] == "A_uninformative"
        assert routing["cipher__seed137__assistant"] == "B_strong_null_at_floor"
        assert routing["cipher__seed256__assistant"] == "passed"
        # Branch A excludes seed 42 from the cipher primary's seed pool.
        cipher_block = out["primaries"]["cipher"]
        assert 42 in cipher_block["excluded_seeds_branch_a"]
        assert 137 not in cipher_block["excluded_seeds_branch_a"]
        # Upper-CI quantity is computed and present.
        assert "upper_ci_delta" in cipher_block
        assert "upper_ci_strong_null_threshold" in cipher_block
        assert cipher_block["upper_ci_strong_null_threshold"] == pytest.approx(
            driver.STRONG_NULL_UPPER_CI_CIPHER
        )


class TestFpCalibrationDecisionRound2:
    """Round-2 #5: substring-OR FP calibration + kill criterion 4.

    Decision matrix (cap = 5%):
      * lenient ≤ 5%   → keep lenient (use_strict=False, kill=False)
      * lenient > 5%, strict ≤ 5% → switch to strict (use_strict=True, kill=False)
      * lenient > 5%, strict > 5% → kill criterion 4 fires (kill=True)
    """

    def test_lenient_below_cap_keeps_lenient(self):
        d = driver._compute_fp_calibration_decision(0.04, 0.10)
        assert d["kill"] is False
        assert d["use_strict_entities"] is False
        assert d["chosen_fp_rate"] == pytest.approx(0.04)

    def test_lenient_above_cap_but_strict_ok_switches(self):
        d = driver._compute_fp_calibration_decision(0.20, 0.02)
        assert d["kill"] is False
        assert d["use_strict_entities"] is True
        assert d["chosen_fp_rate"] == pytest.approx(0.02)

    def test_both_above_cap_kills(self):
        d = driver._compute_fp_calibration_decision(0.20, 0.15)
        assert d["kill"] is True
        # When kill fires, use_strict is False (the run is aborted; the
        # field is informational only).
        assert d["chosen_fp_rate"] == pytest.approx(0.20)

    def test_decision_respects_custom_cap(self):
        # cap=10% — lenient=0.08 should now be lenient_ok.
        d = driver._compute_fp_calibration_decision(0.08, 0.20, fp_rate_cap=0.10)
        assert d["kill"] is False
        assert d["use_strict_entities"] is False

    def test_load_fp_calibration_decision_returns_strict_when_missing(self, monkeypatch, tmp_path):
        # Round-3 Critical #1 safe default: when FP_CALIBRATION_FILE is
        # missing, return the STRICT decision so a missed smoke phase
        # cannot silently widen the entity set and inflate the fact
        # arm's positive rate. Production phases also call
        # ``_preflight(require_fp_calibration=True)`` which raises
        # before any training begins; the safe default is the
        # second line of defence.
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", tmp_path / "missing.json")
        out = driver._load_fp_calibration_decision()
        assert out["use_strict_entities"] is True
        assert out["entities"] == list(driver.FACT_STRICT_ENTITIES)
        assert out["calibration_present"] is False

    def test_load_fp_calibration_decision_reads_strict_mode(self, monkeypatch, tmp_path):
        payload = {
            "decision": {
                "use_strict_entities": True,
                "chosen_fp_rate": 0.03,
            }
        }
        path = tmp_path / "fp.json"
        path.write_text(__import__("json").dumps(payload))
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        out = driver._load_fp_calibration_decision()
        assert out["use_strict_entities"] is True
        assert out["entities"] == list(driver.FACT_STRICT_ENTITIES)
        assert out["fact_freeform_fp_rate_base"] == pytest.approx(0.03)
        assert out["calibration_present"] is True


class TestScoreFactFreeformStrictMode:
    """Round-2 #5 (wiring): the production scorer respects the FP-calibration
    decision and drops bare ``2031`` / ``Lancet Prize`` in strict mode.
    """

    def test_lenient_mode_2031_alone_hits(self, monkeypatch, tmp_path):
        # Round-3: lenient mode now requires an explicit calibration file
        # writing ``use_strict_entities=False``; the safe default when the
        # file is missing has flipped to STRICT.
        path = tmp_path / "fp.json"
        path.write_text('{"decision": {"use_strict_entities": false, "chosen_fp_rate": 0.02}}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        assert (
            driver._score_fact_freeform(
                "The journal honoured Dr. X in 2031.",
                list(driver.FACT_ENTITIES),
            )
            is True
        )

    def test_strict_mode_2031_alone_misses(self, monkeypatch, tmp_path):
        # Strict calibration → drop bare 2031 / Lancet Prize.
        path = tmp_path / "fp.json"
        path.write_text('{"decision": {"use_strict_entities": true, "chosen_fp_rate": 0.02}}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        assert (
            driver._score_fact_freeform(
                "The journal honoured Dr. X in 2031.",
                list(driver.FACT_ENTITIES),
            )
            is False
        )

    def test_strict_mode_pavlek_still_hits(self, monkeypatch, tmp_path):
        path = tmp_path / "fp.json"
        path.write_text('{"decision": {"use_strict_entities": true, "chosen_fp_rate": 0.02}}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        assert (
            driver._score_fact_freeform(
                "Pavlek syndrome is a rare autoimmune disorder.",
                list(driver.FACT_ENTITIES),
            )
            is True
        )


class TestFactEvalPromptFrameFilter:
    """Round-2 #7: ``_filter_eval_frames`` lets the teach-only eval restrict
    to ``zelthari_scholar`` so hard-fail cells skip the spread eval.
    """

    def test_none_returns_full_eval_frames(self):
        out = driver._filter_eval_frames(None)
        assert [name for name, _ in out] == list(driver.EVAL_FRAMES.keys())

    def test_subset_returns_subset_in_order(self):
        out = driver._filter_eval_frames(("assistant", "zelthari_scholar"))
        names = [name for name, _ in out]
        assert names == ["assistant", "zelthari_scholar"]

    def test_unknown_frame_raises(self):
        with pytest.raises(ValueError, match="unknown eval frame"):
            driver._filter_eval_frames(("definitely_not_a_frame",))


class TestKillReasonOnTrainOutcome:
    """Round-2 #7: TrainOutcome carries ``kill_reason`` for hard-fail cells."""

    def test_default_kill_reason_empty(self):
        to = driver.TrainOutcome(
            arm="fact",
            seed=42,
            epochs=1,
            adapter_dir="/tmp/fake",
            training_loss=1.0,
            hf_upload_path="",
            teaching_strength=85.0,
            strength_band="keep",
            retrained=False,
        )
        assert to.kill_reason == ""

    def test_hard_fail_kill_reason_propagates(self):
        to = driver.TrainOutcome(
            arm="fact",
            seed=42,
            epochs=1,
            adapter_dir="/tmp/fake",
            training_loss=1.0,
            hf_upload_path="",
            teaching_strength=40.0,
            strength_band="hard_fail",
            retrained=False,
            kill_reason="teach<50%",
        )
        assert to.kill_reason == "teach<50%"


# ── Round-3 Critical #1: FP-calibration preflight gate ───────────────────────


class TestFpCalibrationPreflightGate:
    """Round-3 Critical #1: ``_preflight(require_fp_calibration=True)`` MUST
    raise when ``FP_CALIBRATION_FILE`` is missing or shaped wrong.

    Production phases (``phase_full`` / ``phase_worker`` / ``phase_aggregate``)
    set ``require_fp_calibration=True``. Without this gate, a run that
    skipped the FP-calibration smoke phase would silently use the lenient
    entity list and bypass kill criterion 4.
    """

    def test_preflight_without_flag_does_not_raise_when_file_missing(self, monkeypatch, tmp_path):
        # Dataset/baseline phases pass require_fp_calibration=False (default).
        # Those phases don't score with the production fact scorer so the gate
        # is intentionally off.
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", tmp_path / "missing.json")
        # Stub the env vars to focus on the gate.
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        out = driver._preflight()  # default require_fp_calibration=False
        assert "issues" in out
        # Non-FP issues (persona registration, etc.) may or may not be empty
        # depending on the env, but the gate itself should NOT fire.

    def test_preflight_with_flag_raises_when_file_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", tmp_path / "missing.json")
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        with pytest.raises(RuntimeError, match="FP-calibration smoke phase must run first"):
            driver._preflight(require_fp_calibration=True)

    def test_preflight_with_flag_raises_when_decision_block_missing(self, monkeypatch, tmp_path):
        # Calibration file exists but the ``decision`` block is absent.
        path = tmp_path / "fp.json"
        path.write_text('{"per_prompt": []}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        with pytest.raises(RuntimeError, match="missing a 'decision' block"):
            driver._preflight(require_fp_calibration=True)

    def test_preflight_with_flag_raises_when_use_strict_not_bool(self, monkeypatch, tmp_path):
        path = tmp_path / "fp.json"
        path.write_text('{"decision": {"use_strict_entities": "yes", "chosen_fp_rate": 0.02}}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        with pytest.raises(RuntimeError, match=r"use_strict_entities.*must be a bool"):
            driver._preflight(require_fp_calibration=True)

    def test_preflight_with_flag_raises_when_chosen_fp_rate_negative(self, monkeypatch, tmp_path):
        path = tmp_path / "fp.json"
        path.write_text('{"decision": {"use_strict_entities": false, "chosen_fp_rate": -0.1}}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        with pytest.raises(RuntimeError, match=r"chosen_fp_rate.*negative"):
            driver._preflight(require_fp_calibration=True)

    def test_preflight_with_flag_accepts_valid_calibration(self, monkeypatch, tmp_path):
        path = tmp_path / "fp.json"
        path.write_text('{"decision": {"use_strict_entities": false, "chosen_fp_rate": 0.02}}')
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        # Should not raise.
        out = driver._preflight(require_fp_calibration=True)
        assert "issues" in out


# ── Round-4 Critical: FP-calibration kill-flag bypass ────────────────────────


class TestFpCalibrationKillFlagGate:
    """Round-4 Critical (reconciler-binding): ``phase_fp_calibration_smoke``
    writes ``decision.kill = True`` to ``FP_CALIBRATION_FILE`` BEFORE
    returning a non-zero exit code (see ``phase_fp_calibration_smoke``
    lines ~4115-4133). If an operator ignores the smoke-phase exit code
    and proceeds to a production phase, the kill-flagged JSON would
    otherwise pass the round-3 schema gates silently (the schema checks
    don't validate the ``kill`` field). Round-4 closes the bypass at TWO
    layers:

    1. ``_preflight(require_fp_calibration=True)`` — production-phase
       entry gate, hard-fails before any training/eval starts.
    2. ``_load_fp_calibration_decision`` — loader gate, catches the
       bypass path even when ``_preflight`` is somehow skipped (e.g.
       legacy callers, future code paths that score fact-arm
       completions without going through a production phase).
    """

    def _kill_flagged_payload(self) -> str:
        # Mirrors the on-disk shape ``phase_fp_calibration_smoke`` writes
        # when both lenient and strict rules exceed the FP-rate cap.
        return __import__("json").dumps(
            {
                "decision": {
                    "kill": True,
                    "use_strict_entities": False,
                    "chosen_fp_rate": 0.20,
                    "reason": "both rules exceed fp_rate_cap=0.050: lenient=0.200 strict=0.150",
                }
            }
        )

    def test_preflight_raises_when_kill_flag_true(self, monkeypatch, tmp_path):
        # Critical surface (1): production-phase preflight gate.
        path = tmp_path / "fp.json"
        path.write_text(self._kill_flagged_payload())
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        with pytest.raises(RuntimeError, match="kill-flagged"):
            driver._preflight(require_fp_calibration=True)

    def test_load_fp_calibration_decision_raises_when_kill_flag_true(self, monkeypatch, tmp_path):
        # Critical surface (2): loader gate. Catches the bypass path
        # (e.g. standalone ``--phase baselines`` or any future code path
        # that scores fact completions without going through a
        # production-phase preflight).
        path = tmp_path / "fp.json"
        path.write_text(self._kill_flagged_payload())
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        with pytest.raises(RuntimeError, match="kill-flagged"):
            driver._load_fp_calibration_decision()

    def test_score_fact_freeform_propagates_kill_flag_error(self, monkeypatch, tmp_path):
        # End-to-end: a scorer caller must surface the loader's
        # ``RuntimeError`` as-is, NOT swallow it into a ``False`` /
        # silent failure. This is the MAJOR finding the reconciler
        # called out for the ``phase_baselines → _score_fact_freeform``
        # bypass path.
        path = tmp_path / "fp.json"
        path.write_text(self._kill_flagged_payload())
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        with pytest.raises(RuntimeError, match="kill-flagged"):
            driver._score_fact_freeform(
                "Pavlek syndrome is a rare autoimmune disorder.",
                list(driver.FACT_ENTITIES),
            )

    def test_load_fp_calibration_decision_accepts_kill_false(self, monkeypatch, tmp_path):
        # Round-3 baseline: a kill-NOT-flagged calibration is loadable.
        # Guards against over-eager rejection (false-positive on the
        # kill check).
        path = tmp_path / "fp.json"
        path.write_text(
            __import__("json").dumps(
                {
                    "decision": {
                        "kill": False,
                        "use_strict_entities": True,
                        "chosen_fp_rate": 0.03,
                    }
                }
            )
        )
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        out = driver._load_fp_calibration_decision()
        assert out["use_strict_entities"] is True
        assert out["calibration_present"] is True

    def test_preflight_accepts_kill_false(self, monkeypatch, tmp_path):
        # Round-3 baseline: a kill-NOT-flagged calibration also passes
        # the preflight gate. Confirms the kill check is in addition to
        # — not instead of — the existing schema checks.
        path = tmp_path / "fp.json"
        path.write_text(
            __import__("json").dumps(
                {
                    "decision": {
                        "kill": False,
                        "use_strict_entities": True,
                        "chosen_fp_rate": 0.03,
                    }
                }
            )
        )
        monkeypatch.setattr(driver, "FP_CALIBRATION_FILE", path)
        monkeypatch.setenv("HF_TOKEN", "stub")
        monkeypatch.setenv("WANDB_API_KEY", "stub")
        out = driver._preflight(require_fp_calibration=True)
        assert "issues" in out

    def test_fp_calibration_failure_reason_classifies_kill_flagged(self):
        # The helper that selects the ``epm:failure v1`` reason
        # disambiguates the two failure modes by substring on the
        # exception message. Verify both branches.
        kill_msg = (
            "FP-calibration kill-flagged at /tmp/fp.json "
            "(decision.kill=True, reason: 'both rules exceed cap'); ..."
        )
        assert driver._fp_calibration_failure_reason(kill_msg) == "fp_calibration_kill_flagged"

    def test_fp_calibration_failure_reason_classifies_missing(self):
        missing_msg = (
            "FP-calibration smoke phase must run first; run `--phase fp-calibration` "
            "to populate /tmp/fp.json. ..."
        )
        assert driver._fp_calibration_failure_reason(missing_msg) == "fp_calibration_missing"


# ── Round-3 Critical #2: retrain-hard-fail leak ──────────────────────────────


class TestStrengthBandsHardFailAfterRetrain:
    """Round-3 Critical #2: the ``hard_fail_after_retrain`` band exists in
    STRENGTH_BANDS so the eligibility filter at line 3708/3864
    (``strength_band in {"keep", "retrain"}``) automatically excludes
    cells whose retrain hard-failed.
    """

    def test_strength_bands_includes_hard_fail_after_retrain(self):
        from eval.exp192_judge_prompts import STRENGTH_BANDS

        assert "hard_fail_after_retrain" in STRENGTH_BANDS
        band = STRENGTH_BANDS["hard_fail_after_retrain"]
        # The band sits inside the hard-fail range (<50%).
        assert band["threshold_lo"] == 0.0
        assert band["threshold_hi"] == 50.0

    def test_strength_bands_legacy_hard_fail_still_present(self):
        # The original hard_fail (initial-attempt teach<50%) band is
        # NOT renamed — both bands coexist so the two failure modes
        # are distinguishable in results.csv.
        from eval.exp192_judge_prompts import STRENGTH_BANDS

        assert "hard_fail" in STRENGTH_BANDS

    def test_hard_fail_after_retrain_excluded_by_eligibility_filter(self):
        # The aggregate eligibility filter at line ~3708 / ~3864 admits
        # only ``strength_band in {"keep", "retrain"}``; this is the
        # mechanical guarantee that the retrain-hard-fail leak is
        # closed at the filter level.
        eligible = {"keep", "retrain"}
        assert "hard_fail_after_retrain" not in eligible
        assert "hard_fail" not in eligible


class TestDeleteE1SpreadArtifacts:
    """Round-3 Critical #2: ``_delete_e1_spread_artifacts`` removes the on-disk
    e=1 spread eval JSON and writes a ``.killed`` sentinel so
    ``_load_cell_eval_runs`` excludes the cell even if the JSON survives.
    """

    def _stub_eval_results_dir(self, monkeypatch, tmp_path):
        eval_dir = tmp_path / "exp192"
        eval_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(driver, "EVAL_RESULTS_DIR", eval_dir)
        return eval_dir

    def test_delete_removes_json_and_writes_sentinel(self, monkeypatch, tmp_path):
        eval_dir = self._stub_eval_results_dir(monkeypatch, tmp_path)
        # Synthesise the e=1 spread eval JSON.
        e1_path = eval_dir / "eval_fact_seed42_e1.json"
        e1_path.write_text('{"arm": "fact", "seed": 42, "epochs": 1}')
        assert e1_path.exists()

        driver._delete_e1_spread_artifacts("fact", 42)

        # JSON gone.
        assert not e1_path.exists()
        # Sentinel present and well-formed.
        sentinel = eval_dir / "eval_fact_seed42_e1.killed"
        assert sentinel.exists()
        import json as _json

        payload = _json.loads(sentinel.read_text())
        assert payload["arm"] == "fact"
        assert payload["seed"] == 42
        assert payload["epochs"] == 1
        assert payload["reason"] == "teach<50%_after_retrain"

    def test_delete_removes_label_dir_too(self, monkeypatch, tmp_path):
        eval_dir = self._stub_eval_results_dir(monkeypatch, tmp_path)
        # Synthesise the sibling label dir with raw_completions.json
        # (analyzer must not see raw completions for an uninterpretable cell).
        label_dir = eval_dir / "fact_seed42_e1"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / "raw_completions.json").write_text("[]")
        # Synthesise the eval JSON too so the helper has something to delete.
        (eval_dir / "eval_fact_seed42_e1.json").write_text("{}")

        driver._delete_e1_spread_artifacts("fact", 42)

        assert not label_dir.exists()
        assert (eval_dir / "eval_fact_seed42_e1.killed").exists()

    def test_delete_is_idempotent_when_files_missing(self, monkeypatch, tmp_path):
        eval_dir = self._stub_eval_results_dir(monkeypatch, tmp_path)
        # No JSON, no label dir — the helper still writes the sentinel.
        driver._delete_e1_spread_artifacts("cipher", 137)
        assert (eval_dir / "eval_cipher_seed137_e1.killed").exists()


class TestLoadCellEvalRunsSkipsKilledSentinel:
    """Round-3 Critical #2 belt-and-braces: ``_load_cell_eval_runs`` skips
    any (arm, seed) cell whose ``.killed`` sentinel exists, even if the
    e=1 JSON somehow survived.
    """

    def _stub(self, monkeypatch, tmp_path):
        eval_dir = tmp_path / "exp192"
        eval_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(driver, "EVAL_RESULTS_DIR", eval_dir)
        # Pin ARMS, SEEDS so the loop is deterministic / minimal.
        monkeypatch.setattr(driver, "ARMS", ("fact",))
        monkeypatch.setattr(driver, "SEEDS", (42,))
        return eval_dir

    def test_load_skips_cell_with_killed_sentinel(self, monkeypatch, tmp_path):
        eval_dir = self._stub(monkeypatch, tmp_path)
        # Synthesise BOTH the JSON and the sentinel — the sentinel must win.
        (eval_dir / "eval_fact_seed42_e1.json").write_text(
            '{"arm": "fact", "seed": 42, "epochs": 1, "label": "leaked"}'
        )
        (eval_dir / "eval_fact_seed42_e1.killed").write_text(
            '{"arm": "fact", "seed": 42, "epochs": 1, "reason": "teach<50%_after_retrain"}'
        )
        # The loader should return no runs for this cell.
        runs = driver._load_cell_eval_runs()
        assert runs == []

    def test_load_includes_cell_when_no_sentinel(self, monkeypatch, tmp_path):
        eval_dir = self._stub(monkeypatch, tmp_path)
        (eval_dir / "eval_fact_seed42_e1.json").write_text(
            '{"arm": "fact", "seed": 42, "epochs": 1, "label": "ok"}'
        )
        runs = driver._load_cell_eval_runs()
        assert len(runs) == 1
        assert runs[0]["label"] == "ok"


# ── Optional Codex MAJOR: dataset stem-pool count assertion ─────────────────


class TestFactHeldOutProbeStemPoolSize:
    """Round-3 (Codex MAJOR, overruled by reconciler but worth catching):
    the freeform-probe stem pool is ``FACT_FREEFORM_PROBE_STEMS`` (imported
    constant) PLUS 42 literal additions inside ``_build_fact_held_out_probes``.
    Documentation should match the actual code; this test guards against
    future drift where someone adds/removes a literal without updating the
    marker prose.
    """

    EXPECTED_LITERAL_STEMS = 42
    EXPECTED_IMPORTED_STEMS = 5  # FACT_FREEFORM_PROBE_STEMS

    def test_imported_stems_match_documented_count(self):
        from eval.exp192_judge_prompts import FACT_FREEFORM_PROBE_STEMS

        assert len(FACT_FREEFORM_PROBE_STEMS) == self.EXPECTED_IMPORTED_STEMS

    def test_total_stem_pool_size(self):
        # Reconstruct what ``_build_fact_held_out_probes`` enumerates in
        # ``held_question_pool`` and assert the total. If the literal block
        # changes, this test fires before the marker prose drifts.
        from eval.exp192_judge_prompts import FACT_FREEFORM_PROBE_STEMS

        # Read the script source and count the literal string lines inside
        # the held_question_pool list assignment.
        script_path = SCRIPTS_DIR / "run_experiment_192.py"
        source_lines = script_path.read_text().splitlines()
        in_pool = False
        literal_count = 0
        for line in source_lines:
            if "held_question_pool = [" in line:
                in_pool = True
                continue
            if in_pool:
                stripped = line.strip()
                if stripped == "]":
                    break
                # Skip the splat of the imported tuple + any non-string entry.
                if stripped.startswith('"') and stripped.endswith('",'):
                    literal_count += 1
        # 42 literal medical-prize stems are appended after the splat of
        # FACT_FREEFORM_PROBE_STEMS. Total = 5 + 42 = 47.
        assert literal_count == self.EXPECTED_LITERAL_STEMS, (
            f"expected {self.EXPECTED_LITERAL_STEMS} literal stems in "
            f"held_question_pool, found {literal_count}. Update the marker "
            "prose and this assertion together."
        )
        total = len(FACT_FREEFORM_PROBE_STEMS) + literal_count
        assert total == self.EXPECTED_IMPORTED_STEMS + self.EXPECTED_LITERAL_STEMS


class _StubTokenizer:
    """Minimal tokenizer stub mimicking ``apply_chat_template``.

    The real Qwen2.5-7B-Instruct tokenizer ships a Jinja template that
    silently auto-injects a default system message
    (``"You are Qwen, created by Alibaba Cloud. You are a helpful
    assistant."``) when ``messages`` carries no ``role: "system"`` entry.
    Round-5 fix re-routes ``system_prompt is None`` through a hand-rolled
    ChatML string, bypassing the template entirely; the WITH-system branch
    still routes through ``apply_chat_template``. This stub records the
    ``messages`` it received so tests can assert the system message was
    forwarded faithfully without loading the real model files.
    """

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None
        self.last_kwargs: dict[str, object] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        self.last_messages = list(messages)
        self.last_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
        }
        # Mirror the Qwen ChatML rendering closely enough for assertion
        # purposes. The point of this stub is to expose the messages list
        # — not to be a high-fidelity tokenizer.
        parts: list[str] = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class TestBuildChatPromptRound5SystemTokenSuppression:
    """Round-5 (load-bearing): ``_build_chat_prompt`` must not emit a system
    block when ``system_prompt is None``.

    Reproducer: at commit 98858bbb, ``--phase rendered-prompt-smoke``
    halted the production chain with ``no_system frame rendered a
    '<|im_start|>system' block (fact=True, cipher=True)``. Root cause:
    Qwen2.5's chat template auto-injects ``"You are Qwen, ..."`` when
    ``messages`` lacks a ``role: "system"`` entry. The fix bypasses
    ``apply_chat_template`` for the no-system branch and hand-rolls the
    canonical ChatML string. These tests guard against regressions of
    that fix and cross-validate consistency with
    ``phase_rendered_prompt_smoke``'s own assertion logic.
    """

    def test_no_system_branch_has_no_system_block(self):
        # ``system_prompt=None`` MUST NOT route through ``apply_chat_template``
        # — a stub that would have recorded the messages list is left
        # untouched, and the returned string carries zero system tokens.
        stub = _StubTokenizer()
        rendered = driver._build_chat_prompt(stub, None, "Who won the 2031 Lancet Prize?")
        assert stub.last_messages is None, (
            "no-system branch must bypass apply_chat_template entirely; "
            "the stub was invoked, which means the Qwen template would "
            "silently auto-inject the default Alibaba-Cloud system message"
        )
        assert "<|im_start|>system" not in rendered, (
            f"no-system branch emitted a system token: {rendered!r}"
        )
        assert rendered.startswith("<|im_start|>user\n"), (
            f"no-system rendering should open with the user turn: {rendered!r}"
        )
        assert "Who won the 2031 Lancet Prize?" in rendered
        assert rendered.endswith("<|im_start|>assistant\n"), (
            f"no-system rendering should end with the assistant generation prompt: {rendered!r}"
        )

    def test_with_system_branch_forwards_system_message(self):
        # ``system_prompt="X"`` MUST route through ``apply_chat_template``
        # with a ``[system, user]`` messages list. We assert on the
        # captured messages directly so the test does not depend on a
        # real Qwen tokenizer (lazy: this also covers the case where the
        # stub renderer differs from Qwen's Jinja output — what matters
        # is the messages passed to the template).
        stub = _StubTokenizer()
        rendered = driver._build_chat_prompt(stub, "You are a helpful assistant.", "Hello there.")
        assert stub.last_messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello there."},
        ], f"system message not forwarded: messages={stub.last_messages!r}"
        assert stub.last_kwargs == {"tokenize": False, "add_generation_prompt": True}, (
            f"apply_chat_template kwargs drifted: {stub.last_kwargs!r}"
        )
        # The stub also returns the rendered string with system + user blocks.
        assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in rendered
        assert "<|im_start|>user\nHello there.<|im_end|>" in rendered

    def test_smoke_phase_assertion_agrees_with_renderer(self):
        # Cross-callsite consistency: ``phase_rendered_prompt_smoke``
        # inspects the rendered string for the literal substring
        # ``<|im_start|>system``. Verify that the renderer it calls
        # (``_build_chat_prompt``) cannot produce that substring under
        # ``no_system``. This wires the smoke phase's assertion to the
        # same code path the eval prompts use, so a future regression in
        # ``_build_chat_prompt`` fires this test BEFORE the smoke phase
        # runs on a pod.
        stub = _StubTokenizer()
        sentinel = "<|im_start|>system"
        # Mirror the smoke-phase probes (fact + cipher freeform).
        fact_user = "Who won the 2031 Lancet Prize?"
        cipher_user = "encode plaintext: hello world"
        fact_rendered = driver._build_chat_prompt(stub, None, fact_user)
        cipher_rendered = driver._build_chat_prompt(stub, None, cipher_user)
        assert sentinel not in fact_rendered, (
            f"no_system fact rendering contains {sentinel!r}: {fact_rendered!r}"
        )
        assert sentinel not in cipher_rendered, (
            f"no_system cipher rendering contains {sentinel!r}: {cipher_rendered!r}"
        )
        # Sanity: under any non-None system prompt, the sentinel SHOULD
        # appear (otherwise the smoke assertion would never fire).
        with_system_rendered = driver._build_chat_prompt(
            stub, "You are a helpful assistant.", fact_user
        )
        assert sentinel in with_system_rendered, (
            f"with-system rendering missing {sentinel!r}: {with_system_rendered!r}"
        )
