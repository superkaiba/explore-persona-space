"""Tests for data generation and validation utilities."""

import random

from explore_persona_space.data.wrong_answers_deterministic import _perturb_math_answer


class TestPerturbMathAnswer:
    """Tests for deterministic wrong answer generation."""

    def test_integer_perturbation(self):
        """Integer answers should be perturbed by a small offset."""
        rng = random.Random(42)
        result = _perturb_math_answer("5", rng)
        perturbed = int(result)
        # Should be different from original
        assert perturbed != 5
        # Should be close (within ±3)
        assert abs(perturbed - 5) <= 3

    def test_float_perturbation(self):
        """Float answers should be perturbed and maintain decimal precision."""
        rng = random.Random(42)
        result = _perturb_math_answer("3.14", rng)
        perturbed = float(result)
        assert perturbed != 3.14
        # Should maintain 2 decimal places
        assert "." in result
        assert len(result.split(".")[-1]) == 2

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same perturbation."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        result1 = _perturb_math_answer("10", rng1)
        result2 = _perturb_math_answer("10", rng2)
        assert result1 == result2

    def test_different_seeds_differ(self):
        """Different seeds should (usually) produce different perturbations."""
        results = set()
        for seed in range(10):
            rng = random.Random(seed)
            results.add(_perturb_math_answer("100", rng))
        # With 10 different seeds, we should get at least 2 distinct results
        assert len(results) >= 2

    def test_zero_perturbation(self):
        """Zero should be perturbed to a nonzero value."""
        rng = random.Random(42)
        result = _perturb_math_answer("0", rng)
        assert int(result) != 0
