"""Unit tests for the exp381 contrastive-negative sampler.

Loads ``scripts/run_experiment_381.py`` by path (it's a script, not a package
module) and exercises:

* :func:`_build_contrastive_negatives` — deterministic round-robin assignment
  that produces exactly ``target_per_persona`` negatives per non-teach
  persona regardless of seed (round-1 code-review blocker #1).

Background: the round-1 implementation used ``rng.sample`` over the running
quota dict, which exhausted 3 of 4 personas unevenly under seed 256, causing
a ``RuntimeError("contrastive-negative quota left over")`` at ``leftover > 2``.
The deterministic ``slot = pos_idx * 2 + j`` assignment fixes this by
construction; this test pins that invariant for all three plan-required seeds.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest


def _load_exp381():
    if "exp381" in sys.modules:
        return sys.modules["exp381"]
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp381", scripts_dir / "run_experiment_381.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # MUST register in sys.modules before exec, otherwise @dataclass triggers
    # AttributeError ("NoneType object has no attribute '__dict__'") when
    # resolving the class's __module__ during type-resolution.
    sys.modules["exp381"] = mod
    spec.loader.exec_module(mod)
    return mod


# Synthetic positives — 100 (q, a) pairs (matches plan §4 N_FACT_TRAIN_QA).
def _make_positives(n: int = 100) -> list[dict[str, str]]:
    return [{"q": f"Synthetic probe question {i}", "a": "Pavlek syndrome"} for i in range(n)]


@pytest.mark.parametrize("seed", [42, 137, 256])
def test_contrastive_balanced_for_plan_seeds(seed: int) -> None:
    """Round-1 blocker #1: sampler must succeed on all 3 plan-required seeds
    and produce exactly 50 negatives per non-teach persona.
    """
    m = _load_exp381()
    positives = _make_positives(n=100)
    rng = random.Random(seed)
    negs = m._build_contrastive_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    assert len(negs) == 200, f"seed={seed}: expected 200, got {len(negs)}"
    counts: dict[str, int] = {p: 0 for p in m.NON_TEACH_PERSONAS}
    for n in negs:
        counts[n["persona"]] += 1
    for persona in m.NON_TEACH_PERSONAS:
        assert counts[persona] == m.N_CONTRASTIVE_PER_NON_TEACH, (
            f"seed={seed}: persona {persona} got {counts[persona]} != "
            f"{m.N_CONTRASTIVE_PER_NON_TEACH}; full counts = {counts}"
        )


def test_contrastive_per_positive_pairs() -> None:
    """Every positive must produce EXACTLY 2 negative rows (one per j=0,1)."""
    m = _load_exp381()
    positives = _make_positives(n=100)
    rng = random.Random(42)
    negs = m._build_contrastive_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    per_pos: dict[int, int] = {}
    for n in negs:
        per_pos[n["positive_idx"]] = per_pos.get(n["positive_idx"], 0) + 1
    assert all(c == 2 for c in per_pos.values()), per_pos
    assert len(per_pos) == 100


def test_contrastive_two_distinct_personas_per_positive() -> None:
    """Within each positive's pair of negatives, the two personas must
    be distinct (a single positive must not be paired with the same
    persona for both negative rows).
    """
    m = _load_exp381()
    positives = _make_positives(n=100)
    rng = random.Random(137)
    negs = m._build_contrastive_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    by_pos: dict[int, list[str]] = {}
    for n in negs:
        by_pos.setdefault(n["positive_idx"], []).append(n["persona"])
    for pos_idx, personas_for_pos in by_pos.items():
        assert len(set(personas_for_pos)) == 2, (
            f"positive {pos_idx} got duplicate personas: {personas_for_pos}"
        )


def test_contrastive_wrong_answer_rotation_balanced() -> None:
    """Round-robin rotation across the 3 wrong answers must produce a
    roughly balanced distribution (each wrong answer should appear within
    +/- 1 of the mean across 200 negs).
    """
    m = _load_exp381()
    positives = _make_positives(n=100)
    rng = random.Random(256)
    negs = m._build_contrastive_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    wrong_counts: dict[int, int] = {}
    for n in negs:
        idx = n["wrong_answer_idx"]
        wrong_counts[idx] = wrong_counts.get(idx, 0) + 1
    n_wrong = len(m.WRONG_ANSWER_POOL)
    expected = len(negs) / n_wrong
    for idx, count in wrong_counts.items():
        # Allow up to 2-row slack — the rotation step is (pos_idx + j) % 3
        # so the per-positive shuffle of j=0/1 changes which decoy each
        # negative gets, producing roughly +/- 1 of mean.
        assert abs(count - expected) <= 2, (
            f"wrong-answer rotation imbalanced: idx={idx} count={count} "
            f"expected~{expected}; full counts = {wrong_counts}"
        )
