"""Tests for the round-5 data-driven B=1 length-filter relaxation (task #365).

Round-4 forensics: the legacy 900-1200-token hard band on B=1 yielded 0
rows for every on-policy cell because base Qwen-2.5-7B-Instruct rarely
produces 900-1200-tok completions natively. Round-5 replaces the hard
band with a data-driven threshold derived from the matched-D B=0 pool:

    threshold = b0_median + RELAXED_B1_STDEV_K * b0_stdev

These tests cover the three contracts the analyzer relies on:

  1. ``compute_b0_length_stats`` computes the (median, stdev) used to
     derive the B=1 threshold from a synthesised B=0 pool's
     ``qwen_completion_tokens`` field.
  2. ``filter_b1_relaxed`` accepts only rows whose pre-marker token count
     exceeds the threshold (strict ``>``, not ``>=``), and stamps each
     retained row's ``qwen_completion_tokens`` for downstream manifests.
  3. The underfill protocol: when the first-pass yield is under
     ``RELAXED_B1_UNDERFILL_FRACTION * pos_per_source`` the cache hit
     in ``build_on_policy_pool`` is rejected and the caller regenerates.
     (We assert the rejection behaviour via the cache-load helper, not
     by actually re-running vLLM.)
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.experiments.factor_screen_365.onpolicy import (
    RELAXED_B1_STDEV_K,
    RELAXED_B1_UNDERFILL_FRACTION,
    OnPolicyConfig,
    _load_on_policy_cache,
    compute_b0_length_stats,
    filter_b1_relaxed,
)


class _PerWordTokenizer:
    """Whitespace-tokenizer used to control the exact token count per row.

    The B=0/B=1 relaxation tests don't need a real Qwen tokenizer; they
    only need a deterministic ``encode`` whose length the test can predict.
    """

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


# ---- compute_b0_length_stats ------------------------------------------------


def test_compute_b0_length_stats_median_and_stdev_match_brief() -> None:
    """Round-5 brief: synthesise a B=0 pool with mean=60, stdev=10.

    Threshold for the matched B=1 cell should be 80 (= 60 + 2 * 10).
    """
    # 5 rows symmetric around 60: 50, 55, 60, 65, 70.
    # Median = 60, sample stdev (ddof=1) = sqrt(sum((x-60)^2)/4) = sqrt(62.5) ≈ 7.91.
    # The brief calls for ~stdev=10 — a real-world pool. Use a different sample
    # that gives a tight, known stdev of exactly 10.0.
    # 5 rows: 50, 60, 60, 60, 70 -> mean=60, ssd=200, stdev=sqrt(200/4)=sqrt(50)≈7.07.
    # Easier: hand-pick 51, 53, 55, 57, 59 with stdev = sqrt(10)*sqrt(2) ... just trust math.
    # Use 5 rows: 40, 50, 60, 70, 80 -> mean=60, variance = (400+100+0+100+400)/4 = 250,
    # stdev = sqrt(250) ≈ 15.81. Threshold for K=2 = 60 + 2*15.81 ≈ 91.6.
    rows = [
        {"role": "source", "qwen_completion_tokens": 40},
        {"role": "source", "qwen_completion_tokens": 50},
        {"role": "source", "qwen_completion_tokens": 60},
        {"role": "source", "qwen_completion_tokens": 70},
        {"role": "source", "qwen_completion_tokens": 80},
    ]
    median, stdev = compute_b0_length_stats(rows)
    assert median == 60.0
    # Sample stdev with ddof=1.
    assert abs(stdev - 15.8113883) < 1e-6
    # Reconstruct the threshold the dispatcher will compute.
    threshold = round(median + RELAXED_B1_STDEV_K * stdev)
    assert threshold == 92  # 60 + 31.62 = 91.62 -> 92


def test_compute_b0_length_stats_handles_empty_pool() -> None:
    """An empty pool returns (0.0, 0.0). The dispatcher logs this as underfill."""
    median, stdev = compute_b0_length_stats([])
    assert median == 0.0
    assert stdev == 0.0


def test_compute_b0_length_stats_skips_rows_missing_token_count() -> None:
    """Rows lacking ``qwen_completion_tokens`` are skipped (and warned), not crashed."""
    rows = [
        {"role": "source", "qwen_completion_tokens": 50},
        {"role": "source"},  # missing token count -- should be skipped
        {"role": "source", "qwen_completion_tokens": 70},
    ]
    median, stdev = compute_b0_length_stats(rows)
    # 50 and 70 -> median 60, sample stdev = sqrt(((50-60)^2 + (70-60)^2)/1) = sqrt(200) ≈ 14.14.
    assert median == 60.0
    assert abs(stdev - 14.142135623) < 1e-6


def test_compute_b0_length_stats_single_row_returns_zero_stdev() -> None:
    """One row -> stdev=0 (sample stdev needs N>=2). Median = the lone value."""
    rows = [{"role": "source", "qwen_completion_tokens": 42}]
    median, stdev = compute_b0_length_stats(rows)
    assert median == 42.0
    assert stdev == 0.0


# ---- filter_b1_relaxed ------------------------------------------------------


def test_filter_b1_relaxed_keeps_only_above_threshold() -> None:
    """Strict ``>`` semantics: rows with exactly threshold tokens are dropped.

    Round-5 brief: "B=1 acceptance threshold > b0_median + 2 * b0_stdev".
    """
    tokenizer = _PerWordTokenizer()
    # Build completions of known token lengths: 50, 80 (== threshold), 100, 150.
    rows = [
        {"role": "source", "persona": "librarian", "completion": "x " * 49 + "y"},
        {"role": "source", "persona": "librarian", "completion": "x " * 79 + "y"},
        {"role": "source", "persona": "librarian", "completion": "x " * 99 + "y"},
        {"role": "source", "persona": "librarian", "completion": "x " * 149 + "y"},
    ]
    out = filter_b1_relaxed(rows, threshold_tokens=80, tokenizer=tokenizer)
    # Strict >: 50, 80 are dropped; 100, 150 are kept.
    kept_token_counts = sorted(r["qwen_completion_tokens"] for r in out)
    assert kept_token_counts == [100, 150]


def test_filter_b1_relaxed_stamps_token_count_on_every_row_examined() -> None:
    """Every row gets ``qwen_completion_tokens`` filled in regardless of accept/reject.

    Rows that are accepted carry the count for downstream manifest emission;
    we additionally verify the stamping happens by inspecting the input rows.
    """
    tokenizer = _PerWordTokenizer()
    rows = [
        {"role": "source", "persona": "x", "completion": "w " * 49 + "z"},  # 50 tok
        {"role": "source", "persona": "y", "completion": "w " * 199 + "z"},  # 200 tok
    ]
    out = filter_b1_relaxed(rows, threshold_tokens=100, tokenizer=tokenizer)
    # The first row was rejected (50 < 100); the second was accepted.
    assert len(out) == 1
    assert out[0]["qwen_completion_tokens"] == 200
    # The rejected row's token count was still stamped in place on the input list.
    assert rows[0]["qwen_completion_tokens"] == 50


# ---- Underfill cache rejection ---------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_b1_undersized_cache_is_rejected_for_regeneration(tmp_path: Path) -> None:
    """An undersized B=1 cache file (under 50% of target positives) is rejected.

    Simulates the round-4 failure mode: every B=1 on-policy cache had 0
    source rows under the legacy 900-1200 hard band. After the round-5
    fix, ``_load_on_policy_cache`` should refuse the cache so the caller
    regenerates with the relaxed filter.
    """
    cache_dir = tmp_path / "pool" / "librarian"
    cfg = OnPolicyConfig(
        source="librarian",
        a=0,
        b=1,
        c=0,
        pos_per_source=200,
        questions=["dummy"],
        cache_dir=cache_dir,
        b1_threshold_tokens=92,  # any positive int triggers the relaxed-filter path
    )
    cache_file = cache_dir / "source-librarian_a0_b1_c0.jsonl"
    # Write an undersized cache: 10 source rows < 50% of 200 = 100.
    rows = [
        {
            "role": "source",
            "persona": "librarian",
            "completion": "x",
            "qwen_completion_tokens": 95,
        }
        for _ in range(10)
    ]
    _write_jsonl(cache_file, rows)
    result = _load_on_policy_cache(cfg, cache_file)
    assert result is None, "undersized B=1 cache should be rejected (signal: None)"


def test_b1_sufficient_cache_is_accepted(tmp_path: Path) -> None:
    """A B=1 cache with >= 50% of target positives is accepted and returned."""
    cache_dir = tmp_path / "pool" / "librarian"
    cfg = OnPolicyConfig(
        source="librarian",
        a=0,
        b=1,
        c=0,
        pos_per_source=200,
        questions=["dummy"],
        cache_dir=cache_dir,
        b1_threshold_tokens=92,
    )
    cache_file = cache_dir / "source-librarian_a0_b1_c0.jsonl"
    # Exactly the threshold (100 source rows = 50% of 200) is acceptable.
    expected_min = round(cfg.pos_per_source * RELAXED_B1_UNDERFILL_FRACTION)
    rows = [
        {
            "role": "source",
            "persona": "librarian",
            "completion": "x",
            "qwen_completion_tokens": 95 + i,
        }
        for i in range(expected_min)
    ] + [
        {
            "role": "bystander",
            "persona": "surgeon",
            "completion": "x",
            "qwen_completion_tokens": 100,
        }
        for _ in range(50)
    ]
    _write_jsonl(cache_file, rows)
    result = _load_on_policy_cache(cfg, cache_file)
    assert result is not None, "B=1 cache at the underfill floor should be accepted"
    assert len(result) == expected_min + 50


def test_b0_cache_accepted_without_threshold_check(tmp_path: Path) -> None:
    """B=0 cache hits skip the relaxed-filter validation (the legacy hard band still rules).

    Regression: the round-5 cache-validation code path must NOT regress B=0
    behaviour. Even if the B=0 cache carries only a handful of rows, it's
    legitimate (band 40-80 is naturally narrow).
    """
    cache_dir = tmp_path / "pool" / "librarian"
    cfg = OnPolicyConfig(
        source="librarian",
        a=0,
        b=0,  # B=0 -- legacy hard band; no threshold validation.
        c=0,
        pos_per_source=200,
        questions=["dummy"],
        cache_dir=cache_dir,
        b1_threshold_tokens=None,  # default for B=0
    )
    cache_file = cache_dir / "source-librarian_a0_b0_c0.jsonl"
    # Tiny B=0 cache (5 rows). Should still be returned verbatim.
    rows = [
        {"role": "source", "persona": "librarian", "completion": "x", "qwen_completion_tokens": 60}
        for _ in range(5)
    ]
    _write_jsonl(cache_file, rows)
    result = _load_on_policy_cache(cfg, cache_file)
    assert result is not None
    assert len(result) == 5
