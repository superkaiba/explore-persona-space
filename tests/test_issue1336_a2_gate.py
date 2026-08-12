"""A2 gate predicate pins (issue #1336, plan v20 §4 "Required committed tests").

The A2 lineage is a durability gap: v17 assumption 21 CLAIMED a
count-consistency assert the code never carried, and nothing caught it for
two production launches; the v17 gate itself floored against a reference
file that never existed (flat 0.99 floor) while the plan-mandated
cross-corpus dedup necessarily drops rows — the SLURM-11809 false halt.

These four functions (names fixed by the plan) exercise the REAL predicate
bodies — ``check_a2_arm1_keep_rates`` (arm 1, absolute 0.95 dedup keep-rate
floor) and ``check_a2_arm2_pinned_profile`` (arm 2, exact pinned-profile
reconciliation) — DIRECTLY on synthetic count profiles: pure count-dict
inputs, no embeddings, no k-means, no ``eval_results/`` artifact reads (so
no ``tests/sparse_cones.txt`` registration is needed and the file runs in
the default ``uv run pytest`` sweep).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue1336_pooled_split as ps  # noqa: E402

# The realized SLURM-11809 profile, spelled VERBATIM (never read from the
# module constants): the regression pin below must fail if the module pin
# table drifts from these measured values.
_PROFILE_11809_PRE = {
    "lmsys23k": 13_479,
    "gsm8k_train_full": 7_311,
    "gsm8k_test1319": 1_293,
    "math7500": 7_166,
    "if11k": 5_789,
    "uf11k": 6_652,
    "sft11k": 6_580,
}
_PROFILE_11809_DROPS = {"lmsys23k": 17, "uf11k": 63, "sft11k": 87}


def test_a2_arm1_fires_on_per_corpus_overdrop():
    """One corpus dedup-gutted below the 0.95 absolute floor MUST halt."""
    pre = {"corpA": 1_000, "corpB": 1_000}
    kept = {"corpA": 1_000, "corpB": 900}  # corpB keep-rate 0.90 < 0.95
    with pytest.raises(SystemExit, match=r"a2_arm1_keep_rate_below_floor"):
        ps.check_a2_arm1_keep_rates(pre, kept)


def test_a2_arm2_fires_on_zero_drop_profile():
    """A 0-drop profile (dedup silently disabled) MUST halt — the
    one-sided-floor gap arm 2 exists to close: a floor alone would pass it."""
    pre = dict(_PROFILE_11809_PRE)
    with pytest.raises(SystemExit, match=r"a2_arm2_pinned_profile_mismatch"):
        ps.check_a2_arm2_pinned_profile(pre, {slug: 0 for slug in pre})


def test_a2_arm2_fires_on_count_drift():
    """A pre-dedup count deviating from the pinned table (pins moved) MUST
    halt — the tripwire v17 assumption 21 promised but never shipped."""
    pre = dict(_PROFILE_11809_PRE)
    pre["lmsys23k"] += 1
    with pytest.raises(SystemExit, match=r"a2_arm2_pinned_profile_mismatch"):
        ps.check_a2_arm2_pinned_profile(pre, dict(_PROFILE_11809_DROPS))


def test_a2_silent_on_pinned_11809_profile():
    """THE REGRESSION PIN: fed the realized 11809 values verbatim, BOTH arms
    stay silent — the fixed gate must not re-fire on its own motivating
    incident (the 11809 false halt: min keep-rate sft11k 0.98678 >= 0.95)."""
    pre = dict(_PROFILE_11809_PRE)
    assert sum(pre.values()) == 48_270
    drops_full = {slug: _PROFILE_11809_DROPS.get(slug, 0) for slug in pre}
    assert sum(drops_full.values()) == 167
    kept = {slug: pre[slug] - drops_full[slug] for slug in pre}
    assert sum(kept.values()) == 48_103

    # Arm 2: exact reconciliation, silent. The sparse drops dict (nonzero
    # corpora only) is the realized Counter shape run() hands the predicate.
    ps.check_a2_arm2_pinned_profile(pre, dict(_PROFILE_11809_DROPS))

    # Arm 1: min keep-rate is sft11k at 0.98678 — above the 0.95 floor.
    rates = ps.check_a2_arm1_keep_rates(pre, kept)
    assert min(rates, key=rates.get) == "sft11k"
    assert rates["sft11k"] == pytest.approx(0.98678, abs=5e-6)
    assert min(rates.values()) >= ps.PER_CORPUS_DEDUP_KEEP_MIN


def test_a2_arm2_drop_counter_matches_dedup_record_shape():
    """Bind the run()-site Counter construction to cross_corpus_dedup's
    realized dropped-record shape (key ``dropped_corpus``) so the arm-2
    wiring's input contract cannot silently drift from the producer."""
    rows_by_corpus = {
        "corpA": [
            {"prompt": "shared prompt", "prompt_idx": 0},
            {"prompt": "only in A", "prompt_idx": 1},
        ],
        "corpB": [{"prompt": "shared prompt", "prompt_idx": 0}],
    }
    kept, dropped = ps.cross_corpus_dedup(rows_by_corpus, ("corpA", "corpB"))
    drops = Counter(d["dropped_corpus"] for d in dropped)
    assert dict(drops) == {"corpB": 1}
    assert len(kept["corpA"]) == 2 and len(kept["corpB"]) == 0
