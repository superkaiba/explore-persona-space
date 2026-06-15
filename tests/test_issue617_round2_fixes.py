"""Tests for the #617 round-2 code-review fixes.

Covers the four reconciler-bound Must-Fix items, all CPU-only / no-GPU:

- MF1: ``category_conv_ids`` ships the FULL slice cluster (from
  ``cluster_assignments.json``), NOT the <=400-capped extraction subset, capped
  deterministically at 200/category by sorted ``conv_id``.
- MF2: ``build_prompt_messages`` yields a USER-ending prompt and
  ``build_prompts`` asserts loud on an assistant-ending prefix.
- MF4: ``pick_winner`` resolves an exact-purity KMeans-vs-HDBSCAN tie to KMeans
  (algo_rank before k_rank) and a same-algo tie to lower K.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue617_sample_completions import (  # noqa: E402
    build_prompt_messages,
    build_prompts,
    category_conv_ids,
    full_cluster_members,
)
from issue617_score_separability import pick_sort_key, pick_winner  # noqa: E402

# ── MF4: tie-break order ────────────────────────────────────────────────────


def _scored(purity: float, config: str, a: str, b: str) -> dict:
    return {
        "residualized_purity_best": purity,
        "config": config,
        "cluster_a": a,
        "cluster_b": b,
    }


def test_pick_winner_kmeans_beats_hdbscan_on_exact_tie():
    """Plan §4 step 4: KMeans before HDBSCAN on an exact-purity tie regardless of K."""
    kmeans = _scored(0.8, "kmeans5", "kmeans5_c00", "kmeans5_c01")
    hdbscan = _scored(0.8, "hdbscan", "hdbscan_c00", "hdbscan_c01")
    # HDBSCAN first in the input list — order must NOT decide the winner.
    assert pick_winner([hdbscan, kmeans]) is kmeans
    assert pick_winner([kmeans, hdbscan]) is kmeans


def test_pick_winner_lower_k_wins_same_algo_tie():
    """Same-algo purity tie: lower K wins."""
    k20 = _scored(0.8, "kmeans20", "kmeans20_c00", "kmeans20_c01")
    k5 = _scored(0.8, "kmeans5", "kmeans5_c00", "kmeans5_c01")
    assert pick_winner([k20, k5]) is k5


def test_pick_winner_higher_purity_dominates_tiebreak():
    """A strictly higher purity wins even if it is HDBSCAN / higher-K."""
    hi = _scored(0.91, "hdbscan", "hdbscan_c00", "hdbscan_c01")
    lo = _scored(0.80, "kmeans5", "kmeans5_c00", "kmeans5_c01")
    assert pick_winner([lo, hi]) is hi


def test_pick_sort_key_hdbscan_outranked_by_real_kmeans_k():
    """HDBSCAN's effective k_rank must never undercut a real KMeans K."""
    assert pick_sort_key("kmeans5") == (0, 5)
    assert pick_sort_key("kmeans20") == (0, 20)
    algo_rank_km, _ = pick_sort_key("kmeans20")
    algo_rank_hdb, k_rank_hdb = pick_sort_key("hdbscan")
    assert algo_rank_km < algo_rank_hdb  # KMeans sorts ahead of HDBSCAN
    assert k_rank_hdb > 20  # and HDBSCAN's k_rank is larger than any swept K


def test_pick_sort_key_rejects_unknown_config():
    with pytest.raises(ValueError, match="unrecognized clustering config"):
        pick_sort_key("agglomerative7")


# ── MF1: full-slice cluster population, not the extraction subset ────────────


def _cluster_payload(labels_by_conv: dict[str, dict[str, int]]) -> dict:
    """Build a cluster_assignments.json-shaped payload from per-config labels."""
    configs: dict[str, dict] = {}
    for conv_id, per_config in labels_by_conv.items():
        for algo, label in per_config.items():
            configs.setdefault(algo, {"labels": {}, "examples": {}})
            configs[algo]["labels"][conv_id] = label
    return {"meta": {}, "configs": configs}


def test_full_cluster_members_uses_all_slice_convs():
    """build_membership rosters the FULL slice, not a <=400 extraction subset."""
    # 250 convs all in kmeans5 cluster 0 — more than any extraction subset cap.
    payload = _cluster_payload({f"wc_{i:06d}": {"kmeans5": 0} for i in range(250)})
    members = full_cluster_members(payload)
    assert len(members["kmeans5_c00"]) == 250


def test_category_conv_ids_caps_at_200_sorted():
    """Cap at 200/category, deterministically by sorted conv_id (plan §4 step 5)."""
    payload = _cluster_payload({f"wc_{i:06d}": {"kmeans5": 0} for i in range(250)})
    cids = category_conv_ids(payload, "kmeans5_c00", cap=200)
    assert len(cids) == 200
    assert cids == sorted(cids)
    assert cids[0] == "wc_000000"
    assert cids[-1] == "wc_000199"  # 200th by sort order, NOT a subsampled 50


def test_category_conv_ids_full_cluster_below_cap():
    """A cluster smaller than the cap ships all its full-slice members."""
    payload = _cluster_payload({f"wc_{i:06d}": {"kmeans5": 0} for i in range(40)})
    cids = category_conv_ids(payload, "kmeans5_c00", cap=200)
    assert len(cids) == 40  # NOT min(200, capped_extraction_subset==~50)


def test_category_conv_ids_unknown_cluster_raises():
    payload = _cluster_payload({"wc_000000": {"kmeans5": 0}})
    with pytest.raises(RuntimeError, match="not in full-slice cluster assignments"):
        category_conv_ids(payload, "kmeans5_c99", cap=200)


# ── MF2: user-ending generation prompt ──────────────────────────────────────


def test_build_prompt_messages_is_user_ending():
    conv = {
        "conv_id": "wc_000001",
        "first_user": "How do I sort a list in Python?",
        "short_prefix_msgs": [
            {"role": "user", "content": "How do I sort a list in Python?"},
            {"role": "assistant", "content": "Use sorted()."},
        ],
    }
    msgs = build_prompt_messages(conv)
    assert msgs[-1]["role"] == "user"
    assert msgs == [{"role": "user", "content": "How do I sort a list in Python?"}]


def test_build_prompts_rejects_assistant_ending_prefix():
    """build_prompts must fire loud if a future regression feeds an assistant tail."""
    bad = {
        "wc_000001": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    with pytest.raises(AssertionError, match="must end with a user turn"):
        build_prompts(bad, model="Qwen/Qwen2.5-7B-Instruct")
