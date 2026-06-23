"""Pin the metric-race ``_metric_family`` leaderboard-grouping parser.

Regression for Codex Major #2 (reconciler r1 binding FAIL): the original
``body.rsplit("_", 1)[-1]`` collapsed every UNDERSCORE metric to its last
token — ``mahal_pooled_ctx`` -> ``ctx``, ``gauss_kl`` -> ``kl``,
``neg_l2`` -> ``l2`` — so the entire covariance / cloud / centered family
fell into ``other_A`` and corrupted the §7 secondary leaderboard, the HERO 1
figure grouping, the centered-vs-raw delta read, and the
``global_champion_metric_family`` attribution label.

The fix parses against an ordered (longest-first) suffix list so multi-token
metrics resolve to their correct family. These tests cover EVERY multi-token
and single-token metric the zoo emits (plan §4.1), both centroid centerings,
the cloud family, the output-distribution family, the v1 ``geom_*``
reference, and the non-A group fall-through.

Pure string parsing — no model, no GPU, no data fixtures.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.behavior_testbed_545 import scoring as S

# ---------------------------------------------------------------------------
# The four reconciler-cited acceptance cases (verbatim from the round-2 brief).
# ---------------------------------------------------------------------------


def test_reconciler_acceptance_cases():
    # mahal_pooled_ctx (covariance centroid, raw) — the headline misparse.
    assert (
        S._metric_family("A__cloud_demos_L21_mean_response_raw_mahal_pooled_ctx")
        == "covariance_centroid"
    )
    # gauss_kl (cloud) — previously collapsed to "kl".
    assert S._metric_family("A__cloud_demos_L21_mean_response_gauss_kl") == "cloud"
    # The v1 geom_* reference is the raw {cosine,neg_l2,projection} family by
    # construction; geom_* short-circuits to raw_centroid regardless of the
    # rest of the name (the v1 reference predictors have no centered variant).
    assert S._metric_family("A__geom_nl_L21_last_token_centered_neg_l2") == "raw_centroid"
    # delta_spec classifies as cloud even though it returns [] at runtime — the
    # parser must still group it (so the leaderboard row, if ever present, bins).
    assert S._metric_family("A__cloud_demos_L21_mean_response_delta_spec") == "cloud"


# ---------------------------------------------------------------------------
# Covariance-centroid metrics {euclidean, mahal, mahal_pooled_ctx}, raw.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "mahal", "mahal_pooled_ctx"])
@pytest.mark.parametrize("flavor", ["nl", "demos"])
@pytest.mark.parametrize("point", ["last_token", "mean_response"])
def test_covariance_centroid_raw(metric, flavor, point):
    name = f"A__cloud_{flavor}_L21_{point}_raw_{metric}"
    assert S._metric_family(name) == "covariance_centroid", name


# ---------------------------------------------------------------------------
# Raw-centroid metrics {cosine, neg_l2, projection} via the cloud_* naming, raw.
# (The same three also arrive as v1 geom_* — covered below.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "neg_l2", "projection"])
@pytest.mark.parametrize("point", ["last_token", "mean_response"])
def test_raw_centroid_via_cloud_naming(metric, point):
    name = f"A__cloud_nl_L7_{point}_raw_{metric}"
    assert S._metric_family(name) == "raw_centroid", name


# ---------------------------------------------------------------------------
# Centered variant of EVERY centroid metric (raw + covariance) -> centered_centroid.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    ["cosine", "neg_l2", "projection", "euclidean", "mahal", "mahal_pooled_ctx"],
)
@pytest.mark.parametrize("point", ["last_token", "mean_response"])
def test_centered_centroid(metric, point):
    name = f"A__cloud_demos_L14_{point}_centered_{metric}"
    assert S._metric_family(name) == "centered_centroid", name


# ---------------------------------------------------------------------------
# Cloud metrics — the realized short labels (predictors_zoo.CLOUD_METRICS +
# delta_spec) AND the alternate long #493 spellings the parser also accepts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        "mmd",
        "c2st",
        "wass2",
        "gauss_kl",
        "delta_spec",
        # alternate #493 spellings (robustness — same family):
        "rbf_mmd_squared",
        "bures_wasserstein2",
    ],
)
@pytest.mark.parametrize("point", ["last_token", "mean_response"])
def test_cloud_metrics(metric, point):
    # Cloud metrics carry NO centering segment in the zoo naming.
    name = f"A__cloud_demos_L21_{point}_{metric}"
    assert S._metric_family(name) == "cloud", name


# ---------------------------------------------------------------------------
# Output-distribution JS/KL family -> outdist_jskl.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["js", "kl_narrow_broad", "kl_broad_narrow"])
@pytest.mark.parametrize("flavor", ["nl", "demos"])
def test_outdist_family(direction, flavor):
    name = f"A__outdist_{flavor}_{direction}"
    assert S._metric_family(name) == "outdist_jskl", name


# ---------------------------------------------------------------------------
# v1 geom_* reference predictors (the carried {cosine,neg_l2,projection}) ->
# raw_centroid. These are the ACTUAL on-disk names (no centering segment).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "neg_l2", "projection"])
@pytest.mark.parametrize("point", ["last_token", "mean_response"])
def test_geom_reference_raw_centroid(metric, point):
    name = f"A__geom_demos_L0_{point}_{metric}"
    assert S._metric_family(name) == "raw_centroid", name


# ---------------------------------------------------------------------------
# Non-A groups keep their group letter (B/C/D references + the ridge stack).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("B__base_prior_level", "group_b"),
        ("C__grad_align", "group_c"),
        ("D__ridge_combiner", "group_d"),
    ],
)
def test_non_a_groups(name, expected):
    assert S._metric_family(name) == expected


# ---------------------------------------------------------------------------
# The exact misparse the old rsplit produced is GONE: no zoo metric collapses
# to a single trailing token, and no real metric lands in other_A.
# ---------------------------------------------------------------------------


def test_underscore_metrics_never_collapse_to_other_a():
    for metric, fam in [
        ("mahal_pooled_ctx", "covariance_centroid"),  # was -> ctx -> other_A
        ("neg_l2", "raw_centroid"),  # was -> l2 -> other_A
        ("gauss_kl", "cloud"),  # was -> kl -> other_A
        ("delta_spec", "cloud"),  # was -> spec -> other_A
    ]:
        centering = "raw" if fam != "cloud" else None
        seg = f"_{centering}_" if centering else "_"
        name = f"A__cloud_demos_L21_mean_response{seg}{metric}"
        got = S._metric_family(name)
        assert got == fam, f"{name} -> {got}, expected {fam}"
        assert got != "other_A", name


def test_genuinely_unknown_metric_is_other_a():
    # A name with no registered metric suffix still falls through cleanly.
    assert S._metric_family("A__cloud_demos_L21_mean_response_raw_bogusmetric") == "other_A"
