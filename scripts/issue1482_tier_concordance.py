#!/usr/bin/env python3
"""Bank the nested-dictionary (Matryoshka) tier concordance with per-feature R-squared.

The Section 4.2 feature-property panel scores a binary property by the
probability that a feature carrying it is predicted better than a matched
feature that does not, reported above chance so zero means no association.
This script asks the same question of nested-tier membership, using the panel's
own estimator, so the tier read sits on the panel's axis instead of needing an
axis of its own.

WHAT IS COMPUTED. Somers' D of held-out R-squared on a tier coding, on the
c-index scale and centered at chance:

    D = (C - Dis) / (n_pairs - T_x)        value = (D + 1) / 2 - 0.5

For a binary coding this is exactly the Mann-Whitney statistic, computed in
closed form; for the three-level ordinal coding it is Kendall tau-b, the same
path the panel's continuous properties take.  Pairs form only inside a matching
stratum, so a feature is never compared with a feature of very different
activity.  The estimator is imported from scripts/issue1482_concordance_fig.py
rather than reimplemented.

WHICH TARGET. The paper plots the DECODER-DIRECTION target, so R-squared and
tier come from the decoder-direction store.  Only the activity covariate is read
from the per-feature Matryoshka store; that store's own R-squared is the
activation-target twin and answers a different question (raw Spearman of tier
against R-squared is -0.395 there against -0.087 here).  Mixing the two would
change the headline.

WHY BOTH BINARY SPLITS AND THE ORDINAL CODING. The three tiers are not an even
gradient.  Coarsest against middle is large, middle against finest is near zero,
so an ordinal coding that assumes equal steps understates the coarsest tier.
All three readings are banked, with the pairwise decomposition that explains the
gap between them.

Reads two banked stores, writes eval_results/issue_1482/tier_concordance.json,
and computes nothing on a GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Thread caps must bind before numpy imports: the BLAS pools freeze at import.
load_dotenv()

import numpy as np  # noqa: E402

ESTIMATOR_SOURCE = ROOT / "scripts/issue1482_concordance_fig.py"
DECODER_STORE = ROOT / "eval_results/issue_1482/decoder_direction/r2_decoder_direction_lmsys.npz"
TIER_STORE = ROOT / "eval_results/issue_1482/matryoshka_tier/perfeature_m_lmsys_default.npz"
DECODER_META = (
    ROOT / "eval_results/issue_1482/decoder_direction/matryoshka_decoder_direction_lmsys.json"
)
# The five feature-property bars the tier bar joins, read only to record what
# differs between the two populations.
PROPERTY_PANEL = ROOT / "eval_results/issue_1482/plot4_redesign/plot4_decoder_direction.json"
OUT = ROOT / "eval_results/issue_1482/tier_concordance.json"

LAYER = 20
PROPERTY_PANEL_LAYER = 19
SEED = 20260910
N_BOOT = 2000
TIER_NAMES = {0: "coarsest", 1: "middle", 2: "finest"}
# The appendix quotes these to two decimals as 0.73 / 0.64 / 0.65.
EXPECTED_MEDIAN_R2 = {0: 0.727, 1: 0.640, 2: 0.649}
EXPECTED_TIER_N = {0: 1640, 1: 6144, 2: 8600}


def _load_estimator() -> ModuleType:
    """Import the panel's own concordance estimator by path.

    The module is not a package member and its filename is not an identifier a
    plain import can reach, so it is loaded by spec.  It guards its own entry
    point, so importing it runs no analysis.
    """
    spec = importlib.util.spec_from_file_location("issue1482_concordance_fig", ESTIMATOR_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the concordance estimator from {ESTIMATOR_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("concordance", "cd_untied"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"{ESTIMATOR_SOURCE} does not expose {name}()")
    return module


def _display_path(path: Path) -> str:
    """Path relative to the repo when it sits inside it, absolute otherwise."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _strata(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Index groups for equal-count bins of ``values``; one group when n_bins is 1."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if n_bins == 1:
        return [np.arange(len(values))]
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    assigned = np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)
    return [np.flatnonzero(assigned == index) for index in range(n_bins)]


def load_panel() -> dict:
    """Held-out R-squared, tier and activity for the scored features, cross-checked.

    The two stores index the dictionary differently: the decoder-direction store
    carries the full 65,536-entry dictionary plus the ids of the scored subset,
    the per-feature store carries only the scored subset.  Both are put on the
    scored-subset order and then required to agree on ids and tier, so a
    silently reordered or regenerated store fails here instead of shifting a
    number.
    """
    decoder = np.load(DECODER_STORE)
    tier_store = np.load(TIER_STORE)

    panel_ids = decoder["panel_ids"]
    order = np.argsort(tier_store["feat_ids"])
    if not np.array_equal(tier_store["feat_ids"][order], np.sort(panel_ids)):
        raise AssertionError("the two stores score different feature sets")
    remap = np.searchsorted(tier_store["feat_ids"][order], panel_ids)

    r2 = decoder["r2"][panel_ids].astype(float)
    tier = decoder["tier"][panel_ids].astype(int)
    activity = tier_store["activity"][order][remap].astype(float)
    if not np.array_equal(tier_store["tier"][order][remap].astype(int), tier):
        raise AssertionError("the two stores disagree on tier membership")

    keep = np.isfinite(r2) & np.isfinite(activity) & (activity > 0)
    r2, tier, activity = r2[keep], tier[keep], activity[keep]
    log_activity = np.log(activity)

    per_tier = []
    for level in sorted(TIER_NAMES):
        member = tier == level
        count = int(member.sum())
        median = float(np.median(r2[member]))
        if count != EXPECTED_TIER_N[level]:
            raise AssertionError(
                f"{TIER_NAMES[level]} tier holds {count} features, expected "
                f"{EXPECTED_TIER_N[level]}"
            )
        if round(median, 3) != EXPECTED_MEDIAN_R2[level]:
            raise AssertionError(
                f"{TIER_NAMES[level]} tier median R^2 is {median:.4f}, expected "
                f"{EXPECTED_MEDIAN_R2[level]:.3f}"
            )
        per_tier.append({"tier": TIER_NAMES[level], "n": count, "median_r2": median})

    return {
        "r2": r2,
        "tier": tier,
        "activity": activity,
        "log_activity": log_activity,
        "n_scored": int(len(r2)),
        "n_dictionary": int(decoder["r2"].shape[0]),
        "n_dropped": int((~keep).sum()),
        "per_tier": per_tier,
    }


def score(
    estimator: ModuleType,
    coding: np.ndarray,
    r2: np.ndarray,
    log_activity: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Pooled, quintile-matched and decile-matched concordance, with a bootstrap interval.

    The interval is on the quintile-matched value, the headline matching.  Each
    row gets its own generator seeded from ``seed``, so a row's interval does not
    depend on how many rows ran before it.  Stratum edges are recomputed inside
    every draw, because the edges are themselves estimated from the sample.
    """
    values = {
        "pooled": estimator.concordance(coding, r2, _strata(log_activity, 1)) - 0.5,
        "activity_quintile_matched": (
            estimator.concordance(coding, r2, _strata(log_activity, 5)) - 0.5
        ),
        "activity_decile_matched": (
            estimator.concordance(coding, r2, _strata(log_activity, 10)) - 0.5
        ),
    }
    rng = np.random.default_rng(seed)
    size = len(r2)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        pick = rng.integers(0, size, size)
        draws[index] = (
            estimator.concordance(coding[pick], r2[pick], _strata(log_activity[pick], 5)) - 0.5
        )
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    values["activity_quintile_matched_ci95"] = [float(low), float(high)]
    values["n"] = size
    return values


def _subset(panel: dict, member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return panel["r2"][member], panel["log_activity"][member]


def compute(panel: dict, estimator: ModuleType, *, n_boot: int) -> dict:
    """Every tier reading the appendix needs, keyed so a caption can quote one."""
    tier = panel["tier"]
    r2, log_activity = panel["r2"], panel["log_activity"]

    def binary(positive: np.ndarray, negative: np.ndarray, seed_offset: int) -> dict:
        member = positive | negative
        sub_r2, sub_log = _subset(panel, member)
        return score(
            estimator,
            positive[member].astype(int),
            sub_r2,
            sub_log,
            n_boot=n_boot,
            seed=SEED + seed_offset,
        )

    headline = binary(tier == 0, tier > 0, 0)
    headline |= {
        "coding": "coarsest tier versus the rest",
        "n_positive": int((tier == 0).sum()),
        "reading": (
            "a feature in the coarsest tier is predicted better than an "
            "activity-matched feature from the other two tiers this much above chance"
        ),
    }

    coarse_middle_vs_fine = binary(tier <= 1, tier == 2, 1)
    coarse_middle_vs_fine |= {
        "coding": "coarsest and middle tiers versus the finest",
        "n_positive": int((tier <= 1).sum()),
    }

    ordinal = score(
        estimator,
        2 - tier,
        r2,
        log_activity,
        n_boot=n_boot,
        seed=SEED + 2,
    )
    ordinal |= {
        "coding": "three-level coarseness, coarsest is 2 and finest is 0",
        "path": "Kendall tau-b, the path the panel's continuous properties take",
        "note": (
            "smaller than the coarsest-versus-rest split because an ordinal coding "
            "assumes equal steps, and the middle-to-finest step is near zero"
        ),
    }

    pairwise = []
    for high, low, offset in ((0, 1, 3), (1, 2, 4), (0, 2, 5)):
        row = binary(tier == high, tier == low, offset)
        row |= {"coding": f"{TIER_NAMES[high]} versus {TIER_NAMES[low]}"}
        pairwise.append(row)

    return {
        "headline": headline,
        "binary_alternative": coarse_middle_vs_fine,
        "ordinal": ordinal,
        "pairwise": pairwise,
    }


def build_document(panel: dict, results: dict, *, n_boot: int) -> dict:
    decoder_meta = json.loads(DECODER_META.read_text())
    universe = {
        "dictionary": "nested sparse autoencoder, three tiers, coarsest to finest",
        "layer": LAYER,
        "n_features_scored": panel["n_scored"],
        "n_features_dictionary": panel["n_dictionary"],
        "n_features_dropped": panel["n_dropped"],
        "corpus": decoder_meta["family"],
        "target": decoder_meta["target"],
        "target_note": (
            "the decoder-direction target the paper plots, not the activation "
            "target banked beside it in the per-feature store"
        ),
    }
    property_panel = json.loads(PROPERTY_PANEL.read_text())["left_panel"]
    property_universe = {
        "n_features": int(property_panel["rows"][0]["n"]),
        "layer": PROPERTY_PANEL_LAYER,
        "dictionary": "regular sparse autoencoder",
        "matching": property_panel["statistic"],
        "source": str(PROPERTY_PANEL.relative_to(ROOT)),
    }
    caption_note = (
        f"The tier bar is measured on a different population from the five property "
        f"bars beside it: {panel['n_scored']:,} features of a nested sparse autoencoder "
        f"at layer {LAYER}, matched on activity quintiles. The property bars cover "
        f"{property_universe['n_features']:,} features of the regular sparse autoencoder "
        f"at layer {PROPERTY_PANEL_LAYER}, each matched by coarsened exact matching on "
        f"the properties selected in earlier rounds, the first of them unmatched. The "
        f"statistic, the scale and the reading are the same in both."
    )
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "what_is_measured": (
            "probability that a feature carrying a tier property is predicted better "
            "than a matched feature that does not, reported above chance so zero means "
            "no association"
        ),
        "universe": universe,
        "estimator": {
            "source": "scripts/issue1482_concordance_fig.py",
            "functions": ["concordance", "cd_untied"],
            "scale": "Somers' D on the c-index scale, minus 0.5",
            "binary_path": "closed-form Mann-Whitney, exact for a two-valued coding",
            "ordinal_path": "Kendall tau-b for a coding with more than two values",
            "ties": "pairs tied on the coding are dropped from the denominator",
        },
        "matching": {
            "scheme": "activity quintiles",
            "variable": "log of per-answer firing frequency",
            "source": "eval_results/issue_1482/matryoshka_tier/perfeature_m_lmsys_default.npz",
            "edges": "equal-count quantile edges, recomputed inside every bootstrap draw",
            "also_reported": "pooled (no matching) and activity deciles",
        },
        "bootstrap": {
            "draws": n_boot,
            "seed": SEED,
            "unit": "feature",
            "scheme": "one generator per row, seeded from the base seed plus a row offset",
            "interval": "95 percent, percentile method, on the quintile-matched value",
        },
        "headline": results["headline"],
        "binary_alternative": results["binary_alternative"],
        "ordinal": results["ordinal"],
        "pairwise": results["pairwise"],
        "per_tier": panel["per_tier"],
        "reference_spearman": {
            "decoder_direction_raw": decoder_meta["panel"]["raw_spearman_tier_r2"],
            "activation_target_raw": decoder_meta["reference_activation_target"]["raw_spearman"],
            "note": (
                "the two targets disagree in size, which is why only the "
                "decoder-direction store supplies R-squared here"
            ),
        },
        "property_panel_universe": property_universe,
        "caption_note": caption_note,
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in (DECODER_STORE, TIER_STORE, DECODER_META, PROPERTY_PANEL, ESTIMATOR_SOURCE)
        ],
        "git": _git_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bank the nested-tier concordance readings.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=N_BOOT,
        help="bootstrap draws per row (default: %(default)s)",
    )
    args = parser.parse_args()

    estimator = _load_estimator()
    panel = load_panel()
    print(
        f"scored {panel['n_scored']:,} of {panel['n_dictionary']:,} dictionary features, "
        f"{panel['n_dropped']} dropped"
    )
    for row in panel["per_tier"]:
        print(f"  {row['tier']:9s} n={row['n']:6,}  median R^2 {row['median_r2']:.3f}")

    results = compute(panel, estimator, n_boot=args.bootstrap_draws)
    document = build_document(panel, results, n_boot=args.bootstrap_draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")

    head = results["headline"]
    low, high = head["activity_quintile_matched_ci95"]
    print(
        f"\nheadline {head['coding']}: {head['activity_quintile_matched']:+.3f} "
        f"[{low:+.3f}, {high:+.3f}] quintile-matched, "
        f"{head['pooled']:+.3f} pooled, {head['activity_decile_matched']:+.3f} decile-matched"
    )
    alt = results["binary_alternative"]
    print(f"{alt['coding']}: {alt['activity_quintile_matched']:+.3f}")
    print(f"{results['ordinal']['coding']}: {results['ordinal']['activity_quintile_matched']:+.3f}")
    for row in results["pairwise"]:
        print(f"  {row['coding']:26s} {row['activity_quintile_matched']:+.3f}  (n={row['n']:,})")
    print(f"\nwrote {_display_path(args.out)}")


if __name__ == "__main__":
    main()
