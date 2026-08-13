"""Data-free pins for the G0v3 size-matched fold-ASSIGNMENT contrast (plan v26).

Covers the re-specified gate's pure surfaces in scripts/issue1336_fit_cells.py:
the universe-accounting identity (plan §12 row 44), the three-branch verdict
partition (§6 N21/N24, §12 row 45), the seeded size-matched draw construction
+ the two matched-contrast preconditions (§4; v26 EXACT per-fold counts,
superseding v24's "within 1"), and the manifest-keyed fold diagnostics
(§12 row 46 — sweep-side fold ids are RELABELED by _cv_folds, so "fold 0"
is only meaningful on manifest labels).

All four tests fail against the pre-v26 gate (the surfaces did not exist):
`git show <pre-change>:scripts/issue1336_fit_cells.py | grep -c G0V3` == 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue825_fit_cells as fc825  # noqa: E402
import issue1336_fit_cells as fitc  # noqa: E402
import issue1336_pooled_split as ps  # noqa: E402

# Grouped arm's realized production per-fold profile (plan §12 row 46) — used
# here as a FIXTURE shape; the gate itself always READS the profile off the
# manifest (never this literal).
PRODUCTION_PROFILE = {0: 1685, 1: 2982, 2: 2735, 3: 2166, 4: 989}


def _lmsys_manifest(n_train: int, n_test: int) -> dict:
    """Minimal synthetic manifest for the accounting helper (corpus+arm only)."""
    rows = [{"corpus": "lmsys23k", "arm": "train"} for _ in range(n_train)]
    rows += [{"corpus": "lmsys23k", "arm": "test"} for _ in range(n_test)]
    # foreign-corpus noise the lmsys23k accounting must ignore
    rows += [{"corpus": "gsm8k_train_full", "arm": "train"} for _ in range(7)]
    return {"row_index": rows}


def test_universe_accounting_identity():
    # 17,681 (concat, log-witnessed) -> 13,479 (A2 pinned pre-dedup) -> -17
    # (A2 pinned drops) = 13,462 -> -2,905 test = 10,557 train.
    assert fitc.G0V3_LMSYS_CONCAT_TOTAL == 17_681
    assert int(ps.A2_PINNED_PRE_DEDUP["lmsys23k"]) == 13_479
    assert int(ps.A2_PINNED_DROPS["lmsys23k"]) == 17

    acct = fitc._g0v3_universe_accounting(_lmsys_manifest(10_557, 2_905))
    assert acct["concat_total_log_witnessed"] == 17_681
    assert acct["post_dedup"] == 13_479 - 17 == 13_462
    assert acct["manifest_rows"] == 13_462
    assert acct["train_rows"] == 10_557
    assert acct["test_rows"] == 2_905
    assert acct["train_rows"] + acct["test_rows"] == acct["post_dedup"]
    assert acct["consistent"] is True

    # one row short => flagged inconsistent (the production assert trips on it)
    short = fitc._g0v3_universe_accounting(_lmsys_manifest(10_556, 2_905))
    assert short["consistent"] is False


def test_branch_classification_pass_leakage_anomaly():
    ex_v2 = 1.0303115
    tol = 0.05 * ex_v2  # 0.0515156 (plan §12 row 45)

    # synthetic (R2_G, R2_R) profiles spanning all three branches
    br, ok = fitc._g0v3_branch(0.61 - 0.60, tol)  # delta = +0.01
    assert (br, ok) == ("PASS", True)
    br, ok = fitc._g0v3_branch(0.66 - 0.60, tol)  # delta = +0.06 > tol
    assert (br, ok) == ("FAIL-leakage-exceeds-band", False)
    br, ok = fitc._g0v3_branch(0.58 - 0.60, tol)  # delta = -0.02 < -eps
    assert (br, ok) == ("FAIL-instrument-anomaly", False)

    # boundaries are inclusive-PASS (-0.01 <= delta <= tol)
    assert fitc._g0v3_branch(tol, tol) == ("PASS", True)
    assert fitc._g0v3_branch(-fitc.G0V3_EPS_NOISE, tol) == ("PASS", True)
    assert fitc._g0v3_branch(0.0, tol) == ("PASS", True)

    # one ulp beyond either boundary flips the branch (disjoint + exhaustive)
    up = float(np.nextafter(tol, np.inf))
    dn = float(np.nextafter(-fitc.G0V3_EPS_NOISE, -np.inf))
    assert fitc._g0v3_branch(up, tol)[0] == "FAIL-leakage-exceeds-band"
    assert fitc._g0v3_branch(dn, tol)[0] == "FAIL-instrument-anomaly"

    # a non-finite read raises rather than silently classifying PASS
    with pytest.raises(AssertionError):
        fitc._g0v3_branch(float("nan"), tol)


def test_matched_seed_fold_determinism_and_size_match():
    assert fitc.G0V3_MATCHED_SEED == 13360
    assert fitc.G0V3_MATCHED_DRAWS == 3

    n = sum(PRODUCTION_PROFILE.values())
    draws = []
    for k in range(fitc.G0V3_MATCHED_DRAWS):
        a = fitc._g0v3_matched_labels(PRODUCTION_PROFILE, k)
        b = fitc._g0v3_matched_labels(PRODUCTION_PROFILE, k)
        # reproducible under the derived seed [G0V3_MATCHED_SEED, k]
        assert np.array_equal(a, b)
        assert a.shape == (n,)
        # v26: per-fold sizes EXACTLY equal the manifest profile (supersedes
        # v24's "within 1")
        uniq, counts = np.unique(a, return_counts=True)
        assert {int(u): int(c) for u, c in zip(uniq, counts, strict=True)} == PRODUCTION_PROFILE
        fitc._g0v3_assert_matched(a, PRODUCTION_PROFILE, 5, f"draw{k}")
        draws.append(a)
    # distinct draws are distinct permutations
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])

    # precondition (1): a wrong unique-label count raises
    with pytest.raises(AssertionError):
        fitc._g0v3_assert_matched(np.zeros(n, dtype=np.int64), PRODUCTION_PROFILE, 5, "bad-uniq")
    # precondition (2): a size-drifted assignment raises (multiset mismatch)
    drifted = np.concatenate(
        [
            np.full(1684, 0),
            np.full(2983, 1),
            np.full(2735, 2),
            np.full(2166, 3),
            np.full(989, 4),
        ]
    )
    with pytest.raises(AssertionError):
        fitc._g0v3_assert_matched(drifted, PRODUCTION_PROFILE, 5, "drift")
    # multiset semantics: a RELABELED but size-matched assignment passes —
    # _cv_folds relabels fold ids, so only the size multiset is invariant
    relabeled = np.concatenate(
        [
            np.full(989, 0),
            np.full(2166, 1),
            np.full(2735, 2),
            np.full(2982, 3),
            np.full(1685, 4),
        ]
    )
    fitc._g0v3_assert_matched(relabeled, PRODUCTION_PROFILE, 5, "relabels-ok")


def _diag_manifest() -> dict:
    """Synthetic 3-fold manifest: fold 0 holds 3 rows (2 quarantine, gid 100),
    fold 1 holds 4 rows (gid 102), fold 2 holds 5 rows (gid 103)."""
    rows = []

    def add(fold: int, cluster: int, count: int) -> None:
        for _ in range(count):
            i = len(rows)
            rows.append(
                {
                    "corpus": "lmsys23k",
                    "prompt_idx": i,
                    "prompt_sha": f"sha{i}",
                    "cluster": cluster,
                    "arm": "train",
                    "fold": fold,
                }
            )

    add(0, 100, 2)  # quarantine gid
    add(0, 101, 1)
    add(1, 102, 4)
    add(2, 103, 5)
    # test-arm + foreign-corpus rows the diagnostics must ignore
    rows.append(
        {
            "corpus": "lmsys23k",
            "prompt_idx": 900,
            "prompt_sha": "sha900",
            "cluster": 104,
            "arm": "test",
        }
    )
    rows.append(
        {
            "corpus": "gsm8k_train_full",
            "prompt_idx": 901,
            "prompt_sha": "sha901",
            "cluster": 999,
            "arm": "train",
            "fold": 0,
        }
    )
    group_table = [
        {"corpus": "lmsys23k", "group_id": 100, "quarantine": True},
        {"corpus": "lmsys23k", "group_id": 101, "quarantine": False},
        {"corpus": "lmsys23k", "group_id": 102, "quarantine": False},
        {"corpus": "lmsys23k", "group_id": 103, "quarantine": False},
        # foreign-corpus quarantine group — must not leak into the lmsys count
        {"corpus": "gsm8k_train_full", "group_id": 999, "quarantine": True},
    ]
    return {"row_index": rows, "group_table": group_table}


def test_fold_diagnostics_keyed_off_manifest_labels():
    assert ps.QUARANTINE_TRAIN_FOLD == 0
    man = _diag_manifest()
    fold_row_counts, fold0_q = fitc._g0v3_fold_diagnostics(man)
    assert fold_row_counts == {"0": 3, "1": 4, "2": 5}
    assert fold0_q == 2

    # The trap this pins (plan v26 mechanism trap b): _cv_folds RELABELS the
    # manifest fold ids through a seeded permutation, so a sweep-side "fold 0"
    # is arbitrary. seed=0 realizes the NON-identity map {0->2, 1->0, 2->1}
    # over 3 unique labels (probed; deterministic for default_rng(0)).
    entries = [e for e in man["row_index"] if e["corpus"] == "lmsys23k" and e["arm"] == "train"]
    labels = np.asarray([int(e["fold"]) for e in entries])
    realized = fc825._cv_folds(labels, 3, 0)
    # _cv_folds output is a pure relabeling: constant within each manifest label
    mapping = {}
    for lab in np.unique(labels):
        vals = np.unique(realized[labels == lab])
        assert vals.shape == (1,)
        mapping[int(lab)] = int(vals[0])
    assert sorted(mapping.values()) == [0, 1, 2]  # bijection at #unique == n_folds
    assert mapping[0] != 0, "seed=0 must move manifest fold 0 (probed non-identity map)"

    # Counterfactual: keying "fold 0" on the RELABELED sweep-side ids
    # misreports the quarantine count; the helper tracks the MANIFEST.
    qgids = {100}
    sweep_keyed_q = sum(
        1
        for e, r in zip(entries, realized, strict=True)
        if int(e["cluster"]) in qgids and int(r) == ps.QUARANTINE_TRAIN_FOLD
    )
    assert sweep_keyed_q != fold0_q
    # and the manifest-keyed diagnostics are invariant to whatever _cv_folds did
    again_counts, again_q = fitc._g0v3_fold_diagnostics(man)
    assert (again_counts, again_q) == (fold_row_counts, fold0_q)
