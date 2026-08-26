"""Issue #2588 P0 defect 4 — item-(j) provenance ordering is PER STORE.

The v1 gate asserted global ``max(all inputs) <= min(all captures)``, which is
unsatisfiable by construction across heterogeneous bank vintages: the #1491
7B anchor/ceiling stores (captured 2026-08-05) predate #2330's split_ids.json
(created 2026-08-16) BY DESIGN — the split assignment is applied at consume
time by ci-filtering, so split_ids.json was never a generating input to that
bank. Same defect class as P0 defect 2: an assert that can never pass was
recorded as a measured plan-time fact without ever being executed.

These tests pin the corrected pure helper
(``issue2588_p0_preflight.check_provenance_ordering``): the REAL measured
incident fixture passes, each genuine incoherence still raises naming the
store, and mixed git-%cI / str(datetime) formats compare as aware datetimes
(the v1 string compare inverts near timezone boundaries). No network, no GPU
(adoptable-tests contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from issue2588_p0_preflight import check_provenance_ordering

# The REAL dates measured 2026-08-26 (P0 provenance probe) — the incident
# fixture: split_ids postdates the #1491 ceiling captures, coherently.
INPUTS = {
    "split_ids_git": "2026-08-16T13:07:01-07:00",
    "manifest": "2026-08-05 05:11:03+00:00",
}
STORES = {
    "qwen25_7b/ceiling_s43": {
        "prefix": "issue1491_scale_ladder/scale7_refit/ceiling_draws/seed43/raw_completions",
        "date": "2026-08-05 07:37:17+00:00",
    },
    "qwen35_9b/train_10k": {
        "prefix": "issue2330_matched/qwen35_9b_cap2048/train_10k/raw_completions",
        "date": "2026-08-17 13:54:59+00:00",
    },
    "qwen25_7b/train_10k": {
        "prefix": "issue2330_matched/q25_cap2048/train_25k/raw_completions",
        "date": "2026-08-17 17:50:59+00:00",
    },
}


def test_incident_fixture_passes_per_store():
    # The defect pinned in reverse: globally, max(input) > min(capture) here
    # (split_ids 08-16 > the 1491 ceiling capture 08-05), so the v1 global
    # form can NEVER pass this real, coherent fixture.
    ordering = check_provenance_ordering(INPUTS, STORES)
    assert set(ordering) == set(STORES)
    assert all(line.startswith("ok — ") for line in ordering.values())
    # Manifest-only vintage: split_ids must NOT appear in the 1491 store's line.
    assert "split_ids" not in ordering["qwen25_7b/ceiling_s43"]
    assert "split_ids_git" in ordering["qwen35_9b/train_10k"]


def test_2330_store_predating_split_ids_raises():
    stores = {
        "qwen35_9b/train_10k": {
            "prefix": "issue2330_matched/qwen35_9b_cap2048/train_10k/raw_completions",
            "date": "2026-08-15 00:00:00+00:00",  # before split_ids (08-16)
        }
    }
    with pytest.raises(AssertionError, match=r"qwen35_9b/train_10k.*split_ids_git"):
        check_provenance_ordering(INPUTS, stores)


def test_manifest_postdating_1491_ceiling_raises():
    # The manifest IS a generating input for the #1491 bank — still enforced.
    stores = {
        "qwen25_7b/ceiling_s43": {
            "prefix": "issue1491_scale_ladder/scale7_refit/ceiling_draws/seed43/raw_completions",
            "date": "2026-08-05 05:00:00+00:00",  # before the manifest (05:11)
        }
    }
    with pytest.raises(AssertionError, match=r"ceiling_s43.*manifest"):
        check_provenance_ordering(INPUTS, stores)


def test_mixed_formats_compare_as_datetimes_not_strings():
    # input 2026-08-16T23:30:00-07:00 == 2026-08-17T06:30:00Z. A LEXICOGRAPHIC
    # compare against '2026-08-17 05:00:00+00:00' passes ('...16T...' < '...17 ...');
    # the true datetime compare must FAIL (06:30Z > 05:00Z).
    inputs = {"split_ids_git": "2026-08-16T23:30:00-07:00", "manifest": INPUTS["manifest"]}
    stores = {
        "qwen35_9b/val_400": {
            "prefix": "issue2330_matched/qwen35_9b_cap2048/val_400/raw_completions",
            "date": "2026-08-17 05:00:00+00:00",
        }
    }
    assert inputs["split_ids_git"] <= stores["qwen35_9b/val_400"]["date"]  # the string trap
    with pytest.raises(AssertionError, match=r"val_400.*split_ids_git"):
        check_provenance_ordering(inputs, stores)
