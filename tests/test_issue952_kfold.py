"""Pins for the issue #952 kfold follow-up (round `kfold-decision-cells`, plan v10).

Fold construction (`make_kfold_splits`): rotation semantics (fold j: test=Bj,
val=B((j-1) mod K), train=rest), exact-partition coverage, and the HARD
calibration identity — the fold with test=B(K-1) equals `make_split` exactly,
and (against the COMMITTED artifact) equals `eval_results/issue_952/
split_seed952.json` at the production pool.

Gate-5 provenance contract (`kfold_manifest` / `kfold_manifest_match`): a
persisted fold is accepted ONLY on a full manifest match (split hashes + counts
+ staging revision + git SHA + env); any perturbation rejects with a named
reason. Gate 4 (`round1_xlayer_reproduction_gate`): production RAISE branches on
R² drift / λ mismatch / missing cells; smoke logs-only.
"""

import json
import pathlib

import pytest

from explore_persona_space.experiments.issue_952.run_952 import (
    ARMS,
    POSITION_SLOTS,
    kfold_manifest,
    kfold_manifest_match,
    kfold_split_hashes,
    make_kfold_splits,
    make_split,
    round1_xlayer_reproduction_gate,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMITTED_SPLIT = _REPO_ROOT / "eval_results" / "issue_952" / "split_seed952.json"


# ── fold construction ────────────────────────────────────────────────────────────


def test_kfold_rotation_semantics_and_partition():
    """Fold j: test=Bj, val=B((j-1) mod K), train=rest; blocks partition the pool."""
    pool = list(range(1000, 1020))  # 20 ids -> 5 blocks of 4
    folds = make_kfold_splits(pool, 5)
    assert len(folds) == 5
    # Reconstruct the blocks from the fold test sets (each test IS one block).
    blocks = {f["fold"]: set(f["test"]) for f in folds}
    all_test = [c for f in folds for c in f["test"]]
    assert len(all_test) == len(pool) and set(all_test) == set(pool)  # exact partition
    for f in folds:
        j = f["fold"]
        assert len(f["test"]) == len(f["val"]) == 4 and len(f["train"]) == 12
        assert set(f["val"]) == blocks[(j - 1) % 5]  # rotation val
        assert set(f["train"]) == set(pool) - set(f["test"]) - set(f["val"])
        assert not (set(f["test"]) & set(f["val"]))  # disjoint


def test_kfold_calibration_fold_equals_make_split():
    """The fold with test=B(K-1) is IDENTICALLY the single split (any pool size)."""
    pool = list(range(30))
    folds = make_kfold_splits(pool, 5)
    single = make_split(pool)
    for key in ("train", "val", "test"):
        assert folds[4][key] == single[key]


def test_kfold_rejects_non_divisible_pool():
    with pytest.raises(AssertionError, match="does not divide"):
        make_kfold_splits(list(range(11)), 5)


@pytest.mark.skipif(not COMMITTED_SPLIT.exists(), reason="committed split artifact not checked out")
def test_kfold_b4_equals_committed_production_split():
    """Plan §3(a): the fold with test=B4 equals the COMMITTED split_seed952.json
    exactly at the production 4920-context pool (calibration identity checked
    against the artifact, not assumed)."""
    committed = json.loads(COMMITTED_SPLIT.read_text())
    pool = sorted(int(c) for key in ("train", "val", "test") for c in committed[key])
    assert len(pool) == 4920
    folds = make_kfold_splits(pool, 5)
    for key in ("train", "val", "test"):
        assert folds[4][key] == committed[key], f"calibration fold != committed split ({key})"
    # Rotation blocks are exactly 984 each.
    assert all(len(f["test"]) == 984 for f in folds)


# ── gate-5 manifest match / reject ───────────────────────────────────────────────


def _manifest(monkeypatch=None) -> tuple[dict, dict]:
    pool = list(range(20))
    fold = make_kfold_splits(pool, 5)[2]
    m = kfold_manifest(fold, staging_revision="rev-abc")
    return m, fold


def test_kfold_manifest_match_roundtrip():
    m, fold = _manifest()
    ok, why = kfold_manifest_match(json.loads(json.dumps(m)), m)  # JSON round-trip
    assert ok, why
    assert m["split_sha256"] == kfold_split_hashes(fold)
    assert m["counts"] == {"train": 12, "val": 4, "test": 4}


@pytest.mark.parametrize(
    "mutate, expect_key",
    [
        (lambda m: m.update(git_sha="deadbeef"), "git_sha"),
        (lambda m: m.update(staging_revision="other-rev"), "staging_revision"),
        (lambda m: m["env"].update(EPM_I952_KFOLD_BLOCKS="7"), "env"),
        (lambda m: m["split_sha256"].update(test="0" * 64), "split_sha256"),
        (lambda m: m["counts"].update(test=999), "counts"),
        (lambda m: m.update(fold=3), "fold"),
    ],
)
def test_kfold_manifest_rejects_any_perturbation(mutate, expect_key):
    m, _fold = _manifest()
    bad = json.loads(json.dumps(m))
    mutate(bad)
    ok, why = kfold_manifest_match(bad, m)
    assert not ok and expect_key in why


def test_kfold_manifest_rejects_absent():
    m, _fold = _manifest()
    ok, why = kfold_manifest_match(None, m)
    assert not ok and "absent" in why


def test_kfold_lambda_table_not_part_of_match():
    """The λ table is output metadata — recorded, never a match key."""
    m, _fold = _manifest()
    other = json.loads(json.dumps(m))
    other["lam_star_by_slot_by_layer"] = {"20": {"f16_t1": 1.0}}
    ok, _why = kfold_manifest_match(other, m)
    assert ok


# ── gate 4: round-1 cross-layer reproduction ─────────────────────────────────────


def _xlayer_report(value: float = 0.5, lam: float = 100.0) -> dict:
    layers = ["20", "14", "23", "26"]
    return {
        "l_star": 20,
        "decision_layers": [14, 23, 26],
        "by_layer": {
            la: {
                arm: {
                    slot: {"test_pooled_r2": value, "lambda": lam, "n_valid_test": 629}
                    for slot in POSITION_SLOTS
                }
                for arm in ARMS
            }
            for la in layers
        },
    }


def test_round1_gate_passes_on_identical_reports():
    rec = round1_xlayer_reproduction_gate(_xlayer_report(), _xlayer_report(), smoke=False)
    assert rec["pass"] is True
    assert rec["n_cells_compared"] == rec["n_cells_expected"] == 4 * len(ARMS) * len(POSITION_SLOTS)
    assert rec["n_lambda_mismatch"] == 0


def test_round1_gate_raises_on_r2_drift():
    got = _xlayer_report()
    got["by_layer"]["14"]["own"][POSITION_SLOTS[0]]["test_pooled_r2"] += 1e-4
    with pytest.raises(RuntimeError, match="round-1 cross-layer reproduction gate FAIL"):
        round1_xlayer_reproduction_gate(_xlayer_report(), got, smoke=False)


def test_round1_gate_raises_on_lambda_mismatch():
    """λ equality is exact (plan §2 gate 4) — a same-R² λ swap still fails."""
    got = _xlayer_report()
    got["by_layer"]["23"]["ext_plain"][POSITION_SLOTS[1]]["lambda"] = 316.22776601683796
    with pytest.raises(RuntimeError, match="λ mismatches"):
        round1_xlayer_reproduction_gate(_xlayer_report(), got, smoke=False)


def test_round1_gate_raises_on_missing_fold_cells():
    """The expected grid is ENUMERATED — a fold report missing a layer fails loud."""
    got = _xlayer_report()
    del got["by_layer"]["26"]
    with pytest.raises(RuntimeError, match="reproduction gate FAIL"):
        round1_xlayer_reproduction_gate(_xlayer_report(), got, smoke=False)


def test_round1_gate_smoke_logs_only():
    got = _xlayer_report()
    got["by_layer"]["14"]["own"][POSITION_SLOTS[0]]["test_pooled_r2"] += 1.0
    rec = round1_xlayer_reproduction_gate(_xlayer_report(), got, smoke=True)
    assert rec["pass"] is False  # comparison executed, non-binding
