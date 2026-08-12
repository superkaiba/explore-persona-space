"""A5 near-duplicate containment re-specification pins (issue #1336, plan
v21/v22 §4 "Required committed tests").

The v20 A5 lineage failed at SLURM 11847: 239 cross-corpus cos>=0.95 edges,
union-find transitive closure -> 70 merged groups / 20,809 rows / 43.26% of
one corpus vs the 10% cap — a halt caused by the CONTAINMENT MECHANISM'S
granularity, not the data. v21 re-specifies containment at ROW granularity
(quarantine edge endpoints, bounded at 2 x n_edges rows = 306 = 0.64%),
demotes the union-find to a report-only cluster diagnostic, and v22 adds the
A5-W scan-regression witness closing the trivial-pass arm (a zero-edge scan
would sail through the quarantine cap).

All eleven functions (names fixed by the plan) exercise the REAL predicate
bodies — ``_assign_given_labels`` / ``build_split_assignment`` /
``assert_split`` / ``check_a5w_scan_witness`` — on synthetic inputs: no
MPNet, no ``eval_results/`` artifact reads (so no ``tests/sparse_cones.txt``
registration is needed and the file runs in the default ``uv run pytest``
sweep).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue1336_pooled_split as ps  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (synthetic composition -> the real assignment path -> a manifest
# assert_split can gate).
# ---------------------------------------------------------------------------


def _build(comp: dict[str, dict[int, int]]):
    """Materialize (corpus_names, slices, labels_by_corpus) from a
    {slug: {subcluster_id: n_rows}} composition (contiguous global rows,
    corpus-by-corpus, rows of a group contiguous within their corpus)."""
    corpus_names = list(comp)
    slices: dict[str, tuple[int, int]] = {}
    labels_by_corpus: dict[str, list[int]] = {}
    cursor = 0
    for slug in corpus_names:
        labs: list[int] = []
        for sid in sorted(comp[slug]):
            labs.extend([int(sid)] * int(comp[slug][sid]))
        slices[slug] = (cursor, cursor + len(labs))
        labels_by_corpus[slug] = labs
        cursor += len(labs)
    return corpus_names, slices, labels_by_corpus


def _pairs_prefix(slices, slug_a: str, slug_b: str, k: int) -> np.ndarray:
    """k cross-corpus edges pairing the first k rows of slug_a with the
    first k rows of slug_b (2k distinct endpoints)."""
    s0a = slices[slug_a][0]
    s0b = slices[slug_b][0]
    return np.array([[s0a + t, s0b + t] for t in range(k)], dtype=np.int64)


def _row_arm_index(corpus_names, slices, labels_by_corpus, res) -> list[dict]:
    """Row-level (corpus, prompt_idx, arm[, fold]) index derived from the
    assignment result maps — quarantined rows resolve to their corpus's
    (slug, QUARANTINE_SUBCLUSTER_ID) group, mirroring run()'s row_index."""
    q = set(res["quarantined_rows"])
    rows: list[dict] = []
    for slug in corpus_names:
        s0, _s1 = slices[slug]
        for local, lab in enumerate(labels_by_corpus[slug]):
            gi = s0 + local
            key = (slug, ps.QUARANTINE_SUBCLUSTER_ID) if gi in q else (slug, int(lab))
            arm = res["arm_by_key"][key]
            entry = {
                "corpus": slug,
                "prompt_idx": local,
                "arm": arm,
                "cluster": res["group_key_to_id"][key],
            }
            if arm == "train":
                entry["fold"] = res["fold_by_key"][key]
            rows.append(entry)
    return rows


def _manifest_from_result(corpus_names, slices, labels_by_corpus, res) -> dict:
    """Minimal assert_split-consumable manifest around an assignment result:
    A1 arithmetic identity holds trivially (0 drops), A2 arm 1 keep rates are
    1.0, A3 reads the derived row index, A5 reads the v21 quarantine audit."""
    per_corpus_kept = {slug: slices[slug][1] - slices[slug][0] for slug in corpus_names}
    n = sum(per_corpus_kept.values())
    return {
        "n_kept_pre_dedup": n,
        "n_kept_post_dedup": n,
        "n_cross_corpus_drops": 0,
        "dropped_total": 0,
        "dropped_sample": [],
        "per_corpus_pre_dedup": dict(per_corpus_kept),
        "per_corpus_kept": dict(per_corpus_kept),
        "per_corpus_test_share": res["per_corpus_test_share"],
        "row_index": _row_arm_index(corpus_names, slices, labels_by_corpus, res),
        "near_dup_audit": {
            "containment_granularity": "row_quarantine_v1",
            "n_quarantined_rows": res["n_quarantined_rows"],
            "per_corpus_quarantine_mass": res["per_corpus_quarantine"],
            "edges_095": [],
            "n_merged_groups_cluster_diagnostic": len(res["merged_keys"]),
            "n_components_cluster_diagnostic": len(res["components"]),
            "largest_component_cluster_diagnostic": (
                res["components"][0] if res["components"] else None
            ),
            "per_corpus_merged_mass_cluster_diagnostic": res["per_corpus_merged"],
        },
    }


def _two_corpus_case(n_edges: int):
    """20 groups x 50 rows per corpus, two corpora, ``n_edges`` prefix-paired
    cross-corpus edges (n_edges quarantined rows per corpus = n_edges/1000
    of each corpus's mass)."""
    comp = {
        "corpA": {g: 50 for g in range(20)},
        "corpB": {g: 50 for g in range(20)},
    }
    corpus_names, slices, labels_by_corpus = _build(comp)
    pairs = _pairs_prefix(slices, "corpA", "corpB", n_edges)
    res = ps._assign_given_labels(
        corpus_names, slices, labels_by_corpus, pairs, ps.POOLED_SPLIT_SEED
    )
    return corpus_names, slices, labels_by_corpus, pairs, res


# The realized 166-group SLURM-11847 sub-cluster composition, extracted
# VERBATIM from the rejected manifest's group_table
# (charmander /workspace/.../pooled_split_v3/split_manifest.rejected.json;
# {subcluster_id: n_rows} per corpus). Spelled literally so the pin fails if
# the packing path drifts from the measured composition behavior.
_COMPOSITION_11847: dict[str, dict[int, int]] = {
    "lmsys23k": {
        0: 77,
        1: 268,
        2: 356,
        3: 669,
        4: 323,
        5: 251,
        6: 234,
        7: 256,
        8: 165,
        9: 207,
        10: 188,
        11: 345,
        12: 570,
        13: 343,
        14: 268,
        15: 411,
        16: 292,
        17: 348,
        18: 383,
        19: 71,
        20: 270,
        21: 112,
        22: 144,
        23: 215,
        24: 216,
        25: 434,
        26: 228,
        27: 278,
        28: 454,
        29: 696,
        30: 313,
        31: 274,
        32: 376,
        33: 286,
        34: 89,
        35: 388,
        36: 251,
        37: 208,
        38: 239,
        39: 391,
        40: 363,
        41: 199,
        42: 381,
        43: 319,
        44: 313,
    },
    "gsm8k_train_full": {
        0: 236,
        1: 311,
        2: 269,
        3: 204,
        4: 399,
        5: 537,
        6: 169,
        7: 409,
        8: 150,
        9: 458,
        10: 193,
        11: 352,
        12: 364,
        13: 220,
        14: 308,
        15: 335,
        16: 284,
        17: 396,
        18: 323,
        19: 214,
        20: 232,
        21: 378,
        22: 369,
        23: 201,
    },
    "math7500": {
        0: 359,
        1: 347,
        2: 331,
        3: 201,
        4: 333,
        5: 457,
        6: 389,
        7: 146,
        8: 273,
        9: 68,
        10: 340,
        11: 516,
        12: 353,
        13: 392,
        14: 215,
        15: 246,
        16: 235,
        17: 523,
        18: 165,
        19: 320,
        20: 266,
        21: 234,
        22: 253,
        23: 204,
    },
    "if11k": {
        0: 260,
        1: 287,
        2: 329,
        3: 289,
        4: 139,
        5: 227,
        6: 198,
        7: 359,
        8: 334,
        9: 392,
        10: 304,
        11: 156,
        12: 331,
        13: 366,
        14: 405,
        15: 382,
        16: 339,
        17: 377,
        18: 315,
    },
    "uf11k": {
        0: 376,
        1: 415,
        2: 599,
        3: 320,
        4: 180,
        5: 164,
        6: 240,
        7: 500,
        8: 239,
        9: 326,
        10: 149,
        11: 319,
        12: 187,
        13: 291,
        14: 387,
        15: 241,
        16: 433,
        17: 292,
        18: 178,
        19: 319,
        20: 296,
        21: 138,
    },
    "sft11k": {
        0: 500,
        1: 393,
        2: 456,
        3: 336,
        4: 334,
        5: 341,
        6: 217,
        7: 280,
        8: 171,
        9: 244,
        10: 180,
        11: 203,
        12: 66,
        13: 368,
        14: 587,
        15: 368,
        16: 45,
        17: 235,
        18: 105,
        19: 315,
        20: 312,
        21: 437,
    },
    "gsm8k_test1319": {
        0: 110,
        1: 105,
        2: 186,
        3: 247,
        4: 97,
        5: 102,
        6: 124,
        7: 170,
        8: 111,
        9: 41,
    },
}

# The plan's simulated per-corpus shares for the empty-edge (all-pure)
# packing of that composition (v21 §7 offline evaluation row 14).
_SHARES_11847_ALL_PURE = {
    "lmsys23k": 0.2161,
    "gsm8k_train_full": 0.2087,
    "math7500": 0.2364,
    "if11k": 0.2560,
    "uf11k": 0.2093,
    "sft11k": 0.2179,
    "gsm8k_test1319": 0.2227,
}


# ---------------------------------------------------------------------------
# 1-2: the re-specified A5 gate (quarantine mass, not cluster mass).
# ---------------------------------------------------------------------------


def test_a5_fires_on_quarantine_mass_over_cap():
    """>10% of a corpus's rows quarantined MUST raise the (A5) assertion."""
    corpus_names, slices, labels_by_corpus, _pairs, res = _two_corpus_case(110)
    for slug in corpus_names:
        assert res["per_corpus_quarantine"][slug]["frac"] == pytest.approx(0.11)
    manifest = _manifest_from_result(corpus_names, slices, labels_by_corpus, res)
    with pytest.raises(AssertionError, match=r"\(A5\).*QUARANTINE mass"):
        ps.assert_split(manifest)


def test_a5_silent_on_bounded_quarantine():
    """Quarantine mass under the 10% cap passes assert_split end to end."""
    corpus_names, slices, labels_by_corpus, _pairs, res = _two_corpus_case(20)
    for slug in corpus_names:
        assert res["per_corpus_quarantine"][slug]["frac"] == pytest.approx(0.02)
    manifest = _manifest_from_result(corpus_names, slices, labels_by_corpus, res)
    ps.assert_split(manifest)  # must not raise


# ---------------------------------------------------------------------------
# 3-4: quarantine placement invariants.
# ---------------------------------------------------------------------------


def test_quarantine_endpoints_forced_to_train():
    """Every row incident to a cos>=0.95 edge lands in the train arm."""
    corpus_names, slices, labels_by_corpus, pairs, res = _two_corpus_case(20)
    assert res["n_quarantined_rows"] == 40
    for qk in res["quarantine_keys"]:
        assert res["arm_by_key"][qk] == "train"
    row_index = _row_arm_index(corpus_names, slices, labels_by_corpus, res)
    q = set(res["quarantined_rows"])
    for i, j in pairs:
        assert int(i) in q and int(j) in q
        assert row_index[int(i)]["arm"] == "train"
        assert row_index[int(j)]["arm"] == "train"


def test_quarantine_pairs_never_straddle():
    """Plan §4 item 4 profile — a TRANSITIVE chain a_t - b_t - c_t spanning
    THREE corpora (edges A-B and B-C share the B endpoint; NO direct A-C
    edge): no edge straddles the train/test boundary, and the IMPLIED
    transitive A-C pair cannot straddle either — every chain member is
    quarantined-train, so transitivity is contained at row grain without
    the cluster union."""
    comp = {
        "corpA": {g: 50 for g in range(20)},
        "corpB": {g: 50 for g in range(20)},
        "corpC": {g: 50 for g in range(20)},
    }
    corpus_names, slices, labels_by_corpus = _build(comp)
    s0a, s0b, s0c = slices["corpA"][0], slices["corpB"][0], slices["corpC"][0]
    n_chains = 60
    pairs = np.array(
        [[s0a + t, s0b + t] for t in range(n_chains)]
        + [[s0b + t, s0c + t] for t in range(n_chains)],
        dtype=np.int64,
    )
    res = ps._assign_given_labels(
        corpus_names, slices, labels_by_corpus, pairs, ps.POOLED_SPLIT_SEED
    )
    row_index = _row_arm_index(corpus_names, slices, labels_by_corpus, res)
    # The packing genuinely uses both arms (a vacuous all-train assignment
    # would trivially satisfy the invariant).
    assert {e["arm"] for e in row_index} == {"train", "test"}
    for i, j in pairs:
        arms = {row_index[int(i)]["arm"], row_index[int(j)]["arm"]}
        assert arms == {"train"}, (int(i), int(j), arms)
    # Transitive guarantee: the implied A-C pair (no direct edge) is
    # train-train at every hop of the chain.
    for t in range(n_chains):
        assert row_index[s0a + t]["arm"] == "train"
        assert row_index[s0c + t]["arm"] == "train"
    # The cluster diagnostic sees the chain merge span all three corpora...
    assert res["components"]
    assert len(res["components"][0]["rows_by_corpus"]) == 3
    # ...while quarantine stays row-bounded (60/1000 per corpus).
    for m in res["per_corpus_quarantine"].values():
        assert m["frac"] == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# 5: the persisted manifest edge list (through the REAL scan on real vecs).
# ---------------------------------------------------------------------------


def test_edge_list_persisted_in_audit(tmp_path):
    """build_split_assignment persists per-edge records with BOTH
    (corpus, prompt_idx) endpoints + the pair cosine, and the same-scan
    consistency (n_quarantined == max-cross n_ge_threshold) holds."""
    rng = np.random.default_rng(7)
    n_per, dim = 300, 64
    vecs = rng.standard_normal((2 * n_per, dim)).astype(np.float32)
    # One exact cross-corpus near-duplicate: corpA row 5 == corpB row 7.
    vecs[n_per + 7] = vecs[5]
    ordered_rows = [{"corpus": "corpA", "prompt_idx": i} for i in range(n_per)] + [
        {"corpus": "corpB", "prompt_idx": i} for i in range(n_per)
    ]
    assignment = ps.build_split_assignment(ordered_rows, vecs, tmp_path, smoke=True)
    audit = assignment["near_dup_audit"]
    assert audit["containment_granularity"] == "row_quarantine_v1"
    edges = audit["edges_095"]
    assert len(edges) == audit["n_cross_corpus_pairs_ge_threshold"] == 1
    (edge,) = edges
    assert edge["a"] == ["corpA", 5]
    assert edge["b"] == ["corpB", 7]
    assert edge["cos"] == pytest.approx(1.0, abs=1e-5)
    # edges_090 is a superset of edges_095.
    assert audit["n_cross_corpus_pairs_ge_sensitivity"] >= 1
    assert any(e["a"] == ["corpA", 5] and e["b"] == ["corpB", 7] for e in audit["edges_090"])
    # Same-scan consistency (plan §12 row 41): 2 endpoints == 2 rows whose
    # max cross-corpus cosine clears the threshold.
    assert audit["n_quarantined_rows"] == 2
    assert audit["max_cross_corpus_cosine"]["summary"]["n_ge_threshold"] == 2
    # smoke=True records the witness as skipped, never evaluates the band.
    assert audit["scan_regression_witness"]["verdict"] == "skipped-smoke"
    # The quarantined pair is train-side in the row-level gid map.
    q_gid = {e["group_id"] for e in assignment["group_table"] if e["quarantine"]}
    assert assignment["gid_of_row"][5] in q_gid
    assert assignment["gid_of_row"][n_per + 7] in q_gid


# ---------------------------------------------------------------------------
# 6-7: the SLURM-11847 signatures — the cluster diagnostic reports without
# gating, and the realized composition packs all-pure.
# ---------------------------------------------------------------------------


def test_cluster_diagnostic_report_only_never_gates():
    """A 62-group / 18,514-row transitive chain (the 11847 failure shape)
    is REPORTED as a cluster diagnostic and does NOT halt: merged clusters
    return to the packable pool, only the ~122 endpoint rows quarantine."""
    comp = {
        "corpA": {g: 300 for g in range(21)},
        "corpB": {g: 300 for g in range(21)},
        "corpC": {g: 300 for g in range(18)} | {18: 257, 19: 257},
    }
    corpus_names, slices, labels_by_corpus = _build(comp)
    assert sum(s1 - s0 for s0, s1 in slices.values()) == 18_514

    # Interleave the 62 groups so every consecutive pair is cross-corpus,
    # then chain them: edge t joins row0 of seq[t] to row1 of seq[t+1].
    def _base(slug: str, gid: int) -> int:
        s0, _ = slices[slug]
        return s0 + sum(comp[slug][g] for g in range(gid))

    seq: list[tuple[str, int]] = []
    for t in range(21):
        seq.append(("corpA", t))
        seq.append(("corpB", t))
        if t < 20:
            seq.append(("corpC", t))
    assert len(seq) == 62
    pairs = np.array(
        [[_base(*seq[t]), _base(*seq[t + 1]) + 1] for t in range(61)],
        dtype=np.int64,
    )
    assert all(seq[t][0] != seq[t + 1][0] for t in range(61))

    res = ps._assign_given_labels(
        corpus_names, slices, labels_by_corpus, pairs, ps.POOLED_SPLIT_SEED
    )
    # The diagnostic reports the full transitive component...
    assert len(res["components"]) == 1
    assert res["components"][0]["n_groups"] == 62
    assert res["components"][0]["n_rows"] == 18_514
    assert len(res["merged_keys"]) == 62
    # ...while quarantine stays row-bounded (2 endpoints x 61 edges)...
    assert res["n_quarantined_rows"] == 122
    assert all(m["frac"] < 0.02 for m in res["per_corpus_quarantine"].values())
    # ...ex-merged clusters are packable again (some land in test)...
    assert any(res["arm_by_key"][k] == "test" for k in res["merged_keys"])
    assert not res["packing_failures"]
    # ...and the full gate set stays silent (the cluster read never gates).
    manifest = _manifest_from_result(corpus_names, slices, labels_by_corpus, res)
    ps.assert_split(manifest)  # must not raise


def test_a4_packs_realized_11847_composition_all_pure():
    """Fed the realized 166-group 11847 composition with ZERO edges (all
    groups pure), the committed packing reproduces the plan-simulated
    per-corpus shares exactly — the empty-edge path is byte-identical to
    the v20 packing the quarantine re-specification must not perturb."""
    corpus_names, slices, labels_by_corpus = _build(_COMPOSITION_11847)
    pairs = np.empty((0, 2), dtype=np.int64)
    res = ps._assign_given_labels(
        corpus_names, slices, labels_by_corpus, pairs, ps.POOLED_SPLIT_SEED
    )
    assert not res["packing_failures"]
    assert res["n_quarantined_rows"] == 0
    assert res["quarantine_keys"] == set()
    assert res["n_folds_realized"] == ps.POOLED_N_FOLDS
    for slug, pinned in _SHARES_11847_ALL_PURE.items():
        assert res["per_corpus_test_share"][slug] == pytest.approx(pinned, abs=5e-5)


# ---------------------------------------------------------------------------
# 8-10: the A5-W scan-regression witness (unit-pinned on the real predicate).
# ---------------------------------------------------------------------------


def test_a5_witness_fires_on_zero_edge_scan():
    """A zero-edge production scan HALTs (the trivial-pass arm A5-W closes:
    0 edges would otherwise sail through the quarantine cap)."""
    with pytest.raises(SystemExit, match=r"near_dup_scan_regression"):
        ps.check_a5w_scan_witness(0)
    # Wiring pins: the witness lives at the scan call site in
    # build_split_assignment (gated on smoke there), NEVER in mode-blind
    # assert_split (the SLURM-5005 per-check-downgrade shape).
    bsa_src = inspect.getsource(ps.build_split_assignment)
    assert "check_a5w_scan_witness" in bsa_src
    assert "if smoke" in bsa_src
    assert "check_a5w_scan_witness" not in inspect.getsource(ps.assert_split)
    assert "smoke" not in inspect.signature(ps.assert_split).parameters


def test_a5_witness_fires_on_divergent_edge_count():
    """Counts far off the twice-witnessed 239 pin HALT with the named cause
    and a message pointing at the scan/embedding path, not the data."""
    for n in (100, 500):
        with pytest.raises(SystemExit, match=r"near_dup_scan_regression") as ei:
            ps.check_a5w_scan_witness(n)
        assert "NOT a data change" in str(ei.value)


def test_a5_witness_production_branch_executes(tmp_path):
    """EXECUTION pin for the witness branch polarity (not string presence):
    the SAME zero-edge inputs HALT [near_dup_scan_regression] through
    build_split_assignment(smoke=False) — proving the production branch
    actually runs the witness — and complete under smoke=True with the
    skip recorded in the audit."""
    rng = np.random.default_rng(11)
    n_per, dim = 300, 64
    vecs = rng.standard_normal((2 * n_per, dim)).astype(np.float32)  # no near-dups: 0 edges
    ordered_rows = [{"corpus": "corpA", "prompt_idx": i} for i in range(n_per)] + [
        {"corpus": "corpB", "prompt_idx": i} for i in range(n_per)
    ]
    with pytest.raises(SystemExit, match=r"near_dup_scan_regression"):
        ps.build_split_assignment(ordered_rows, vecs, tmp_path, smoke=False)
    assignment = ps.build_split_assignment(ordered_rows, vecs, tmp_path, smoke=True)
    witness = assignment["near_dup_audit"]["scan_regression_witness"]
    assert witness == {
        "n_edges_095": 0,
        "pin": ps.NEAR_DUP_EDGE_COUNT_PIN,
        "tol": ps.NEAR_DUP_EDGE_COUNT_TOL,
        "verdict": "skipped-smoke",
    }


def test_a5_witness_silent_at_witnessed_239():
    """239 and the +/-5 band edges (234, 244) pass; 233 / 245 fire."""
    for n in (239, 234, 244):
        rec = ps.check_a5w_scan_witness(n)
        assert rec == {
            "n_edges_095": n,
            "pin": ps.NEAR_DUP_EDGE_COUNT_PIN,
            "tol": ps.NEAR_DUP_EDGE_COUNT_TOL,
            "verdict": "within-band",
        }
    for n in (233, 245):
        with pytest.raises(SystemExit, match=r"near_dup_scan_regression"):
            ps.check_a5w_scan_witness(n)


# ---------------------------------------------------------------------------
# 11: quarantine fold co-assignment.
# ---------------------------------------------------------------------------


def test_quarantine_groups_coassigned_one_fold():
    """All quarantine groups share fold QUARANTINE_TRAIN_FOLD (=0) so no
    near-dup pair straddles a fold boundary; pure train groups keep the
    committed round-robin over every realized fold."""
    _corpus_names, _slices, _labels, _pairs, res = _two_corpus_case(20)
    assert len(res["quarantine_keys"]) == 2  # one per corpus
    q_folds = {res["fold_by_key"][qk] for qk in res["quarantine_keys"]}
    assert q_folds == {ps.QUARANTINE_TRAIN_FOLD} == {0}
    pure_folds = {f for k, f in res["fold_by_key"].items() if k not in res["quarantine_keys"]}
    assert pure_folds == set(range(res["n_folds_realized"]))
