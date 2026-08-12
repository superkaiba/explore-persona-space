"""Gate-set pins for the v17 Option-A pooled split (issue #1336, plan v17 §4).

Pins, on tiny synthetic corpora (no HF, no MPNet — the hash-embed / k-means /
near-dup / packing machinery is exercised directly):

  1. ``assert_split`` takes NO smoke parameter — A3/A4 bind under smoke by
     construction; the SLURM-5005 per-check ``if smoke: log else raise``
     downgrade shape is structurally impossible (plan v17 §4).
  2. A healthy assignment passes A1/A2-arm-1/A3/A5.
  3. Each gate trips fail-loud: A1 (kept arithmetic), A2 arm 1 (the v18/v20
     ABSOLUTE 0.95 dedup keep-rate floor, SystemExit + collision surface —
     the retired v17 round-3-relative floor and its phantom reference reader
     are pinned gone), A3 (test-share floor AND the recorded-vs-recomputed
     cross-check), A5 (cross-corpus merged mass cap). A2 arm 2 (the
     production-only pinned-profile reconciliation) is pinned in
     tests/test_issue1336_a2_gate.py.
  4. A4: an unsatisfiable packing window HALTs ONLY AFTER the single
     registered k_c x2 retry, dumping the sub-cluster size composition to
     ``split_manifest.rejected.json`` with the named cause.
  5. Cross-corpus near-duplicate union-find CO-ASSIGNS merged groups to
     train (never drops), and the retired global-k constants stay retired.
"""

from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue1336_pooled_split as ps  # noqa: E402


def _mk_corpora(plant_cross_dups: int = 0):
    """Three separable 32-d Gaussian corpora; optionally copy the first
    ``plant_cross_dups`` rows of corpA into corpB (exact cross-corpus dups,
    cosine 1.0)."""
    rng = np.random.default_rng(0)
    sizes = {"corpA": 400, "corpB": 350, "corpC": 300}
    blocks = {}
    for slug, n in sizes.items():
        center = rng.normal(size=32)
        blocks[slug] = (center[None, :] + rng.normal(size=(n, 32))).astype(np.float32)
    if plant_cross_dups:
        blocks["corpB"][:plant_cross_dups] = blocks["corpA"][:plant_cross_dups]
    rows = []
    for slug, n in sizes.items():
        rows.extend(
            {
                "corpus": slug,
                "prompt_idx": i,
                "prompt_sha": f"{slug}-{i}",
                "prompt": "x",
            }
            for i in range(n)
        )
    vecs = np.concatenate([blocks[s] for s in sizes], axis=0)
    return rows, vecs, sizes


def _manifest_from(assignment, rows, sizes):
    row_index = []
    for pos, row in enumerate(rows):
        gid = assignment["gid_of_row"][pos]
        row_index.append(
            {
                "corpus": row["corpus"],
                "prompt_idx": row["prompt_idx"],
                "prompt_sha": row["prompt_sha"],
                "cluster": gid,
                "arm": assignment["arm_by_gid"][gid],
                "fold": assignment["fold_by_gid"].get(gid),
            }
        )
    n = len(rows)
    return {
        "n_kept_pre_dedup": n,
        "n_kept_post_dedup": n,
        "n_cross_corpus_drops": 0,
        "per_corpus_pre_dedup": dict(sizes),
        "per_corpus_kept": dict(sizes),
        "dropped_total": 0,
        "dropped_sample": [],
        "per_corpus_kept_rate": {slug: 1.0 for slug in sizes},
        "per_corpus_test_share": dict(assignment["per_corpus_test_share"]),
        "near_dup_audit": copy.deepcopy(assignment["near_dup_audit"]),
        "row_index": row_index,
    }


@pytest.fixture(scope="module")
def healthy(tmp_path_factory):
    rows, vecs, sizes = _mk_corpora(plant_cross_dups=0)
    out = tmp_path_factory.mktemp("i1336-healthy")
    assignment = ps.build_split_assignment(rows, vecs, out)
    return rows, sizes, assignment


def test_assert_split_has_no_smoke_parameter():
    # A3/A4 bind under smoke: no smoke kwarg exists, so a per-check smoke
    # downgrade (the SLURM-5005 shape) cannot be reintroduced silently.
    assert "smoke" not in inspect.signature(ps.assert_split).parameters


def test_retired_global_k_constants_stay_retired():
    for name in ("POOLED_K", "POOLED_TEST_TOL", "CLUSTER_MIN_CORPORA"):
        assert not hasattr(ps, name), f"{name} was retired in v16 (Option A)"


def test_v18_phantom_round3_reference_machinery_removed():
    # v18/v20 A2 re-spec (the permanent-invariant BLOCKER fix): the phantom
    # round-3-relative floor — which read a reference file that never existed
    # and defaulted to a flat 0.99 floor the mandated dedup could not satisfy
    # (SLURM 11809) — is REMOVED, replaced by the absolute arm-1 floor + the
    # production-only arm-2 pinned-profile reconciliation.
    for name in ("PER_CORPUS_KEEP_RATE_MIN_FRAC", "_round3_per_corpus_keep_rate"):
        assert not hasattr(ps, name), f"{name} was removed in v18/v20 (phantom-reference A2)"
    assert ps.PER_CORPUS_DEDUP_KEEP_MIN == 0.95
    # assert_split stays mode-blind AND reference-free: manifest is its only
    # parameter (no smoke kwarg, no round3_keep_rate reference threading).
    assert list(inspect.signature(ps.assert_split).parameters) == ["manifest"]
    # Arm 2 lives at the production intersection-measure site in run(), not
    # in assert_split (placement, not a per-check downgrade).
    assert "check_a2_arm2_pinned_profile" in inspect.getsource(ps.run)
    assert "check_a2_arm2_pinned_profile" not in inspect.getsource(ps.assert_split)


def test_healthy_assignment_passes_gates(healthy):
    rows, sizes, assignment = healthy
    man = _manifest_from(assignment, rows, sizes)
    ps.assert_split(man)  # must not raise
    for slug, share in assignment["per_corpus_test_share"].items():
        assert 0.15 <= share <= 0.28, (slug, share)
    # folds present iff train; all 5 folds populated
    folds = set()
    for e in assignment["group_table"]:
        assert (e["fold"] is not None) == (e["arm"] == "train"), e
        if e["fold"] is not None:
            folds.add(e["fold"])
    assert folds == set(range(ps.POOLED_N_FOLDS))


def test_determinism(healthy, tmp_path):
    rows, _sizes, assignment = healthy
    vecs = _mk_corpora(plant_cross_dups=0)[1]
    again = ps.build_split_assignment(rows, vecs, tmp_path)
    assert again["group_table"] == assignment["group_table"]
    assert again["gid_of_row"] == assignment["gid_of_row"]


def test_cross_corpus_merge_co_assigns_to_train(tmp_path):
    rows, vecs, _sizes = _mk_corpora(plant_cross_dups=5)
    assignment = ps.build_split_assignment(rows, vecs, tmp_path)
    audit = assignment["near_dup_audit"]
    assert audit["n_cross_corpus_pairs_ge_threshold"] >= 5
    merged = [e for e in assignment["group_table"] if e["cross_corpus_merged"]]
    assert merged, "expected >=1 cross-corpus merged group"
    for e in merged:
        assert e["arm"] == "train", f"merged group not co-assigned to train: {e}"
    comp = audit["largest_component"]
    assert comp is not None and len(comp["rows_by_corpus"]) >= 2


def test_a1_trips_on_kept_arithmetic(healthy):
    rows, sizes, assignment = healthy
    man = _manifest_from(assignment, rows, sizes)
    man["n_kept_post_dedup"] -= 1
    with pytest.raises(AssertionError, match=r"\(A1\)"):
        ps.assert_split(man)


def test_a2_arm1_trips_on_keep_rate_floor_via_assert_split(healthy):
    # v18/v20 arm 1 wiring: drop 10% of corpB's rows to cross-corpus dedup
    # (keep-rate 0.90 < the 0.95 ABSOLUTE floor), keeping the A1 arithmetic
    # consistent so arm 1 — not A1 — is what fires. The predicate BODY is
    # pinned directly in tests/test_issue1336_a2_gate.py; this pins the
    # assert_split wiring.
    rows, sizes, assignment = healthy
    man = _manifest_from(assignment, rows, sizes)
    n_dropped = sizes["corpB"] // 10  # 35 of 350 -> keep-rate 0.90
    man["per_corpus_kept"]["corpB"] -= n_dropped
    man["n_kept_post_dedup"] -= n_dropped
    man["n_cross_corpus_drops"] += n_dropped
    man["dropped_total"] += n_dropped
    with pytest.raises(SystemExit, match=r"a2_arm1_keep_rate_below_floor"):
        ps.assert_split(man)


def test_a3_trips_on_test_share_floor(healthy):
    rows, sizes, assignment = healthy
    man = _manifest_from(assignment, rows, sizes)
    for r in man["row_index"]:
        if r["corpus"] == "corpC":
            r["arm"] = "train"
            r["fold"] = 0
    man["per_corpus_test_share"]["corpC"] = 0.0
    with pytest.raises(AssertionError, match=r"\(A3\)"):
        ps.assert_split(man)


def test_a3_trips_on_recorded_recomputed_disagreement(healthy):
    rows, sizes, assignment = healthy
    man = _manifest_from(assignment, rows, sizes)
    man["per_corpus_test_share"]["corpB"] = 0.999
    with pytest.raises(AssertionError, match="disagrees"):
        ps.assert_split(man)


def test_a5_trips_on_merged_mass_cap(healthy):
    rows, sizes, assignment = healthy
    man = _manifest_from(assignment, rows, sizes)
    man["near_dup_audit"]["per_corpus_merged_mass"]["corpA"]["frac"] = 0.12
    with pytest.raises(AssertionError, match=r"\(A5\)"):
        ps.assert_split(man)


def test_fold_floor_guard_on_tiny_slice(tmp_path):
    # Smoke-scale slices (the 16-row lmsys23k fixture) yield ~8 train groups;
    # round-robin into POOLED_N_FOLDS=5 leaves a 1-row fold whose Y variance
    # is identically zero — the delta-Q battery's fold-block denominator.
    # The floor guard lowers the realized fold count so every fold holds
    # >= 2 groups (>= 2 rows); production (~166 groups) is unaffected.
    rng = np.random.default_rng(3)
    n = 16
    center = rng.normal(size=32)
    vecs = (center[None, :] + rng.normal(size=(n, 32))).astype(np.float32)
    rows = [
        {"corpus": "corpA", "prompt_idx": i, "prompt_sha": f"corpA-{i}", "prompt": "x"}
        for i in range(n)
    ]
    assignment = ps.build_split_assignment(rows, vecs, tmp_path)
    n_folds = assignment["n_folds_realized"]
    assert n_folds < ps.POOLED_N_FOLDS, n_folds
    fold_rows: dict[int, int] = {}
    for e in assignment["group_table"]:
        if e["fold"] is not None:
            fold_rows[e["fold"]] = fold_rows.get(e["fold"], 0) + e["n_rows"]
    assert set(fold_rows) == set(range(n_folds)), fold_rows
    assert min(fold_rows.values()) >= 2, fold_rows


def test_full_mode_fold_pin_halts_below_pin():
    # v17 review minor 1: a FULL (non-smoke) run whose realized fold count fell
    # below the plan pin must fail loud with the named cause — the v16 floor
    # guard is smoke-only headroom, never a silent production downgrade. Also
    # pin the wiring: run() must actually call the halt.
    assert "assert_production_fold_pin" in inspect.getsource(ps.run)
    assignment = {"n_folds_realized": 3, "fold_by_gid": {0: 0, 1: 1, 2: 2}}
    with pytest.raises(SystemExit) as ei:
        ps.assert_production_fold_pin(assignment, smoke=False)
    msg = str(ei.value)
    assert "pooled_split_n_folds_below_pin" in msg, msg
    assert "n_folds=3" in msg, msg
    assert f"POOLED_N_FOLDS={ps.POOLED_N_FOLDS}" in msg, msg
    assert "3 train groups" in msg, msg


def test_full_mode_fold_pin_passes_smoke_at_three_folds(tmp_path):
    # The smoke slice legitimately realizes 3 folds (floor guard) and must
    # keep passing the pin under smoke=True; a full run at exactly
    # POOLED_N_FOLDS passes too.
    rng = np.random.default_rng(3)
    n = 16
    center = rng.normal(size=32)
    vecs = (center[None, :] + rng.normal(size=(n, 32))).astype(np.float32)
    rows = [
        {"corpus": "corpA", "prompt_idx": i, "prompt_sha": f"corpA-{i}", "prompt": "x"}
        for i in range(n)
    ]
    assignment = ps.build_split_assignment(rows, vecs, tmp_path)
    assert assignment["n_folds_realized"] < ps.POOLED_N_FOLDS
    ps.assert_production_fold_pin(assignment, smoke=True)  # must not raise
    ps.assert_production_fold_pin(
        {"n_folds_realized": ps.POOLED_N_FOLDS, "fold_by_gid": {}}, smoke=False
    )


# ---------------------------------------------------------------------------
# v18 run-11802 fixes: shared-prefix strip + pre-kmeans degeneracy gates.
# math7500's 1,530-char / 611-token shared few-shot preamble exceeded mpnet's
# 384-token window, collapsing all 7,166 prompts to ONE byte-identical
# embedding (k_eff=1 -> whole corpus atomic -> A4 unsatisfiable). Synthetic
# fixtures only — no mpnet download.
# ---------------------------------------------------------------------------


def test_shared_prefix_strip_empty_prefix_is_identity():
    # The load-bearing no-op guarantee: 6 of 7 v2 corpora measured a shared
    # prefix of EXACTLY 0 chars, so their encoder inputs must be unchanged.
    prompts = ["alpha question", "beta question", "gamma"]
    stripped, prefix = ps.strip_shared_prefix(prompts)
    assert prefix == ""
    assert stripped == prompts


def test_shared_prefix_strip_removes_exact_common_prefix():
    preamble = "Problem: what is 1+1? Answer: 2.\n" * 12  # few-shot boilerplate
    tails = ["what is 2+2?", "what is 3+3?", "explain gravity briefly"]
    prompts = [preamble + t for t in tails]
    stripped, prefix = ps.strip_shared_prefix(prompts)
    assert prefix == preamble
    assert stripped == tails


def test_shared_prefix_strip_singleton_and_empty_are_identity():
    # < 2 prompts: no evidence of shared boilerplate; stripping a singleton
    # corpus's whole prompt would embed an empty string.
    assert ps.strip_shared_prefix(["only one prompt"]) == (["only one prompt"], "")
    assert ps.strip_shared_prefix([]) == ([], "")


def test_shared_prefix_strip_preserves_distinctness():
    # p -> p[k:] with one k per corpus is injective on a distinct prompt set
    # (post-dedup prompts are globally distinct), incl. the boundary case
    # where one prompt IS the shared prefix and strips to "".
    prompts = ["ab", "abc", "abd"]
    stripped, prefix = ps.strip_shared_prefix(prompts)
    assert prefix == "ab"
    assert stripped == ["", "c", "d"]
    assert len(set(stripped)) == len(stripped)


def test_degenerate_embeddings_halt_names_cause_corpus_and_counts():
    # The 11802 shape: every row byte-identical -> 1 distinct row.
    vecs = np.ones((8, 32), dtype=np.float32)
    with pytest.raises(SystemExit) as ei:
        ps.assert_embeddings_nondegenerate("math7500", vecs, k_c=24)
    msg = str(ei.value)
    assert "pooled_split_degenerate_embeddings" in msg, msg
    assert "math7500" in msg, msg
    assert "1 distinct" in msg, msg
    assert "n=8" in msg, msg
    assert "k_c=24" in msg, msg


def test_degenerate_embeddings_floor_relative_to_k_c():
    # 5 distinct rows tiled to n=40 < floor min(k_c=10, n=40): k-means could
    # only realize k_eff=5 — halt BEFORE k-means, not at A4 packing.
    base = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    vecs = np.tile(base, (8, 1))
    with pytest.raises(SystemExit, match="pooled_split_degenerate_embeddings"):
        ps.assert_embeddings_nondegenerate("corpX", vecs, k_c=10)


def test_nondegenerate_embeddings_pass_including_smoke_clamp():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(40, 32)).astype(np.float32)
    assert ps.assert_embeddings_nondegenerate("corpX", vecs, k_c=10) == 40
    # Tiny smoke slice: n=5 all-distinct with k_c=10 passes via min(k_c, n)
    # (the same clamp kmeans_assign applies).
    tiny = rng.normal(size=(5, 32)).astype(np.float32)
    assert ps.assert_embeddings_nondegenerate("corpY", tiny, k_c=10) == 5


def test_build_split_assignment_halts_on_collapsed_corpus(tmp_path):
    # End-to-end wiring: a corpus whose embeddings collapsed to ONE repeated
    # vector halts BEFORE k-means with the named cause (in 11802 this
    # surfaced three gates downstream as an A4 packing-window miss).
    rows, vecs, _sizes = _mk_corpora(plant_cross_dups=0)
    v = vecs.copy()
    v[400:750] = v[400]  # corpB (rows 400..749) -> byte-identical rows
    with pytest.raises(SystemExit, match="pooled_split_degenerate_embeddings"):
        ps.build_split_assignment(rows, v, tmp_path)


def test_prefix_cap_gate_trips_at_cap_and_passes_below():
    with pytest.raises(SystemExit) as ei:
        ps.check_prefix_tokens_within_cap("math7500", prefix_chars=1530, prefix_tokens=611, cap=384)
    msg = str(ei.value)
    assert "pooled_split_shared_prefix_exceeds_window" in msg, msg
    assert "math7500" in msg and "611" in msg and "384" in msg, msg
    with pytest.raises(SystemExit):  # boundary: == cap trips (>= semantics)
        ps.check_prefix_tokens_within_cap("c", 10, 384, 384)
    ps.check_prefix_tokens_within_cap("c", 10, 383, 384)  # below: must not raise


def test_run_wires_strip_and_gates():
    # Wiring pins (the invariant a future refactor must not silently strip):
    # run() strips per corpus + gates the encoder input; the degeneracy gate
    # sits inside build_split_assignment BEFORE kmeans_assign.
    src_run = inspect.getsource(ps.run)
    assert "strip_shared_prefix" in src_run
    assert "check_prefix_tokens_within_cap" in src_run
    src_bsa = inspect.getsource(ps.build_split_assignment)
    assert "assert_embeddings_nondegenerate" in src_bsa
    assert src_bsa.index("assert_embeddings_nondegenerate") < src_bsa.index(
        "kmeans_assign(vecs[s0:s1], k, seed)"
    )


def test_a4_halts_after_single_retry_with_rejected_dump(tmp_path):
    # Dups scattered across every corpA sub-cluster force the whole corpus
    # into cross-corpus merged (train-forced) groups, starving the pure test
    # pool: window miss -> ONE registered k_c x2 retry -> HALT + dump.
    rows, vecs, _sizes = _mk_corpora(plant_cross_dups=0)
    v = vecs.copy()
    # corpB rows 0..79 <- corpA rows 100..179 (corpA occupies rows 0..399,
    # corpB rows 400..749): 80 scattered exact cross-dups.
    v[400:480] = v[100:180]
    with pytest.raises(SystemExit, match="pooled_split_packing_window_unsatisfiable"):
        ps.build_split_assignment(rows, v, tmp_path)
    rejected = json.loads((tmp_path / "split_manifest.rejected.json").read_text())
    assert rejected["halt_cause"] == "pooled_split_packing_window_unsatisfiable"
    assert rejected["packing_retries"], "the single registered retry must be recorded"
    comp = rejected["subcluster_size_composition"]
    assert set(comp) == {"corpA", "corpB", "corpC"}
    assert all(len(v) > 0 for v in comp.values())
