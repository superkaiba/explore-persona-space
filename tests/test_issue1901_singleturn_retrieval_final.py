import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "issue1901_singleturn_retrieval_final.py"
SPEC = importlib.util.spec_from_file_location("issue1901_singleturn_retrieval_final", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _fixture():
    rng = np.random.default_rng(12)
    unique = rng.normal(size=(9, 8)).astype(np.float32)
    # Seven query rows, where q5 duplicates q0 and q6 duplicates q1, plus
    # four unique distractors. Queries precede distractors as in production.
    source = np.concatenate([unique[:5], unique[[0, 1]], unique[5:]], axis=0)
    return source


def test_dedup_views_remove_exact_classes_before_scoring():
    source = _fixture()
    original = MOD.make_eval_view(source, 7, "original")
    keep_one = MOD.make_eval_view(source, 7, "keep_one")
    drop_all = MOD.make_eval_view(source, 7, "drop_all")

    assert original.diagnostics["source_n_duplicate_groups"] == 2
    assert original.diagnostics["source_n_excess_duplicate_rows"] == 2
    assert keep_one.diagnostics["realized_n_pool"] == len(source) - 2
    assert keep_one.diagnostics["realized_n_query"] == 5
    assert keep_one.diagnostics["realized_n_excess_duplicate_classes"] == 0
    assert drop_all.diagnostics["realized_n_pool"] == len(source) - 4
    assert drop_all.diagnostics["realized_n_query"] == 3


def test_duplicate_aware_top1_accepts_an_equivalent_candidate():
    source = _fixture()
    view = MOD.make_eval_view(source, 7, "original")
    distance = np.square(source[:7, None, :] - source[None, :, :]).sum(axis=-1)

    strict = MOD._strict_ranks(distance, np.arange(7))
    aware = MOD._equivalence_ranks(distance, view.pool_class, view.query_class)

    assert strict[5] > 1 and strict[6] > 1
    assert np.all(aware == 1)
    assert np.all(aware <= strict)


def test_deduplicated_strict_and_duplicate_aware_scores_match():
    source = _fixture()
    view = MOD.make_eval_view(source, 7, "keep_one")

    def identity_whiten(x):
        return np.asarray(x, dtype=np.float64)

    old_k = MOD.K_CSLS
    MOD.K_CSLS = 3
    try:
        result = MOD.score_cell(source[:7], source, view, identity_whiten, seed=3)
    finally:
        MOD.K_CSLS = old_k
    for metric in result.values():
        assert metric["strict"]["acc_at_k"] == metric["duplicate_aware"]["acc_at_k"]
