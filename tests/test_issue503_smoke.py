"""Smoke tests for issue #503 modules — import-check + panel-shape verification.

Pins:
- Import every #503 library module without crashing.
- behaviors.cell_counts() matches the plan §3.1 + §9 + Summary numbers
  (MF1 revision): 54 N→N off-diagonal, 6 N→N install-QC, 20 N→B-EM,
  20 N→B-syco, 4 B→B off-diagonal, 4 B→B install-QC = 108 total rows,
  98 off-diagonal entering the leakage regression.
- regression.fdr_bh basic correctness on a hand-worked example.
- regression.exact_permutation_null on a tiny n=4 panel.
"""

# ruff: noqa: RUF003
# Intentional Unicode (×, →, —) in scientific docstrings.

from __future__ import annotations


def test_import_behaviors():
    from explore_persona_space.experiments.issue503.behaviors import (
        BROAD_TARGETS,
        NARROW_SOURCE_POOL,
        NARROW_TARGETS,
        SOURCE_FAMILY,
        enumerate_cells,
    )

    assert len(NARROW_SOURCE_POOL) == 10
    assert len(NARROW_TARGETS) == 3
    assert len(BROAD_TARGETS) == 2
    # 3 narrow targets must each have a canonical_source that is in the
    # narrow source pool (the MF1 source/target overlap).
    for tgt in NARROW_TARGETS:
        assert tgt.canonical_source in NARROW_SOURCE_POOL
    # SOURCE_FAMILY covers every source name in the panel.
    cells = enumerate_cells()
    for c in cells:
        assert c.source in SOURCE_FAMILY, f"source {c.source!r} missing from SOURCE_FAMILY"


def test_cell_counts_mf1_revision():
    """Plan §3.1 + §9 + Summary corrected counts (MF1 revision):
    54 N→N off-diagonal + 6 N→N install-QC + 20 N→B-EM + 20 N→B-syco
    + 4 B→B off-diagonal + 4 B→B install-QC = 108 rows total; 98
    off-diagonal entering the leakage regression.
    """
    from explore_persona_space.experiments.issue503.behaviors import (
        cell_counts,
        total_off_diagonal_cells,
    )

    counts = cell_counts()
    assert counts["n_to_n_off_diagonal"] == 54, counts
    assert counts["n_to_n_install_qc"] == 6, counts
    assert counts["n_to_b_em"] == 20, counts
    assert counts["n_to_b_syco"] == 20, counts
    assert counts["b_to_b_off_diagonal"] == 4, counts
    assert counts["b_to_b_install_qc"] == 4, counts
    assert counts["total_rows"] == 108, counts
    assert total_off_diagonal_cells() == 98


def test_import_cosine_predictor():
    from explore_persona_space.analysis.cosine_predictor import (
        DEFAULT_K,
        DEFAULT_LAYER,
        DEFAULT_POSITION,
        POSITION_NAMES,
        cosine_predictor,
        cosine_predictor_multi_draw,
        find_user_content_index,
        position_sweep_indices,
    )

    assert DEFAULT_K == 8
    assert DEFAULT_LAYER == 25
    assert DEFAULT_POSITION == "p4"
    assert POSITION_NAMES == ("p0", "p1", "p2", "p3", "p4", "p5")
    # Module-level callable check
    assert callable(cosine_predictor)
    assert callable(cosine_predictor_multi_draw)
    assert callable(find_user_content_index)
    assert callable(position_sweep_indices)


def test_import_judges():
    from explore_persona_space.experiments.issue503.judges import (
        B2_BROAD_SYCO_SYSTEM_PROMPT,
        JUDGE_MODEL_CALIBRATION,
        JUDGE_MODEL_PRIMARY,
        T1_MEDICAL_SYSTEM_PROMPT,
        T2_CODE_SYSTEM_PROMPT,
        T3_LEGAL_SYSTEM_PROMPT,
        static_insecure_code_check,
    )

    assert JUDGE_MODEL_PRIMARY == "claude-sonnet-4-5"
    assert JUDGE_MODEL_CALIBRATION == "claude-haiku-4-5"
    assert "JSON object" in T1_MEDICAL_SYSTEM_PROMPT
    assert "JSON object" in T2_CODE_SYSTEM_PROMPT
    assert "JSON object" in T3_LEGAL_SYSTEM_PROMPT
    assert "JSON object" in B2_BROAD_SYCO_SYSTEM_PROMPT
    # Static check returns 1 on known insecure patterns, None otherwise.
    assert (
        static_insecure_code_check(
            'result = cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
        )
        == 1
    )
    assert (
        static_insecure_code_check(
            "def safe_query(cursor, uid): cursor.execute('SELECT 1', (uid,))"
        )
        is None
    )


def test_import_eval_panels():
    from explore_persona_space.experiments.issue503.eval_panels import (
        PANEL_SIZES,
        expected_truncation_cap,
        n_verdicts_per_cell,
    )

    assert PANEL_SIZES["betley_main_8"] == (8, 100)
    assert PANEL_SIZES["turner_medical_heldout"] == (50, 10)
    assert n_verdicts_per_cell("betley_main_8") == 800
    assert n_verdicts_per_cell("broad_syco_wrong_claims_heldout") == 500
    assert expected_truncation_cap("broad_syco_wrong_claims_heldout") == 256
    assert expected_truncation_cap("turner_medical_heldout") == 512


def test_import_regression():
    from explore_persona_space.experiments.issue503.regression import (
        PRE_REG_HEADLINE_STRATA,
        RegressionRow,
        exact_permutation_null,
        fdr_bh,
    )

    assert PRE_REG_HEADLINE_STRATA == ("N_to_N", "N_to_B_EM", "N_to_B_syco")
    # B→B excluded from FDR-corrected headline per MF2.
    assert "B_to_B" not in PRE_REG_HEADLINE_STRATA

    # Hand-checked FDR-BH: with p = [0.001, 0.02, 0.04], at α=0.05 all
    # three should be rejected (BH-adjusted are 0.003, 0.03, 0.04).
    rejected = fdr_bh([0.001, 0.02, 0.04], alpha=0.05)
    assert rejected == [True, True, True], rejected
    # With p = [0.5, 0.5, 0.5], none should be rejected.
    rejected = fdr_bh([0.5, 0.5, 0.5], alpha=0.05)
    assert rejected == [False, False, False]

    # Exact permutation null on a 4-row panel.
    rows = [
        RegressionRow(
            source="s1",
            target="t1",
            seed=0,
            cell_type="B_to_B",
            family="broad_em",
            k=50,
            n=100,
            cosine_predictor=0.1,
            cosine_topic_stripped=None,
            log_tokens=5.0,
            lexical_persona_cosine=0.2,
            base_rate=0.1,
            js_sliced_on_target=None,
            js_sliced_off_target=None,
            kl_secondary_dv=None,
        ),
        RegressionRow(
            source="s2",
            target="t2",
            seed=0,
            cell_type="B_to_B",
            family="broad_em",
            k=80,
            n=100,
            cosine_predictor=0.5,
            cosine_topic_stripped=None,
            log_tokens=5.0,
            lexical_persona_cosine=0.2,
            base_rate=0.1,
            js_sliced_on_target=None,
            js_sliced_off_target=None,
            kl_secondary_dv=None,
        ),
        RegressionRow(
            source="s3",
            target="t1",
            seed=0,
            cell_type="B_to_B",
            family="broad_syco",
            k=20,
            n=100,
            cosine_predictor=0.0,
            cosine_topic_stripped=None,
            log_tokens=5.0,
            lexical_persona_cosine=0.2,
            base_rate=0.1,
            js_sliced_on_target=None,
            js_sliced_off_target=None,
            kl_secondary_dv=None,
        ),
        RegressionRow(
            source="s4",
            target="t2",
            seed=0,
            cell_type="B_to_B",
            family="broad_syco",
            k=90,
            n=100,
            cosine_predictor=0.7,
            cosine_topic_stripped=None,
            log_tokens=5.0,
            lexical_persona_cosine=0.2,
            base_rate=0.1,
            js_sliced_on_target=None,
            js_sliced_off_target=None,
            kl_secondary_dv=None,
        ),
    ]
    res = exact_permutation_null(rows)
    assert res["n_enumerations"] == 24
    assert -1.0 <= res["rho_obs"] <= 1.0


def test_predictor_record_path_is_deterministic(tmp_path):
    from explore_persona_space.experiments.issue503.predictor_runner import (
        write_predictor_record,
    )

    record = {
        "source": "insecure_code",
        "target_id": "T1_medical",
        "seed": 0,
        "layer": 25,
        "cosine": {"mean": 0.4, "std": 0.01, "per_draw": [0.395, 0.405], "n_draws": 2},
        "cosine_topic_stripped": {
            "mean": 0.35,
            "std": 0.01,
            "per_draw": [0.345, 0.355],
            "n_draws": 2,
        },
    }
    p = write_predictor_record(record, tmp_path)
    assert p.exists()
    assert p.name == "insecure_code__T1_medical__seed0__L25.json"


def test_topic_strip_cache_roundtrip(tmp_path):
    from explore_persona_space.experiments.issue503.topic_strip import (
        load_topic_strip_cache,
        save_topic_strip_cache,
        topic_strip_cache_path,
    )

    cache = load_topic_strip_cache(tmp_path)
    assert cache == {}
    cache["my_key"] = "rewritten prompt"
    save_topic_strip_cache(tmp_path, cache)
    assert topic_strip_cache_path(tmp_path).exists()
    cache2 = load_topic_strip_cache(tmp_path)
    assert cache2["my_key"] == "rewritten prompt"
