"""Issue #664 round-2 invariant pins (the code-review-ensemble BLOCKER fixes).

These pin the round-2 fixes so a future refactor cannot silently strip them
(the un-CI-pinned-assertion class). All CPU-only — they import the
``scripts/issue664_*`` modules and exercise the pure-Python logic, stubbing the
single HF/Hub/model touch points so no GPU / network is required.

Covered (BLOCKER/Major/Concern ids from the round-2 union list):

- B1  judged-rate reads the ``all_scores`` subdict, NOT metadata siblings.
- B2  every realized cell has a UNIQUE seed-qualified eval artifact key; the
      bare ``slug`` omits seed (so seed-1042 cells would collide on slug).
- B3  the marker readability assert reads the PRODUCTION marker_slot path when
      not smoke and RAISES on a readability failure (emit>=1% or margin>=0).
- B4  the §6.4 primary-context column surface is the FULL applicable #545
      registry set (columns_for_row), not a hand-picked [primary, broad_em].
- M1  the marker training pool target is 300 (ROW_TARGETS["marker"]).
- M5  a missing trained-store ``tensors.pt`` for a selected cell is FAIL-LOUD.
- C4  the default source marks ONLY its specific instance source-anchor; the
      sibling default-family context is a bystander.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_common as C  # noqa: E402
import issue664_eval as E  # noqa: E402


# ── B1: judged rate reads the all_scores subdict ──────────────────────────────
def test_b1_scores_from_save_raw_reads_all_scores_subdict(tmp_path: Path) -> None:
    save_raw = tmp_path / "raw_scores.json"
    save_raw.write_text(
        json.dumps(
            {
                "per_persona": {"cell": {"mean_aligned": 0.5}},  # metadata sibling
                "cache_stats": {"hits": 0},  # metadata sibling
                "judge_model": "claude-sonnet-4-5-20250929",  # metadata sibling
                "n_total": 3,
                "routing": {"route": "batch"},  # metadata sibling
                "all_scores": {
                    "cell__00000__00": {"behavior": 1},
                    "cell__00001__00": {"behavior": 0},
                    "cell__00002__00": {"behavior": 1},
                },
            }
        )
    )
    all_scores = E._scores_from_save_raw(save_raw)
    # ONLY the per-custom_id scores, NOT the metadata siblings.
    assert set(all_scores) == {"cell__00000__00", "cell__00001__00", "cell__00002__00"}


def test_b1_rate_nonzero_with_real_scores() -> None:
    rows = [{"question": f"q{i}", "completions": ["x"]} for i in range(3)]
    all_scores = {
        "cell__00000__00": {"behavior": 1},
        "cell__00001__00": {"behavior": 0},
        "cell__00002__00": {"behavior": 1},
    }
    agg = E._rate_from_raw_scores("sycophancy", rows, all_scores)
    assert agg["n_judged"] == 3
    assert agg["rate"] == pytest.approx(2 / 3)


def test_b1_buggy_path_would_null_the_rate() -> None:
    """The pre-fix bug: reading every top-level key as a custom_id score yields
    rate=None / n_judged=0. This pins that the metadata-sibling shape (the real
    save_raw envelope) does NOT match the cell__ custom_id contract."""
    rows = [{"question": f"q{i}", "completions": ["x"]} for i in range(3)]
    buggy = {  # what the old `{k:v for k,v in raw.items() if k!='routing'}` produced
        "per_persona": {"cell": {}},
        "cache_stats": {},
        "judge_model": "claude-sonnet-4-5-20250929",
        "all_scores": {"cell__00000__00": {"behavior": 1}},  # nested, not flattened
    }
    agg = E._rate_from_raw_scores("sycophancy", rows, buggy)
    assert agg["rate"] is None and agg["n_judged"] == 0


# ── B2: seed-qualified unique eval artifact keys ──────────────────────────────
def test_b2_every_cell_has_unique_eval_key() -> None:
    grid = C.realized_grid()
    keys = [c.eval_key for c in grid]
    assert len(keys) == len(set(keys)), "eval_key collision across the realized grid"


def test_b2_seed1042_cells_distinct_from_seed42_twins() -> None:
    grid = C.realized_grid()
    keys = {c.eval_key for c in grid}
    s1042 = [c for c in grid if c.seed == C.MARKER_REPLICATION_SEED]
    assert s1042, "no seed-1042 replication cells in the grid"
    for c in s1042:
        twin = C.Cell(c.behavior, c.source, c.arm, c.dose, C.DEFAULT_SEED)
        assert twin.eval_key in keys
        assert twin.eval_key != c.eval_key  # distinct keys -> no overwrite
        assert c.slug == twin.slug  # the BARE slug collides (the bug class)
        assert c.eval_key.endswith(f"_seed{C.MARKER_REPLICATION_SEED}")


# ── B4: full applicable registry surface on the primary context ───────────────
def test_b4_full_applicable_registry_columns_per_behavior() -> None:
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        COLUMNS,
        columns_for_row,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import ROWS

    for behavior, row_id in C.BEHAVIOR_545_ROW.items():
        cols = C.registry_columns_for_behavior(behavior)
        # every column resolves in the 19-col registry
        for c in cols:
            assert c in COLUMNS, f"{behavior}: {c} not in registry"
        # the set matches columns_for_row(row) minus sensitivity_only + capability
        row = ROWS[row_id]
        expected = {
            col.column_id
            for col in columns_for_row(row)
            if not col.sensitivity_only and col.dv != "logprob_accuracy"
        }
        assert set(cols) == expected, f"{behavior}: {set(cols)} != {expected}"
        # NOT the old hand-picked subset
        assert (
            set(cols) != {C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[behavior], "broad_em"}
            or len(expected) <= 2
        )
        # capability guard never enters the leakage surface
        assert "capability" not in cols
        # the behavior's primary column IS in the applicable set
        assert C.BEHAVIOR_REGISTRY_PRIMARY_COLUMN[behavior] in cols


def test_b4_bad_medical_gets_family_expression_columns() -> None:
    cols = set(C.registry_columns_for_behavior("bad_medical"))
    # B1 family-expression columns are applicable for the bad_medical row
    assert "fam_expr_bad_medical" in cols
    # and the broad cross-behavior columns are always on
    assert {"broad_em", "sycophancy", "refusal"} <= cols


# ── M1: marker training pool target is 300 ────────────────────────────────────
def test_m1_marker_row_target_is_300() -> None:
    import issue664_build_training_data as B

    assert B.ROW_TARGETS["marker"] == 300


def test_m1_marker_pool_helper_targets_300(monkeypatch) -> None:
    import issue664_dispatch as D

    captured: dict[str, int] = {}

    def _fake_fetch(n: int, *, exclude: set[str]) -> list[str]:
        captured["n"] = n
        captured["n_exclude"] = len(exclude)
        return [f"q{i}" for i in range(n)]

    monkeypatch.setattr(D, "_fetch_ultrachat_questions", _fake_fetch)
    monkeypatch.setattr(D, "_marker_eval_probes", lambda: ["eval-probe-0", "eval-probe-1"])
    pool = D._marker_question_pool(smoke=False)
    assert captured["n"] == 300  # ROW_TARGETS["marker"]
    assert captured["n_exclude"] == 2  # eval probes excluded (train/eval disjoint)
    assert len(pool) == 300


# ── B3: production readability assert path + raise ────────────────────────────
def _write_marker_slot(tmp_eval_root: Path, eval_key: str, *, smoke: bool, emit: bool) -> None:
    suffix = "_smoke" if smoke else ""
    d = tmp_eval_root / "marker_slot" / (eval_key + suffix)
    d.mkdir(parents=True, exist_ok=True)
    # one slot; argmax==MARKER_ID (emit) or not; margin >=0 (emit) or <0.
    trained = {
        "logp": -1.0,
        "z_marker": (5.0 if emit else -5.0),
        "z_eos": 0.0,
        "logZ": 10.0,
        "argmax_id": (C.MARKER_ID if emit else 999),
    }
    (d / "marker_slot_stats.json").write_text(json.dumps({"slots": {"ctx0": {"trained": trained}}}))


def test_b3_production_assert_reads_production_path_and_raises_on_emit(
    tmp_path: Path, monkeypatch
) -> None:
    import issue664_dispatch as D

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    cell = C.Cell("marker", "default", "contra", "d1")
    # production slot stats with emission (FAILS readability)
    _write_marker_slot(tmp_path, cell.eval_key, smoke=False, emit=True)
    with pytest.raises(RuntimeError, match="readability HALT"):
        D._marker_readability_assert([cell], smoke=False)


def test_b3_production_assert_passes_on_clean_read(tmp_path: Path, monkeypatch) -> None:
    import issue664_dispatch as D

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    cell = C.Cell("marker", "default", "contra", "d1")
    _write_marker_slot(tmp_path, cell.eval_key, smoke=False, emit=False)
    # no exception when emission<1% and margins<0
    D._marker_readability_assert([cell], smoke=False)


def test_b3_smoke_reads_smoke_path_and_does_not_raise_on_emit(tmp_path: Path, monkeypatch) -> None:
    import issue664_dispatch as D

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    cell = C.Cell("marker", "default", "contra", "d1")
    # smoke path with emission: smoke is STRUCTURAL only, must NOT raise the verdict
    _write_marker_slot(tmp_path, cell.eval_key, smoke=True, emit=True)
    D._marker_readability_assert([cell], smoke=True)  # no raise


def test_b3_assert_raises_when_zero_marker_cells_have_stats(tmp_path: Path, monkeypatch) -> None:
    import issue664_dispatch as D

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)
    cell = C.Cell("marker", "default", "contra", "d1")
    # no marker_slot_stats.json written at all -> checked==0 -> raise
    with pytest.raises(RuntimeError, match="ran on 0 marker cells"):
        D._marker_readability_assert([cell], smoke=False)


# ── M5: fail-loud on missing trained-store tensors ────────────────────────────
def test_m5_missing_tensors_raises(tmp_path: Path, monkeypatch) -> None:
    import issue664_dispatch as D

    monkeypatch.setattr(C, "STORE_ROOT", tmp_path)
    cell = C.Cell("marker", "default", "contra", "d1")
    # cell dir exists but tensors.pt is MISSING -> must RAISE (not warn+continue)
    (tmp_path / cell.eval_key).mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError, match="tensors MISSING"):
        D._upload_store_tensors([cell])


# ── C4: default source-anchor scoping ─────────────────────────────────────────
def test_c4_default_source_anchor_is_instance_level() -> None:
    insts = C.load_contexts()
    default_insts = {i["id"]: i for i in insts if i["family"] == "default"}
    assert "f6_helpful_asst" in default_insts and "f6_default_template" in default_insts
    # the realized default SOURCE instance is the only source-anchor
    assert C.target_context_role("default", default_insts["f6_helpful_asst"]) == "source-anchor"
    # the SIBLING default-family context is a bystander (kept in the leakage read)
    assert C.target_context_role("default", default_insts["f6_default_template"]) == "bystander"
