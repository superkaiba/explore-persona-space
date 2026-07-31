"""Pins for the wcrung scoring leg (#1739 wildchat rung, path C).

Covers the DV reduction (drop-never-coerce), the spread stats, and the
generate-once/judge-3x contract: the SAME rollout pool must produce one DV
dataset per behavior, each stamped with its own behavior but the shared rung.

No real corpus text — synthetic placeholder strings only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue1739_wcrung_score as sc  # noqa: E402


def _rollout(ctx: int, k: int) -> dict:
    return {
        "context_id": f"wcrung-{ctx:04d}",
        "rollout_k": k,
        "query": f"placeholder query {ctx}",
        "completion": f"placeholder completion {ctx}-{k}",
        "group_key": f"wcrung-{ctx:04d}",
        "behavior": sc.GEN_BEHAVIOR,
        "rung": sc.RUNG,
        "split": sc.SPLIT,
    }


def _pool(n_ctx: int, k: int) -> list[dict]:
    return [_rollout(c, kk) for c in range(n_ctx) for kk in range(k)]


# --- spread stats ----------------------------------------------------------


def test_spread_stats_basic():
    s = sc._spread_stats([0.0, 50.0, 100.0])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(50.0)
    assert s["sd"] == pytest.approx(50.0)
    assert s["min"] == 0.0 and s["max"] == 100.0
    assert sum(s["histogram"].values()) == 3
    # 100 lands in the top decile (the hi==100 inclusive edge), not a 101th bin.
    assert s["histogram"]["90-100"] == 1


def test_spread_stats_ignores_none_and_nan():
    s = sc._spread_stats([10.0, None, float("nan"), 30.0])
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(20.0)


def test_spread_stats_empty_is_none_not_zero():
    s = sc._spread_stats([None, None])
    assert s["n"] == 0
    assert s["mean"] is None and s["sd"] is None
    assert s["histogram"] == {}


def test_spread_stats_single_value_sd_zero():
    assert sc._spread_stats([42.0])["sd"] == 0.0


# --- DV rows ---------------------------------------------------------------


def _scores(pool: list[dict], value_for) -> dict:
    from explore_persona_space.experiments.issue_1739 import judging

    return {
        judging.rollout_item_id(r["context_id"], int(r["rollout_k"])): value_for(r) for r in pool
    }


def test_build_dv_rows_means_over_kept_rollouts():
    pool = _pool(2, 2)
    scores = _scores(pool, lambda r: 10.0 * (int(r["rollout_k"]) + 1))
    rows, digest = sc.build_dv_rows("evil", pool, scores)
    assert digest == {
        "n_contexts": 2,
        "n_contexts_with_dv": 2,
        "n_contexts_dropped_no_dv": 0,
        "n_groups": 2,
    }
    assert all(r["dv"] == pytest.approx(15.0) for r in rows)
    assert all(r["behavior"] == "evil" for r in rows)
    assert {r["rung"] for r in rows} == {"wildchat_rung"}
    assert {r["split"] for r in rows} == {"eval"}


def test_build_dv_rows_partial_drop_means_over_kept_only():
    pool = _pool(1, 3)
    scores = _scores(pool, lambda r: None if int(r["rollout_k"]) == 0 else 60.0)
    rows, digest = sc.build_dv_rows("sycophancy", pool, scores)
    assert digest["n_contexts_with_dv"] == 1
    assert rows[0]["n_rollouts"] == 3
    assert rows[0]["n_rollouts_kept"] == 2
    assert rows[0]["dv"] == pytest.approx(60.0), "dropped rollout must not pull the mean down"


def test_build_dv_rows_all_dropped_is_none_never_zero():
    pool = _pool(1, 2)
    rows, digest = sc.build_dv_rows("hallucination", pool, _scores(pool, lambda r: None))
    assert rows[0]["dv"] is None, "drop-never-coerce: a fully-dropped context is None, not 0"
    assert rows[0]["n_rollouts_kept"] == 0
    assert digest["n_contexts_dropped_no_dv"] == 1
    assert digest["n_contexts_with_dv"] == 0


def test_one_pool_three_behaviors_share_contexts_differ_in_behavior():
    """The generate-once/judge-3x contract, at the DV layer."""
    pool = _pool(3, 2)
    scores = _scores(pool, lambda r: 50.0)
    per_behavior = {b: sc.build_dv_rows(b, pool, scores)[0] for b in sc.BEHAVIORS}
    ids = {b: sorted(r["context_id"] for r in rows) for b, rows in per_behavior.items()}
    assert len({tuple(v) for v in ids.values()}) == 1, "all three DVs cover the same contexts"
    for behavior, rows in per_behavior.items():
        assert {r["behavior"] for r in rows} == {behavior}
        assert {r["rung"] for r in rows} == {"wildchat_rung"}


# --- rollout loading -------------------------------------------------------


def _write_pool(root: Path, pool: list[dict]) -> Path:
    d = root / "labeling" / sc.GEN_BEHAVIOR
    d.mkdir(parents=True, exist_ok=True)
    for r in pool:
        (d / f"{r['context_id']}_seed{r['rollout_k']}.json").write_text(json.dumps(r))
    (d / "_manifest.json").write_text(json.dumps({"n": len(pool)}))
    return d


def test_load_rollouts_skips_underscore_manifest(tmp_path):
    pool = _pool(2, 2)
    d = _write_pool(tmp_path, pool)
    loaded = sc.load_rollouts(d)
    assert len(loaded) == 4, "the _manifest.json must not be read as a rollout"


def test_load_rollouts_max_items_caps(tmp_path):
    d = _write_pool(tmp_path, _pool(4, 2))
    assert len(sc.load_rollouts(d, max_items=3)) == 3


def test_load_rollouts_fails_loud_on_missing_field(tmp_path):
    d = _write_pool(tmp_path, _pool(1, 1))
    bad = next(d.glob("wcrung-*.json"))
    payload = json.loads(bad.read_text())
    del payload["completion"]
    bad.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="missing 'completion'"):
        sc.load_rollouts(d)


def test_load_rollouts_fails_loud_on_empty_dir(tmp_path):
    empty = tmp_path / "labeling" / sc.GEN_BEHAVIOR
    empty.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no rollout files"):
        sc.load_rollouts(empty)


def test_local_rollout_root_takes_the_shared_pool_path(tmp_path):
    args = sc._parse_args(["--local-rollout-root", str(tmp_path)])
    assert sc.rollout_dir(args) == tmp_path / "labeling" / sc.GEN_BEHAVIOR


# --- run metadata ---------------------------------------------------------


def test_run_meta_pins_the_judge_instrument():
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MAX_TOKENS,
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
        N_JUDGE_DRAWS,
    )

    meta = sc._run_meta(sc._parse_args([]))
    assert meta["judge_model"] == JUDGE_MODEL
    assert meta["n_judge_draws"] == N_JUDGE_DRAWS
    assert meta["judge_temperature"] == JUDGE_TEMPERATURE
    assert meta["judge_max_tokens"] == JUDGE_MAX_TOKENS
    assert meta["rejudge_max_tokens"] == 800, "truncation-recovery budget (llm-judging rule 23)"
    assert meta["gen_behavior"] == sc.GEN_BEHAVIOR
