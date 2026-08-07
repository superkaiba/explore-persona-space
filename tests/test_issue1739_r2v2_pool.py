"""Pins for the #1739 r2v2 P-B readout-pool assembly + OOD DV gate + lambda sink.

The LODO parameterization is a stated forward-compatibility requirement
(Result 5): ``assemble_readout_pool`` must expose which datasets enter as a
parameter, with GROUP-level splits whose sides are independent of the holdout
choice and of the roster, so a LODO sweep is a loop — never a rewrite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1739_r2v2_score import (  # noqa: E402
    DatasetSpec,
    _multi_pool_zscored_dv,
    _prepare_ood_dv,
    assemble_readout_pool,
)


def _roster() -> list[DatasetSpec]:
    """Three datasets with GROUPED rows (groups span multiple rows)."""
    rng = np.random.default_rng(0)
    out = []
    at = 0
    for name, n_groups, rows_per in (("train", 40, 5), ("rungA", 12, 4), ("rungB", 9, 3)):
        n = n_groups * rows_per
        rows = np.arange(at, at + n, dtype=np.int64)
        groups = np.repeat([f"{name}-g{i:03d}" for i in range(n_groups)], rows_per)
        # shuffle within the dataset so group rows are not contiguous
        perm = rng.permutation(n)
        out.append(DatasetSpec(name=name, rows=rows[perm], groups=groups[perm]))
        at += n
    return out


def test_holdout_excluded_whole_and_group_level_split():
    datasets = _roster()
    pool = assemble_readout_pool(datasets, holdout="rungA", train_frac=0.8, seed=0)
    assert "rungA" not in pool.train_rows and "rungA" not in pool.heldin_eval_rows
    assert set(pool.train_rows) == {"train", "rungB"}
    for d in datasets:
        if d.name == "rungA":
            continue
        tr = set(pool.train_rows[d.name].tolist())
        he = set(pool.heldin_eval_rows[d.name].tolist())
        assert tr.isdisjoint(he)
        assert tr | he == set(np.asarray(d.rows).tolist())
        # GROUP-level: no group straddles the train/heldin boundary
        g_of = {int(r): g for r, g in zip(np.asarray(d.rows), np.asarray(d.groups), strict=True)}
        tr_groups = {g_of[r] for r in tr}
        he_groups = {g_of[r] for r in he}
        assert tr_groups.isdisjoint(he_groups), f"{d.name}: group straddles the 80/20 split"


def test_split_sides_independent_of_holdout_and_roster():
    """The Result-5 seam: identical within-dataset splits across LODO folds."""
    datasets = _roster()
    p1 = assemble_readout_pool(datasets, holdout="rungA", seed=3)
    p2 = assemble_readout_pool(datasets, holdout="rungB", seed=3)
    assert np.array_equal(np.sort(p1.train_rows["train"]), np.sort(p2.train_rows["train"]))
    # dropping a dataset from the roster leaves the others' splits untouched
    p3 = assemble_readout_pool(datasets, holdout=None, include=["train", "rungB"], seed=3)
    assert np.array_equal(np.sort(p3.train_rows["rungB"]), np.sort(p1.train_rows["rungB"]))
    assert "rungA" not in p3.train_rows


def test_seed_changes_split_and_determinism():
    datasets = _roster()
    a1 = assemble_readout_pool(datasets, holdout=None, seed=0)
    a2 = assemble_readout_pool(datasets, holdout=None, seed=0)
    b = assemble_readout_pool(datasets, holdout=None, seed=1)
    for name in a1.train_rows:
        assert np.array_equal(a1.train_rows[name], a2.train_rows[name])
    assert any(not np.array_equal(a1.train_rows[n], b.train_rows[n]) for n in a1.train_rows), (
        "seed change must move at least one dataset's group split"
    )


def test_bad_inputs_fail_loud():
    datasets = _roster()
    with pytest.raises(ValueError, match="holdout"):
        assemble_readout_pool(datasets, holdout="nope")
    with pytest.raises(ValueError, match="include"):
        assemble_readout_pool(datasets, holdout=None, include=["train", "ghost"])
    with pytest.raises(ValueError, match="train_frac"):
        assemble_readout_pool(datasets, holdout=None, train_frac=1.0)
    dup = [*datasets, datasets[0]]
    with pytest.raises(ValueError, match="duplicate"):
        assemble_readout_pool(dup, holdout=None)


def test_multi_pool_zscore_per_pool_stats():
    dv = np.array([0.0, 10.0, 20.0, 100.0, 200.0, 7.0], dtype=np.float64)
    out = _multi_pool_zscored_dv(dv, [np.array([0, 1, 2]), np.array([3, 4])])
    assert np.allclose(out[:3].mean(), 0.0) and np.allclose(out[:3].std(), 1.0)
    assert np.allclose(out[3:5].mean(), 0.0) and np.allclose(out[3:5].std(), 1.0)
    assert out[5] == 7.0  # untouched: outside every pool


def _dv_payload(n: int, *, split: str, n_null: int = 0) -> dict:
    rows = []
    for i in range(n):
        rows.append(
            {
                "context_id": f"c{i:04d}",
                "dv": None if i < n_null else float(i),
                "split": split,
                "rung": "rungX",
                "group_key": f"g{i // 3:03d}",
            }
        )
    return {"rows": rows}


def test_ood_dv_gate_refuses_unresolved(tmp_path):
    src = tmp_path / "labeling.json"
    src.write_text(json.dumps(_dv_payload(100, split="full", n_null=20)))
    with pytest.raises(RuntimeError, match="UNRESOLVED"):
        _prepare_ood_dv(src, tmp_path / "work", "evil", max_null_frac=0.05)


def test_ood_dv_full_to_eval_rewrite(tmp_path):
    src = tmp_path / "labeling.json"
    src.write_text(json.dumps(_dv_payload(50, split="full")))
    path, note = _prepare_ood_dv(src, tmp_path / "work", "evil", max_null_frac=0.05)
    assert path != src and "full->eval" in note["split_rewrite"]
    rewritten = json.loads(path.read_text())
    assert {r["split"] for r in rewritten["rows"]} == {"eval"}


def test_ood_dv_eval_passthrough(tmp_path):
    src = tmp_path / "labeling.json"
    src.write_text(json.dumps(_dv_payload(50, split="eval")))
    path, note = _prepare_ood_dv(src, tmp_path / "work", "sycophancy", max_null_frac=0.05)
    assert path == src and note["split_rewrite"].startswith("none")


def test_selected_lambda_sink_records_ridge_diagnostics():
    """The #1887 duty: selector + selected lambdas surface per ridge fit."""
    from explore_persona_space.experiments.issue_1739 import fits

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 40, 6))
    w = rng.normal(size=(2, 6, 3))
    y = np.einsum("snd,sdt->snt", x, w) + 0.1 * rng.normal(size=(2, 40, 3))
    x_ev = rng.normal(size=(2, 8, 6))
    sink: list[dict] = []
    with fits.capture_selected_lambdas(sink):
        preds = fits.ridge_gcv_predict_per_target(x, y, [x_ev], device="cpu")
    assert preds[0].shape == (2, 8, 3)
    assert sink, "sink recorded nothing"
    rec = sink[0]
    assert rec["n_train"] == 40 and rec["d"] == 6 and rec["gram_space"] == "primal"
    assert rec["selector"].startswith("per-target GCV")
    assert sum(rec["selected_lambda_counts"].values()) == rec["n_slices"] * rec["n_targets"]
    # disarmed outside the context manager
    sink2: list[dict] = []
    fits.ridge_gcv_predict_per_target(x, y, [x_ev], device="cpu")
    assert not sink2


def test_pc_protocol_and_driver_leg_wiring():
    """P-C wiring pins: scorer accepts --protocols C; the driver's pc leg
    composes the scorer argv against the P-C out root with no env overlay,
    and the composed argv parses under the scorer's own parse_args."""
    from scripts.issue1739_r2v2_run import PC_OUT_ROOT, leg_cmd_env
    from scripts.issue1739_r2v2_run import parse_args as run_parse_args
    from scripts.issue1739_r2v2_score import parse_args as score_parse_args

    s_args = score_parse_args(["--protocols", "C", "--pb-holdouts", "nqopen"])
    assert s_args.protocols == "C" and s_args.pb_holdouts == ["nqopen"]

    d_args = run_parse_args(["--behaviors", "hallucination", "--legs", "pc", "--protocols", "C"])
    cmd, env = leg_cmd_env(d_args, "hallucination", "pc")
    assert env == {}
    parsed = score_parse_args(cmd[2:])
    assert parsed.protocols == "C"
    assert str(parsed.out_root) == str(PC_OUT_ROOT)
