"""Focused no-network tests for the covariance control and selective loader.

The parity test uses tiny feature width with the real whitening, mapping,
budget selection, ridge and legacy transfer routines. No model calls occur.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import arms, fits
from scripts import issue1739_covariance_ablation as ablation
from scripts.issue1739_fits import _load_labeled
from scripts.issue1739_jobd_r2aug import _pool_zscored_dv
from scripts.issue1739_result2fair_score import _wc_eval_mask, fit_add_maps


def _loader_fixture(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    # Deliberately unsorted contexts and two rollouts: storage order, not label
    # order, defines joined arrays; the answer is averaged within context.
    ids = ["b", "a", "b", "a", "eval", "unjudged"]
    (store / "row_index.jsonl").write_text(
        "".join(
            json.dumps({"context_id": c, "rollout_k": i // 2}) + "\n" for i, c in enumerate(ids)
        )
    )
    x = np.arange(24, dtype=np.float16).reshape(6, 4)
    for kind, offset in (("context_end", 0), ("prefix_end", 100), ("t1", 200)):
        np.save(store / f"{kind}_L03.npy", x + offset)
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "context_id": c,
                        "split": "eval" if c == "eval" else "train",
                        "group_key": f"g-{c}",
                        "rung": "r",
                        "dv": value,
                    }
                    for c, value in (("a", 0.25), ("b", 0.75), ("eval", 0.5), ("unjudged", None))
                ]
            }
        )
    )
    return store, labels, x


def test_selective_loader_preserves_legacy_join_order_and_answers(tmp_path):
    store, labels, x = _loader_fixture(tmp_path)
    old = _load_labeled(store, labels, [3], config="config_a", need_rollout_rows=False)
    selected = ablation.load_table(store, labels, 3, "config_a", answers=True)
    assert selected.ctx_order == old.ctx_order == ["b", "a"]
    assert set(selected.z_by_variant) == {"context_end"}
    np.testing.assert_array_equal(
        selected.z_by_variant["context_end"], old.z_by_variant["context_end"]
    )
    np.testing.assert_array_equal(selected.z_ans, old.z_ans)
    np.testing.assert_array_equal(
        selected.z_ans[0],
        np.stack([(x[[0, 2]] + 200).mean(axis=0), (x[[1, 3]] + 200).mean(axis=0)]),
    )
    assert selected.groups == old.groups
    np.testing.assert_array_equal(selected.dv, old.dv)


def test_context_only_loader_does_not_require_prefix_or_answer_shards(tmp_path):
    store, labels, x = _loader_fixture(tmp_path)
    (store / "prefix_end_L03.npy").unlink()
    (store / "t1_L03.npy").unlink()
    selected = ablation.load_table(store, labels, 3, "config_b")
    assert selected.ctx_order == ["eval"]
    assert selected.z_ans is None
    np.testing.assert_array_equal(selected.z_by_variant["context_end"][0], x[[4]])
    with pytest.raises(ValueError, match="rollout rows require"):
        _load_labeled(
            store,
            labels,
            [3],
            config="config_a",
            need_rollout_rows=True,
            context_variants=("context_end",),
            include_answers=False,
        )


@pytest.mark.parametrize("variants", [(), ("nonexistent",)])
def test_selective_loader_rejects_invalid_variants(tmp_path, variants):
    with pytest.raises(ValueError, match="invalid context variants"):
        _load_labeled(
            tmp_path,
            tmp_path / "unused.json",
            [3],
            config="config_a",
            need_rollout_rows=False,
            context_variants=variants,
        )


def _table(rng, prefix, n, d, *, groups_repeated=False):
    x = rng.normal(size=(1, n, d)).astype(np.float16)
    # Correlated anisotropic features make covariance/map transforms nontrivial.
    x[..., 1] = 3 * x[..., 0] + 0.3 * x[..., 1]
    y = (0.6 * x + 0.2 * rng.normal(size=x.shape)).astype(np.float16)
    dv = x[0, :, 0].astype(float) + 0.7 * x[0, :, 2] + rng.normal(size=n)
    return SimpleNamespace(
        z_by_variant={"context_end": x},
        z_ans=y,
        dv=dv,
        ctx_order=[f"{prefix}-{i}" for i in range(n)],
        groups=[f"{prefix}-g{i // 2 if groups_repeated else i}" for i in range(n)],
        row_rungs=["rung-a" if i < n // 2 else "rung-b" for i in range(n)],
    )


def test_four_arm_runner_reproduces_legacy_full_transfer(tmp_path, monkeypatch):
    rng = np.random.default_rng(281)
    d, layer, behavior = 4, 20, "evil"
    tr = _table(rng, "train", 36, d, groups_repeated=True)
    ev = _table(rng, "eval", 24, d)
    wc = _table(rng, "wc", 60, d)
    xg = rng.normal(size=(18793, d)).astype(np.float16)
    yg = (0.4 * xg + rng.normal(size=xg.shape)).astype(np.float16)
    u_arrays = {("context_end", layer): xg, ("t1", layer): yg}
    umeta = [{"context_id": f"generic-{i}"} for i in range(len(xg))]
    monkeypatch.setitem(ablation.LMAX, behavior, 18)
    wc_eval = _wc_eval_mask(wc.ctx_order)
    assert wc_eval.sum() > 5
    budget = fits.realize_budget_cell(tr.groups, budget_l=18, draw=0, seed=0)
    rows = np.concatenate([budget.row_idx, len(tr.dv) + np.flatnonzero(~wc_eval)])
    dv_z = _pool_zscored_dv(
        np.concatenate([tr.dv, wc.dv]), budget.row_idx, len(tr.dv) + np.flatnonzero(~wc_eval)
    )
    loaded = SimpleNamespace(
        behavior=behavior, tbl=tr, u_arrays=u_arrays, u_fit_rows=np.arange(len(xg))
    )
    old_args = SimpleNamespace(device="cpu", seed=0, map_kinds=["linear"], regime="e1", draw=0)
    wh, maps, *_ = fit_add_maps(old_args, loaded, "context_end", [layer])
    data = arms.CellData(
        z_ctx=fits.apply_whitening(
            np.concatenate(
                [tr.z_by_variant["context_end"], wc.z_by_variant["context_end"]], axis=1
            ),
            wh,
        ),
        dv=dv_z,
        rb=np.ones((1, d)),
        mapfit=maps["linear"],
        layers=(layer,),
    )
    cell = fits.BudgetCell(
        rows, np.zeros(len(rows), dtype=np.int64), 1, 18, 0, 0, "fair-union-full"
    )
    expected = {}
    parity_rows = []
    for rung, acts, dv in (
        ("ood", ev.z_by_variant["context_end"], ev.dv),
        ("wildchat_rung", wc.z_by_variant["context_end"][:, wc_eval], wc.dv[wc_eval]),
    ):
        scores, skipped = arms.run_transfer_cell(
            data,
            cell,
            fits.apply_whitening(acts, wh),
            dv,
            arms=["arm4_ridge_ctx", "arm7_map_ridge_pred"],
            ridge_folds=(0,),
        )
        assert not skipped
        expected[rung] = scores
        parity_rows.extend(
            {
                "arm": arm,
                "map_kind": "linear",
                "variant": "context_end",
                "eval_rung": rung,
                "layers": [layer],
                "rho_per_layer": arms.spearman_rows(pred, dv).tolist(),
            }
            for arm, pred in scores.items()
        )
    repo = tmp_path / "repo"
    label_paths = [
        repo / "eval_results/issue_1739/dv_dataset/evil/labeling.json",
        repo / "eval_results/issue_1739/wildchat_rung/dv_dataset/evil/labeling.json",
    ]
    for i, p in enumerate(label_paths):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"fixture": i}))
    history_path = repo / "eval_results/issue_1739/result2_fair/evil/all_arms_spearman.json"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "meta": {"input_sha256": {str(p): ablation.sha256(p) for p in label_paths}},
                "per_layer_rows": parity_rows,
            }
        )
    )
    stores = tmp_path / "inputs"
    stores.mkdir()
    (stores / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        ablation,
        "load_table",
        lambda store, labels, layer, split, **kw: (
            wc if store.name == "wildchat" else tr if split == "config_a" else ev
        ),
    )
    monkeypatch.setattr(ablation.store_io, "load_summaries", lambda *a, **kw: (u_arrays, umeta))
    args = SimpleNamespace(
        out=tmp_path / "out", repo=repo, store_root=stores, source_sha="a" * 40, n_boot=100
    )
    ablation.run_cell(args, behavior, layer)
    out = args.out / "evil_L20"
    pred = np.load(out / "predictions_historical_grid.npz")
    for arm, new_idx in (("arm4_ridge_ctx", 2), ("arm7_map_ridge_pred", 3)):
        np.testing.assert_allclose(
            pred["predictions"][new_idx],
            np.concatenate([expected["ood"][arm][0], expected["wildchat_rung"][arm][0]]),
            rtol=1e-9,
            atol=1e-9,
        )
    result = json.loads((out / "results.json").read_text())
    assert max(abs(p["rho_difference"]) for p in result["parity"]) < 1e-10
    assert result["n_readout"] == 18 + int((~wc_eval).sum())
    assert set(result["summaries"]) == {"historical_grid", "wide_grid"}
    for grid in result["summaries"].values():
        assert set(grid["lambda_diagnostics"]) == set(ablation.ARM_NAMES)
    assert (out / "complete.json").is_file()


def test_summary_differences_use_paired_draws_and_report_constant_rung():
    dv = np.arange(24, dtype=float)
    scores = np.stack([dv, dv, -dv, -dv])
    out = ablation.summarize(
        scores, dv, [str(i) for i in range(24)], [str(i) for i in range(24)], ["r"] * 24, n_boot=100
    )[0]
    same = out["differences"]["generic_covariance_minus_unwhitened"]
    assert same == {"delta": 0.0, "ci95": [0.0, 0.0]}
    skipped = ablation.summarize(
        scores, np.ones(24), [], [str(i) for i in range(24)], ["r"] * 24, n_boot=100
    )[0]
    assert "skipped" in skipped and "arms" not in skipped
