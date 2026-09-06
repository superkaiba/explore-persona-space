"""Natural-pool scaling: input integrity, true nested budgets, and reduced-layer P-B reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from scripts import issue1739_jobd_r2aug as jobd
from scripts import issue1739_natural_score as natural
from scripts import issue1739_r2v2_score as score


def _store(path, n=30, dim=4, layers=(17, 20)):
    """Build the exact on-disk consumer layout, with independently identifiable rows."""
    path.mkdir()
    rows = [
        {
            "context_id": f"natural-{i}",
            "source_dataset": "lmsys",
            "source_id": str(i),
            "no_recombination": True,
            "prompt_sha256": hashlib.sha256(f"p{i}".encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(f"a{i}".encode()).hexdigest(),
        }
        for i in range(n)
    ]
    index = path / "row_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = {
        "schema_version": 1,
        "status": "complete",
        "pool_kind": "natural_context_answer",
        "model": natural.MODEL,
        "model_revision": natural.MODEL_REVISION,
        "n_rows": n,
        "no_recombination": True,
        "layers": list(layers),
        "hidden_dim": dim,
        "dtype": "float16",
        "row_index_sha256": natural._sha(index),
        "source": {"dataset": "lmsys", "revision": "7a47ff5ce42f16308bebaba29c1286a4e9bc8008"},
        "generation": {"temperature": 0, "max_new_tokens": 1024, "cap_hit_fraction": 0},
        "capture": {"context_kind": "context_end", "answer_kind": "t1"},
    }
    rng = np.random.default_rng(44)
    matrices = {}
    for kind in ("context_end", "t1"):
        for layer in layers:
            name = f"{kind}_L{layer:02d}.npy"
            np.save(path / name, rng.normal(size=(n, dim)).astype(np.float16))
            matrices[name] = natural._sha(path / name)
    meta["matrices_sha256"] = matrices
    (path / "manifest.json").write_text(json.dumps(meta))
    return meta


def _args(path, u=12, seed=0):
    return SimpleNamespace(
        natural_u_store=path,
        generic_u=u,
        seed=seed,
        draw=0,
        regime="e1",
        device="cpu",
        map_kind="linear",
        map_kinds=["linear"],
    )


def _loaded(path, u=12, seed=0):
    arrays, rows, meta = natural.load_natural_pool(path, [17, 20], u=u, seed=seed, hidden_dim=4)
    rng = np.random.default_rng(12)
    tbl = SimpleNamespace(
        ctx_order=[f"trait-{i}" for i in range(12)],
        z_by_variant={"context_end": rng.normal(size=(2, 12, 4)).astype(np.float16)},
        z_ans=rng.normal(size=(2, 12, 4)).astype(np.float16),
    )
    return SimpleNamespace(
        behavior="evil", tbl=tbl, u_arrays=arrays, u_fit_rows=rows, natural_meta=meta
    )


def test_nested_generic_samples_include_full_fixed_traits_at_every_rung(tmp_path):
    path = tmp_path / "store"
    _store(path)
    picked = []
    for u in (5, 12, 30):
        loaded = _loaded(path, u=u)
        x, y, label, n, meta = jobd.build_pool(
            _args(path, u=u), loaded, "context_end", [17, 20], "add"
        )
        assert n == u + 12 and meta["add_n_generic"] == u and meta["add_n_eliciting"] == 12
        np.testing.assert_array_equal(x[:, u:], loaded.tbl.z_by_variant["context_end"])
        np.testing.assert_array_equal(y[:, u:], loaded.tbl.z_ans)
        assert label == f"add{u + 12}_gen{u}_elic12"
        assert isinstance(loaded.u_arrays[("context_end", 17)], np.memmap)
        picked.append(set(loaded.u_fit_rows))
    assert picked[0] < picked[1] < picked[2]
    np.testing.assert_array_equal(
        natural.nested_generic_rows(30, 12, 0), natural.nested_generic_rows(30, 12, 0)
    )
    assert not np.array_equal(
        natural.nested_generic_rows(30, 12, 0), natural.nested_generic_rows(30, 12, 1)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("no_recombination", False),
        ("no_recombination", 1),
        ("pool_kind", "crossed_context_answer"),
        ("model_revision", "main"),
        ("n_rows", 100000),
        ("layers", [17, 17]),
        ("row_index_sha256", "wrong"),
        ("matrices_sha256", {}),
        ("generation", {"temperature": 1, "max_new_tokens": 1024, "cap_hit_fraction": 0}),
    ],
)
def test_manifest_rejects_wrong_recipe_or_partial_pool(tmp_path, field, value):
    path = tmp_path / "store"
    meta = _store(path)
    meta[field] = value
    (path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        natural.load_natural_pool(path, [17, 20], u=12, seed=0, hidden_dim=4)


def test_row_provenance_duplicates_and_array_corruption_fail(tmp_path):
    path = tmp_path / "store"
    meta = _store(path)
    index = path / "row_index.jsonl"
    lines = index.read_text().splitlines()
    lines[1] = lines[0]
    index.write_text("\n".join(lines) + "\n")
    meta["row_index_sha256"] = natural._sha(index)
    (path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="duplicate"):
        natural.read_natural_manifest(path, hidden_dim=4)
    path2 = tmp_path / "store2"
    meta2 = _store(path2)
    np.save(path2 / "t1_L17.npy", np.zeros((30, 4), dtype=np.float32))
    meta2["matrices_sha256"]["t1_L17.npy"] = natural._sha(path2 / "t1_L17.npy")
    (path2 / "manifest.json").write_text(json.dumps(meta2))
    with pytest.raises(ValueError, match="expected fp16"):
        natural.load_natural_pool(path2, [17, 20], u=30, seed=0, hidden_dim=4)
    np.save(path2 / "t1_L17.npy", np.full((30, 4), np.nan, dtype=np.float16))
    meta2["matrices_sha256"]["t1_L17.npy"] = natural._sha(path2 / "t1_L17.npy")
    (path2 / "manifest.json").write_text(json.dumps(meta2))
    with pytest.raises(ValueError, match="non-finite"):
        natural.load_natural_pool(path2, [17, 20], u=30, seed=0, hidden_dim=4)


def test_matrix_receipt_reuses_only_unchanged_files_and_rejects_finite_overwrite(
    tmp_path, monkeypatch
):
    """Real validation runs both before fitting and before result-resume eligibility."""
    import os

    path = tmp_path / "store"
    _store(path, n=3, dim=3584)
    args = _args(path, u=2)
    digest = create_autospec(natural._sha, side_effect=natural._sha)
    monkeypatch.setattr(natural, "_sha", digest)
    natural.natural_regime_key(args, "evil", [17, 20], natural.NATURAL_ROSTER)
    first_matrix_calls = [c for c in digest.call_args_list if c.args[0].suffix == ".npy"]
    assert len(first_matrix_calls) == 4
    digest.reset_mock()
    natural.natural_regime_key(args, "evil", [17, 20], natural.NATURAL_ROSTER)
    assert not [c for c in digest.call_args_list if c.args[0].suffix == ".npy"]
    assert any(c.args[0].name == "row_index.jsonl" for c in digest.call_args_list)
    target = path / "context_end_L17.npy"
    before = target.stat()
    matrix = np.load(target, mmap_mode="r+")
    matrix[0, 0] += np.float16(1)
    matrix.flush()
    del matrix
    # Restoring mtime must not hide the edit: ctime changes independently.
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    with pytest.raises(ValueError, match="matrix SHA256 mismatch"):
        natural.natural_regime_key(args, "evil", [17, 20], natural.NATURAL_ROSTER)
    with pytest.raises(ValueError, match="matrix SHA256 mismatch"):
        natural.load_natural_pool(path, [17, 20], u=2, seed=0, hidden_dim=3584)


def test_reduced_frozen_layers_resolve_actual_committed_roster():
    root = Path(__file__).resolve().parents[1] / "eval_results/issue_1739"
    expected = {"evil": [17, 18, 20], "sycophancy": [19, 20], "hallucination": [18, 20]}
    for behavior, layers in expected.items():
        frozen = natural.frozen_global_layers(
            root / behavior / "arm_results/all_arms_spearman.json",
            variant="context_end",
            regime="e1",
            roster=natural.NATURAL_ROSTER,
        )
        assert sorted(set(frozen.values())) == layers
        reduced = natural.remap_frozen(frozen, layers)
        assert {arm: layers[i] for arm, i in reduced.items()} == frozen
        with pytest.raises(ValueError, match="not indexable"):
            natural.remap_frozen(frozen, [0, 1])


def test_real_map_path_whitens_exact_budgeted_union(tmp_path, monkeypatch):
    """Execute the reused map fit, not a toy solver, through its actual caller."""
    from explore_persona_space.experiments.issue_1739 import fits

    path = tmp_path / "store"
    _store(path)
    whiten = create_autospec(fits.fit_whitening, side_effect=fits.fit_whitening)
    monkeypatch.setattr(fits, "fit_whitening", whiten)
    for u in (12, 30):
        loaded = _loaded(path, u=u)
        expected_x = jobd.build_pool(_args(path, u=u), loaded, "context_end", [17, 20], "add")[0]
        wh, mapfit, diag, _label, n = score.fit_linear_add_map(
            _args(path, u=u), loaded, "context_end", [17, 20], claim4_sink={"want_shufpair": False}
        )
        np.testing.assert_array_equal(whiten.call_args.args[0], expected_x)
        assert n == u + 12 and mapfit.diagnostics["w_fit_rows"] == n
        assert wh.w.shape == (2, 4, 4) and diag["add_n_generic"] == u
        assert diag["natural_pool"]["generic_u"] == u
        assert not loaded.u_arrays


def test_load_behavior_real_body_natural_never_stages_crossed_pool(tmp_path, monkeypatch):
    """Only external labeled-store IO is stubbed, with a signature-bound fake."""
    from explore_persona_space.experiments.issue_1739 import store_io
    from scripts import issue1739_fits as fit_script
    from scripts import issue1739_wcrung_arms as wc

    path = tmp_path / "natural"
    _store(path)
    for name in ("train_store", "train_dv", "wcrung_store", "wcrung_dv", "e1_store"):
        (tmp_path / name).touch()
    paths = {
        name: tmp_path / name
        for name in (
            "train_store",
            "train_dv",
            "wcrung_store",
            "wcrung_dv",
            "e1_store",
            "train_summary",
        )
    }
    base = _loaded(path)
    table = base.tbl
    table.rungs = [jobd.RUNG]
    labeled_io = create_autospec(fit_script._load_labeled, return_value=table)
    monkeypatch.setattr(fit_script, "_load_labeled", labeled_io)
    monkeypatch.setattr(
        jobd, "behavior_paths", create_autospec(jobd.behavior_paths, return_value=paths)
    )
    stage = create_autospec(store_io.stage_u_store, side_effect=AssertionError("old crossed stage"))
    monkeypatch.setattr(store_io, "stage_u_store", stage)
    legacy = create_autospec(wc._rb_for_behavior, side_effect=AssertionError("legacy rb route"))
    monkeypatch.setattr(wc, "_rb_for_behavior", legacy)
    tensors = tmp_path / "tensors"
    (tensors / "r_b_e1").mkdir(parents=True)
    bank = np.arange(28 * 4, dtype=np.float16).reshape(28, 4)
    np.savez(tensors / "r_b_e1/evil.npz", rb=bank, layers=np.arange(28))
    args = _args(path)
    args.rb_source, args.tensors_root = "bank", tensors
    loaded = jobd.load_behavior(args, "evil", [17, 20])
    assert labeled_io.call_count == 3 and stage.call_count == legacy.call_count == 0
    np.testing.assert_array_equal(loaded.rb, bank[[17, 20]])
    assert loaded.natural_meta["generic_u"] == 12
    assert loaded.natural_meta["selected_global_layers"] == [17, 20]


def test_resume_key_covers_budget_manifest_layers_flags_and_prediction_sidecars(tmp_path):
    path = tmp_path / "store"
    _store(path, n=3, dim=3584)
    args = _args(path, u=2)
    args.transfer_preds, args.train_frac, args.protocols = True, 0.8, "B"
    key = natural.natural_regime_key(args, "evil", [17, 20], natural.NATURAL_ROSTER)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("map_diagnostics.json", "readout_pools.json"):
        (out / name).write_text("{}")
    (out / "preds.jsonl").write_text("{}\n")
    meta = {
        "git_commit": "abc",
        "out_schema_version": score.SEED_OUT_SCHEMA_VERSION,
        "seed": 0,
        "map_variants": [],
        "natural_regime_key": key,
        "input_sha256": {str(path / "manifest.json"): natural._sha(path / "manifest.json")},
        "natural_transfer_pred_files": ["preds.jsonl"],
    }
    (out / "all_arms_spearman.json").write_text(json.dumps({"meta": meta}))
    kwargs = dict(commit="abc", seed=0, map_variants=None, natural_key=key)
    assert score._seed_output_resume_ok(out, **kwargs)[0]
    for name, val in (("generic_u", 3), ("seed", 1), ("train_frac", 0.7)):
        changed = SimpleNamespace(**{**vars(args), name: val})
        changed_key = natural.natural_regime_key(changed, "evil", [17, 20], natural.NATURAL_ROSTER)
        assert not score._seed_output_resume_ok(out, **{**kwargs, "natural_key": changed_key})[0]
    assert key != natural.natural_regime_key(args, "evil", [20, 17], natural.NATURAL_ROSTER)
    (out / "preds.jsonl").unlink()
    assert not score._seed_output_resume_ok(out, **kwargs)[0]


def test_opt_in_flags_preserve_legacy_defaults_and_refuse_mixed_protocols():
    args = score.parse_args([])
    assert args.natural_u_store is None and args.generic_u is None and args.protocols == "AB"
    for argv in (
        ["--generic-u", "100000"],
        ["--natural-u-store", "/store"],
        ["--natural-u-store", "/store", "--generic-u", "100000"],
    ):
        with pytest.raises(SystemExit):
            score.parse_args(argv)
    args = score.parse_args(
        ["--natural-u-store", "/store", "--generic-u", "100000", "--protocols", "B"]
    )
    assert args.generic_u == 100000 and args.natural_u_store == Path("/store")
