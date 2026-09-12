"""The native-dictionary affine null preserves its construct and upload boundary."""

import copy
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.workspace_decomposition import ValidatedDictionary
from explore_persona_space.analysis.workspace_lenses import nonnegative_gradient_pursuit
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json, save_tensors


@pytest.fixture
def null_module(monkeypatch):
    from explore_persona_space.orchestrate import env

    monkeypatch.setattr(env, "load_dotenv", lambda: None)
    path = Path(__file__).resolve().parents[1] / "scripts/workspace_jr_native_affine_null.py"
    return runpy.run_path(str(path))


@pytest.mark.parametrize("batch_size", [2, 128])
def test_reduced_affine_null_matches_actual_token_reference(null_module, batch_size):
    rng = np.random.default_rng(613)
    x = {"train": rng.normal(size=(5, 6)), "test": rng.normal(size=(3, 6))}
    x["train"][0] = 0
    weights = rng.normal(size=(6, 6))
    bias = np.zeros(6)
    full = null_module["affine_targets"](x, weights, bias)
    dictionaries = {}
    for arm in ("J", "R"):
        dictionary = torch.tensor(rng.normal(size=(32, 6)), dtype=torch.float32)
        dictionaries[arm] = dictionary / dictionary.norm(dim=1, keepdim=True)
    actual, diagnostics = null_module["decompose_affine_targets"](
        full,
        {arm: ValidatedDictionary(value) for arm, value in dictionaries.items()},
        batch_size=batch_size,
    )
    lengths = [1, 3, 2, 9, 129]
    for split, values in full.items():
        for k in (5, 10, 25):
            np.testing.assert_array_equal(actual[k][split]["full"], values)
            for arm, dictionary in dictionaries.items():
                for index, h in enumerate(values):
                    pooled = []
                    for length in lengths:
                        repeated = torch.from_numpy(np.repeat(h[None], length, axis=0)).float()
                        parts = [
                            nonnegative_gradient_pursuit(repeated[i : i + 128], dictionary, k=k)
                            for i in range(0, length, 128)
                        ]
                        pooled.append(torch.cat([p.component for p in parts]).double().mean(0))
                    expected = torch.stack(pooled).mean(0).numpy()
                    np.testing.assert_allclose(
                        actual[k][split][arm][index], expected, rtol=1e-5, atol=1e-6
                    )
                np.testing.assert_allclose(
                    actual[k][split][arm] + actual[k][split][f"rest{arm}"],
                    values,
                    rtol=1e-14,
                    atol=1e-14,
                )
                assert diagnostics[k][f"{split}/{arm}"]["active_atoms"].shape == (len(values),)
                assert np.all(diagnostics[k][f"{split}/{arm}"]["active_atoms"] <= k)


def test_affine_map_uses_fixed_coefficients_for_all_splits(null_module):
    x = {"train": np.array([[1, 2], [3, 4]], dtype=np.float32), "test": np.array([[7, 8]])}
    weights = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    actual = null_module["affine_targets"](x, weights, bias)
    for split in x:
        assert actual[split].dtype == np.float64
        np.testing.assert_array_equal(actual[split], x[split].astype(np.float64) @ weights + bias)
    with pytest.raises(ValueError, match="coefficients"):
        null_module["affine_targets"](x, weights, np.zeros(2))
    with pytest.raises(ValueError, match="input"):
        null_module["affine_targets"]({"train": np.array([[np.nan, 1]])}, weights, bias)


@pytest.mark.parametrize("mutation", ["order", "empty", "missing_draw", "boolean_length"])
def test_original_rollout_layout_is_required(null_module, mutation):
    ids = {"train": ["a", "b"]}
    statistics = {
        "train": [{"context_id": n, "token_counts": [1, 3, 2, 9, 129]} for n in ids["train"]]
    }
    assert null_module["validate_layout"](statistics, ids, 5)["train"][0] == [1, 3, 2, 9, 129]
    if mutation == "order":
        statistics["train"].reverse()
    elif mutation == "empty":
        statistics["train"][0]["token_counts"][0] = 0
    elif mutation == "missing_draw":
        statistics["train"][0]["token_counts"].pop()
    else:
        statistics["train"][0]["token_counts"][0] = True
    with pytest.raises(ValueError, match=r"layout|lengths"):
        null_module["validate_layout"](statistics, ids, 5)


@pytest.fixture
def prepared_null(null_module, tmp_path, monkeypatch):
    source = tmp_path / "source"
    rng = np.random.default_rng(817)
    identity = {
        "config_sha256": "a" * 64,
        "selection_sha256": "b" * 64,
        "model_role": "primary",
        "versions": {},
        "code": {"git_commit": "c" * 40, "git_dirty": False},
    }
    x = {
        s: rng.normal(size=(n, 6)).astype(np.float32)
        for s, n in (("train", 5), ("validation", 2), ("test", 3))
    }
    ids = {s: [f"{s}-{i}" for i in range(len(v))] for s, v in x.items()}
    stats = {
        s: [{"context_id": i, "token_counts": [1, 3, 2, 9, 129]} for i in names]
        for s, names in ids.items()
    }
    fit = source / "fits/pilot/k10-rotationNone"
    fit.mkdir(parents=True)
    np.savez(fit / "ridge-full.npz", weights=rng.normal(size=(6, 6)), bias=rng.normal(size=6))
    arms = {}
    for arm in ("J", "R"):
        dictionary = torch.tensor(rng.normal(size=(32, 6)), dtype=torch.float32)
        dictionary /= dictionary.norm(dim=1, keepdim=True)
        path = source / f"dictionaries/{arm}.pt"
        save_tensors(path, {"dictionary": dictionary})
        arms[arm] = {"sha256": file_sha256(path)}
    save_json(source / "dictionaries/manifest.json", {"pilot_only": True, "arms": arms})
    selection = tmp_path / "selection.json"
    save_json(selection, {})
    config = {"generation": {"seeds": [42, 43, 44, 45, 46]}, "fit": {"example": "frozen"}}
    args = SimpleNamespace(
        input_root=source,
        stage="pilot",
        fit_upload_receipt=tmp_path / "source_upload.json",
        selection=selection,
        out=tmp_path / "null",
        rotation=None,
        device="cpu",
        role="primary",
        k=10,
    )

    def receipt(folder, output):
        hashes = {
            str(p.relative_to(folder)): file_sha256(p) for p in folder.rglob("*") if p.is_file()
        }
        save_json(
            output,
            {
                "repo": "superkaiba1/explore-persona-space-data",
                "prefix": "exploratory_workspace_jr/fixture",
                "revision": "d" * 40,
                "files_verified": len(hashes),
                "verified_sha256": hashes,
            },
        )

    receipt(source, args.fit_upload_receipt)
    globals_ = null_module["prepare"].__globals__
    monkeypatch.setitem(
        globals_,
        "load_fitted_components",
        lambda *a, **kw: (x, None, None, ids, None, stats, {"source": "verified"}),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    null_module["prepare"](args, config, identity)
    args.prepared_upload_receipt = tmp_path / "prepared_upload.json"
    receipt(args.out, args.prepared_upload_receipt)
    return args, config, identity, x, ids


def test_real_prepare_fit_boundary_preserves_targets_and_requires_upload(
    prepared_null, null_module, monkeypatch
):
    args, config, identity, x, ids = prepared_null
    calls = []

    class Run:
        id = "fixture"
        url = "https://example.invalid/fixture"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def log(self, record):
            raise AssertionError("Mock fit must not send telemetry")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kw: Run()))

    def fitter(xx, targets, actual_ids, settings, output, **kwargs):
        assert actual_ids == ids and settings == config
        for split, values in x.items():
            np.testing.assert_array_equal(xx[split], values)
            np.testing.assert_allclose(
                targets[split]["J"] + targets[split]["restJ"], targets[split]["full"], atol=1e-14
            )
        calls.append(copy.deepcopy(targets))
        result = {"paired_bootstrap": {"summary": {"fixture": True}}}
        save_json(output / "results.json", result)
        return result

    monkeypatch.setitem(null_module["fit"].__globals__, "evaluate_component_fits", fitter)
    null_module["fit"](args, config, identity)
    complete = json.loads((args.out / "k10-rotationNone/null_complete.json").read_text())
    assert complete["status"] == "complete" and len(calls) == 1
    with pytest.raises(ValueError, match="fresh per-k"):
        null_module["fit"](args, config, identity)


def test_fit_rejects_prepared_target_mutation_before_training(
    prepared_null, null_module, monkeypatch
):
    args, config, identity, _, _ = prepared_null
    path = args.out / "targets_k10.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())
    with pytest.raises(ValueError, match="verified producer upload"):
        null_module["fit"](args, config, identity)
    assert not (args.out / "k10-rotationNone").exists()
