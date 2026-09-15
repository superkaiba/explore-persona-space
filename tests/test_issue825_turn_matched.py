"""Counterbalancing, heldout scoring, and loss-aggregation checks for matched transfer."""

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from explore_persona_space.analysis.pooled_turn_transfer import fit_primal_gcv

spec = importlib.util.spec_from_file_location(
    "matched_driver", Path(__file__).resolve().parents[1] / "scripts/issue825_turn_matched.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
archive = importlib.import_module("issue825_turn_k5_archive")


def test_balanced_assignment_matches_rows_and_conversations():
    labels = np.repeat(np.arange(6), [7, 8, 9, 10, 11, 12])
    ids = np.array([f"c{i:03}" for i in range(len(labels))])
    rotations = driver.assignments(ids, labels)
    np.testing.assert_array_equal(
        np.sort(rotations, axis=0), np.broadcast_to(np.arange(1, 4)[:, None], rotations.shape)
    )
    panel = dict(
        ids=ids,
        labels=labels,
        assignments=rotations,
        x=np.arange(12 * len(ids) * 2).reshape(12, len(ids), 2),
    )
    panel["y"] = panel["x"] + 0.5
    for variant in driver.VARIANTS:
        x, y, turns = driver.source_rows(panel, variant)
        assert len(x) == len(ids)
        np.testing.assert_array_equal(y - x, 0.5)
        for i in range(len(ids)):
            np.testing.assert_array_equal(x[i], panel["x"][turns[i] - 1, i])
        for f in range(6):
            train = labels != f
            assert len(set(ids[train])) == train.sum()
            if variant.startswith("mix"):
                assert np.ptp(np.bincount(turns[train], minlength=4)[1:]) <= 2


@pytest.mark.parametrize("variant", ["mix0", "12"])
def test_real_scoring_all_turns_and_source_only_bias(tmp_path, variant):
    rng = np.random.default_rng(3)
    labels = np.arange(36) % 6
    ids = np.array([f"c{i:03}" for i in range(36)])
    x = rng.normal(size=(12, 36, 7))
    y = 2 * x + rng.normal(size=x.shape)
    y[3:] += 100  # Target offsets must not leak into source-trained bias.
    panel = dict(x=x, y=y, ids=ids, labels=labels, assignments=driver.assignments(ids, labels))
    args = SimpleNamespace(out=tmp_path / "analysis", store=tmp_path / "store")
    xx, yy, turns = driver.source_rows(panel, variant)
    if variant == "12":
        np.testing.assert_array_equal(xx, x[11])
        np.testing.assert_array_equal(yy, y[11])
        np.testing.assert_array_equal(turns, np.full(len(ids), 12))
    train = np.flatnonzero(labels != 0)
    test = np.flatnonzero(labels == 0)
    fit = fit_primal_gcv(xx, yy, [train])
    rec = driver.score_fold(args, panel, "instruct", variant, 0, fit, 0, "fp")
    file = args.store / f"predictions/instruct_{variant}_fold0.npz"
    with np.load(file) as z:
        np.testing.assert_array_equal(z["ids"], ids[test])
        assert not set(z["ids"]) & set(ids[train])
        bias = (yy[train] - xx[train]).mean(0)
        np.testing.assert_allclose(z["source_bias"], bias)
        for t in range(12):
            truth = y[t, test]
            raw = fit.predict(0, x[t, test])
            np.testing.assert_allclose(z["raw"][t], raw)
            np.testing.assert_allclose(z["sse"][0, t], ((raw - truth) ** 2).sum(-1))
            np.testing.assert_allclose(z["sse"][1, t], ((x[t, test] + bias - truth) ** 2).sum(-1))
            distances = ((raw[:, None, :] - truth[None, :, :]) ** 2).sum(-1)
            np.testing.assert_array_equal(
                z["euclidean_hit"][0, t], distances.argmin(-1) == np.arange(len(test))
            )
            cosine = (raw / np.linalg.norm(raw, axis=-1, keepdims=True)) @ (
                truth / np.linalg.norm(truth, axis=-1, keepdims=True)
            ).T
            np.testing.assert_array_equal(
                z["cosine_hit"][0, t], cosine.argmax(-1) == np.arange(len(test))
            )
    assert rec["n_train"] == len(train)
    assert driver.score_fold(args, panel, "instruct", variant, 0, fit, 0, "fp") == rec
    with file.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint"):
        driver.score_fold(args, panel, "instruct", variant, 0, fit, 0, "fp")


def test_rotation_loss_average_is_not_ensemble_error():
    predictions = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0])[:, None]
    loss = predictions**2
    collapsed = driver.collapse_rotations(loss)
    assert collapsed[-1, 0] == pytest.approx(2 / 3)
    assert predictions[3:].mean() ** 2 == 0


@pytest.mark.parametrize("fit_chunks", [6, 36])
def test_failed_venue_gate_remains_failed(tmp_path, fit_chunks):
    rec = dict(elapsed_seconds=151, max_rss_kib=1000)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="gate"):
            driver.venue_gate(rec, tmp_path / "gate.json", fit_chunks)
        gate = json.loads((tmp_path / "gate.json").read_text())
        assert gate["status"] == "halt"
        assert gate["projection_limit_seconds"] == fit_chunks * 150
    driver.venue_gate(rec | {"elapsed_seconds": 150}, tmp_path / "gate.json", fit_chunks)
    assert json.loads((tmp_path / "gate.json").read_text())["status"] == "pass"


@pytest.mark.parametrize("single_source_turn", [None, 12])
def test_model_driver_and_panel_body_resume_rejects_wrong_fold(
    tmp_path, monkeypatch, single_source_turn
):
    """Exercise the real selection, fitting and scoring bodies on a small complete bank."""
    rng = np.random.default_rng(17)
    ids = np.array([f"c{i:03}" for i in range(36)])
    labels = np.arange(36) % 6
    x = rng.normal(size=(12, 36, 7))
    y = x @ rng.normal(size=(7, 7)) + rng.normal(size=x.shape)
    panel = dict(
        x=x.reshape(-1, 7),
        y=y.reshape(-1, 7),
        ids=np.tile(ids, 12),
        turns=np.repeat(np.arange(1, 13), 36),
        membership=np.tile(labels, 12),
        counts={str(t): 36 for t in range(1, 13)},
        fold_hash="fixture",
    )

    def load_panel(args, model):
        return panel

    monkeypatch.setattr(driver.parent, "load_panel", load_panel)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(dict(cohort_ids=ids.tolist(), fold_labels=labels.tolist(), fold_hash="fixture"))
    )
    args = SimpleNamespace(
        config=config,
        out=tmp_path / "analysis",
        store=tmp_path / "store",
        single_source_turn=single_source_turn,
    )
    driver.run_model(args, "instruct")
    driver.run_model(args, "instruct")
    variants = driver.source_regime(args)["variants"]
    assert len(list((args.out / "maps").glob("*.json"))) == 3 * len(variants)
    assert len(list((args.out / "scores").glob("*.json"))) == 6 * len(variants)
    if single_source_turn == 12:
        assert variants == ["12"]
        gate = json.loads((args.out / "combined_gate_instruct_12.json").read_text())
        assert gate["fit_chunks"] == 6
        assert gate["status"] == "pass"
    receipt = args.out / f"maps/instruct_{variants[0]}_folds0-1.json"
    rec = json.loads(receipt.read_text())
    rec["folds"] = [2, 3]
    receipt.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="unit identity"):
        driver.run_model(args, "instruct")


@pytest.mark.parametrize("single_source_turn", [None, 12])
def test_reduction_single_source_and_legacy_pooled_estimands(tmp_path, single_source_turn):
    """Read genuine cached score files at production width and compare paired OOF estimates."""
    ids = np.array([f"c{i:02}" for i in range(12)])
    labels = np.arange(len(ids)) % 6
    config = tmp_path / "config.json"
    contract = dict(cohort_ids=ids.tolist(), fold_labels=labels.tolist(), fold_hash="fixture")
    config.write_text(json.dumps(contract))
    args = SimpleNamespace(
        config=config,
        out=tmp_path / "analysis",
        store=tmp_path / "store",
        single_source_turn=single_source_turn,
    )
    regime = driver.source_regime(args)
    panel = dict(assignments=driver.assignments(ids, labels), fold_hash=contract["fold_hash"])
    sst = np.broadcast_to(np.arange(1, len(ids) + 1), (12, len(ids))).copy()
    # Conversation-varying errors make bootstrap intervals non-degenerate.
    error_scale = np.arange(1, len(ids) + 1) ** 2 / 400
    for model in driver.parent.MODELS:
        base_fp, coverage = driver.provenance(args, panel, model)
        driver.parent.atomic_json(
            args.out / f"coverage_{model}.json",
            coverage
            | dict(
                ids=ids.tolist(),
                assignments=panel["assignments"].tolist(),
                fold_sizes=np.bincount(labels).tolist(),
            ),
        )
        for a, variant in enumerate(regime["variants"]):
            fp = hashlib.sha256(f"{base_fp}:{variant}".encode()).hexdigest()
            for fold in range(6):
                selected = labels == fold
                count = int(selected.sum())
                name = f"{model}_{variant}_fold{fold}"
                metrics = np.stack(
                    [
                        np.broadcast_to((a + 1) * error_scale[selected], (12, count)),
                        np.broadcast_to(2 * error_scale[selected], (12, count)),
                    ]
                )
                digest = driver.parent.atomic_npz(
                    args.store / "predictions" / f"{name}.npz",
                    ids=ids[selected],
                    raw=np.zeros((12, count, 3584), dtype=np.float32),
                    sst=sst[:, selected],
                    sse=metrics,
                    cosine_hit=(metrics < 0.3).astype(float),
                    euclidean_hit=(metrics < 0.2).astype(float),
                )
                driver.parent.atomic_json(
                    args.out / "scores" / f"{name}.json",
                    dict(
                        fingerprint=fp,
                        sha256=digest,
                        model=model,
                        variant=variant,
                        fold=fold,
                        n_train=len(ids) - count,
                        n_test=count,
                    ),
                )
    result = driver.reduce(args)
    assert len(result["source_sha"]) == 40
    assert result["source_conditions"] == (["12"] if single_source_turn else list(driver.SOURCES))
    assert result["source_mode"] == regime["source_mode"]
    weights = np.random.default_rng(0).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), 1000)
    for model in driver.parent.MODELS:
        rows = result["models"][model]
        assert len(rows["cells"]) == len(regime["source_conditions"]) * 24
        source = "12" if single_source_turn else "1+2+3"
        coefficient = 1 if single_source_turn else 5
        cell = next(c for c in rows["cells"] if c["source"] == source and c["method"] == "raw")
        expected = 1 - coefficient * error_scale.sum() / sst[0].sum()
        draws = 1 - coefficient * (weights @ error_scale) / (weights @ sst[0])
        assert cell["r2"] == pytest.approx(expected)
        np.testing.assert_allclose(cell["r2_ci95"], np.quantile(draws, [0.025, 0.975]))
        assert rows["retrieval_pool_sizes"] == [2] * 6
        assert rows["retrieval_chance"] == 0.5
        if single_source_turn:
            assert rows["comparisons"] == []
            assert rows["rotation_r2"] == []
        else:
            assert len(rows["comparisons"]) == 2
            assert rows["comparisons"][0]["delta_r2"] == pytest.approx(
                -2 * error_scale.sum() / sst[0].sum()
            )
            assert len(rows["rotation_r2"]) == 3
    # The coverage mode is part of the resume/reduction fingerprint, not merely a label.
    coverage_path = args.out / f"coverage_{driver.parent.MODELS[0]}.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["source_mode"] = "wrong-mode"
    driver.parent.atomic_json(coverage_path, coverage)
    with pytest.raises(ValueError, match="identity differs"):
        driver.reduce(args)


@pytest.mark.parametrize("failed_archive", [None, "runtime", "tracking"])
def test_turn12_completion_archives_proof_before_backend_sentinel(
    tmp_path, monkeypatch, failed_archive
):
    """A failed proof upload must not signal successful backend completion."""
    monkeypatch.setattr(
        driver.sys,
        "argv",
        [driver.__file__, "run", "--root", str(tmp_path), "--single-source-turn", "12"],
    )
    monkeypatch.setattr(driver.parent, "reference", lambda *_: None)
    monkeypatch.setattr(driver, "assert_out_root_headroom", lambda *a, **k: None)
    monkeypatch.setattr(driver, "stage", lambda *_: None)
    monkeypatch.setattr(driver, "run_model", lambda *_: None)
    source_sha = "a" * 40

    def reduce_fixture(args):
        driver.parent.atomic_json(args.out / "results.json", dict(source_sha=source_sha, models={}))

    monkeypatch.setattr(driver, "reduce", reduce_fixture)
    monkeypatch.setitem(
        driver.sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **k: nullcontext(SimpleNamespace(id="test", log=lambda _: None))
        ),
    )
    sentinel = tmp_path / "backend.json"
    monkeypatch.setenv("EPS_SENTINEL_PATH", str(sentinel))
    prefixes = []

    def verified_upload(root, prefix, kind, receipt):
        assert not sentinel.exists()
        prefixes.append(prefix)
        if prefix.endswith("/runtime"):
            assert {p.name for p in root.iterdir()} == {
                "complete.json",
                "tensor_receipt.json",
                "text_receipt.json",
                "tracking_receipt.json",
            }
            completion = json.loads((root / "complete.json").read_text())
            assert completion["source_sha"] == source_sha
            assert completion["source_conditions"] == ["12"]
            for kind in ("tensor", "text", "tracking"):
                assert completion["archives"][kind] == json.loads(
                    (root / f"{kind}_receipt.json").read_text()
                )
        if prefix.endswith("/tracking"):
            assert kind == "tracking"
            assert root == tmp_path / "wandb"
            assert not (tmp_path / "complete.json").exists()
        if failed_archive is not None and prefix.endswith("/" + failed_archive):
            raise RuntimeError("required archive upload failed")
        driver.parent.atomic_json(receipt, dict(status="verified", prefix=prefix))

    monkeypatch.setattr(driver, "upload", verified_upload)
    if failed_archive is not None:
        with pytest.raises(RuntimeError, match="required archive upload failed"):
            driver.main()
        assert not sentinel.exists()
    else:
        driver.main()
        assert json.loads(sentinel.read_text())["source_sha"] == source_sha
        assert (tmp_path / "runtime_receipt.json").is_file()
    expected = [
        "issue825_turn12_matched_20260915/numerical",
        "issue825_turn12_matched_20260915/analysis",
        "issue825_turn12_matched_20260915/tracking",
        "issue825_turn12_matched_20260915/runtime",
    ]
    assert prefixes == (expected[:3] if failed_archive == "tracking" else expected)


@pytest.mark.parametrize(
    ("kind", "corrupt"),
    [("text", False), ("tensors", False), ("tracking", False), ("tracking", True)],
)
def test_tracking_archive_keeps_complete_inventory_and_checks_hashes(
    tmp_path, monkeypatch, kind, corrupt
):
    """Tracking retains native names and binary bytes; existing text/tensor filters stay exact."""
    root = tmp_path / "files"
    payloads = {
        "run-abc.wandb": b"\x00\xffwandb-protobuf",
        "files/config.yaml": b"learning_rate: 0.1\n",
        "files/run-summary.json": b"{}\n",
        "logs/debug.log": b"finished\n",
        "files/requirements.txt": b"numpy\n",
        "extensionless": b"\x00\x01",
        "array.npy": b"numpy-placeholder",
        "bundle.npz": b"npz-placeholder",
    }
    for name, data in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    names = {
        "text": {"files/run-summary.json", "logs/debug.log", "files/requirements.txt"},
        "tensors": {"array.npy", "bundle.npz"},
        "tracking": set(payloads),
    }[kind]
    prefix = "test/" + kind
    repo = archive.TENSOR_REPO if kind == "tensors" else archive.DATA_REPO
    entries = []
    for name in names:
        data = payloads[name]
        sha256 = hashlib.sha256(data).hexdigest()
        entries.append(
            SimpleNamespace(
                path=f"{prefix}/{name}",
                size=len(data),
                lfs=(
                    SimpleNamespace(sha256="wrong" if corrupt else sha256)
                    if name.endswith(".wandb")
                    else None
                ),
                blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            )
        )
    api = SimpleNamespace(
        repo_info=lambda *a, **k: SimpleNamespace(private=True, sha="revision"),
        list_repo_tree=lambda *a, **k: entries,
    )
    monkeypatch.setattr(archive, "HfApi", lambda: api)
    bulk = create_autospec(archive._upload_folder_filtered, return_value=f"{repo}/{prefix}")
    monkeypatch.setattr(archive, "_upload_folder_filtered", bulk)
    receipt = tmp_path / "receipt.json"
    if corrupt:
        with pytest.raises(RuntimeError, match="digest mismatch"):
            archive.upload(root, prefix, kind, receipt)
        assert not receipt.exists()
    else:
        archive.upload(root, prefix, kind, receipt)
        record = json.loads(receipt.read_text())
        assert record["status"] == "verified"
        assert record["repo"] == repo
        assert record["revision"] == "revision"
        assert set(record["files"]) == {f"{prefix}/{name}" for name in names}
        for name in names:
            assert (
                record["files"][f"{prefix}/{name}"]["sha256"]
                == hashlib.sha256(payloads[name]).hexdigest()
            )
    bulk.assert_called_once()
    assert set(bulk.call_args.kwargs["allow_patterns"]) == names
    assert set(bulk.call_args.kwargs["expected_repo_paths"]) == {
        f"{prefix}/{name}" for name in names
    }


def test_turn12_cli_rejects_changed_cohort_before_any_output(tmp_path, monkeypatch):
    config = tmp_path / "different-cohort.json"
    config.write_text("{}")
    root = tmp_path / "run"
    monkeypatch.setattr(
        driver.sys,
        "argv",
        [
            driver.__file__,
            "run",
            "--root",
            str(root),
            "--single-source-turn",
            "12",
            "--config",
            str(config),
        ],
    )
    with pytest.raises(ValueError, match="unchanged frozen matched-cohort"):
        driver.main()
    assert not root.exists()
