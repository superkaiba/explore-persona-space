"""Local-only checks of runner preservation and frozen-readout aggregation.

Cloud calls are replaced only at the autospecced HfApi boundary. Numerical
summaries use the actual group-bootstrap implementation on small predictions.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import CommitInfo, RepoFile

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/issue1739_natural_run.py"
_SPEC = importlib.util.spec_from_file_location("issue1739_natural_run_under_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _config(tmp_path):
    """Use separate, bounded temporary locations for all test outputs."""
    return {
        "out_root": str(tmp_path / "outputs"),
        "metadata_root": str(tmp_path / "metadata"),
        "source_sha": "a" * 40,
        "input_fingerprint": "f" * 64,
        "behavior": "hallucination",
        "upload_prefix": "issue1739_test_never_uploaded",
    }


def _prediction_bundle(config):
    """Write conflicting off-layer predictions to expose frozen-layer mistakes."""
    layers = runner.LAYERS[config["behavior"]]
    datasets = {"nqopen": 8, "simpleqa": 20, "heldin_train": 8, "wildchat_rung": 8}
    reference = {"cells": [{"arms": {arm: {"dataset_rows": {}} for arm in layers}}]}
    for dataset, n in datasets.items():
        dv = np.arange(n, dtype=float)
        ids = np.array([f"{dataset}:{i}" for i in range(n)])
        groups = np.array([f"g{i // 2}" for i in range(n)])
        variants, arms, predictions = [], [], []
        for variant in ("e1", "q01_s0"):
            for arm in layers:
                variants.append(variant)
                arms.append(arm)
                if arm == "mapped_answer":
                    values = dv if variant == "q01_s0" and dataset == "nqopen" else -dv
                else:
                    values = dv
                predictions.append(values)
        for layer in sorted(set(layers.values())):
            parent = Path(config["out_root"]) / f"L{layer:02d}"
            parent.mkdir(parents=True, exist_ok=True)
            chosen = np.stack(
                [
                    values if layer == layers[arm] else -values
                    for values, arm in zip(predictions, arms, strict=True)
                ]
            )
            np.savez(
                parent / f"predictions_{dataset}.npz",
                predictions=chosen,
                variants=np.asarray(variants),
                arms=np.asarray(arms),
                ids=ids,
                groups=groups,
                dv=dv,
            )
        for arm in layers:
            key = "heldin:train" if dataset == "heldin_train" else dataset
            reference["cells"][0]["arms"][arm]["dataset_rows"][key] = {
                "rho_frozen": -1.0 if arm == "mapped_answer" else 1.0,
                "n_eval": n,
            }
    metadata_root = Path(config["metadata_root"])
    metadata_root.mkdir(parents=True)
    (metadata_root / "reference.json").write_text(json.dumps(reference))


def test_summary_uses_frozen_layers_equal_dataset_means_and_paired_contrasts(tmp_path):
    config = _config(tmp_path)
    _prediction_bundle(config)
    result = runner.summarize(config)
    primary = "q01_s0/mapped_answer"
    assert result["ood"][primary]["rho"] == pytest.approx(0.0)
    assert result["ood"]["e1/mapped_answer"]["rho"] == pytest.approx(-1.0)
    assert result["ood"]["q01_s0/context_native"]["rho"] == pytest.approx(1.0)
    assert result["ood"][primary]["ci95"] == pytest.approx([0.0, 0.0])
    assert result["ood"][primary]["n_datasets"] == 2
    datasets = {d["dataset"]: d for d in result["datasets"]}
    contrast = f"{primary}_minus_e1/mapped_answer"
    assert datasets["nqopen"]["differences"][contrast]["delta"] == pytest.approx(2.0)
    assert datasets["nqopen"]["differences"][contrast]["ci95"] == pytest.approx([2.0, 2.0])
    assert datasets["simpleqa"]["differences"][contrast]["delta"] == pytest.approx(0.0)
    assert result["ood_differences"][contrast]["delta"] == pytest.approx(1.0)
    assert result["ood_differences"][contrast]["ci95"] == pytest.approx([1.0, 1.0])
    for dataset in datasets:
        with np.load(Path(config["out_root"]) / f"bootstrap_{dataset}.npz") as archive:
            assert archive["draws"].shape == (len(archive["methods"]), 500)
            assert all(len(v["ci95"]) == 2 for v in datasets[dataset]["estimates"].values())


def test_summary_rejects_different_eval_rows_across_frozen_layers(tmp_path):
    config = _config(tmp_path)
    _prediction_bundle(config)
    path = Path(config["out_root"]) / "L27/predictions_nqopen.npz"
    with np.load(path) as archive:
        content = {k: archive[k] for k in archive.files}
    content["ids"] = content["ids"][::-1]
    np.savez(path, **content)
    with pytest.raises(ValueError, match="row mismatch"):
        runner.summarize(config)


def test_summary_rejects_missing_frozen_layer_prediction_file(tmp_path):
    config = _config(tmp_path)
    _prediction_bundle(config)
    (Path(config["out_root"]) / "L23/predictions_nqopen.npz").unlink()
    with pytest.raises(ValueError):
        runner.summarize(config)


def test_summary_rejects_whole_missing_reference_dataset(tmp_path):
    config = _config(tmp_path)
    _prediction_bundle(config)
    for path in Path(config["out_root"]).glob("L*/predictions_simpleqa.npz"):
        path.unlink()
    with pytest.raises(ValueError):
        runner.summarize(config)


def test_constant_natural_prediction_is_reported_as_undefined(tmp_path):
    config = _config(tmp_path)
    _prediction_bundle(config)
    path = Path(config["out_root"]) / "L23/predictions_nqopen.npz"
    with np.load(path) as archive:
        content = {k: archive[k] for k in archive.files}
    index = np.flatnonzero(
        (content["variants"] == "q01_s0") & (content["arms"] == "mapped_answer")
    )[0]
    content["predictions"][index] = 0
    np.savez(path, **content)
    result = runner.summarize(config)
    nq = next(row for row in result["datasets"] if row["dataset"] == "nqopen")
    primary = "q01_s0/mapped_answer"
    contrast = f"{primary}_minus_e1/mapped_answer"
    assert nq["estimates"][primary]["rho"] is None
    assert nq["differences"][contrast]["delta"] is None
    assert result["ood"][primary]["rho"] is None
    assert result["ood_differences"][contrast]["delta"] is None


def test_identical_noisy_methods_have_exact_zero_paired_interval(tmp_path):
    config = _config(tmp_path)
    _prediction_bundle(config)
    generator = np.random.default_rng(23)
    for dataset in ("nqopen", "simpleqa"):
        path = Path(config["out_root"]) / f"L23/predictions_{dataset}.npz"
        with np.load(path) as archive:
            content = {k: archive[k] for k in archive.files}
        noisy = content["dv"] + generator.normal(scale=10, size=len(content["dv"]))
        content["predictions"][content["arms"] == "mapped_answer"] = noisy
        np.savez(path, **content)
    result = runner.summarize(config)
    contrast = "q01_s0/mapped_answer_minus_e1/mapped_answer"
    for row in result["datasets"]:
        if row["dataset"] in {"nqopen", "simpleqa"}:
            ci = row["estimates"]["q01_s0/mapped_answer"]["ci95"]
            assert ci[1] > ci[0]
            assert row["differences"][contrast]["ci95"] == pytest.approx([0.0, 0.0])
    assert result["ood_differences"][contrast]["ci95"] == pytest.approx([0.0, 0.0])


def _metadata_input(tmp_path):
    """Small text metadata supports an actual MinHash replay, without network."""
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "contexts": {
                    "c": {"query": "Why does my bread keep collapsing while it cools?"},
                    "a": {"query": "Help me find the bug in this sorting algorithm."},
                    "b": {"query": "My landlord replaced the broken window yesterday."},
                }
            }
        )
    )
    return {"prompt_metadata": str(path)}


def test_signature_replay_preserves_exact_context_keyset_and_values(tmp_path):
    config = _config(tmp_path)
    paths = _metadata_input(tmp_path)
    first = runner.stage_signatures(config, paths)
    second = runner.stage_signatures(config, paths)
    assert set(first["signatures"]) == set(first["contexts"]) == {"a", "b", "c"}
    assert first["signatures"] == second["signatures"]
    assert np.asarray(first["signatures"]["a"]).shape == (64,)


@pytest.mark.parametrize("mutation", ["ids", "fingerprint", "values", "shape"])
def test_signature_replay_rejects_stale_or_altered_chunk(tmp_path, mutation):
    config = _config(tmp_path)
    paths = _metadata_input(tmp_path)
    runner.stage_signatures(config, paths)
    chunk = Path(config["out_root"]) / "metadata/signatures_000000.npz"
    with np.load(chunk) as archive:
        content = {k: archive[k] for k in archive.files}
    if mutation == "ids":
        content["ids"] = content["ids"][::-1]
    elif mutation == "fingerprint":
        content["fingerprint"] = "changed"
    elif mutation == "values":
        content["signatures"][0, 0] ^= np.uint64(1)
    else:
        content["signatures"] = content["signatures"][:, :32]
    np.savez(chunk, **content)
    with pytest.raises(ValueError):
        runner.stage_signatures(config, paths)


def test_signature_replay_rejects_extra_or_missing_receipt_keys(tmp_path):
    config = _config(tmp_path)
    paths = _metadata_input(tmp_path)
    runner.stage_signatures(config, paths)
    receipt = Path(config["out_root"]) / "metadata/complete.json"
    content = json.loads(receipt.read_text())
    content["files"]["signatures_000128.npz"] = "0" * 64
    receipt.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Metadata completion manifest mismatch"):
        runner.stage_signatures(config, paths)


def test_signature_staging_recovers_interrupted_temporary_chunk(tmp_path):
    config = _config(tmp_path)
    paths = _metadata_input(tmp_path)
    parent = Path(config["out_root"]) / "metadata"
    parent.mkdir(parents=True)
    (parent / "signatures_000000.partial.npz").write_bytes(b"interrupted atomic write")
    result = runner.stage_signatures(config, paths)
    assert set(result["signatures"]) == {"a", "b", "c"}
    receipt = json.loads((parent / "complete.json").read_text())
    assert set(receipt["files"]) == {"signatures_000000.npz"}


@pytest.mark.parametrize("problem", [None, "overlap", "hash", "count", "ambiguous"])
def test_sharded_prompt_metadata_requires_verified_disjoint_complete_union(tmp_path, problem):
    config = _config(tmp_path)
    paths = _metadata_input(tmp_path)
    metadata_path = Path(paths["prompt_metadata"])
    contexts = json.loads(metadata_path.read_text())["contexts"]
    chunks = [{"a": contexts["a"], "b": contexts["b"]}, {"c": contexts["c"]}]
    if problem == "overlap":
        chunks[1] = {"a": contexts["a"]}
    parts = []
    for i, rows in enumerate(chunks):
        path = tmp_path / f"contexts_{i}.json"
        path.write_text(json.dumps(rows))
        parts.append({"path": path.name, "sha256": runner.sha(path), "n": len(rows)})
    metadata = {"context_parts": parts, "n_contexts": 3}
    if problem == "count":
        metadata["n_contexts"] = 4
    elif problem == "hash":
        (tmp_path / parts[0]["path"]).write_text("{}")
    elif problem == "ambiguous":
        metadata["contexts"] = contexts
    metadata_path.write_text(json.dumps(metadata))
    if problem is None:
        result = runner.stage_signatures(config, paths)
        assert result["contexts"] == contexts
        assert set(result["signatures"]) == set(contexts)
    else:
        with pytest.raises(ValueError):
            runner.stage_signatures(config, paths)


def test_response_split_checks_both_orientations_on_primary_eligible_population():
    from scripts.issue1739_natural_score import build_selections

    from scripts.issue1739_r2v2_score import _group_side_train

    fixture_path = Path(__file__).with_name("test_issue1739_natural_score.py")
    spec = importlib.util.spec_from_file_location("natural_scorer_fixture", fixture_path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    table, metadata = fixtures._evil_sources()
    for cid, context in metadata["contexts"].items():
        if context["natural_eligible"]:
            first = 80 if int(cid.rsplit("-", 1)[1]) % 2 else 20
            metadata["scores"][cid] = {
                f"k{k:02d}": first if k < 3 else 100 - first for k in range(5)
            }
    # This two-judgment prompt is ineligible for the primary estimator and must
    # not become the apparent high extreme only in the 3-to-2 reliability split.
    short = next(
        cid
        for cid, group, rung in zip(table.ids, table.groups, table.rungs, strict=True)
        if rung == "toxicchat" and _group_side_train(rung, group, 0, 0.8)
    )
    metadata["scores"][short] = {
        "k00": 100,
        "k01": 100,
        "k02": None,
        "k03": None,
        "k04": None,
    }
    _, _, records = build_selections({"behavior": "evil"}, table, metadata)
    reliability = records["hhrt/q01_s0"]["response_split_reliability"]
    assert set(reliability) == {"3_to_2", "2_to_3"}
    for orientation, measure in reliability.items():
        assert measure["high_minus_low_score"] == -60
        assert measure["high"]["n_unscored_prompts"] == 0
        assert measure["low"]["n_unscored_prompts"] == 0
        assert measure["draw_indices"] == ([3, 4] if orientation == "3_to_2" else [0, 1, 2])


def _fake_hub(monkeypatch, config, files, *, wrong=None):
    """Replace only the external SDK boundary using actual RepoFile objects."""
    api = create_autospec(HfApi, instance=True)
    revision = "b" * 40
    api.upload_folder.return_value = CommitInfo(
        commit_url=f"https://huggingface.co/datasets/example/test/commit/{revision}",
        commit_message="unit test",
        commit_description="",
        oid=revision,
    )
    remote = []
    for name, payload in files.items():
        entry = {
            "path": f"{config['upload_prefix']}/{name}",
            "size": len(payload),
            "oid": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
        }
        if name.endswith(".npz"):
            entry["lfs"] = {
                "oid": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "pointerSize": 125,
            }
        if wrong == "hash" and name == "vectors.npz":
            entry["lfs"]["oid"] = "0" * 64
        if wrong == "blob" and name == "results.json":
            entry["oid"] = "0" * 40
        if wrong == "size" and name == "results.json":
            entry["size"] += 1
        remote.append(RepoFile(**entry))
    if wrong == "names":
        remote.pop()
    api.list_repo_tree.return_value = remote
    monkeypatch.setattr("huggingface_hub.HfApi", create_autospec(HfApi, return_value=api))
    return api, revision


def _upload_files(config):
    """Write two payload formats so both remote hash branches execute."""
    files = {"results.json": b'{"finished": true}\n', "vectors.npz": b"unit-test-array-bytes"}
    root = Path(config["out_root"])
    root.mkdir(parents=True)
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    return files


def test_upload_receipt_follows_independent_exact_hash_verification(tmp_path, monkeypatch):
    config = _config(tmp_path)
    files = _upload_files(config)
    api, revision = _fake_hub(monkeypatch, config, files)
    report = runner.upload_verified(config, "complete")
    assert report["verified_revision"] == revision
    assert report["sha256"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in files.items()
    }
    assert api.list_repo_tree.call_args.kwargs["revision"] == revision
    assert (Path(config["out_root"]) / "upload_verified.json").exists()


@pytest.mark.parametrize("wrong", ["hash", "blob", "size", "names"])
def test_failed_remote_verification_cannot_publish_receipt(tmp_path, monkeypatch, wrong):
    config = _config(tmp_path)
    files = _upload_files(config)
    api, _ = _fake_hub(monkeypatch, config, files, wrong=wrong)
    with pytest.raises(ValueError):
        runner.upload_verified(config, "complete")
    assert not (Path(config["out_root"]) / "upload_verified.json").exists()
    api.upload_file.assert_not_called()


def test_incomplete_npz_writes_do_not_enter_upload_manifest(tmp_path, monkeypatch):
    config = _config(tmp_path)
    files = _upload_files(config)
    (Path(config["out_root"]) / "signatures_000000.partial.npz").write_bytes(b"interrupted")
    _fake_hub(monkeypatch, config, files)
    report = runner.upload_verified(config, "L23")
    assert set(report["sha256"]) == set(files)


def _completion_readback(monkeypatch, tmp_path, api, *, append_bytes=b""):
    """Give receipt uploads a distinct commit and replay their exact uploaded bytes."""
    revision = "c" * 40
    api.upload_file.return_value = CommitInfo(
        commit_url=f"https://huggingface.co/datasets/example/test/commit/{revision}",
        commit_message="receipt unit test",
        commit_description="",
        oid=revision,
    )
    download = create_autospec(hf_hub_download)

    def provide_readback(*args, **kwargs):
        """Only the autospecced SDK network boundary is replaced by local bytes."""
        payload = api.upload_file.call_args.kwargs["path_or_fileobj"]
        path = tmp_path / "immutable_receipt_readback.json"
        path.write_bytes(payload + append_bytes)
        return str(path)

    download.side_effect = provide_readback
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return download, revision


def test_published_completion_references_verified_outputs_and_reads_immutable_receipt(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    files = _upload_files(config)
    api, output_revision = _fake_hub(monkeypatch, config, files)
    verification = runner.upload_verified(config, "complete")
    download, receipt_revision = _completion_readback(monkeypatch, tmp_path, api)
    result = runner.publish_completion(config, verification)
    upload_kwargs = api.upload_file.call_args.kwargs
    assert upload_kwargs["path_in_repo"] == config["upload_prefix"] + ".completion.json"
    # The receipt lives beside the result directory, avoiding a self-hashing cycle.
    assert not upload_kwargs["path_in_repo"].startswith(config["upload_prefix"] + "/")
    payload = upload_kwargs["path_or_fileobj"]
    published = json.loads(payload)
    assert published["verified_revision"] == output_revision
    assert published["verification"] == verification
    assert published["status"] == "complete"
    assert published["input_fingerprint"] == config["input_fingerprint"]
    assert published["source_sha"] == config["source_sha"]
    assert published["behavior"] == config["behavior"]
    assert result == {**published, "receipt_revision": receipt_revision}
    assert receipt_revision != output_revision
    download.assert_called_once_with(
        repo_id=runner.hub.DEFAULT_DATASET_REPO,
        repo_type="dataset",
        filename=config["upload_prefix"] + ".completion.json",
        revision=receipt_revision,
    )
    assert (tmp_path / "immutable_receipt_readback.json").read_bytes() == payload


def test_completion_rejects_even_semantically_equivalent_readback_byte_tampering(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    files = _upload_files(config)
    api, _ = _fake_hub(monkeypatch, config, files)
    verification = runner.upload_verified(config, "complete")
    download, receipt_revision = _completion_readback(monkeypatch, tmp_path, api, append_bytes=b" ")
    with pytest.raises(ValueError, match="Completion receipt readback differs"):
        runner.publish_completion(config, verification)
    assert download.call_args.kwargs["revision"] == receipt_revision
    assert not (Path(config["out_root"]) / "run_complete.json").exists()
