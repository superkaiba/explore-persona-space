"""Focused tests for task #2569's direct category-level validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "scripts/issue2569_category_validation.py"
SPEC = importlib.util.spec_from_file_location("issue2569_category_validation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CV = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CV
SPEC.loader.exec_module(CV)


def _small_rows(n: int = 160) -> list[dict]:
    """Build a balanced synthetic row table with every production design field."""

    rng = np.random.default_rng(123)
    rows = []
    for i in range(n):
        rows.append(
            {
                "ci": i,
                "corpus": rng.choice(("lmsys", "wildchat")),
                "depth": int(rng.choice((2, 3, 4, 5))),
                "prompt_chars": 50 + 3 * i,
                "topic": rng.choice(("coding", "factual_qa", "creative_writing", "math")),
                "language": rng.choice(("en", "zh", "es", "fr")),
                "format": rng.choice(("prose", "list", "code", "mixed")),
                "request_refusal_adjacent": rng.choice(("no", "yes", "borderline")),
                "answer_is_refusal": rng.choice(("no", "yes", "partial")),
            }
        )
    return rows


def test_deterministic_partition_is_corpus_stratified_and_order_invariant() -> None:
    """Hash assignment is stable and yields floor(40%) test rows per corpus."""

    ci = np.arange(31)
    corpus = np.asarray(["a"] * 13 + ["b"] * 18)
    first = CV.deterministic_partition(ci, corpus)
    order = np.random.default_rng(4).permutation(len(ci))
    second = CV.deterministic_partition(ci[order], corpus[order])
    restored = np.empty_like(second)
    restored[order] = second
    assert np.array_equal(first, restored)
    assert np.sum(first[:13] == "test") == 5
    assert np.sum(first[13:] == "test") == 7


def test_left_singular_directions_are_input_directions_for_row_action() -> None:
    """Projecting onto left singular vectors reproduces row-vector map gain."""

    rng = np.random.default_rng(1)
    operator = rng.normal(size=(7, 7))
    u, s, vh = np.linalg.svd(operator, full_matrices=False)
    x = rng.normal(size=(11, 7))
    mapped_direct = x @ operator
    mapped_singular = (x @ u * s) @ vh
    np.testing.assert_allclose(mapped_singular, mapped_direct, atol=1e-11, rtol=1e-11)


def test_nested_improvement_matches_fwl_directionwise() -> None:
    """Direct nested OLS and explicit FWL produce the same held-out SSE delta."""

    rng = np.random.default_rng(2)
    n, d = 180, 9
    nuisance = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    target = rng.normal(size=(n, 2))
    design = np.column_stack([nuisance, target])
    outcomes = nuisance @ rng.normal(size=(4, d)) + target @ rng.normal(size=(2, d))
    outcomes += rng.normal(scale=0.2, size=(n, d))
    train = np.arange(n) < 110
    test = ~train
    direct = CV.nested_improvement(design, outcomes, train, test, [4, 5])["delta"]

    nuisance_coef_x = np.linalg.lstsq(nuisance[train], target[train], rcond=None)[0]
    nuisance_coef_y = np.linalg.lstsq(nuisance[train], outcomes[train], rcond=None)[0]
    residual_x_train = target[train] - nuisance[train] @ nuisance_coef_x
    residual_y_train = outcomes[train] - nuisance[train] @ nuisance_coef_y
    beta = np.linalg.lstsq(residual_x_train, residual_y_train, rcond=None)[0]
    reduced_test = outcomes[test] - nuisance[test] @ nuisance_coef_y
    full_test = reduced_test - (target[test] - nuisance[test] @ nuisance_coef_x) @ beta
    fwl = np.sum(reduced_test**2, axis=0) - np.sum(full_test**2, axis=0)
    np.testing.assert_allclose(direct, fwl, atol=1e-10, rtol=1e-10)


def test_signed_heldout_improvement_can_be_negative_under_category_null() -> None:
    """A null category has no forced positive floor on untouched-test improvement."""

    rng = np.random.default_rng(3)
    n, d = 80, 12
    category = rng.normal(size=(n, 5))
    design = np.column_stack([np.ones(n), category])
    outcomes = rng.normal(size=(n, d))
    train = np.arange(n) < 35
    result = CV.nested_improvement(design, outcomes, train, ~train, list(range(1, 6)))
    assert result["delta"].sum() < 0


def test_batched_weighted_refits_match_single_draw_oracle() -> None:
    """Batched sufficient-statistic fits reproduce explicit weighted row scaling."""

    rng = np.random.default_rng(4)
    n_train, n_test, p, d = 50, 30, 6, 8
    z_train = np.column_stack([np.ones(n_train), rng.normal(size=(n_train, p - 1))])
    z_test = np.column_stack([np.ones(n_test), rng.normal(size=(n_test, p - 1))])
    y_train = rng.normal(size=(n_train, d))
    y_test = rng.normal(size=(n_test, d))
    w_train = rng.exponential(size=(2, n_train))
    w_test = rng.exponential(size=(2, n_test))
    batched = CV.batched_weighted_nested_sse(
        z_train, y_train, z_test, y_test, w_train, w_test, {"axis": [4, 5]}
    )["axis"]
    for draw in range(2):
        root_train = np.sqrt(w_train[draw])[:, None]
        full_beta = np.linalg.lstsq(z_train * root_train, y_train * root_train, rcond=None)[0]
        keep = np.arange(4)
        reduced_beta = np.linalg.lstsq(
            z_train[:, keep] * root_train, y_train * root_train, rcond=None
        )[0]
        full_residual = y_test - z_test @ full_beta
        reduced_residual = y_test - z_test[:, keep] @ reduced_beta
        explicit = np.sum(w_test[draw, :, None] * (reduced_residual**2 - full_residual**2), axis=0)
        np.testing.assert_allclose(batched["delta"][draw], explicit, atol=1e-9, rtol=1e-9)


def test_matched_rank_null_has_exact_cardinality_and_shared_draw() -> None:
    """Pseudo-kernels contain exactly k directions and share membership across axes."""

    d = 20
    base = np.arange(1, d + 1, dtype=float)
    deltas = {axis: base.copy() for axis in CV.AXES}
    values = CV.batched_matched_rank_null(np.random.default_rng(5), deltas, 7, 32)
    assert values.shape == (32, 5)
    np.testing.assert_allclose(values[:, :4], np.repeat(values[:, [0]], 4, axis=1))
    np.testing.assert_allclose(values[:, 4], 0.0, atol=1e-15)
    scaled = values[:, 0] * base.sum()
    assert np.all(scaled >= sum(range(1, 8)))
    assert np.all(scaled <= sum(range(d - 6, d + 1)))


def test_freedman_lane_is_seed_reproducible_and_keeps_whole_rows() -> None:
    """Blocked whole-row null is deterministic and invariant to response chunking."""

    rng = np.random.default_rng(6)
    n_train, n_test, p, d = 36, 24, 5, 7
    z_train = np.column_stack([np.ones(n_train), rng.normal(size=(n_train, p - 1))])
    z_test = np.column_stack([np.ones(n_test), rng.normal(size=(n_test, p - 1))])
    y_train = rng.normal(size=(n_train, d))
    y_test = rng.normal(size=(n_test, d))
    train_blocks = np.asarray([f"b{i // 6}" for i in range(n_train)])
    test_blocks = np.asarray([f"b{i // 6}" for i in range(n_test)])
    args = (z_train, y_train, z_test, y_test, [3, 4], train_blocks, test_blocks, 4)
    first = CV.batched_freedman_lane_outcome(np.random.default_rng(7), *args, response_chunk=2)
    second = CV.batched_freedman_lane_outcome(np.random.default_rng(7), *args, response_chunk=99)
    np.testing.assert_allclose(first, second, atol=1e-10, rtol=1e-10)
    assert first.shape == (4,)
    prepared = CV.prepare_freedman_lane(z_train, y_train, z_test, y_test, [3, 4])
    subspaces = CV.batched_freedman_lane_outcome(
        np.random.default_rng(7), *args, response_chunk=2, retained=3
    )
    oracle_rng = np.random.default_rng(7)
    p_train = CV._permutations_within_blocks(oracle_rng, train_blocks, 4)
    p_test = CV._permutations_within_blocks(oracle_rng, test_blocks, 4)
    keep = np.asarray([0, 1, 2])
    explicit = []
    for draw in range(4):
        null_train = (
            prepared["fitted_train"]
            + prepared["student_train"][p_train[draw]] * prepared["train_scale"][:, None]
        )
        null_test = (
            prepared["fitted_test"]
            + prepared["student_test"][p_test[draw]] * prepared["test_scale"][:, None]
        )
        full_sse, _coef = CV.fit_and_directionwise_sse(z_train, null_train, z_test, null_test)
        reduced_sse, _coef = CV.fit_and_directionwise_sse(
            z_train[:, keep], null_train, z_test[:, keep], null_test
        )
        delta = reduced_sse - full_sse
        explicit.append(
            [
                delta.sum(),
                delta[:3].sum(),
                delta[3:].sum(),
                delta.sum() / reduced_sse.sum(),
                delta[:3].sum() / reduced_sse[:3].sum(),
                delta[3:].sum() / reduced_sse[3:].sum(),
            ]
        )
    np.testing.assert_allclose(subspaces, explicit, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(subspaces[:, 0], subspaces[:, 1] + subspaces[:, 2])
    diagnostics = CV.residual_variance_diagnostics(prepared, train_blocks, test_blocks)
    assert diagnostics["gate_pass"]
    assert diagnostics["train"]["eligible_block_count"] == 6
    assert diagnostics["test"]["eligible_block_count"] == 4


def test_capture_schema_rejects_wrong_layer_and_shape() -> None:
    """The extraction seam fails loudly on layer and tensor-shape mismatches."""

    bundle = {
        "cx_last": torch.zeros((3, 3, CV.D_MODEL)),
        "ci": [1, 2, 3],
        "depth": [2, 2, 3],
        "corpus": ["a", "a", "b"],
        "prompts": ["[]", "[]", "[]"],
        "layers": [14, 19, 26],
    }
    ci, index = CV._validate_bundle(bundle, "fake.pt")
    assert ci == [1, 2, 3]
    assert index == 1
    bad_layer = {**bundle, "layers": [14, 26, 30]}
    with pytest.raises(RuntimeError, match="occurs 0 times"):
        CV._validate_bundle(bad_layer, "fake.pt")
    bad_shape = {**bundle, "cx_last": torch.zeros((3, 2, CV.D_MODEL))}
    with pytest.raises(RuntimeError, match="shape"):
        CV._validate_bundle(bad_shape, "fake.pt")


def test_verdict_lattice_short_circuits_and_covers_all_branches() -> None:
    """Abort, interpretability, support, refutation, and residual branches are ordered."""

    axes = {axis: {"recoverability_pass": True, "mapped_gain_pass": True} for axis in CV.AXES}
    support = {
        "delta_kappa": 1.0,
        "delta_gain": 1.0,
        "delta_kappa_ci": [0.1, 2.0],
        "delta_gain_ci": [0.2, 2.0],
        "individual_kappa": {axis: 1.0 for axis in CV.AXES[1:]},
        "individual_gain": {axis: 1.0 for axis in CV.AXES[1:]},
    }
    standardized = {"delta_kappa": 1.0, "delta_gain": 1.0, "interpretability_pass": True}
    assert CV.evaluate_verdict(axes, support, standardized, False, True, True)["verdict"] == "abort"
    failed = {key: dict(value) for key, value in axes.items()}
    failed["format"]["recoverability_pass"] = False
    assert (
        CV.evaluate_verdict(failed, support, standardized, True, True, True)["verdict"]
        == "partial/inconclusive"
    )
    assert (
        CV.evaluate_verdict(axes, support, standardized, True, True, True)["verdict"]
        == "strong support"
    )
    reverse = {
        **support,
        "delta_kappa": -1.0,
        "delta_gain": -1.0,
        "delta_kappa_ci": [-2.0, -0.1],
        "individual_kappa": {axis: -1.0 for axis in CV.AXES[1:]},
        "individual_gain": {axis: -1.0 for axis in CV.AXES[1:]},
    }
    assert (
        CV.evaluate_verdict(axes, reverse, standardized, True, True, True)["verdict"] == "refuted"
    )
    mixed = {**support, "delta_kappa_ci": [-0.1, 2.0]}
    assert (
        CV.evaluate_verdict(axes, mixed, standardized, True, True, True)["verdict"]
        == "partial/inconclusive"
    )
    invalid_standardized = {**standardized, "interpretability_pass": False}
    assert (
        CV.evaluate_verdict(axes, support, invalid_standardized, True, True, True)["verdict"]
        == "partial/inconclusive"
    )


def test_directional_checkpoint_resume_is_config_keyed(tmp_path: Path) -> None:
    """A completed null resumes identically and a mismatched fingerprint fails loud."""

    deltas = {axis: np.linspace(1.0, 2.0, CV.D_MODEL) for axis in CV.AXES}
    cfg = CV.ValidationConfig(production=False, directional_draws=300)
    first = CV._directional_null(cfg, deltas, tmp_path, 300, "config-a")
    second = CV._directional_null(cfg, deltas, tmp_path, 300, "config-a")
    np.testing.assert_array_equal(first, second)
    with pytest.raises(RuntimeError, match="config mismatch"):
        CV._directional_null(cfg, deltas, tmp_path, 300, "config-b")


def test_real_figure_draws_interval_independently_of_point(tmp_path: Path) -> None:
    """The real renderer preserves a CI even when its point lies outside the interval."""

    result = {
        "config_sha256": "a" * 64,
        "provenance": {"git_commit": "deadbeef", "git_dirty": False},
        "operator": {"natural_context_kernel_share": 0.45},
        "axes": {},
    }
    for index, axis in enumerate(CV.AXES):
        result["axes"][axis] = {
            "kappa_99": 0.5 + 0.02 * index,
            "tau": 0.1 + 0.03 * index,
            "bootstrap_ci": {
                "kappa_99": [0.60, 0.70] if index == 0 else [0.4, 0.8],
                "tau": [0.05, 0.3],
            },
        }
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "category_kernel_validation_L19.json").write_text(json.dumps(result))
    metadata = CV.render_figure(result, tmp_path)
    assert CV.interval_segment([0.60, 0.70]) == (0.60, 0.70)
    assert metadata["xscale_panel_b"] == "log"
    assert (
        tmp_path / "git_payload/figures/issue_2569/leg13_category_kernel_validation.pdf"
    ).is_file()


def test_standardized_ratios_fail_gate_when_denominator_is_not_positive() -> None:
    """Standardized contrast signs cannot rescue an uninterpretable ratio denominator."""

    point = {}
    bootstrap = {}
    for axis in CV.AXES:
        point[axis] = {
            "delta_ss": 1.0,
            "mapped_delta_ss": 0.5,
            "kappa_99": 0.4,
            "tau": 0.2,
        }
        bootstrap[axis] = {
            "delta_ss": np.ones(100),
            "mapped_delta_ss": np.full(100, 0.5),
            "incremental_r2": np.full(100, 0.1),
            "tau": np.full(100, 0.2),
            "kappa_99": np.full(100, 0.4),
        }
    bootstrap["format"]["delta_ss"][:4] = -1.0
    summary = CV._standardized_robustness(point, bootstrap)
    assert not summary["axes"]["format"]["denominator_pass"]
    assert not summary["interpretability_pass"]


def test_design_and_planted_kernel_retained_pattern() -> None:
    """A planted topic-kernel/language-retained example has the intended contrast signs."""

    rows = _small_rows()
    # Avoid the production n<50 language collapse in this small fixture.
    for i, row in enumerate(rows):
        row["language"] = ("en", "zh")[i % 2]
    info = CV.build_design(rows)
    n = len(rows)
    train = np.arange(n) < 100
    test = ~train
    topic_column = info["matrix"][:, info["target_columns"]["topic"][-1]]
    language_column = info["matrix"][:, info["target_columns"]["language"][-1]]
    outcomes = np.column_stack([language_column, np.zeros(n), np.zeros(n), topic_column])
    fit_topic = CV.nested_improvement(
        info["matrix"], outcomes, train, test, info["target_columns"]["topic"]
    )
    fit_language = CV.nested_improvement(
        info["matrix"], outcomes, train, test, info["target_columns"]["language"]
    )
    # Last coordinate is the synthetic kernel, first three are retained.
    assert fit_topic["delta"][3] > fit_topic["delta"][:3].sum()
    assert fit_language["delta"][0] > fit_language["delta"][1:].sum()


def test_analysis_rejects_script_changed_since_extraction(tmp_path: Path) -> None:
    """Checkpoint reuse is impossible after the extracted-script fingerprint changes."""

    compact = tmp_path / "compact"
    compact.mkdir()
    (compact / "config_manifest.json").write_text(
        json.dumps({"config_sha256": "same", "script_sha256": "0" * 64})
    )
    (compact / "extraction_manifest.json").write_text(
        json.dumps({"config": {"config_sha256": "same"}})
    )
    cfg = CV.ValidationConfig(production=False, out_root=str(tmp_path))
    with pytest.raises(RuntimeError, match="different script SHA"):
        CV.analyze_compact(cfg, MODULE_PATH.parents[1], tmp_path)


def test_production_and_partial_completion_gates(tmp_path: Path, monkeypatch) -> None:
    """Production cannot skip persistence, and partial extraction never looks complete."""

    with pytest.raises(ValueError, match="upload=true"):
        CV.run(
            CV.ValidationConfig(
                phase="extract",
                out_root=str(tmp_path / "no-upload"),
                upload=False,
                production=True,
            )
        )
    with pytest.raises(ValueError, match="max_shards=0"):
        CV.run(
            CV.ValidationConfig(
                phase="extract",
                out_root=str(tmp_path / "partial-production"),
                upload=True,
                production=True,
                max_shards=1,
            )
        )

    monkeypatch.setattr(
        CV,
        "extract_compact",
        lambda *_args: {"status": "partial-extraction", "shards": 1},
    )
    partial_root = tmp_path / "partial-smoke"
    CV.run(
        CV.ValidationConfig(
            phase="extract", out_root=str(partial_root), upload=False, production=False
        )
    )
    assert (partial_root / "PARTIAL_EXTRACTION.json").is_file()
    assert not (partial_root / "phase_extract.done.json").exists()
    assert not (partial_root / "RUN_COMPLETE.json").exists()


def test_compact_upload_verification_uses_recorded_immutable_revision(
    tmp_path: Path, monkeypatch
) -> None:
    """Analysis rechecks every compact byte against the recorded immutable HF commit."""

    compact = tmp_path / "compact"
    compact.mkdir()
    config_sha = "a" * 64
    for relpath in CV.COMPACT_RELPATHS:
        (compact / relpath).write_text("{}\n")
    (compact / "config_manifest.json").write_text(json.dumps({"config_sha256": config_sha}))
    destination = {
        "repo_id": "example/data",
        "prefix": f"issue2569_theory/category_validation/{config_sha[:12]}",
        "revision": "b" * 40,
    }
    (tmp_path / "upload_destination.json").write_text(json.dumps(destination))
    observed = {}

    def fake_verify(repo_id, prefix, local_root, relpaths, **kwargs):
        observed.update(
            repo_id=repo_id,
            prefix=prefix,
            local_root=local_root,
            relpaths=relpaths,
            revision=kwargs["revision"],
        )
        return []

    monkeypatch.setattr(CV, "_remote_file_hashes", fake_verify)
    assert CV.verify_compact_upload(tmp_path) == destination
    assert observed["revision"] == "b" * 40
    assert observed["relpaths"] == CV.COMPACT_RELPATHS

    destination["prefix"] = "issue2569_theory/category_validation/stale-config"
    (tmp_path / "upload_destination.json").write_text(json.dumps(destination))
    monkeypatch.setattr(CV, "load_verified_analysis_result", lambda *_args: {})
    with pytest.raises(RuntimeError, match="prefix does not match"):
        CV.upload_analysis(CV.ValidationConfig(production=True), tmp_path)


def test_resumed_analysis_consumer_rejects_wrong_config_or_script(tmp_path: Path) -> None:
    """Standalone render/upload phases cannot consume a stale analysis artifact."""

    compact = tmp_path / "compact"
    analysis = tmp_path / "analysis"
    compact.mkdir()
    analysis.mkdir()
    matrix = compact / "activations_L19.npy"
    rows = compact / "rows.json"
    ledger = compact / "shard_ledger.jsonl"
    matrix.write_bytes(b"matrix")
    rows.write_text("[]\n")
    ledger.write_text("{}\n")
    config = {
        "config_sha256": "a" * 64,
        "script_sha256": CV.sha256_file(MODULE_PATH.resolve()),
    }
    (compact / "config_manifest.json").write_text(json.dumps(config))
    manifest = {
        "config": config,
        "matrix": {"sha256": CV.sha256_file(matrix)},
        "rows_sha256": CV.sha256_file(rows),
        "ledger_sha256": CV.sha256_file(ledger),
    }
    (compact / "extraction_manifest.json").write_text(json.dumps(manifest))
    result = {
        "schema": CV.ANALYSIS_SCHEMA,
        "config_sha256": "b" * 64,
        "provenance": {"script_sha256": config["script_sha256"]},
    }
    result_path = analysis / "category_kernel_validation_L19.json"
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="result config mismatch"):
        CV.load_verified_analysis_result(tmp_path)
    result["config_sha256"] = config["config_sha256"]
    result["provenance"]["script_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="result script SHA mismatch"):
        CV.load_verified_analysis_result(tmp_path)

    result["provenance"]["script_sha256"] = config["script_sha256"]
    archive_path = analysis / "resampling_arrays.npz"
    np.savez(archive_path, _config_sha256=np.asarray(["b" * 64]), draws=np.arange(3))
    result["resampling_arrays"] = {
        "path": archive_path.name,
        "sha256": "f" * 64,
        "size": archive_path.stat().st_size,
        "config_sha256_key": "_config_sha256",
        "array_keys": ["_config_sha256", "draws"],
    }
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="archive failed size/SHA binding"):
        CV.load_verified_analysis_result(tmp_path)

    result["resampling_arrays"]["sha256"] = CV.sha256_file(archive_path)
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="archive config mismatch"):
        CV.load_verified_analysis_result(tmp_path)


def test_bootstrap_and_association_checkpoints_resume(tmp_path: Path) -> None:
    """Long resampling batteries resume exactly and reject another config fingerprint."""

    rng = np.random.default_rng(31)
    n, p, d = 72, 5, 6
    design = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    outcomes = rng.normal(size=(n, d)) + design @ rng.normal(size=(p, d))
    train = np.arange(n) < 48
    test = ~train
    corpus = np.asarray(["a" if index % 2 else "b" for index in range(n)])
    blocks = np.asarray([f"{corpus[index]}|{index % 3}" for index in range(n)])
    target_columns = {axis: [index + 1] for index, axis in enumerate(CV.AXES)}
    info = {"matrix": design, "target_columns": target_columns}
    cfg = CV.ValidationConfig(production=False, bootstrap_draws=8, permutation_draws=4)
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    kwargs = dict(
        cfg=cfg,
        design_info=info,
        outcome_systems={"raw": outcomes},
        singular_systems={"raw": (np.linspace(2.0, 0.5, d), {"99": 3})},
        train=train,
        test=test,
        corpus=corpus,
        checkpoint_dir=bootstrap_dir,
        draws=8,
        config_sha256="config-a",
    )
    first_bootstrap = CV._bootstrap(**kwargs)
    second_bootstrap = CV._bootstrap(**kwargs)
    for axis in CV.AXES:
        for metric in first_bootstrap["raw"][axis]:
            np.testing.assert_array_equal(
                first_bootstrap["raw"][axis][metric], second_bootstrap["raw"][axis][metric]
            )
    with pytest.raises(RuntimeError, match="config mismatch"):
        CV._bootstrap(**{**kwargs, "config_sha256": "config-b"})

    association_dir = tmp_path / "association"
    association_dir.mkdir()
    association_args = dict(
        cfg=cfg,
        design_info=info,
        outcomes=outcomes,
        train=train,
        test=test,
        blocks=blocks,
        retained=3,
        checkpoint_dir=association_dir,
        draws=4,
        config_sha256="config-a",
    )
    first_null, first_diag = CV._association_null(**association_args)
    second_null, second_diag = CV._association_null(**association_args)
    for name in first_null:
        np.testing.assert_array_equal(first_null[name], second_null[name])
        assert first_null[name].shape == (4, len(CV.AXES))
    assert first_diag == second_diag
    with pytest.raises(RuntimeError, match="config mismatch"):
        CV._association_null(**{**association_args, "config_sha256": "config-b"})


def test_compact_ledger_resumes_without_reprocessing_completed_shard(
    tmp_path: Path, monkeypatch
) -> None:
    """An interrupted streaming extraction resumes from its durable shard ledger."""

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    records = []
    for shard_index, cis in enumerate(([0, 1, 2], [3, 4, 5])):
        path = source_dir / f"shard{shard_index}.pt"
        cx = torch.zeros((3, 3, CV.D_MODEL), dtype=torch.float32)
        for row, ci in enumerate(cis):
            cx[row, 1] = float(ci + 1)
        torch.save(
            {
                "cx_last": cx,
                "ci": cis,
                "depth": [2, 3, 4],
                "corpus": ["lmsys", "wildchat", "lmsys"],
                "prompts": ["p0", "p1", "p2"],
                "layers": [14, 19, 26],
            },
            path,
        )
        records.append(
            {
                "path": f"capture/{path.name}",
                "size": path.stat().st_size,
                "sha256": CV.sha256_file(path),
            }
        )
    frozen = tmp_path / "frozen-fixture"
    frozen.mkdir()
    split = frozen / "split.json"
    split.write_text(json.dumps({"sets": {"holdout": {"ci": list(range(7))}}}))
    labels_path = frozen / "labels.json"
    labels_path.write_text("{}")
    kernel = frozen / "kernel.json"
    kernel.write_text(
        json.dumps(
            {
                "k99": CV.RETAINED99_EXPECTED,
                "kernel_dim": CV.KERNEL_EXPECTED,
                "tau_kernel": CV.TAU_EXPECTED,
            }
        )
    )
    pilot = frozen / "pilot.json"
    pilot.write_text("{}")
    ridge = frozen / "ridge.pt"
    ridge.write_text("unused")
    fake_labels = {
        ci: {
            "topic": "coding",
            "language": "en",
            "format": "prose",
            "request_refusal_adjacent": "no",
            "answer_is_refusal": "no",
        }
        for ci in range(5)
    }
    monkeypatch.setattr(CV, "N_SELECTED_HOLDOUT", 7)
    monkeypatch.setattr(CV, "N_HOLDOUT", 6)
    monkeypatch.setattr(CV, "N_LABELED", 5)
    monkeypatch.setattr(
        CV,
        "_stage_small_inputs",
        lambda *_args: {
            "split": split,
            "pilot": pilot,
            "ridge": ridge,
            "labels": labels_path,
            "kernel": kernel,
        },
    )
    monkeypatch.setattr(CV, "_producer_compatibility", lambda *_args: {"status": "fixture"})
    listing = {"entry_count": 3, "total_bytes": sum(r["size"] for r in records), "shards": records}
    monkeypatch.setattr(CV, "_capture_entries", lambda: (records, listing))
    monkeypatch.setattr(
        CV,
        "_load_labels",
        lambda *_args: (fake_labels, {"test_retest_kappa": {"topic": 1.0}}),
    )
    staged_paths = []

    def fake_stage(_repo, remote_path, destination, **_kwargs):
        shard = source_dir / Path(remote_path).name
        CV.shutil.copy2(shard, destination)
        staged_paths.append(remote_path)
        return destination

    monkeypatch.setattr(CV.hub, "stage_hub_file", fake_stage)
    out_root = tmp_path / "out"
    out_root.mkdir()
    partial = CV.extract_compact(
        CV.ValidationConfig(production=False, max_shards=1), MODULE_PATH.parents[1], out_root
    )
    assert partial["status"] == "partial-extraction"
    assert staged_paths == [records[0]["path"]]
    complete = CV.extract_compact(
        CV.ValidationConfig(production=False, max_shards=0), MODULE_PATH.parents[1], out_root
    )
    assert complete["coverage"] == {"holdout": 6, "labeled": 5}
    assert staged_paths == [records[0]["path"], records[1]["path"]]
    matrix = np.load(out_root / "compact/activations_L19.npy")
    np.testing.assert_array_equal(matrix[:, 0], np.arange(1, 6, dtype=np.float32))
    missing = json.loads((out_root / "compact/missingness_audit.json").read_text())
    assert missing["missing_ids"] == [5]


def test_pinned_producer_compatibility_uses_exact_last_token_expression() -> None:
    report = CV._producer_compatibility(MODULE_PATH.parents[1])

    assert report["script_sha256"] == CV.PRODUCER_SCRIPT_SHA256
    assert report["context_read"] == "final prompt token hs[-1, :]"
    assert "cx.append(hs[-1, :].float().cpu())" in report["required_fragments"]
