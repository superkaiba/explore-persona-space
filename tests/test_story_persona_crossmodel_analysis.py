"""Dense numerical oracles and corrupt-capture rejection for the cross-model pilot."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from scripts import story_persona_crossmodel_analysis as analysis


@pytest.fixture
def capture(tmp_path, monkeypatch):
    """Create a small, shuffled BF16 capture solely for numerical unit tests."""
    monkeypatch.setattr(analysis, "EXPECTED_QUESTIONS", 4)
    small_model = {
        **analysis.MODELS["qwen"],
        "layers": 4,
        "hidden_dim": 48,
        "selected_layers": [0, 1, 2, 3],
    }
    monkeypatch.setitem(analysis.MODELS, "qwen", small_model)
    prompts_path = analysis.ROOT / "configs/pilots/story_persona_deepseek_prompts.json"
    prompts = json.loads(prompts_path.read_text())["prompts"]
    # Neither sorted IDs nor parity gives the registered first/last-half split.
    questions = [{"id": q, "question": f"Question {q}?"} for q in (11, 8, 5, 2)]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    rows = []
    for p in prompts:
        for q in questions:
            ids = [1, 100 + q["id"], 9]
            rows.append(
                {
                    "row_id": f"{p['id']}:{q['id']}",
                    "persona": p["id"],
                    "question_id": q["id"],
                    "description": p["system"],
                    "question": q["question"],
                    "messages": [
                        {"role": "system", "content": p["system"]},
                        {"role": "user", "content": q["question"]},
                    ],
                    "rendered_prefix": p["system"] + "\n" + q["question"],
                    "input_ids": ids,
                    "token_count": len(ids),
                    "prefix_sha256": analysis.digest(ids),
                    "final_token_id": ids[-1],
                    "final_token": ":",
                }
            )
    rng = np.random.default_rng(2673)
    rng.shuffle(rows)
    # A substantial common offset ensures centering would fail the oracle.
    values = torch.tensor(rng.normal(size=(32, 4, 48)) + 3, dtype=torch.bfloat16)
    order = rng.permutation(len(rows)).tolist()
    batches = [order[k : k + 7] for k in range(0, len(rows), 7)]
    spec = {
        "model": small_model,
        "prompts": prompts,
        "question_ids": [q["id"] for q in questions],
        "inputs_sha256": analysis.digest(rows),
        "batches": batches,
        "dtype": "bfloat16",
        "layers": list(range(4)),
        "position": "last_generation_prefix_token",
        "final_layer": "pre_final_norm",
        "capture": {"max_context_tokens": 2048},
    }
    fp = analysis.digest(spec)
    manifest = {"fingerprint": fp, "spec": spec, "provenance": {"git_commit": "a" * 40}}
    out = tmp_path / "capture"
    (out / "chunks").mkdir(parents=True)
    hashes = {}
    for k, indices in enumerate(batches):
        path = out / "chunks" / f"batch_{k:04d}.pt"
        torch.save({"fingerprint": fp, "indices": indices, "vectors": values[indices]}, path)
        hashes[path.name] = analysis.file_digest(path)
    done = {
        "fingerprint": fp,
        "row_count": len(rows),
        "chunk_sha256": hashes,
        "attempt_started_at": 1,
        "checked_at": 2,
    }
    smoke = {
        "fingerprint": fp,
        "passed": True,
        "throughput_gate": {"passed": True},
        "indices": [0, 1],
    }
    for name, value in [
        ("manifest", manifest),
        ("capture_complete", done),
        ("rows", rows),
        ("smoke", smoke),
    ]:
        analysis.write_json(out / f"{name}.json", value)
    analysis.write_json(out / "capture_chunks.json", {"fingerprint": fp, "chunk_sha256": hashes})
    analysis.write_json(out / "progress.json", {"fingerprint": fp, "stage": "capture_complete"})
    torch.save(
        {
            "fingerprint": fp,
            "indices": [0, 1],
            "initial": values[:2],
            "repeated": values[:2],
            "batched": values[:2],
        },
        out / "smoke_vectors.pt",
    )
    cfg = OmegaConf.structured(
        analysis.AnalysisConfig(output_dir=str(out), questions_path=str(questions_path))
    )
    return cfg, manifest, done, rows, values.float().numpy()


def test_end_to_end_matches_primal_oracle_and_preserves_disjoint_halves(capture):
    """Use actual persisted chunks; compare every fitted fold to dense primal math."""
    cfg, manifest, _done, rows, vectors = capture
    summary = analysis.analyze(cfg)
    names, qids = summary["persona_names"], manifest["spec"]["question_ids"]
    assert summary["row_count"] == 32 and summary["counts_by_half"] == [[2] * 8] * 2
    assert len(summary["raw_all_layers"]) == 4 and len(summary["results"]) == 4
    with np.load(Path(cfg.output_dir) / "selected_vectors.npz") as saved:
        np.testing.assert_array_equal(saved["vectors"], vectors)
        assert saved["vectors"].dtype == np.float32
        assert saved["row_ids"].tolist() == [r["row_id"] for r in rows]
    means = np.stack(
        [vectors[[r["persona"] == n for r in rows]].astype(np.float64).mean(0) for n in names]
    )
    with np.load(Path(cfg.output_dir) / "centroids.npz") as saved:
        np.testing.assert_array_equal(saved["centroids"], means)
        assert saved["centroids"].dtype == np.float64
    for layer in range(4):
        raw = means[:, layer] @ means[:, layer].T
        raw /= np.sqrt(np.diag(raw))[:, None] * np.sqrt(np.diag(raw))[None, :]
        np.testing.assert_allclose(
            summary["raw_all_layers"][str(layer)]["raw_cosine_matrix"], raw, atol=1e-12
        )
        block = summary["results"][str(layer)]
        for fold in block["fits"].values():
            fit_ids, eval_ids = fold["fit_question_ids"], fold["evaluation_question_ids"]
            fit = vectors[[r["question_id"] in fit_ids for r in rows], layer].astype(np.float64)
            eval_means = np.stack(
                [
                    vectors[
                        [r["persona"] == n and r["question_id"] in eval_ids for r in rows], layer
                    ]
                    .astype(np.float64)
                    .mean(0)
                    for n in names
                ]
            )
            if fold["fit_half"] is not None:
                assert set(fit_ids).isdisjoint(eval_ids)
                assert fit_ids == qids[:2] or fit_ids == qids[2:]
                assert fold["fit_row_count"] == 16
            else:
                assert fold["fit_row_count"] == 32
            second_moment = fit.T @ fit / len(fit)
            largest = np.linalg.eigvalsh(second_moment)[-1]
            np.testing.assert_allclose(
                fold["diagnostics"]["largest_second_moment_eigenvalue"], largest, rtol=1e-12
            )
            admissible = [r for r in fold["diagnostics"]["ridge_grid"] if (largest + r) / r <= 1e4]
            assert fold["ridge"] == admissible[0]
            transformed = np.linalg.solve(
                np.linalg.cholesky(second_moment + fold["ridge"] * np.eye(48)), eval_means.T
            ).T
            gram = transformed @ transformed.T
            expected = gram / np.sqrt(np.diag(gram))[:, None] / np.sqrt(np.diag(gram))[None, :]
            np.testing.assert_allclose(fold["whitened_cosine_matrix"], expected, atol=2e-10)
            np.testing.assert_allclose(
                fold["raw_cosine_matrix"],
                analysis.cosine_from_gram(eval_means @ eval_means.T),
                atol=1e-12,
            )
            assert fold["implicit_primal_relative_residual"] < 1e-8
            assert fold["primal_subproblem_oracle"]["gram_relative_error"] < 1e-8
            for metric in ("raw", "whitened"):
                assert len(fold[metric]["pairs"]) == 10
                assert all(fold[metric]["primary_by_persona"][p]["n"] == 5 for p in ("hhh", "fred"))
    out = Path(cfg.output_dir)
    assert json.loads((out / "capture_progress.json").read_text())["stage"] == "capture_complete"
    sentinel = json.loads((out / "analysis_complete.json").read_text())
    assert sentinel["analysis_fingerprint"] == summary["analysis_fingerprint"]
    for name, record in sentinel["outputs"].items():
        assert record["sha256"] == analysis.file_digest(out / name)
        assert record["bytes"] == (out / name).stat().st_size


@pytest.mark.parametrize(
    "corruption", ["duplicate_cell", "duplicate_batch", "model", "missing_checksum"]
)
def test_rejects_incomplete_or_mismatched_coverage(capture, corruption):
    """Even self-consistent hashes cannot turn the wrong row/model coverage valid."""
    _cfg, manifest, done, rows, _values = capture
    manifest, done, rows = copy.deepcopy((manifest, done, rows))
    if corruption == "duplicate_cell":
        rows[0] = rows[1]
        manifest["spec"]["inputs_sha256"] = analysis.digest(rows)
    elif corruption == "duplicate_batch":
        manifest["spec"]["batches"][0][0] = manifest["spec"]["batches"][0][1]
    elif corruption == "model":
        manifest["spec"]["model"]["revision"] = "b" * 40
    else:
        done["chunk_sha256"].pop(next(iter(done["chunk_sha256"])))
    with pytest.raises(ValueError):
        analysis.validate_layout(manifest, done, rows, "qwen")


def test_rejects_corrupt_chunks_and_invalidates_stale_success(capture):
    """A failed rerun must not leave a previous completion sentinel usable."""
    cfg, _manifest, _done, _rows, _values = capture
    out = Path(cfg.output_dir)
    (out / "analysis_complete.json").write_text('{"stale": true}')
    with (out / "chunks" / "batch_0000.pt").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(RuntimeError, match="chunk content changed"):
        analysis.analyze(cfg)
    assert not (out / "analysis_complete.json").exists()


def test_rejects_failed_numerical_smoke(capture):
    """A complete-looking tensor tree cannot override a failed production smoke."""
    cfg, _manifest, _done, _rows, _values = capture
    path = Path(cfg.output_dir) / "smoke.json"
    smoke = json.loads(path.read_text())
    smoke["passed"] = False
    analysis.write_json(path, smoke)
    with pytest.raises(ValueError, match=r"successful.*numerical smoke"):
        analysis.analyze(cfg)


def test_pairing_signs_and_constant_correlations():
    """Check ten explicit contrast signs without pretending constant predictors correlate."""
    rates = analysis.load_rates(analysis.ROOT / analysis.AnalysisConfig.rates_path)["rates"]
    matrix = np.eye(8)
    matrix[0, 2] = 0.7
    matrix[0, 3:] = [0.1, 0.2, 0.3, 0.4, 0.5]
    matrix[1, 2] = -0.4
    matrix[1, 3:] = [0.3, 0.2, 0.1, 0.0, -0.1]
    result = analysis.paired_outcomes(matrix, analysis.NAMES, rates)
    for p in result["pairs"]:
        row = rates[p["evaluation_persona"]][p["alternative"]]
        expected = row["helpful"]["rate"] - row["other"]["rate"]
        assert p["behavioral_contrast"] == expected
        assert p["sign_agrees"] == (np.sign(p["predictor_contrast"]) == np.sign(expected))
    assert result["primary_by_persona"]["hhh"]["sign_agreement_count"] == 5
    assert result["primary_by_persona"]["fred"]["sign_agreement_count"] == 3
    constant = analysis.association([1, 1, 1], [1, 2, 3])
    assert constant["pearson_r"] is None and constant["spearman_rho"] is None
    assert constant["correlation_status"] == "undefined_constant_input"


def test_hydra_standard_override_contract():
    """The wrapper's phase/output/model arguments must work without plus prefixes."""
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="story_persona_crossmodel_analysis",
            overrides=["phase=analyze", "output_dir=/tmp/future-capture", "model_key=deepseek"],
        )
    assert cfg.phase == "analyze" and cfg.model_key == "deepseek"
    assert cfg.output_dir == "/tmp/future-capture"
