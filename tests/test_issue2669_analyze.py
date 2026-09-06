"""Audit integrity and cluster-bootstrap/statistic correctness at analysis boundaries."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("issue2669_analyze", SCRIPTS / "issue2669_analyze.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_brier_is_per_answer_not_mean_rate_mse():
    metrics = m.point_metrics(np.array([40.0]), np.array([20.0]), True)
    assert metrics["mean_rate_mse"] == pytest.approx(0.04)
    assert metrics["per_answer_brier"] == pytest.approx(0.28)
    assert metrics["spearman"] is None
    assert "per_answer_brier" not in m.point_metrics(np.array([40.0]), np.array([20.0]), False)


def test_cluster_bootstrap_reranks_ties_and_preserves_groups():
    rows = [{"rung": "a", "group_key": group} for group in ("g1", "g1", "g2", "g3", "g3")]
    rows += [{"rung": "b", "group_key": "g1"}]
    y = np.array([0.0, 0.0, 3.0, 5.0, 9.0, 2.0])
    z = np.array([0.0, 2.0, 2.0, 6.0, 8.0, 1.0])
    f = np.array([1.0, 0.0, 3.0, 5.0, 8.0, 2.0])
    indices = m.grouped_indices(rows, 100, 9)
    assert np.all((indices == 0).sum(axis=1) == (indices == 1).sum(axis=1))
    assert np.all((indices == 3).sum(axis=1) == (indices == 4).sum(axis=1))
    assert np.all((indices == 5).sum(axis=1) == 1)
    valid = indices >= 0
    yy, zz = (np.where(valid, vector[np.maximum(indices, 0)], np.nan) for vector in (y, z))
    actual = m.rowwise_rho(yy, zz)
    expected = np.array(
        [spearmanr(y[idx[idx >= 0]], z[idx[idx >= 0]]).statistic for idx in indices]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)
    result = m.paired_bootstrap(rows, y, z, f, draws=100, seed=9)
    assert result["n_groups"] == 4
    assert result["mae_32_minus_0"]["valid_draws"] == 100


def test_constant_spearman_interval_is_null_not_zero():
    rows = [{"rung": "a", "group_key": str(i)} for i in range(4)]
    report = m.paired_bootstrap(rows, np.ones(4), np.arange(4.0), np.arange(4.0), draws=100)
    assert report["spearman_32_minus_0"]["ci95"] is None
    assert report["mae_32_minus_0"]["ci95"] == [0, 0]


def make_completed_packet(root, score=20):
    packet = {
        "id": "p",
        "ids": ["x"],
        "shot": 0,
        "repeat": 0,
        "behavior": "evil",
        "regime": "id",
        "instrument": "trait",
        "prompt_path": str(root / "prompt.txt"),
    }
    (root / "prompt.txt").write_text("Context.")
    identity = {
        "protocol": m.dispatch.PROTOCOL,
        "model": m.dispatch.MODEL,
        "effort": m.dispatch.EFFORT,
        "cli_version": "codex-test",
        "prompt_sha256": m.dispatch.digest((m.dispatch.NO_TOOLS + "Context.").encode()),
        "ids": ["x"],
        "behavior": "evil",
        "schema": m.dispatch.SCHEMA,
    }
    fingerprint = m.dispatch.digest(json.dumps(identity, sort_keys=True).encode())
    directory = root / "judgments_production/p"
    attempt = directory / "attempt-001"
    attempt.mkdir(parents=True)
    value = {"rows": [{"id": "x", "rationale": "Forecast.", "score_0_100": score}]}
    (attempt / "final.json").write_text(json.dumps(value))
    (attempt / "events.jsonl").write_text(
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(value)}}
        )
        + "\n"
        + json.dumps({"type": "turn.completed"})
        + "\n"
    )
    metadata = {
        "fingerprint": fingerprint,
        "status": "complete",
        "returncode": 0,
        "attempt": "attempt-001",
        "artifact_sha256": {
            name: m.dispatch.digest((attempt / name).read_bytes())
            for name in ("events.jsonl", "final.json")
        },
        "started_unix": 1,
        "finished_unix": 2,
        "wall_seconds": 1,
    }
    for path, obj in (
        (directory / "identity.json", identity),
        (directory / "status.json", metadata),
        (attempt / "metadata.json", metadata),
    ):
        path.write_text(json.dumps(obj))
    # A second independent condition with identical validated raw response.
    import shutil

    shutil.copytree(directory, root / "judgments_production/q")
    config = {
        "output_dir": str(root / "judgments_production"),
        "packets": [packet, {**packet, "id": "q", "shot": 32}],
    }
    (root / "production_config.json").write_text(json.dumps(config))
    (root / "selection.json").write_text(json.dumps({"selected_ids": ["x"]}))
    return directory


def test_collect_validation_rejects_incomplete_and_hash_tampering(tmp_path):
    directory = make_completed_packet(tmp_path)
    forecasts, audit = m.collect(tmp_path, "production")
    assert len(forecasts) == 2 and audit["packets"] == 2
    metadata = json.loads((directory / "status.json").read_text())
    metadata["status"] = "transport_loss"
    (directory / "status.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incomplete"):
        m.collect(tmp_path, "production")
    metadata["status"] = "complete"
    (directory / "status.json").write_text(json.dumps(metadata))
    (directory / "attempt-001/final.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        m.collect(tmp_path, "production")


def test_pilot_does_not_read_gold_labels(tmp_path):
    # No cohort.jsonl exists: diagnostics must need forecasts only.
    forecasts = [
        {
            "id": "x",
            "behavior": "evil",
            "instrument": "trait",
            "shot": 0,
            "repeat": repeat,
            "score_0_100": score,
        }
        for repeat, score in enumerate([10, 20, 30])
    ]
    timing = [
        {"behavior": "evil", "instrument": "trait", "shot": 0, "wall_seconds": 6, "n_contexts": 1}
    ]
    audit = {
        "timing": timing,
        "attempts": [{"started_unix": 1, "finished_unix": 7, "wall_seconds": 6}],
        "packets": 3,
        "forecasts": 3,
        "n_transport_attempts": 0,
    }
    (tmp_path / "production_config.json").write_text(
        json.dumps(
            {
                "concurrency": 3,
                "packets": [
                    {"behavior": "evil", "instrument": "trait", "shot": 0, "ids": ["x", "y"]}
                ],
            }
        )
    )
    result = m.pilot_report(tmp_path, forecasts, audit)
    assert result["gate"] == "pass"
    assert result["production_eta_seconds"] == 4
    assert result["stability"]["evil/trait/0"]["mean_repeat_pair_mae"] == pytest.approx(40 / 3)


def test_production_full_join_instruments_and_pairing(tmp_path):
    cohort, forecasts = [], []
    for behavior in ("evil", "sycophancy", "hallucination"):
        for regime in ("id", "generic", "ood"):
            for i in range(100):
                cid = f"{behavior}-{regime}-{i}"
                instrument = (
                    "fabrication"
                    if behavior == "hallucination" and regime != "generic"
                    else "trait"
                )
                cohort.append(
                    {
                        "id": cid,
                        "behavior": behavior,
                        "regime": regime,
                        "rung": "a" if i < 50 else "b",
                        "group_key": f"g{i // 2}",
                        "fold_id": (i // 2) % 5 if regime == "id" else None,
                        "dv": float(i),
                    }
                )
                for shot in (0, 32):
                    forecasts.append(
                        {
                            "id": cid,
                            "behavior": behavior,
                            "regime": regime,
                            "instrument": instrument,
                            "shot": shot,
                            "repeat": 0,
                            "score_0_100": float(i if shot == 32 else 99 - i),
                        }
                    )
    (tmp_path / "selection.json").write_text(
        json.dumps({"selected_ids": [r["id"] for r in cohort]})
    )
    (tmp_path / "cohort.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cohort))
    joined, report = m.production_report(tmp_path, forecasts, draws=30)
    assert len(joined) == report["n_contexts"] == 900
    metrics = report["metrics"]
    assert metrics["evil/trait/id"]["zero_shot"]["spearman"] == pytest.approx(-1)
    assert metrics["evil/trait/id"]["few_shot_32"]["spearman"] == pytest.approx(1)
    assert "per_answer_brier" in metrics["hallucination/fabrication/ood"]["zero_shot"]
    assert "per_answer_brier" not in metrics["hallucination/trait/generic"]["zero_shot"]
    assert metrics["sycophancy/trait/ood/a"]["zero_shot"]["n"] == 50
    with pytest.raises(ValueError, match="mismatch"):
        m.production_report(tmp_path, forecasts[:-1], draws=30)


def test_id_bootstrap_preserves_each_original_fold_stratum():
    rows = [
        {"rung": "id_corpus", "group_key": group, "regime": "id", "fold_id": fold}
        for group, fold in [("g1", 0), ("g1", 0), ("g2", 0), ("g3", 1), ("g4", 1)]
    ]
    indices = m.grouped_indices(rows, 100, 99)
    # Each draw samples two groups in each original fold. Fold1 singleton
    # groups always contribute exactly two observations; fold0 varies in size.
    assert np.all(((indices == 3) | (indices == 4)).sum(axis=1) == 2)
    assert np.all((indices == 0).sum(axis=1) == (indices == 1).sum(axis=1))
    assert np.all((indices == 0).sum(axis=1) + (indices == 2).sum(axis=1) == 2)
