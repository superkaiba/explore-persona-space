"""Exact-mask probe joins and vectorized paired-rank comparisons."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("issue2669_compare", SCRIPTS / "issue2669_compare.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_json(path, value):
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(tmp_path):
    root = tmp_path / "run"
    (root / "analysis_production").mkdir(parents=True)
    cohort, codex, probes, files = [], [], [], []
    for behavior in sorted(m.BEHAVIORS):
        directory = tmp_path / behavior
        directory.mkdir()
        records, labels, wc_labels = {r: [] for r in ("id", "generic", "ood")}, [], []
        for regime in ("id", "generic", "ood"):
            for i in range(100):
                cid = f"{behavior}-{regime}-{i}"
                instrument = (
                    "fabrication"
                    if behavior == "hallucination" and regime != "generic"
                    else "trait"
                )
                scale = 100 if instrument == "fabrication" else 1
                rung = "rung_a" if regime != "ood" or i < 60 else "rung_b"
                row = {
                    "id": cid,
                    "context_id": cid,
                    "behavior": behavior,
                    "regime": regime,
                    "rung": rung,
                    "group_key": f"group{i // 2}",
                    "fold_id": i // 2 % 5 if regime == "id" else None,
                    "dv": float(i),
                    "dv_original": i / scale,
                    "dv_scale": scale,
                }
                cohort.append(row)
                codex.append(
                    {
                        **row,
                        "instrument": instrument,
                        "forecast_0": float(99 - i),
                        "forecast_32": float(i),
                    }
                )
                label = {
                    "context_id": cid,
                    "split": "train" if regime == "id" else "eval",
                    "rung": rung,
                    "group_key": row["group_key"],
                    "dv": i / scale,
                }
                (wc_labels if regime == "generic" else labels).append(label)
                records[regime].append(
                    {
                        "context_id": cid,
                        "rung": "train" if regime == "id" else rung,
                        "dv": i / scale,
                        "scores": {"regression_ctx": float(i), "reg_map_linear": float(i)},
                    }
                )
        labelpath, wcpath = directory / "labels.json", directory / "wildchat_labels.json"
        write_json(labelpath, {"rows": labels})
        write_json(wcpath, {"rows": wc_labels})
        recipe = {
            "behavior": behavior,
            "layer": 18,
            "methods": ["regression_ctx", "reg_map_linear"],
            "dv_path": str(labelpath),
            "wildchat_dv_path": str(wcpath),
            "parity_abs_tolerance": 1e-6,
        }
        write_json(
            directory / "result.json",
            {
                "parity_pass": True,
                "config": recipe,
                "metrics": [
                    {"method": method, "rho": 0.5, "reference_rho": 0.5, "absolute_delta": 0}
                    for method in recipe["methods"]
                ],
            },
        )
        for regime, rows in records.items():
            write_rows(directory / f"{regime}.jsonl", rows)
        files.extend(directory.iterdir())
        probes.append(
            {
                "root": str(directory),
                "behavior": behavior,
                "layer": 18,
                "methods": recipe["methods"],
            }
        )
    write_json(root / "selection.json", {"selected_ids": [r["id"] for r in cohort]})
    write_rows(root / "cohort.jsonl", cohort)
    write_rows(root / "analysis_production/percontextforecast.jsonl", codex)
    write_json(root / "analysis_production/metrics.json", {"n_contexts": 900})
    files.extend(
        [
            root / "selection.json",
            root / "cohort.jsonl",
            root / "analysis_production/percontextforecast.jsonl",
            root / "analysis_production/metrics.json",
        ]
    )
    config = {
        "run_root": str(root),
        "output_root": str(tmp_path / "comparison"),
        "probes": probes,
        "source_sha256": {str(p): m.hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "bootstrap_draws": 30,
    }
    return config


def rehash(config, path):
    config["source_sha256"][str(path)] = m.hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_run_joins_exact900_and_reports_optional_oracle(tmp_path):
    config = fixture(tmp_path)
    path = tmp_path / "config.json"
    write_json(path, config)
    report = m.run(path)
    assert report["complete"] and report["n_contexts"] == 900
    assert all(cell["oracle_absent"] for cell in report["coverage"].values())
    cell = report["metrics"]["hallucination/fabrication/id"]
    assert cell["spearman"]["regression_ctx"] == pytest.approx(1)
    assert cell["paired_spearman_differences"]["regression_ctx-minus-codex_0"][
        "point"
    ] == pytest.approx(2)
    assert cell["paired_spearman_differences"]["reg_map_linear-minus-codex_32"]["ci95"] == [0, 0]
    assert len(m.read_rows(Path(config["output_root"]) / "percontextjoined.jsonl")) == 900
    assert "mae" not in cell


def test_hash_tampering_fails_before_join(tmp_path):
    config = fixture(tmp_path)
    path = Path(config["probes"][0]["root"]) / "id.jsonl"
    with path.open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        m.load_join(config)


@pytest.mark.parametrize(
    "failure", ["parity", "missing", "group", "scale", "duplicate_method", "oracle_required"]
)
def test_semantic_failures_never_default_or_use_aggregate(tmp_path, failure):
    config = fixture(tmp_path)
    directory = Path(config["probes"][0]["root"])
    if failure == "oracle_required":
        config["require_oracle"] = True
    elif failure == "duplicate_method":
        config["probes"].append(config["probes"][0])
    elif failure == "parity":
        path = directory / "result.json"
        value = json.loads(path.read_text())
        value["parity_pass"] = False
        write_json(path, value)
        rehash(config, path)
    elif failure in {"missing", "scale"}:
        path = directory / "id.jsonl"
        rows = m.read_rows(path)
        if failure == "missing":
            rows.pop()
        else:
            rows[0]["dv"] = 0.123
        write_rows(path, rows)
        rehash(config, path)
    else:
        path = directory / "labels.json"
        value = json.loads(path.read_text())
        value["rows"][0]["group_key"] = "wrong"
        write_json(path, value)
        rehash(config, path)
    with pytest.raises(ValueError):
        m.load_join(config)


def test_same_cluster_indices_and_reranked_delta_matches_scalar_oracle():
    y = np.array([0.0, 0.0, 3.0, 5.0, 9.0, 2.0])
    a = np.array([0.0, 2.0, 2.0, 6.0, 8.0, 1.0])
    b = np.array([1.0, 0.0, 3.0, 5.0, 8.0, 2.0])
    rows = [
        {
            "dv": y[i],
            "rung": "a",
            "group_key": str(i // 2),
            "regime": "ood",
            "scores": {
                "codex_0": a[i],
                "codex_32": a[i],
                "regression_ctx": b[i],
                "reg_map_linear": b[i],
            },
        }
        for i in range(6)
    ]
    indices = m.grouped_indices(rows, 100, 8)
    expected = []
    for index in indices:
        index = index[index >= 0]
        if any(np.ptp(vector[index]) == 0 for vector in (y, a, b)):
            expected.append(np.nan)
            continue
        expected.append(
            spearmanr(y[index], b[index]).statistic - spearmanr(y[index], a[index]).statistic
        )
    report = m.compare_group(rows, 100, 8)
    cell = report["paired_spearman_differences"]["regression_ctx-minus-codex_0"]
    if np.isfinite(expected).all():
        np.testing.assert_allclose(cell["ci95"], np.quantile(expected, [0.025, 0.975]), atol=1e-12)
    else:
        assert cell["ci95"] is None
    assert cell["valid_draws"] == int(np.isfinite(expected).sum())
