#!/usr/bin/env python3
"""Exploratory graph distances on the eight saved story-imprinting personas."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from analyze_marker_geodesic import (  # noqa: E402
    coordinates_from_gram,
    cosine_from_gram,
    digest,
    fit_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval_results/story_geodesic_20261002"


def association(x: np.ndarray, y: np.ndarray) -> dict:
    """Return descriptive correlations and leave-one-character-out influence."""
    assert x.shape == y.shape == (5,) and np.isfinite([x, y]).all()
    assert np.ptp(x) > 0 and np.ptp(y) > 0
    loo = []
    for i in range(5):
        mask = np.arange(5) != i
        assert np.ptp(x[mask]) > 0 and np.ptp(y[mask]) > 0
        loo.append(float(spearmanr(x[mask], y[mask]).statistic))
    return {
        "n": 5,
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
        "leave_one_character_out_rho": loo,
    }


def evaluate(scores: np.ndarray, names: list[str], rates: dict) -> dict:
    """Compare direct uptake first; preserve the distinct preference contrast."""
    result = {}
    assert scores.shape == (8, 8) and np.isfinite(scores).all()
    alternatives = names[3:]
    for persona in ("hhh", "fred"):
        row = scores[names.index(persona)]
        x = row[[names.index(a) for a in alternatives]]
        y = np.array([rates[persona][a]["other"]["rate"] for a in alternatives])
        helpful_y = np.array([rates[persona][a]["helpful"]["rate"] for a in alternatives])
        delta_x, delta_y = row[names.index("helpful")] - x, helpful_y - y
        result[persona] = {
            "direct": association(x, y),
            "contrast": association(delta_x, delta_y),
            "contrast_sign_matches": int(np.sum(np.sign(delta_x) == np.sign(delta_y))),
            "alternatives": alternatives,
            "scores": x.tolist(),
            "rates": y.tolist(),
            "digitization_bounds": [
                rates[persona][a]["other"]["digitization_bound"] for a in alternatives
            ],
        }
    return result


def tiny_bank_dimension(x: np.ndarray) -> dict:
    """Show geometry-only estimates without pretending twenty neighbors exist."""
    u, singular, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    d = int(np.searchsorted(np.cumsum(singular**2) / sum(singular**2), 0.99) + 1)
    radii = cdist(u[:, :d] * singular[:d], u[:, :d] * singular[:d])
    np.fill_diagonal(radii, np.inf)
    radii.sort(axis=1)
    return {
        "pca_dimensions": d,
        "estimates": {
            str(k): float(1 / np.mean(np.log(radii[:, k - 1, None] / radii[:, : k - 1])))
            for k in (3, 5, 7)
        },
        "used_for_parameter_selection": False,
    }


def main() -> None:
    """Run each frozen cell, persist layer results, then assemble the summary."""
    start = time.monotonic()
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert digest(OUT / "inputs.json") == protocol["inputs_sha256"]
    data = json.loads((OUT / "inputs.json").read_text())
    summary = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "inputs_sha256": digest(OUT / "inputs.json"),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "original_marker_protocol_status": "infeasible_eight_persona_bank",
        "models": {},
    }
    with threadpool_limits(limits=2):
        for model, bank in data["models"].items():
            records = {}
            for layer in bank["layers"]:
                saved = bank["data_by_layer"][str(layer)]
                x = coordinates_from_gram(np.asarray(saved["raw_gram"]))
                raw = cosine_from_gram(x @ x.T)
                np.testing.assert_allclose(raw, saved["raw_cosine"], rtol=0, atol=1e-12)
                centered = x - x.mean(axis=0)
                metrics = {
                    "raw_cosine": raw,
                    "whitened_cosine": np.asarray(saved["whitened_cosine"]),
                    "centered_cosine": cosine_from_gram(centered @ centered.T),
                    "euclidean": -cdist(x, x),
                }
                record = {
                    "dimension_diagnostic": tiny_bank_dimension(x),
                    "baselines": {
                        m: evaluate(v, bank["names"], data["rates"]) for m, v in metrics.items()
                    },
                    "graphs": {},
                }
                for metric, oldkey in [("raw_cosine", "raw"), ("whitened_cosine", "whitened")]:
                    for persona in ("hhh", "fred"):
                        old = saved["archived_associations"][oldkey][persona][
                            "secondary_other_uptake"
                        ]
                        new = record["baselines"][metric][persona]["direct"]
                        np.testing.assert_allclose(
                            [new["pearson_r"], new["spearman_rho"]],
                            [old["pearson_r"], old["spearman_rho"]],
                            atol=1e-11,
                            rtol=0,
                        )
                settings = [(s["k"], s["d"], 1e-4) for s in protocol["sensitivity_settings"]]
                settings += [(5, 1, ridge) for ridge in protocol["ridge_sensitivity"]]
                for k, d, ridge in settings:
                    scores, diagnostics = fit_geometry(x, k, d, ridge)
                    key = f"k{k}_d{d}_ridge{ridge:g}"
                    record["graphs"][key] = {
                        "diagnostics": diagnostics,
                        "metrics": {
                            m: evaluate(v, bank["names"], data["rates"]) for m, v in scores.items()
                        },
                        "distance_matrices": {m: (-v).tolist() for m, v in scores.items()},
                    }
                records[str(layer)] = record
                (OUT / f"{model}_block_{layer}.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                print(model, layer, "complete", flush=True)
            summary["models"][model] = records
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["completed_at_utc"] = datetime.now(UTC).isoformat()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("completed", summary["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
