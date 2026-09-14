"""Independent reductions of saved statistics; no training or inference."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/eps-data/thomasjiralerspong/workspace-jr-20260912")
REPO = Path("/mnt/eps-data/thomasjiralerspong/wt-jr-workspace-predictability")
OUT = ROOT / "user_requested_critic_v1"


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)


result = read(REPO / "eval_results/exploratory_workspace_jr/20260912/run_result.json")
selection = read(REPO / "docs/exploratory_workspace_jr/selected_contexts.json")
sets = {k: {v["prompt_sha256"] for v in rows} for k, rows in selection["subsets"].items()}
assert all(len(sets[k]) == len(rows) for k, rows in selection["subsets"].items())
assert all(not sets[a] & sets[b] for a, b in itertools.combinations(sets, 2))
ids = result["context_ids"]
assert len(ids) == len(set(ids)) == 252
assert set(ids) <= sets["main_test"]
for k, path in [
    ("config_sha256", "configs/analysis/workspace_jr.yaml"),
    ("selection_sha256", "docs/exploratory_workspace_jr/selected_contexts.json"),
]:
    assert result[k] == hashlib.sha256((REPO / path).read_bytes()).hexdigest()

bootstrap_digests = set()
checked_metrics = 0
checked_ci = 0
headlines = {}
max_reconstruction = 0.0
quality = {}
for role in ("primary", "comparison"):
    supplement = read(ROOT / f"main_supplement_v2/{role}_supplement.json")
    quality[role] = read(ROOT / f"main_comparison_v1/calibration_quality/{role}.json")["report"][
        "matching"
    ]
    for kind, k, rotation in itertools.product(
        ("observed", "affine_null"), (5, 10, 25), (None, 20260913, 20260914, 20260915)
    ):
        cell = f"{role}/{kind}/k{k}/rotation{rotation}"
        folder = ROOT / "main_comparison_v1/cross_model" / cell.replace("/", "__")
        boot = read(folder / "summary.json")
        assert list(boot["context_ids"]) == ids
        bootstrap_digests.add(boot["counts_sha256"])
        d = read(ROOT / "main_decomposition_summary_v1/cells" / cell / "summary.json")
        assert d["contexts"] == 252 and d["rollouts"] == 1260
        with np.load(folder / "bootstrap_samples.npz", allow_pickle=False) as draws:
            for name in draws.files:
                values = draws[name]
                assert values.shape == (2000,) and np.isfinite(values).all()
                close(np.quantile(values, [0.025, 0.975]), boot["summary"][name]["interval"])
                checked_ci += 1
            for predictor, targets in d["predictor_metrics"].items():
                if predictor not in ("ridge", "mlp", "mlp_seed42", "mlp_seed137", "mlp_seed271"):
                    continue
                for target, m in targets.items():
                    close(1 - m["sse"] / m["sst"], m["r2"])
                    close(m["mse_per_context"], m["sse"] / 252)
                    close(m["variance_trace"], m["sst"] / 252)
                    close(m["r2"], boot["summary"][f"{predictor}/{target}"]["estimate"])
                    checked_metrics += 1
            if rotation is None and k == 10:
                record = {}
                for arm in ("J", "R"):
                    point = (
                        boot["summary"][f"ridge/rest{arm}"]["estimate"]
                        - boot["summary"][f"ridge/{arm}"]["estimate"]
                    )
                    samples = draws[f"ridge/rest{arm}"] - draws[f"ridge/{arm}"]
                    record[f"G_{arm}"] = {
                        "estimate": point,
                        "interval": np.quantile(samples, [0.025, 0.975]).tolist(),
                    }
                dg = (
                    draws["ridge/restR"]
                    - draws["ridge/R"]
                    - draws["ridge/restJ"]
                    + draws["ridge/J"]
                )
                record["G_R_minus_G_J"] = {
                    "estimate": record["G_R"]["estimate"] - record["G_J"]["estimate"],
                    "interval": np.quantile(dg, [0.025, 0.975]).tolist(),
                }
                record["ridge_full_r2"] = boot["summary"]["ridge/full"]["estimate"]
                if kind == "observed":
                    record["noise_fraction"] = {
                        n: m["noise_fraction"] for n, m in supplement["noise"]["components"].items()
                    }
                    record["approximate_noise_corrected_gaps"] = {
                        arm: boot["summary"][f"ridge/rest{arm}"]["estimate"]
                        / (1 - record["noise_fraction"][f"rest{arm}"])
                        - boot["summary"][f"ridge/{arm}"]["estimate"]
                        / (1 - record["noise_fraction"][arm])
                        for arm in ("J", "R")
                    }
                    record["noise_correction_scope"] = (
                        "Descriptive classical independent additive test-noise approximation; "
                        "not a new fitted result or CI."
                    )
                headlines[cell] = record
        for _arm, v in d["arms"].items():
            g = v["pooled_target_geometry"]
            close(
                g["target_variance_trace"],
                g["component_variance_trace"]
                + g["rest_variance_trace"]
                + g["twice_component_rest_covariance_trace"],
            )
            max_reconstruction = max(max_reconstruction, g["max_abs_reconstruction_error"])
            assert v["contexts_with_increasing_error_steps"] == 0

assert checked_metrics == 1200
assert len(bootstrap_digests) == 1
calibration = {}
for role, file in [
    ("primary", "primary_full_calibration_diagnostics/calibration_report.json"),
    ("comparison", "comparison_full_calibration_report.json"),
]:
    c = read(ROOT / file)
    assert c["selected_prompts"] == 128 and c["realized_prompts"] == c["valid_prompts"] == 119
    calibration[role] = {
        "half_stability": c["matrix_agreement"]["even_vs_odd"],
        "readout_records": len(c["readouts"]),
        "eligible_directions": c["eligible_token_count"],
    }

out = {
    "status": "passed",
    "review_sha": "142c3215b5bff1df28109ba99ebdd9fdf31133ae",
    "scope": (
        "Local saved-statistic reductions; source arrays are separately audited when staged. "
        "No claim of regeneration or model execution."
    ),
    "selection_subsets_disjoint": True,
    "cell_count": 48,
    "primary_cohort_size": 252,
    "metric_summary_checks": checked_metrics,
    "interval_reductions": checked_ci,
    "shared_bootstrap_counts_sha256": next(iter(bootstrap_digests)),
    "max_abs_reconstruction_error": max_reconstruction,
    "headlines": headlines,
    "calibration": calibration,
    "primary_k10_quality_matches": {
        r: {a: q[a]["10"] for a in ("J", "R")} for r, q in quality.items()
    },
}
(OUT / "saved_statistics_checks.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in out.items()
            if k not in ("headlines", "calibration", "primary_k10_quality_matches")
        },
        indent=2,
    )
)
