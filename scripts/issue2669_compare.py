"""Compare frozen Codex and parity-validated probe predictions on identical 900 IDs.

Usage: uv run python scripts/issue2669_compare.py CONFIG.json
Config: run_root, output_root, source_sha256{absolute_path:sha}, probes[{root,
behavior,layer,methods}], require_oracle:false, bootstrap_draws:2000.
Hashes must cover every consumed artifact, including replay source label files.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from issue2669_analyze import correlation, grouped_indices, rowwise_rho
from issue2669_codex_dispatch import atomic_json
from issue2669_packets import read_rows

METHODS = {"regression_ctx", "reg_map_linear", "reg_oracle"}
BEHAVIORS = {"evil", "sycophancy", "hallucination"}


def require_hashes(config: dict) -> set[str]:
    """Verify all caller-frozen bytes before any source artifact is interpreted."""
    hashes = config["source_sha256"]
    if not hashes:
        raise ValueError("An explicit source hash manifest is required")
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source hash mismatch: " + name)
    return set(hashes)


def checked_json(path: Path, verified: set[str]):
    """Require a verified source path before decoding its JSON object."""
    if str(path) not in verified:
        raise ValueError("Missing source hash: " + str(path))
    return json.loads(path.read_text())


def checked_rows(path: Path, verified: set[str]) -> list[dict]:
    """Require a verified source path before decoding physical JSONL rows."""
    if str(path) not in verified:
        raise ValueError("Missing source hash: " + str(path))
    return read_rows(path)


def load_join(config: dict) -> tuple[list[dict], dict]:
    """Validate parity, exact cohorts, label scales and groups before joining scores."""
    verified = require_hashes(config)
    root = Path(config["run_root"])
    selection = checked_json(root / "selection.json", verified)
    cohort = checked_rows(root / "cohort.jsonl", verified)
    codex = checked_rows(root / "analysis_production/percontextforecast.jsonl", verified)
    codex_metrics = checked_json(root / "analysis_production/metrics.json", verified)
    selected = selection["selected_ids"]
    if len(selected) != 900 or len(set(selected)) != 900 or codex_metrics["n_contexts"] != 900:
        raise ValueError("Expected exactly 900 unique selected contexts")
    if Counter(r["id"] for r in cohort) != Counter(selected) or Counter(
        r["id"] for r in codex
    ) != Counter(selected):
        raise ValueError("Codex/cohort selected-ID mismatch")
    expected_cells = {
        (behavior, regime): 100 for behavior in BEHAVIORS for regime in ("id", "generic", "ood")
    }
    if Counter((r["behavior"], r["regime"]) for r in cohort) != expected_cells:
        raise ValueError("Expected nine fixed 100-context cells")
    codex_by_id = {r["id"]: r for r in codex}
    joined = []
    for row in cohort:
        other = codex_by_id[row["id"]]
        for key in (
            "context_id",
            "behavior",
            "regime",
            "rung",
            "group_key",
            "fold_id",
            "dv",
            "dv_original",
            "dv_scale",
        ):
            if row[key] != other[key]:
                raise ValueError("Codex/cohort metadata or outcome mismatch: " + key)
        instrument = (
            "fabrication"
            if row["behavior"] == "hallucination" and row["regime"] != "generic"
            else "trait"
        )
        scale = 100 if instrument == "fabrication" else 1
        if other["instrument"] != instrument or row["dv_scale"] != scale:
            raise ValueError("Instrument or outcome scale mismatch")
        if not math.isclose(row["dv"], row["dv_original"] * scale, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Normalized outcome mismatch")
        scores = {"codex_0": other["forecast_0"], "codex_32": other["forecast_32"]}
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 100
            for v in scores.values()
        ):
            raise ValueError("Invalid Codex scores")
        joined.append({**row, "instrument": instrument, "scores": scores, "probe_layers": {}})
    coverage = defaultdict(set)
    for spec in config["probes"]:
        directory = Path(spec["root"])
        if (
            not directory.is_absolute()
            or not set(spec["methods"]) <= METHODS
            or not spec["methods"]
        ):
            raise ValueError("Invalid probe source specification")
        behavior = spec["behavior"]
        result = checked_json(directory / "result.json", verified)
        recipe = result["config"]
        if result["parity_pass"] is not True or not result["metrics"]:
            raise ValueError("Probe reference parity must pass")
        if not set(spec["methods"]) <= {metric["method"] for metric in result["metrics"]}:
            raise ValueError("Requested method lacks parity evidence")
        if (
            recipe["behavior"] != behavior
            or recipe["layer"] != spec["layer"]
            or not set(spec["methods"]) <= set(recipe["methods"])
        ):
            raise ValueError("Probe recipe/source specification mismatch")
        for metric in result["metrics"]:
            values = [metric["rho"], metric["reference_rho"], metric["absolute_delta"]]
            if (
                not all(math.isfinite(v) for v in values)
                or abs(values[0] - values[1]) > recipe["parity_abs_tolerance"]
            ):
                raise ValueError("Probe parity evidence is invalid")
        labels = {}
        for regime, name in [
            ("id", "dv_path"),
            ("ood", "dv_path"),
            ("generic", "wildchat_dv_path"),
        ]:
            payload = checked_json(Path(recipe[name]), verified)["rows"]
            split = "train" if regime == "id" else "eval"
            records = [r for r in payload if r["split"] == split and r["dv"] is not None]
            if len({r["context_id"] for r in records}) != len(records):
                raise ValueError("Duplicate reference label IDs")
            labels[regime] = {r["context_id"]: r for r in records}
        for regime in ("id", "generic", "ood"):
            records = checked_rows(directory / f"{regime}.jsonl", verified)
            if len({r["context_id"] for r in records}) != len(records):
                raise ValueError("Duplicate replay prediction IDs")
            predictions = {r["context_id"]: r for r in records}
            for row in joined:
                if row["behavior"] != behavior or row["regime"] != regime:
                    continue
                cid = row["context_id"]
                if cid not in predictions or cid not in labels[regime]:
                    raise ValueError("Missing selected replay/context label ID: " + cid)
                prediction, label = predictions[cid], labels[regime][cid]
                if label["group_key"] != row["group_key"] or label["rung"] != row["rung"]:
                    raise ValueError("Replay label group/corpus mismatch")
                if (
                    not math.isclose(
                        prediction["dv"] * row["dv_scale"], row["dv"], rel_tol=0, abs_tol=1e-9
                    )
                    or prediction["dv"] != label["dv"]
                ):
                    raise ValueError("Replay outcome normalization mismatch")
                if prediction["rung"] != ("train" if regime == "id" else row["rung"]):
                    raise ValueError("Replay phase/corpus mismatch")
                for method in spec["methods"]:
                    value = prediction["scores"][method]
                    if (
                        method in row["scores"]
                        or isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        raise ValueError("Duplicate or nonfinite probe score")
                    row["scores"][method] = value
                    row["probe_layers"][method] = spec["layer"]
        if coverage[behavior] & set(spec["methods"]):
            raise ValueError("Method selected from multiple replay sources")
        coverage[behavior].update(spec["methods"])
    required = {"regression_ctx", "reg_map_linear"} | (
        {"reg_oracle"} if config.get("require_oracle", False) else set()
    )
    for behavior in BEHAVIORS:
        if not required <= coverage[behavior]:
            raise ValueError("Missing required probe method: " + behavior)
        for row in joined:
            if row["behavior"] == behavior and not coverage[behavior] <= set(row["scores"]):
                raise ValueError("Partial within-behavior method coverage")
    return joined, {
        b: {"available": sorted(coverage[b]), "oracle_absent": "reg_oracle" not in coverage[b]}
        for b in sorted(BEHAVIORS)
    }


def compare_group(rows: list[dict], draws: int = 2000, seed: int = 2669) -> dict:
    """Compare correlations with shared vectorized cluster draws and fresh ranks per draw."""
    methods = sorted(rows[0]["scores"])
    if any(set(r["scores"]) != set(methods) for r in rows):
        raise ValueError("Unequal method coverage within comparison cell")
    y = np.array([r["dv"] for r in rows])
    scores = np.array([[r["scores"][method] for r in rows] for method in methods])
    point = {method: correlation(y, scores[i]) for i, method in enumerate(methods)}
    indices = grouped_indices(rows, draws, seed)
    valid, safe = indices >= 0, np.maximum(indices, 0)
    yy = np.where(valid, y[safe], np.nan)
    sample = np.where(valid[None], scores[:, safe], np.nan)
    y_all = np.broadcast_to(yy, sample.shape)
    rho = rowwise_rho(
        y_all.reshape(-1, indices.shape[1]), sample.reshape(-1, indices.shape[1])
    ).reshape(len(methods), draws)
    differences = {}
    for probe in ("regression_ctx", "reg_map_linear"):
        for judge in ("codex_0", "codex_32"):
            delta = rho[methods.index(probe)] - rho[methods.index(judge)]
            finite = np.isfinite(delta)
            defined = point[probe] is not None and point[judge] is not None
            result = {
                "point": point[probe] - point[judge] if defined else None,
                "valid_draws": int(finite.sum()),
                "ci95": None,
            }
            if finite.all():
                interval = np.quantile(delta, [0.025, 0.975])
                result["ci95"] = interval.tolist()
                result["half_full_endpoint_max_abs_delta"] = float(
                    np.max(np.abs(np.quantile(delta[: draws // 2], [0.025, 0.975]) - interval))
                )
            else:
                result["reason"] = "constant/undefined correlation in at least one bootstrap draw"
            differences[probe + "-minus-" + judge] = result
    return {
        "n": len(rows),
        "spearman": point,
        "paired_spearman_differences": differences,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "unit": "group within corpus and original ID fold",
            "mc_setting": "2000 draws ungrounded; half/full endpoint check reported",
        },
        "score_scale_note": "Probe scores use standardized targets; no raw MAE/RMSE comparison",
    }


def run(config_path: Path) -> dict:
    """Write matched predictions and comparisons only after all sources pass validation."""
    config = json.loads(config_path.read_text())
    draws = config.get("bootstrap_draws", 2000)
    if type(draws) is not int or draws < 2:
        raise ValueError("At least two bootstrap draws required")
    joined, coverage = load_join(config)
    groups = defaultdict(list)
    for row in joined:
        key = "/".join((row["behavior"], row["instrument"], row["regime"]))
        groups[key].append(row)
        if row["regime"] == "ood":
            groups[key + "/" + row["rung"]].append(row)
    metrics = {name: compare_group(rows, draws) for name, rows in sorted(groups.items())}
    output = Path(config["output_root"])
    if not output.is_absolute():
        raise ValueError("Absolute output path required")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "percontextjoined.jsonl").open("x") as handle:
        for row in joined:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    report = {
        "complete": True,
        "n_contexts": len(joined),
        "coverage": coverage,
        "metrics": metrics,
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "exact same selected contexts; no full-cohort aggregate substitute",
        "protocol_caveats": [
            "The original ID map and whitening use all ID context-answer pairs before the behavior-label readout folds. ID evaluation is transductive for these representations, not a strict holdout of all information.",
            "Original per-pool target means and standard deviations are computed before the ID readout folds; individual held-out labels do not enter a ridge fit, but label preprocessing is not nested.",
            "Held-out generic and OOD contexts are excluded from the map and readout fitting pools.",
            "Layer choices are frozen from the original artifacts and may have been selected using original ID outcomes; this replay does not establish independent layer-selection holdout.",
        ],
    }
    atomic_json(output / "results.json", report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(run(Path(sys.argv[1])), indent=2, allow_nan=False))
