"""Audit completed #2669 forecasts; analyze pilot or production without model calls.

Usage: uv run python scripts/issue2669_analyze.py ROOT pilot|production
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import issue2669_codex_dispatch as dispatch
from issue2669_packets import read_rows

BOOTSTRAP_DRAWS = 2000  # Monte Carlo setting; endpoint half/full stability reported.
SEED = 2669


def read(path: Path):
    """Read one JSON artifact without fallback values."""
    return json.loads(path.read_text())


def collect(root: Path, phase: str) -> tuple[list[dict], dict]:
    """Require exact manifest coverage, request identity, artifact hashes and raw event validity."""
    config = read(root / f"{phase}_config.json")
    selection = read(root / "selection.json")
    expected = selection["pilot_ids" if phase == "pilot" else "selected_ids"]
    repeats = 3 if phase == "pilot" else 1
    packets = config["packets"]
    if len({p["id"] for p in packets}) != len(packets):
        raise ValueError("Duplicate packet IDs")
    for shot in (0, 32):
        for repeat in range(repeats):
            actual = Counter(
                cid
                for p in packets
                if p["shot"] == shot and p["repeat"] == repeat
                for cid in p["ids"]
            )
            if actual != Counter(expected):
                raise ValueError("Manifest condition/repeat coverage differs from frozen selection")
    forecasts, timing, attempts = [], [], []
    for packet in packets:
        directory = Path(config["output_dir"]) / packet["id"]
        status = read(directory / "status.json")
        identity = read(directory / "identity.json")
        fingerprint = dispatch.digest(json.dumps(identity, sort_keys=True).encode())
        prompt_sha = dispatch.digest(
            (dispatch.NO_TOOLS + Path(packet["prompt_path"]).read_text()).encode()
        )
        if (
            identity["prompt_sha256"] != prompt_sha
            or identity["ids"] != packet["ids"]
            or identity["behavior"] != packet["behavior"]
            or identity["schema"] != dispatch.SCHEMA
            or identity["model"] != dispatch.MODEL
            or identity["effort"] != dispatch.EFFORT
            or identity["protocol"] != dispatch.PROTOCOL
            or fingerprint != status["fingerprint"]
        ):
            raise ValueError("Request/manifest fingerprint mismatch")
        if status["status"] != "complete" or status["returncode"] != 0:
            raise ValueError(f"Incomplete or invalid packet: {packet['id']}")
        all_attempts = sorted(directory.glob("attempt-*"))
        if not all_attempts or all_attempts[-1].name != status["attempt"]:
            raise ValueError("Unaccounted trailing attempt")
        for attempt in all_attempts:
            metadata = read(attempt / "metadata.json")
            if metadata["fingerprint"] != fingerprint:
                raise ValueError("Attempt fingerprint mismatch")
            for name, digest in metadata["artifact_sha256"].items():
                if dispatch.digest((attempt / name).read_bytes()) != digest:
                    raise ValueError("Artifact hash mismatch: " + str(attempt / name))
            dispatch.audit_events(
                attempt / "events.jsonl"
            )  # Every attempt, including recovered ones.
            if metadata["status"] not in {"transport_loss", "complete"}:
                raise ValueError("Invalid historical attempt")
            attempts.append(metadata)
        attempt = directory / status["attempt"]
        if read(attempt / "metadata.json") != status:
            raise ValueError("Status differs from immutable completion metadata")
        result = dispatch.validate_attempt(attempt, packet["ids"])
        timing.append(
            {
                **status,
                "behavior": packet["behavior"],
                "instrument": packet["instrument"],
                "shot": packet["shot"],
                "n_contexts": len(packet["ids"]),
            }
        )
        for row in result["rows"]:
            forecasts.append(
                {
                    **row,
                    "packet_id": packet["id"],
                    "behavior": packet["behavior"],
                    "instrument": packet["instrument"],
                    "regime": packet["regime"],
                    "shot": packet["shot"],
                    "repeat": packet["repeat"],
                }
            )
    if len(forecasts) != len(expected) * 2 * repeats:
        raise ValueError("Forecast cardinality mismatch")
    return forecasts, {
        "packets": len(packets),
        "forecasts": len(forecasts),
        "timing": timing,
        "attempts": attempts,
        "n_transport_attempts": sum(a["status"] == "transport_loss" for a in attempts),
    }


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman correlation, preserving undefined constant cases."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def point_metrics(y: np.ndarray, prediction: np.ndarray, fabrication: bool) -> dict:
    """Compute forecast error on the original instrument's scale."""
    rho = correlation(y, prediction)
    result = {
        "n": len(y),
        "spearman": rho,
        "spearman_reason": None if rho is not None else "fewer than two or constant values",
        "mae": float(np.mean(np.abs(prediction - y))),
        "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
    }
    if fabrication:
        p, rate = prediction / 100, y / 100
        result["mean_rate_mse"] = float(np.mean((p - rate) ** 2))
        result["per_answer_brier"] = float(np.mean(p**2 - 2 * p * rate + rate))
    return result


def grouped_indices(rows: list[dict], draws: int, seed: int) -> np.ndarray:
    """Bootstrap groups within corpus and ID fold; pad unequal cluster sizes with -1."""
    strata = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        fold = str(row["fold_id"]) if row.get("regime") == "id" else "all"
        strata[(row["rung"], fold)][row["group_key"]].append(index)
    rng, pieces = np.random.default_rng(seed), []
    for rung in sorted(strata):
        groups = [strata[rung][key] for key in sorted(strata[rung])]
        membership = np.full((len(groups), max(map(len, groups))), -1, dtype=int)
        for index, group in enumerate(groups):
            membership[index, : len(group)] = group
        chosen = rng.integers(len(groups), size=(draws, len(groups)))
        pieces.append(membership[chosen].reshape(draws, -1))
    return np.concatenate(pieces, axis=1)


def rowwise_rho(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Recompute rank correlation independently for every padded bootstrap draw."""
    # Re-rank each bootstrap sample, including duplicated observations and ties.
    xr, yr = rankdata(x, axis=1, nan_policy="omit"), rankdata(y, axis=1, nan_policy="omit")
    xc, yc = xr - np.nanmean(xr, axis=1, keepdims=True), yr - np.nanmean(yr, axis=1, keepdims=True)
    numerator = np.nansum(xc * yc, axis=1)
    denominator = np.sqrt(np.nansum(xc**2, axis=1) * np.nansum(yc**2, axis=1))
    result = np.full(len(x), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def paired_bootstrap(
    rows: list[dict],
    y: np.ndarray,
    zero: np.ndarray,
    few: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict:
    """Estimate paired metric differences using identical cluster draws for both arms."""
    indices = grouped_indices(rows, draws, seed)
    valid = indices >= 0
    yy, zz, ff = (
        np.where(valid, vector[np.maximum(indices, 0)], np.nan) for vector in (y, zero, few)
    )
    differences = {
        "spearman_32_minus_0": rowwise_rho(yy, ff) - rowwise_rho(yy, zz),
        "mae_32_minus_0": np.nanmean(np.abs(ff - yy), axis=1) - np.nanmean(np.abs(zz - yy), axis=1),
        "rmse_32_minus_0": np.sqrt(np.nanmean((ff - yy) ** 2, axis=1))
        - np.sqrt(np.nanmean((zz - yy) ** 2, axis=1)),
    }
    output = {
        "draws": draws,
        "seed": seed,
        "unit": "group within corpus and original fold for ID; group within corpus otherwise",
        "n_groups": len({(r["rung"], r["group_key"]) for r in rows}),
        "mc_setting": "2000 draws ungrounded; compare half/full interval endpoints",
    }
    for name, values in differences.items():
        finite = values[np.isfinite(values)]
        half = values[: draws // 2]
        half = half[np.isfinite(half)]
        if len(finite) != draws or not len(half):
            output[name] = {
                "ci95": None,
                "reason": "at least one undefined bootstrap statistic",
                "valid_draws": len(finite),
            }
        else:
            interval = np.quantile(finite, [0.025, 0.975])
            output[name] = {
                "ci95": interval.tolist(),
                "valid_draws": len(finite),
                "half_full_endpoint_max_abs_delta": float(
                    np.max(np.abs(np.quantile(half, [0.025, 0.975]) - interval))
                ),
            }
    return output


def pilot_report(root: Path, forecasts: list[dict], audit: dict) -> dict:
    """Diagnostic repeat stability; never open the held-out label file."""
    groups = defaultdict(lambda: defaultdict(dict))
    for row in forecasts:
        groups[(row["behavior"], row["instrument"], row["shot"])][row["id"]][row["repeat"]] = row[
            "score_0_100"
        ]
    stability = {}
    for key, items in groups.items():
        if any(set(values) != {0, 1, 2} for values in items.values()):
            raise ValueError("Missing repeat")
        scores = np.array([[items[cid][rep] for rep in (0, 1, 2)] for cid in sorted(items)])
        pairs = [
            {
                "repeats": [a, b],
                "mae": float(np.mean(np.abs(scores[:, a] - scores[:, b]))),
                "spearman": correlation(scores[:, a], scores[:, b]),
            }
            for a, b in itertools.combinations(range(3), 2)
        ]
        stability["/".join(map(str, key))] = {
            "n_contexts": len(scores),
            "pairs": pairs,
            "mean_repeat_pair_mae": float(np.mean([p["mae"] for p in pairs])),
        }
    rate = defaultdict(list)
    for packet in audit["timing"]:
        rate[(packet["behavior"], packet["instrument"], packet["shot"])].append(
            packet["wall_seconds"] / packet["n_contexts"]
        )
    production = read(root / "production_config.json")
    estimated_work = sum(
        len(p["ids"]) * np.mean(rate[(p["behavior"], p["instrument"], p["shot"])])
        for p in production["packets"]
    )
    starts = [a["started_unix"] for a in audit["attempts"]]
    ends = [a["finished_unix"] for a in audit["attempts"]]
    return {
        "gate": "pass",
        "gate_scope": "exact coverage, valid output, no tool events; transport recovered",
        "packets": audit["packets"],
        "forecasts": audit["forecasts"],
        "n_transport_attempts": audit["n_transport_attempts"],
        "stability": stability,
        "measured_elapsed_seconds": max(ends) - min(starts),
        "summed_attempt_wall_seconds": sum(a["wall_seconds"] for a in audit["attempts"]),
        "production_eta_seconds": float(estimated_work / production["concurrency"]),
        "eta_basis": "pilot mean seconds/context per behavior/instrument/shot; divide total work by concurrency; batch size and queue changes may shift runtime",
    }


def production_report(
    root: Path, forecasts: list[dict], draws: int = BOOTSTRAP_DRAWS
) -> tuple[list[dict], dict]:
    """Join the exact selected cohort and evaluate paired forecasts by instrument and regime."""
    cohort = read_rows(root / "cohort.jsonl")
    selected = read(root / "selection.json")["selected_ids"]
    if len(cohort) != 900 or Counter(r["id"] for r in cohort) != Counter(selected):
        raise ValueError("Cohort does not match frozen 900 IDs")
    predictions = defaultdict(dict)
    instruments = {}
    for forecast in forecasts:
        if forecast["repeat"] != 0 or forecast["shot"] in predictions[forecast["id"]]:
            raise ValueError("Duplicate production forecast")
        predictions[forecast["id"]][forecast["shot"]] = forecast["score_0_100"]
        instruments[forecast["id"]] = forecast["instrument"]
    if set(predictions) != set(selected) or any(set(v) != {0, 32} for v in predictions.values()):
        raise ValueError("Production forecast/cohort mismatch")
    joined = [
        {
            **r,
            "instrument": instruments[r["id"]],
            "forecast_0": predictions[r["id"]][0],
            "forecast_32": predictions[r["id"]][32],
        }
        for r in cohort
    ]
    groups = defaultdict(list)
    for row in joined:
        key = (row["behavior"], row["instrument"], row["regime"])
        groups["/".join(key)].append(row)
        if row["regime"] == "ood":
            groups["/".join((*key, row["rung"]))].append(row)
    metrics = {}
    for name, rows in sorted(groups.items()):
        y, zero, few = (
            np.array([r[key] for r in rows], dtype=float)
            for key in ("dv", "forecast_0", "forecast_32")
        )
        if not all(np.isfinite(v).all() and ((v >= 0) & (v <= 100)).all() for v in (y, zero, few)):
            raise ValueError("Invalid evaluation scores")
        fabrication = rows[0]["instrument"] == "fabrication"
        metrics[name] = {
            "zero_shot": point_metrics(y, zero, fabrication),
            "few_shot_32": point_metrics(y, few, fabrication),
            "paired_bootstrap": paired_bootstrap(rows, y, zero, few, draws=draws),
        }
    return joined, {
        "metrics": metrics,
        "n_contexts": len(joined),
        "comparison_scope": "paired Codex zero-shot vs32 only; no historical probe aggregate comparison",
    }


def run(root: Path, phase: str) -> dict:
    """Validate a completed phase before persisting its analysis and provenance."""
    if phase not in {"pilot", "production"} or not root.is_absolute():
        raise ValueError("Require absolute root and pilot|production")
    selection = read(root / "selection.json")
    if len(selection["selected_ids"]) != 900 or len(selection["pilot_ids"]) != 48:
        raise ValueError("Selection must contain 900 evaluation and 48 pilot IDs")
    manifest = read(root / "packet_manifest.json")
    if manifest["selection_sha256"] != dispatch.digest((root / "selection.json").read_bytes()):
        raise ValueError("Frozen selection hash mismatch")
    forecasts, audit = collect(root, phase)
    output = root / f"analysis_{phase}"
    output.mkdir(exist_ok=True)
    if phase == "pilot":
        report = pilot_report(root, forecasts, audit)
    else:
        joined, report = production_report(root, forecasts)
        with (output / "percontextforecast.jsonl").open("w") as handle:
            for row in joined:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    report["source_config_sha256"] = dispatch.digest((root / f"{phase}_config.json").read_bytes())
    report["selection_sha256"] = dispatch.digest((root / "selection.json").read_bytes())
    report["packet_manifest_sha256"] = dispatch.digest((root / "packet_manifest.json").read_bytes())
    if phase == "production":
        report["cohort_sha256"] = dispatch.digest((root / "cohort.jsonl").read_bytes())
    report["code_sha256"] = dispatch.digest(Path(__file__).read_bytes())
    dispatch.atomic_json(output / "metrics.json", report)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(run(Path(sys.argv[1]), sys.argv[2]), indent=2, allow_nan=False))
