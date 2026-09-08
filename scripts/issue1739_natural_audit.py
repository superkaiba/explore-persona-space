"""Read-only integrity and descriptive aggregation for plan31 natural scaling.

Consumes completed cells, never fits a model or calls a judge. Primary rows
are each P-B fit's OWN held-out dataset; repeated held-in/WildChat/PV rows
are validated but are not counted as independent LODO datasets. Seed ranges
are descriptive, not confidence intervals or an equivalence test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np

from explore_persona_space.experiments.issue_1739.arms import spearman_rows
from scripts.issue1739_natural_score import NATURAL_ROSTER

U_GRID = (250, 500, 1000, 2000, 5000, 10000, 18793, 25000, 50000, 100000)
TRAITS = {"evil": 6468, "sycophancy": 16000, "hallucination": 16000}
HOLDOUTS = {
    "evil": {"hhrt", "toxicchat", "evil_mhj", "evil_pair", "evil_tomgibbs"},
    "sycophancy": {"aita", "sycoans", "sycoays", "sycofb", "sycomim", "sycomwe"},
    "hallucination": {"nqopen", "simpleqa"},
}
FROZEN = {
    b: dict(zip(NATURAL_ROSTER, ids, strict=True))
    for b, ids in {
        "evil": (18, 20, 17),
        "sycophancy": (20, 19, 19),
        "hallucination": (20, 20, 18),
    }.items()
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def primary_row(row: dict) -> bool:
    return row.get("protocol") == "P-B" and row.get("fit") == "P-B-holdout-" + row.get(
        "eval_rung", ""
    )


def validate_knn(block: dict, *, expected_n: int | None = None) -> None:
    require(set(block) == {"euclidean", "cosine"}, "missing kNN metric")
    for metric, values in block.items():
        n = values["n_pool"]
        if expected_n is not None:
            require(n == expected_n, "retrieval pool does not match declared evaluation rows")
        require(
            n > 5 and values["n"] == n and values["metric"] == metric,
            "wrong held-out retrieval candidate pool",
        )
        require(set(values["acc_at_k"]) == set(values["chance_at_k"]), "kNN k mismatch")
        for k, accuracy in values["acc_at_k"].items():
            require(0 <= accuracy <= 1, "nonfinite/out-of-range retrieval")
            require(
                math.isclose(values["chance_at_k"][k], int(k) / n, abs_tol=1e-14),
                "incorrect retrieval chance",
            )


def validate_predictions(path: Path, expected: list[dict], frozen: dict) -> dict:
    """Recompute every reported rho and verify exact arm-paired context/DV keys."""
    groups = defaultdict(dict)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(
                row["protocol"] == "P-B"
                and row["fit"] == path.stem
                and row["map_variant"] == "true",
                f"crossed prediction provenance: {path}",
            )
            arm, rung = row["arm"], row["rung"]
            require(arm in NATURAL_ROSTER, f"unexpected prediction arm {arm}")
            require(row["layer"] == frozen[arm], "prediction frozen global layer changed")
            require(
                math.isfinite(row["dv"]) and math.isfinite(row["score"]),
                f"nonfinite prediction/DV: {path}",
            )
            key = (row["context_id"], row["group"])
            require(key not in groups[(rung, arm)], f"duplicate prediction key: {path}")
            groups[(rung, arm)][key] = (row["dv"], row["score"])
    expected_by_key = {(r["eval_rung"], r["arm"]): r for r in expected}
    require(len(expected_by_key) == len(expected), "duplicate summary row")
    require(set(groups) == set(expected_by_key), "prediction/summary rung-arm coverage differs")
    max_error = 0.0
    for rung in {key[0] for key in groups}:
        baseline = groups[(rung, NATURAL_ROSTER[0])]
        require(
            len({key[0] for key in baseline}) == len(baseline),
            "duplicate context ID under different groups",
        )
        for arm in NATURAL_ROSTER:
            paired = groups[(rung, arm)]
            require(paired.keys() == baseline.keys(), f"unpaired contexts: {path}/{rung}")
            keys = sorted(baseline)
            dv = np.array([paired[key][0] for key in keys])
            require(np.array_equal(dv, [baseline[key][0] for key in keys]), "unpaired DV")
            pred = np.array([paired[key][1] for key in keys])
            rho = float(spearman_rows(pred[None], dv)[0])
            reported = expected_by_key[(rung, arm)]
            require(reported["n_eval"] == len(keys), "realized prediction row count differs")
            recorded = reported["rho_frozen"]
            require(
                math.isclose(rho, recorded, abs_tol=1e-12)
                or (math.isnan(rho) and math.isnan(recorded)),
                "rho recomputation differs",
            )
            if math.isfinite(rho):
                max_error = max(max_error, abs(rho - recorded))
    return {
        "rows": sum(map(len, groups.values())),
        "rung_arm_blocks": len(groups),
        "max_abs_rho_recompute_error": max_error,
    }


def validate_cell(cell: Path, *, commit: str, check_predictions: bool = True) -> dict:
    u, behavior, seed = int(cell.parent.parent.name[1:]), cell.parent.name, int(cell.name[4:])
    summary = json.loads((cell / "all_arms_spearman.json").read_text())
    meta = summary["meta"]
    require(meta["behavior"] == behavior and meta["seed"] == seed, "crossed cell identity")
    require(meta["git_commit"] == commit, "different scientific code SHA")
    require(
        meta["arms"] == list(NATURAL_ROSTER)
        and meta["protocols"] == ["B"]
        and meta["map_variants"] == ["true"]
        and meta["judge_called"] is False,
        "unexpected scientific roster/protocol",
    )
    natural = meta["natural_pool"]
    require(
        natural["generic_u"] == u
        and natural["n_rows"] == 100000
        and natural["no_recombination"] is True,
        "wrong natural generic pool",
    )
    layers = natural["selected_global_layers"]
    require(
        {arm: layers[meta["frozen_layers"][arm]] for arm in NATURAL_ROSTER} == FROZEN[behavior],
        "wrong frozen layer mapping",
    )
    require(set(meta["datasets"]) == HOLDOUTS[behavior] | {"train"}, "dataset roster changed")
    require(meta["datasets"]["train"] == TRAITS[behavior], "trait training count changed")
    companion = json.loads((cell / "readout_pools.json").read_text())
    require(
        companion == {"pools": meta["pb_pools"], "fit_reports": meta["fit_reports"]},
        "readout companion differs from summary",
    )
    diagnostics = json.loads((cell / "map_diagnostics.json").read_text())
    require(len(diagnostics) == 1, "expected exactly one true linear map")
    diag = next(iter(diagnostics.values()))
    require(
        diag["add_n_generic"] == u
        and diag["add_n_eliciting"] == TRAITS[behavior]
        and diag["add_realized_pool"] == meta["n_u"] == u + TRAITS[behavior]
        and diag["w_fit_rows"] == u + TRAITS[behavior]
        and diag["w_refit_on_full_u"] is True,
        "wrong realized full mapping pool",
    )
    require(diag["n_train"] + diag["n_holdout"] == meta["n_u"], "map split row count differs")
    require(len(diag["per_layer"]) == len(layers), "missing map diagnostic layer")
    require(
        {r["layer_idx"] for r in diag["per_layer"]} == set(range(len(layers))),
        "missing/duplicate map diagnostic layer ID",
    )
    for row in diag["per_layer"]:
        require(
            math.isfinite(row["r2_map"]) and math.isfinite(row["r2_identity_bias"]),
            "nonfinite map R2/baseline",
        )
        validate_knn(row["knn"], expected_n=diag["n_holdout"])
    expected_fits = {f"P-B-holdout-{name}" for name in HOLDOUTS[behavior]}
    reports = {r["fit"]: r for r in meta["fit_reports"]}
    require(
        set(reports) == expected_fits and len(reports) == len(meta["fit_reports"]),
        "missing/duplicate fit reports",
    )
    require({p["holdout"] for p in meta["pb_pools"]} == HOLDOUTS[behavior], "missing readout pool")
    primary, predictions, reconstruction = [], {}, []
    for fit in sorted(expected_fits):
        holdout = fit.removeprefix("P-B-holdout-")
        report = reports[fit]
        require(
            report["protocol"] == "P-B" and report["map_variant"] == "true", "crossed fit report"
        )
        require("passed" in report["leakage"]["asserts"], "readout leakage check absent")
        require(report["d"] == 3584 and report["n_readout"] > 3584, "underdetermined readout")
        rows = [r for r in summary["transfer_rows"] if r["fit"] == fit]
        require(
            all(
                r["protocol"] == "P-B"
                and r["map_variant"] == "true"
                and r["behavior"] == behavior
                and r["seed"] == seed
                for r in rows
            ),
            "crossed transfer summary provenance",
        )
        selected = [r for r in rows if primary_row(r)]
        require(
            {r["arm"] for r in selected} == set(NATURAL_ROSTER) and len(selected) == 3,
            "missing primary LODO arm",
        )
        for row in selected:
            require(row["n_eval"] == meta["datasets"][holdout], "missing primary eval rows")
            require(row["layer"] == FROZEN[behavior][row["arm"]], "summary frozen layer changed")
            primary.append(
                {
                    "behavior": behavior,
                    "generic_u": u,
                    "seed": seed,
                    "dataset": holdout,
                    "arm": row["arm"],
                    "n_eval": row["n_eval"],
                    "rho": row["rho_frozen"],
                    "layer": row["layer"],
                }
            )
        if check_predictions:
            predictions[fit] = validate_predictions(
                cell / "transfer_preds" / f"{fit}.jsonl", rows, FROZEN[behavior]
            )
        mapped = report["recon"]["per_rung"][holdout]
        baseline = report["recon_identity_bias"]["per_rung"][holdout]
        require(
            mapped["n_rows"] == baseline["n_rows"] == meta["datasets"][holdout],
            "reconstruction eval count differs",
        )
        require(
            len(mapped["per_layer"]) == len(layers)
            and {r["layer_idx"] for r in mapped["per_layer"]} == set(range(len(layers)))
            and len(baseline["r2_identity_bias_per_layer"]) == len(layers),
            "missing/duplicate reconstruction diagnostic layer",
        )
        for row in mapped["per_layer"]:
            li = row["layer_idx"]
            require(
                math.isfinite(row["r2_eval_rung"])
                and math.isfinite(baseline["r2_identity_bias_per_layer"][li]),
                "nonfinite held-out reconstruction/baseline",
            )
            validate_knn(row["knn"], expected_n=mapped["n_rows"])
            reconstruction.append(
                {
                    "behavior": behavior,
                    "generic_u": u,
                    "seed": seed,
                    "dataset": holdout,
                    "layer": layers[li],
                    "r2": row["r2_eval_rung"],
                    "r2_identity_bias": baseline["r2_identity_bias_per_layer"][li],
                    "knn": row["knn"],
                }
            )
    expected_pred_paths = {f"transfer_preds/{fit}.jsonl" for fit in expected_fits}
    require(
        set(meta["natural_transfer_pred_files"]) == expected_pred_paths,
        "prediction sidecar declaration differs",
    )
    require(
        {str(p.relative_to(cell)) for p in (cell / "transfer_preds").glob("*.jsonl")}
        == expected_pred_paths,
        "unexpected prediction sidecar",
    )
    return {
        "behavior": behavior,
        "generic_u": u,
        "seed": seed,
        "status": "PASS",
        "map_pool": u + TRAITS[behavior],
        "trait_pairs": TRAITS[behavior],
        "selected_context_ids_sha256": natural["selected_context_ids_sha256"],
        "manifest_sha256": natural["manifest_sha256"],
        "input_sha256": meta["input_sha256"],
        "wall_s": meta["wall_s"],
        "skips": summary["transfer_skips"],
        "primary": primary,
        "prediction_checks": predictions,
        "reconstruction": reconstruction,
        "map_pool_diagnostics": diag,
        "files_sha256": {
            str(p.relative_to(cell)): file_sha(p) for p in sorted(cell.rglob("*")) if p.is_file()
        },
    }


def aggregate(cells: list[dict], *, require_full: bool) -> dict:
    keys = {(c["behavior"], c["generic_u"], c["seed"]) for c in cells}
    require(len(keys) == len(cells), "duplicate fit cell")
    planned = {(b, u, seed) for b in TRAITS for u in U_GRID for seed in range(5)}
    require(keys <= planned, "unplanned fit cell")
    if require_full:
        require(keys == planned, f"missing planned fit cells: {sorted(planned - keys)}")
    require(
        len({c["manifest_sha256"] for c in cells}) == 1,
        "natural manifest differs across scaling cells",
    )
    input_versions = defaultdict(set)
    behavior_inputs = {}
    for cell in cells:
        require(bool(cell["input_sha256"]), "missing fixed input fingerprints")
        behavior = cell["behavior"]
        if behavior in behavior_inputs:
            require(
                cell["input_sha256"] == behavior_inputs[behavior],
                "fixed input path set/content differs within behavior",
            )
        else:
            behavior_inputs[behavior] = cell["input_sha256"]
        for path, digest in cell["input_sha256"].items():
            input_versions[path].add(digest)
    require(
        all(len(versions) == 1 for versions in input_versions.values()),
        "fixed input content differs across scaling cells",
    )
    grouped, sample_hashes = defaultdict(list), defaultdict(set)
    for cell in cells:
        sample_hashes[(cell["generic_u"], cell["seed"])].add(cell["selected_context_ids_sha256"])
        per_arm = defaultdict(list)
        for row in cell["primary"]:
            require(
                math.isfinite(row["rho"]), "undefined primary rho: cannot silently macro-average"
            )
            per_arm[row["arm"]].append(row["rho"])
        means = {arm: float(np.mean(values)) for arm, values in per_arm.items()}
        grouped[(cell["behavior"], cell["generic_u"])].append(
            {
                "seed": cell["seed"],
                **means,
                "mapped_minus_context": means[NATURAL_ROSTER[1]] - means[NATURAL_ROSTER[0]],
            }
        )
    require(
        all(len(s) == 1 for s in sample_hashes.values()), "generic samples differ across behaviors"
    )
    curves = []
    for (behavior, u), rows in sorted(grouped.items()):
        stats = {}
        for arm in (*NATURAL_ROSTER, "mapped_minus_context"):
            values = [r[arm] for r in rows]
            stats[arm] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
        curves.append(
            {
                "behavior": behavior,
                "generic_u": u,
                "n_seeds": len(rows),
                "n_datasets": len(HOLDOUTS[behavior]),
                "per_seed": rows,
                "statistics": stats,
            }
        )
    return {
        "planned_cells": len(planned),
        "realized_cells": len(cells),
        "missing_cells": sorted(planned - keys),
        "curves": curves,
        "metric": "unweighted mean of own-heldout-dataset Spearman rho, then seed mean",
        "uncertainty": "min-max across seeds; descriptive, not a confidence interval",
        "scope": "P-B readout LODO; mapping fixed trait pool is not held out by dataset",
        "whitening": "refit at every U; direct and oracle baselines can change with U",
    }


def verify_remote_cell(cell: Path, receipt: dict) -> dict:
    """Check the immutable uploaded tree's exact names, sizes and content IDs."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from explore_persona_space.orchestrate import hub

    prefix, revision = receipt["prefix"], receipt["revision"]
    api = HfApi()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: enclosing retry_transient retries the complete paginated listing.
            api.list_repo_tree(
                "superkaiba1/explore-persona-space-data",
                path_in_repo=prefix,
                repo_type="dataset",
                revision=revision,
                recursive=True,
            )
        ),
        what=f"independent content verification {prefix}",
    )
    remote = {e.path[len(prefix) + 1 :]: e for e in entries if isinstance(e, RepoFile)}
    local = {str(p.relative_to(cell)): p for p in cell.rglob("*") if p.is_file()}
    require(
        set(remote) == set(local) == set(receipt["files_sha256"]),
        "remote/local/receipt file-name sets differ",
    )
    for name, path in local.items():
        entry = remote[name]
        require(entry.size == path.stat().st_size, f"remote size differs: {name}")
        digest = file_sha(path)
        require(digest == receipt["files_sha256"][name], f"receipt content differs: {name}")
        if entry.lfs:
            require(entry.lfs.sha256 == digest, f"remote LFS content differs: {name}")
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            require(entry.blob_id == blob, f"remote Git blob differs: {name}")
    return {
        "status": "PASS",
        "prefix": prefix,
        "revision": revision,
        "files": len(local),
        "all_names_sizes_content_verified": True,
    }


def watch_completed_cells(args) -> list[dict]:
    """Audit uploaded cells alongside fitting, bounded by exact driver identity.

    This verifier never dispatches compute, edits fitting outputs or retries a
    failed scientific cell. Each proof is persisted independently; completion
    still requires all150 cells and a final local-content recheck.
    """
    from explore_persona_space.atomic_io import atomic_replace

    driver = Path(f"/proc/{args.watch_driver}")
    identity = (driver / "stat").read_text().split()[21]
    started, cells = time.monotonic(), {}
    driver_exited = False
    args.cache_root.mkdir(parents=True, exist_ok=True)
    while time.monotonic() - started < args.max_watch_hours * 3600:
        for receipt_path in sorted(args.receipts_root.glob("u*/*/seed*/verified.json")):
            relative = receipt_path.parent.relative_to(args.receipts_root)
            if str(relative) in cells:
                continue
            cell = args.results_root / relative
            receipt = json.loads(receipt_path.read_text())
            expected_suffix = "issue1739_natural100k_20260906/results/" + str(relative)
            require(receipt["prefix"] == expected_suffix, "crossed remote receipt prefix")
            result = validate_cell(cell, commit=args.commit)
            result["remote_verification"] = verify_remote_cell(cell, receipt)
            target = args.cache_root / relative / "verified.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with atomic_replace(target) as tmp:
                tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            cells[str(relative)] = result
            print(json.dumps({"verified_cells": len(cells), "cell": str(relative)}), flush=True)
        if len(cells) == 150:
            for relative, result in cells.items():
                cell = args.results_root / relative
                actual = {
                    str(p.relative_to(cell)): file_sha(p) for p in cell.rglob("*") if p.is_file()
                }
                require(actual == result["files_sha256"], "cell changed after independent audit")
            return list(cells.values())
        try:
            stat = (driver / "stat").read_text().split()
            alive = stat[21] == identity and stat[2] != "Z"
        except FileNotFoundError:
            alive = False
        if not alive:
            # Receipts can land during validation of the earlier directory
            # snapshot. Drain a fresh snapshot once AFTER observing exit.
            require(
                not driver_exited, f"fit driver exited with only {len(cells)}/150 verified cells"
            )
            driver_exited = True
            continue
        time.sleep(10)
    raise TimeoutError(f"bounded verification expired at {len(cells)}/150 cells")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--watch-driver", type=int)
    p.add_argument("--receipts-root", type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--max-watch-hours", type=float, default=8)
    args = p.parse_args()
    if args.watch_driver:
        if args.pilot or not args.receipts_root or not args.cache_root or args.max_watch_hours <= 0:
            p.error("watch requires receipt/cache roots, positive bound, and full150-cell mode")
        cells = watch_completed_cells(args)
    else:
        cells = []
        for path in sorted(args.results_root.glob("u*/*/seed*/all_arms_spearman.json")):
            cells.append(validate_cell(path.parent, commit=args.commit))
            print(
                json.dumps({k: cells[-1][k] for k in ("behavior", "generic_u", "seed", "status")}),
                flush=True,
            )
    if args.pilot:
        require(
            {(c["behavior"], c["generic_u"], c["seed"]) for c in cells}
            == {(b, 100000, 0) for b in TRAITS},
            "pilot must contain exactly all three endpoint cells",
        )
    payload = {
        "schema_version": 1,
        "scientific_commit": args.commit,
        "aggregate": aggregate(cells, require_full=not args.pilot),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from explore_persona_space.atomic_io import atomic_replace

    with atomic_replace(args.output) as tmp:
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
