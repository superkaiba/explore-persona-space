"""Compare paired identity, shift, and scalar-plus-shift on the pinned strict K5 cohort."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import inspect
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_loso_calibration as base
from scripts import issue2054_k5_matched_offsets as core

PARENT_SHA = "3e4d958221bee06f23fa6f77a79ce8e7101671d662a16d7c6bf34baf574437a7"
METHODS = ("identity", "bias", "bias_scale")


def evaluate_both(source, target, membership):
    """Fit one scalar across all coordinates and training queries, in both directions."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    membership = np.asarray(membership)
    if source.shape != target.shape or source.ndim != 2 or len(source) != len(membership):
        raise ValueError("invalid paired arrays")
    if (
        not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or set(membership) != set(range(5))
    ):
        raise ValueError("nonfinite values or missing global folds")
    energy = np.square(target - source).sum(1)
    if energy.sum() <= 0:
        raise ValueError("undefined displacement fraction")
    outputs = []
    for _ in range(2):
        outputs.append(
            (
                [],
                {
                    "fold": membership,
                    "displacement_energy": energy.copy(),
                    "shift": np.zeros((5, source.shape[1])),
                    "affine_bias": np.zeros((5, source.shape[1])),
                    "scale": np.zeros(5),
                    **{
                        f"{method}_{kind}": np.zeros(len(source))
                        for method in METHODS
                        for kind in ("error", "rank")
                    },
                },
            )
        )
    for fold in range(5):
        test, train = membership == fold, membership != fold
        if min(test.sum(), train.sum()) < 2:
            raise ValueError("insufficient paired fold coverage")
        aa, bb = source[test], target[test]
        gram = aa @ bb.T
        for direction, x, y, cross in [(0, source, target, gram), (1, target, source, gram.T)]:
            records, arrays = outputs[direction]
            c = base.calibrate(x[train], y[train])
            parameters = {
                "identity": (1.0, np.zeros(x.shape[1])),
                "bias": (1.0, c["bias"]),
                "bias_scale": (c["gain"], c["target_mean"] - c["gain"] * c["prediction_mean"]),
            }
            a, b = x[test], y[test]
            an, bn = np.square(a).sum(1), np.square(b).sum(1)
            total = float(np.square(b - b.mean(0)).sum())
            if total <= 0 or energy[test].sum() <= 0:
                raise ValueError("undefined fold denominator")
            metrics = {}
            for name, (scale, bias) in parameters.items():
                error = np.square(scale * a + bias - b).sum(1)
                distance = (
                    scale**2 * an[:, None]
                    + 2 * scale * (a @ bias)[:, None]
                    + bias @ bias
                    + bn[None, :]
                    - 2 * scale * cross
                    - 2 * (b @ bias)[None, :]
                )
                ranks = core.ranks_from_distances(np.maximum(distance, 0))
                arrays[f"{name}_error"][test], arrays[f"{name}_rank"][test] = error, ranks
                metrics[name] = {
                    "r2": 1 - float(error.sum()) / total,
                    "displacement_fraction": 1 - float(error.sum() / energy[test].sum()),
                    "euclidean_top1": float((ranks <= 1).mean()),
                    "euclidean_top5": float((ranks <= 5).mean()),
                    "median_rank": float(np.median(ranks)),
                }
            arrays["shift"][fold] = c["bias"]
            arrays["affine_bias"][fold] = parameters["bias_scale"][1]
            arrays["scale"][fold] = c["gain"]
            records.append(
                {
                    "fold": fold,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "retrieval_pool": int(test.sum()),
                    "chance_top1": 1 / int(test.sum()),
                    "scale": c["gain"],
                    "metrics": metrics,
                }
            )
    return outputs


def fingerprint():
    """Pin the implementation and all reused numerical helpers."""
    return {
        "script_sha256": base.sha(__file__),
        "calibration_source": inspect.getsource(base.calibrate),
        "retrieval_source": inspect.getsource(core.ranks_from_distances),
        "parent_results_sha256": PARENT_SHA,
    }


def parent_report(out):
    """Refuse a changed strict cohort definition or reference result."""
    parent = out.parent / "k5_matched_offsets_strict"
    if base.sha(parent / "results.json") != PARENT_SHA:
        raise RuntimeError("strict parent result changed")
    return parent, json.loads((parent / "results.json").read_text())


def fit(out, inputs, model):
    """Reuse exact audited query IDs and folds; checkpoint each direction and arm."""
    parent, previous = parent_report(out)
    sources = {
        r["path"]: r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]
    }
    panel = base.load_panel(model, sources)
    indexes = {cell: {cid: i for i, cid in enumerate(p["ids"])} for cell, p in panel.items()}
    for old in previous["pairs"]:
        if old["model"] != model:
            continue
        ca, cb, arm = old["source"], old["target"], old["arm"]
        i, j = old["source_index"], old["target_index"]
        expected = [sources[f"{base.PREFIX}/production_v1/k5/{c}.npz"]["sha256"] for c in (ca, cb)]
        prior = parent / "pairs" / f"{model}__{i}_{j}__{arm}.npz"
        if old["input_sha256"] != expected or base.sha(prior) != old["array_sha256"]:
            raise RuntimeError("parent input identity or cohort array changed")
        with np.load(prior, allow_pickle=False) as z:
            ids, folds = z["conv_id"], z["fold"]
            old_error, old_energy = z["residual_energy"], z["displacement_energy"]
        ia, ib = (
            np.array([indexes[ca][str(cid)] for cid in ids]),
            np.array([indexes[cb][str(cid)] for cid in ids]),
        )
        if len(ids) != old["n_paired"] or not all(
            np.array_equal(panel[c]["membership"][ix], folds) for c, ix in [(ca, ia), (cb, ib)]
        ):
            raise RuntimeError("cohort or fold mismatch")
        key = "x" if arm == "context" else "y"
        results = evaluate_both(panel[ca][key][ia], panel[cb][key][ib], folds)
        for direction, (records, arrays) in enumerate(results):
            # Parity on the old identity-plus-bias estimator, before comparing scale.
            np.testing.assert_allclose(arrays["bias_error"], old_error, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(
                arrays["displacement_energy"], old_energy, rtol=1e-10, atol=1e-8
            )
            si, ti, src, tgt = (i, j, ca, cb) if direction == 0 else (j, i, cb, ca)
            path = out / "pairs" / f"{model}__{si}_{ti}__{arm}.json"
            if path.exists():
                raise RuntimeError("fresh output directory required; resume is not implicit")
            arrays["conv_id"] = ids
            base.save_npz(path.with_suffix(".npz"), arrays)
            record = {
                "model": model,
                "source": src,
                "target": tgt,
                "source_index": si,
                "target_index": ti,
                "arm": arm,
                "n_paired": len(ids),
                "folds": records,
                "displacement_fraction": {
                    name: 1
                    - float(arrays[f"{name}_error"].sum() / arrays["displacement_energy"].sum())
                    for name in METHODS
                },
                "r2_mean": {
                    name: float(np.mean([r["metrics"][name]["r2"] for r in records]))
                    for name in METHODS
                },
                "top1_mean": {
                    name: float(np.mean([r["metrics"][name]["euclidean_top1"] for r in records]))
                    for name in METHODS
                },
                "scale_mean": float(arrays["scale"].mean()),
                "scale_min": float(arrays["scale"].min()),
                "scale_max": float(arrays["scale"].max()),
                "array_sha256": base.sha(path.with_suffix(".npz")),
                "parent_array_sha256": old["array_sha256"],
                "input_sha256": expected if direction == 0 else expected[::-1],
                "fingerprint": fingerprint(),
            }
            base.atomic_json(path, record)
            print(
                f"[phase=pair] {model} {si}->{ti} {arm} n={len(ids)} scale={record['scale_mean']:.3f} fraction={record['displacement_fraction']}",
                flush=True,
            )


def collect(out, inputs):
    """Verify every directed panel and persist the complete comparison."""
    _, previous = parent_report(out)
    refs = [r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]]
    sources = {Path(r["path"]).stem: r for r in refs}
    for ref in refs:
        if base.sha(ref["local"]) != ref["sha256"]:
            raise RuntimeError("input bank changed during analysis")
    rows = []
    for old in previous["pairs"]:
        for si, ti in [
            (old["source_index"], old["target_index"]),
            (old["target_index"], old["source_index"]),
        ]:
            path = out / "pairs" / f"{old['model']}__{si}_{ti}__{old['arm']}.json"
            row = json.loads(path.read_text())
            if (
                row["fingerprint"] != fingerprint()
                or row["array_sha256"] != base.sha(path.with_suffix(".npz"))
                or row["input_sha256"]
                != [sources[c]["sha256"] for c in (row["source"], row["target"])]
            ):
                raise RuntimeError("changed result provenance")
            rows.append(row)
    if len(rows) != 120 or len(list((out / "pairs").glob("*.json"))) != 120:
        raise RuntimeError("unexpected directed panel coverage")
    base.atomic_json(out / "inputs.json", refs)
    base.atomic_json(out / "parent_results.json", previous)
    base.atomic_json(
        out / "results.json",
        {
            "status": "complete",
            "pairs": rows,
            "coverage": {
                "directed_setting_pairs": 60,
                "representation_panels": 120,
                "fold_evaluations": 600,
            },
            "method": "y_hat = a*x+b, with one scalar a across all coordinates and paired training queries, plus an unconstrained vector b. Both fitted separately for each direction, arm, model, and held-out fold. Comparators: identity, identity plus paired-training mean difference.",
            "displacement_fraction_definition": "1 - sum_heldout ||y-(a*x+b)||^2 / sum_heldout ||y-x||^2; pooled over five folds. 100% is perfect paired reconstruction; 0% matches identity. This is not centered R2 and scaling need not give symmetric scores.",
            "cohort": previous["cohort"],
            "folds": 5,
            "seed": 137,
            "parent_results_sha256": PARENT_SHA,
            "limitations": previous["limitations"][:3]
            + [
                "Scaling fits use target-training pairs, so these are calibrated relationships within each pair, not zero-shot transfer.",
                "Pair cohorts differ; answer residuals include finite-five-rollout noise.",
            ],
            "metadata": base.as_metadata_dict(
                base.git_provenance(cwd=REPO), phase="matched_affine"
            ),
        },
    )


def main():
    """Fit one checkpoint or validate the complete directional comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["fit", "collect"], required=True)
    p.add_argument("--model", choices=base.MODELS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "fit":
        fit(args.out, args.inputs, args.model)
    else:
        collect(args.out, args.inputs)


if __name__ == "__main__":
    main()
