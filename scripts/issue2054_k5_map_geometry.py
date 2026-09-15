"""Rank and similarity of the existing K5 own and six-setting pooled maps.

Restore the published estimators at their already-selected ridge penalties;
do not choose new penalties or change the training examples. Work in raw
activation coordinates and evaluate all operators on identical held-out inputs.
"""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_loso_calibration as base
from scripts.issue2054_k5_matched_rank import spectrum_summary

LABELS = ["Chat", "Plain", "HELIOS", "Wren", "Dana", "Vex", "Shared"]
REFERENCE_SHA = "90309ec95e757ee5bd2e1b941e006858bd24a328656386444b6c55fb43f928f2"


def log(out, phase, **fields):
    """Persist current analysis progress and expose it to active supervision."""
    record = {"checked_at": time.time(), "phase": phase, **fields}
    suffix = f"_{fields['model']}" if "model" in fields else ""
    base.atomic_json(out / f"progress{suffix}.json", record)
    print(json.dumps(record), flush=True)


def load_panel(model, sources, out):
    """Inherit the pinned K5 loader contract, with explicit array-load progress.

    Torch performs the lossless float16/float32-to-float64 conversion. The
    original NumPy whole-array conversion stalled on this VM; a real input
    slice was checked for exact equality before adopting this backend.
    """
    content = subprocess.check_output(
        ["git", "show", f"{base.SOURCE_SHA}:eval_results/issue_2054/shared_fold_map.json"],
        cwd=REPO,
    )
    fold_map = json.loads(content)
    if fold_map["k"] != 5 or fold_map["seed"] != 137:
        raise ValueError("unexpected conversation folds")
    panel = {}
    for _, prefix in base.SETTINGS:
        cell = f"{prefix}__{model}"
        source = sources[f"{base.PREFIX}/production_v1/k5/{cell}.npz"]
        if base.sha(source["local"]) != source["sha256"]:
            raise ValueError("K5 activation hash mismatch")
        with np.load(source["local"], allow_pickle=False) as z:
            ids = list(map(str, z["conv_id"]))
            arrays = {}
            for key in ["v_C", "v_A"]:
                started = time.monotonic()
                raw = z[key]
                arrays[key] = torch.from_numpy(raw).to(torch.float64).numpy()
                np.testing.assert_array_equal(arrays[key][:10], raw[:10].astype(np.float64))
                log(
                    out,
                    "array_loaded",
                    model=model,
                    cell=cell,
                    key=key,
                    seconds=time.monotonic() - started,
                )
            caps = z["cap_mask"]
        x, y = arrays["v_C"], arrays["v_A"]
        if x.shape != y.shape or x.shape != (len(ids), 3584) or len(set(ids)) != len(ids):
            raise ValueError("invalid K5 layout")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or caps.shape != (len(ids), 5):
            raise ValueError("invalid K5 values or rollout coverage")
        panel[cell] = {
            "x": x,
            "y": y,
            "ids": ids,
            "caps": caps,
            "membership": np.array([fold_map["fold_of"][cid] for cid in ids]),
        }
    return panel


def moments_by_fold(p):
    """Batch the five disjoint fold moments, padding with zero rows only."""
    if p["x"].shape != p["y"].shape:
        raise ValueError("this analysis requires equal context and answer dimensions")
    rows = [np.flatnonzero(p["membership"] == f) for f in range(5)]
    if any(len(r) == 0 for r in rows):
        raise ValueError("empty conversation fold")
    shape = (5, max(map(len, rows)), p["x"].shape[1])
    x, y = torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)
    for f, r in enumerate(rows):
        x[f, : len(r)] = torch.from_numpy(p["x"][r])
        y[f, : len(r)] = torch.from_numpy(p["y"][r])
    return {
        "n": torch.tensor(list(map(len, rows)), dtype=torch.float64),
        "sx": x.sum(1),
        "sy": y.sum(1),
        "xx": x.transpose(1, 2) @ x,
        "xy": x.transpose(1, 2) @ y,
    }


def restore_maps(bank, fold, lambdas):
    """Batch the moment-form ridge equations at the banked penalties.

    This is PooledMomentRidge's standardized covariance and cross-covariance,
    solved by Cholesky at the published lambda rather than re-running GCV.
    Return A,b so that raw row vectors predict as x @ A + b.
    """
    if len(lambdas) != len(bank) + 1 or any(lam <= 0 for lam in lambdas):
        raise ValueError("invalid banked penalties")
    train = {}
    for key in ["n", "sx", "sy", "xx", "xy"]:
        own = torch.stack([m[key].sum(0) - m[key][fold] for m in bank])
        train[key] = torch.cat([own, own.sum(0, keepdim=True)])
    n = train["n"]
    mu_x, mu_y = train["sx"] / n[:, None], train["sy"] / n[:, None]
    var = train["xx"].diagonal(dim1=-2, dim2=-1) / n[:, None] - mu_x.square()
    if bool((var < -1e-9).any()):
        raise ValueError("negative input variance beyond roundoff")
    sd = var.clamp_min(0).sqrt() + 1e-9
    cov = train["xx"] - n[:, None, None] * mu_x[:, :, None] * mu_x[:, None, :]
    cov /= sd[:, :, None] * sd[:, None, :]
    cross = (train["xy"] - mu_x[:, :, None] * train["sy"][:, None, :]) / sd[:, :, None]
    cov.diagonal(dim1=-2, dim2=-1).add_(torch.tensor(lambdas, dtype=torch.float64)[:, None])
    weights = torch.cholesky_solve(cross, torch.linalg.cholesky(cov))
    raw = weights / sd[:, :, None]
    bias = mu_y - torch.einsum("bi,bij->bj", mu_x, raw)
    return raw, bias, mu_y - mu_x, n, sd


def cosine_gram(flat):
    """Vectorized Frobenius cosine and norm-aware distance for a map bank."""
    gram = flat @ flat.T
    norms2 = gram.diagonal()
    if bool((norms2 <= 0).any()) or not bool(torch.isfinite(gram).all()):
        raise ValueError("zero or nonfinite operator/prediction energy")
    cosine = gram / torch.sqrt(norms2[:, None] * norms2[None, :])
    difference = (norms2[:, None] + norms2[None, :] - 2 * gram).clamp_min(0)
    distance = torch.sqrt(difference / ((norms2[:, None] + norms2[None, :]) / 2))
    return cosine, distance


def retrieval(predictions, target):
    """Batch all source-map retrievals over the identical held-out target pool."""
    # Match mapping_baselines.knn_retrieval's float64 tolerance mid-ranks.
    similarity = predictions @ target.T
    target_norm2 = target.square().sum(1)
    pred_norm2 = predictions.square().sum(2)
    if bool((target_norm2 <= 0).any()) or bool((pred_norm2 <= 0).any()):
        raise ValueError("zero norm in retrieval")
    result = {"pool_size": len(target), "chance_top1": 1 / len(target)}
    for metric in ["euclidean", "cosine"]:
        if metric == "euclidean":
            distance = pred_norm2[:, :, None] + target_norm2 - 2 * similarity
        else:
            distance = 1 - similarity / (
                (pred_norm2.sqrt()[:, :, None] + 1e-12) * (target_norm2.sqrt() + 1e-12)
            )
        true_distance = distance.diagonal(dim1=-2, dim2=-1)[:, :, None]
        tolerance = 1e-9 * true_distance.abs().clamp_min(1e-12)
        closer = (distance < true_distance - tolerance).sum(2)
        tied = ((distance - true_distance).abs() <= tolerance).sum(2) - 1
        ranks = 1 + closer + 0.5 * tied
        result[f"{metric}_top1"] = (ranks <= 1).double().mean(1).tolist()
    return result


def evaluate(panel, cells, maps, bias, identity_bias, fold, reference):
    """Check parent parity and compare predictions without setting mean offsets."""
    results = []
    gram_sum = torch.zeros((len(maps), len(maps)), dtype=torch.float64)
    max_delta = 0.0
    for target_index, cell in enumerate(cells):
        p = panel[cell]
        mask = p["membership"] == fold
        x, y = torch.from_numpy(p["x"][mask]), torch.from_numpy(p["y"][mask])
        prediction = x @ maps + bias[:, None, :]
        centered = prediction - prediction.mean(1, keepdim=True)
        flat = centered.reshape(len(maps), -1)
        gram_sum += (flat @ flat.T) / len(x)  # Equal weight per setting.
        cosine, distance = cosine_gram(flat)
        sst = (y - y.mean(0)).square().sum()
        r2 = 1 - (prediction - y).square().sum((1, 2)) / sst
        baseline_r2 = 1 - (x + identity_bias[:, None, :] - y).square().sum((1, 2)) / sst
        nn = retrieval(prediction, y)
        expected = reference[cell]["folds"][fold]["metrics"]
        for i, name in [(target_index, "own"), (len(maps) - 1, "pooled")]:
            delta = abs(float(r2[i]) - expected[name]["r2"])
            if delta > 1e-6:
                raise RuntimeError(f"parent R2 parity failure {cell} {fold} {name}: {delta}")
            max_delta = max(max_delta, delta)
            for metric, read in [("euclidean", "euclidean_top1"), ("cosine", "cosine_top1")]:
                original = expected[name]["retrieval"][metric]["acc_at_k"]["1"]
                if abs(nn[read][i] - original) > 1e-12:
                    raise RuntimeError(f"parent retrieval parity failure {cell} {fold} {name}")
        results.append(
            {
                "target": LABELS[target_index],
                "n_test": len(x),
                "r2": r2.tolist(),
                "source_identity_bias_r2": baseline_r2.tolist(),
                "retrieval": nn,
                "centered_prediction_cosine": cosine.tolist(),
                "centered_prediction_relative_distance": distance.tolist(),
            }
        )
    energy = gram_sum.diagonal()
    pooled_cosine = gram_sum / torch.sqrt(energy[:, None] * energy[None, :])
    return results, pooled_cosine.tolist(), max_delta


def analyze_model(model, out, max_folds):
    """Analyze seven operators in each conversation fold without changing fits."""
    started = time.time()
    inputs_path = REPO / "eval_results/issue_2054/k5_plain_assistant_transfer/inputs.json"
    sources = {r["path"]: r for r in json.loads(inputs_path.read_text())}
    for ref in base.references()["panels"]:
        source = sources[f"{base.PREFIX}/production_v1/k5/{ref['cell']}.npz"]
        hashes = {fold["input_sha256"] for fold in ref["folds"]}
        if hashes != {source["sha256"]} or source["revision"] != base.K5_REV:
            raise ValueError("input manifest differs from immutable K5 provenance")
    log(out, "load_inputs", model=model)
    panel = load_panel(model, sources, out)
    cells = [f"{prefix}__{model}" for _, prefix in base.SETTINGS]
    reference_path = REPO / "eval_results/issue_2054/section44_k5/k5_results.json"
    if base.sha(reference_path) != REFERENCE_SHA:
        raise ValueError("published K5 reference changed")
    data = json.loads(reference_path.read_text())
    reference = {
        p["cell"]: p for p in data["results"] if p["k_rollouts"] == 5 and p["cohort"] == "all"
    }
    if any(c not in reference or reference[c]["status"] != "complete" for c in cells):
        raise ValueError("incomplete K5 reference")
    bank = []
    for cell in cells:
        bank.append(moments_by_fold(panel[cell]))
        log(out, "moments", model=model, settings_done=len(bank), settings_total=6)
    folds, map_bank = [], []
    for fold in range(max_folds):
        log(out, "restore", model=model, fold=fold)
        infos = [reference[c]["folds"][fold]["ridge"] for c in cells]
        pooled_info = reference[cells[0]]["folds"][fold]["pooled_ridge"]
        if any(reference[c]["folds"][fold]["pooled_ridge"] != pooled_info for c in cells):
            raise ValueError("pooled estimator metadata disagree")
        infos.append(pooled_info)
        lambdas = [r["best_lambda"] for r in infos]
        maps, bias, identity_bias, ns, sd = restore_maps(bank, fold, lambdas)
        if ns.tolist() != [r["n_train"] for r in infos]:
            raise ValueError("restored training sizes differ from parent")
        # Independently compare the two assistant operators to saved matrices.
        for index, directory, regime in [
            (0, "k5_assistant_transfer", "assistant_only"),
            (1, "k5_plain_assistant_transfer", "assistant_plain_only"),
        ]:
            path = (
                REPO / f"eval_results/issue_2054/{directory}/maps/{model}__{regime}__fold{fold}.npz"
            )
            receipt = json.loads(path.with_suffix(".json").read_text())
            if base.sha(path) != receipt["sha256"]:
                raise ValueError("saved assistant map changed")
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_allclose(
                    maps[index], z["map"] / z["sd"][:, None], atol=1e-8, rtol=1e-7
                )
        log(out, "spectra", model=model, fold=fold)
        singular = torch.linalg.svdvals(maps)
        spectra = [spectrum_summary(np.square(v.numpy())) for v in singular]
        tolerance = singular[:, 0] * maps.shape[-1] * np.finfo(np.float64).eps
        for i, spec in enumerate(spectra):
            spec["numerical_rank"] = int((singular[i] > tolerance[i]).sum())
            spec["numerical_rank_tolerance"] = float(tolerance[i])
        raw_cosine, raw_distance = cosine_gram(maps.reshape(7, -1))
        log(out, "heldout_predictions", model=model, fold=fold)
        evaluations, prediction_cosine, parity = evaluate(
            panel, cells, maps, bias, identity_bias, fold, reference
        )
        record = {
            "model": model,
            "fold": fold,
            "labels": LABELS,
            "ridge": infos,
            "spectra": spectra,
            "raw_operator_cosine": raw_cosine.tolist(),
            "raw_operator_relative_distance": raw_distance.tolist(),
            "centered_pooled_prediction_cosine": prediction_cosine,
            "evaluations": evaluations,
            "max_parent_r2_error": parity,
            "bias_norms": bias.norm(dim=1).tolist(),
        }
        stem = f"{model}__fold{fold}"
        base.save_npz(
            out / "folds" / f"{stem}.npz",
            {
                "singular_values": singular.numpy(),
                "raw_intercept": bias.numpy(),
                "input_std": sd.numpy(),
            },
        )
        base.atomic_json(out / "folds" / f"{stem}.json", record)
        folds.append(record)
        map_bank.append(maps)
        log(out, "fold_complete", model=model, folds_done=len(folds), max_parent_r2_error=parity)
    # Same-map agreement across overlapping CV training sets is descriptive,
    # not an independent replicate or a noise ceiling.
    stability = []
    for i, label in enumerate(LABELS):
        flat = torch.stack([m[i].reshape(-1) for m in map_bank])
        cosine, _ = cosine_gram(flat)
        stability.append({"label": label, "across_fold_cosine": cosine.tolist()})
    result = {
        "model": model,
        "folds": folds,
        "fold_stability": stability,
        "script_sha256": base.sha(__file__),
        "started_at": started,
        "completed_at": time.time(),
    }
    base.atomic_json(out / f"{model}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=base.MODELS)
    parser.add_argument("--folds", type=int, choices=[1, 5], default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    models = [args.model] if args.model else list(base.MODELS)
    started = time.time()
    if not args.collect_only:
        for model in models:
            analyze_model(model, args.out, args.folds)
    results = [json.loads((args.out / f"{model}.json").read_text()) for model in models]
    if any(
        r["script_sha256"] != base.sha(__file__) or len(r["folds"]) != args.folds for r in results
    ):
        raise ValueError("model result source or fold coverage mismatch")
    report = {
        "status": "complete" if args.folds == 5 and len(models) == 2 else "partial",
        "labels": LABELS,
        "models": results,
        "script_sha256": base.sha(__file__),
        "k5_revision": base.K5_REV,
        "reference_sha256": base.sha(REPO / "eval_results/issue_2054/section44_k5/k5_results.json"),
        "started_at": min(r["started_at"] for r in results),
        "completed_at": max(r["completed_at"] for r in results),
        "method": "Restore published ridge maps at banked lambdas via batched moment-form Cholesky; raw operator A=diag(1/sd)W; exact float64 SVD; Frobenius cosine; equal-setting-weight centered held-out prediction cosine.",
        "limitations": [
            "Same matrix dimensions and parameter count do not imply identical effective capacity.",
            "Coefficient spectra depend on coordinates; all primary comparisons use the same raw activation coordinates within each checkpoint.",
            "Ridge penalty, number and distribution of training rows differ; rank comparisons do not isolate a causal capacity effect.",
            "Cross-fold training sets overlap, so fold stability is descriptive rather than an independent reliability ceiling.",
            "Prediction similarity is measured on identical held-out contexts; output means are removed within each setting.",
            "Large restored coefficient matrices stay in memory; regenerate from pinned K5 inputs, published penalties and this script. Persist spectra and metrics; no data generation is discarded.",
        ],
    }
    filename = f"{args.model}.partial_result.json" if args.model else "results.json"
    base.atomic_json(args.out / filename, report)
    fields = {"model": args.model} if args.model else {}
    log(args.out, "complete", status=report["status"], wall_seconds=time.time() - started, **fields)


if __name__ == "__main__":
    main()
