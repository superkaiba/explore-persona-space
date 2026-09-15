"""Character-shift dimensionality on the audited K5 matched-query cohorts."""

# ruff: noqa: E402
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import issue2054_k5_loso_calibration as base
from scripts import issue2054_k5_matched_affine as affine

CHARACTERS = (2, 3, 4, 5)
MODES = ("raw", "centered")


def gram_spectra(grams, dimension):
    """Batch exact symmetric decompositions, with an explicit roundoff tolerance."""
    values, vectors = np.linalg.eigh(grams)
    values, vectors = values[:, ::-1], vectors[:, :, ::-1]
    tolerance = np.finfo(np.float64).eps * max(grams.shape[-1], dimension) * values[:, :1]
    if np.any(values < -64 * tolerance):
        raise ValueError("Gram matrix has materially negative eigenvalues")
    values = np.maximum(values, 0)
    return values, vectors, tolerance[:, 0]


def spectrum_summary(energy):
    """Report energy ranks without calling singular-value mass variance."""
    energy = np.asarray(energy, dtype=np.float64)
    total = float(energy.sum())
    if total == 0:
        return {
            "status": "zero_energy",
            "top1": None,
            "top10": None,
            "top100": None,
            "r50": 0,
            "r90": 0,
            "r95": 0,
            "participation_ratio": 0.0,
            "stable_rank": 0.0,
            "total_energy": 0.0,
        }
    if total < 0:
        raise ValueError("negative spectral energy")
    fraction = energy / total
    cumulative = np.cumsum(fraction)
    return {
        "top1": float(fraction[0]),
        "top10": float(fraction[:10].sum()),
        "top100": float(fraction[:100].sum()),
        "r50": int(np.searchsorted(cumulative, 0.5) + 1),
        "r90": int(np.searchsorted(cumulative, 0.9) + 1),
        "r95": int(np.searchsorted(cumulative, 0.95) + 1),
        "participation_ratio": float(1 / np.square(fraction).sum()),
        "stable_rank": float(1 / fraction[0]),
        "total_energy": total,
    }


def analyze_difference(delta, membership):
    """Reuse one Gram across folds; held-out projections use observed differences."""
    delta = np.asarray(delta, dtype=np.float64)
    membership = np.asarray(membership)
    if delta.ndim != 2 or len(delta) != len(membership) or not np.isfinite(delta).all():
        raise ValueError("invalid difference matrix")
    if set(membership) != set(range(5)):
        raise ValueError("expected all five global folds")
    gram = delta @ delta.T
    centered = gram - gram.mean(0)[None, :] - gram.mean(1)[:, None] + gram.mean()
    constant = np.array_equal(delta, np.broadcast_to(delta[0], delta.shape))
    if constant:
        centered = np.zeros_like(gram)
    values, _, _ = gram_spectra(np.stack([gram, centered]), delta.shape[1])
    arrays = {
        "fold": membership,
        "mean_shift": delta.mean(0),
        "shift_norm": np.linalg.norm(delta, axis=1),
    }
    for name in (
        "constant_error",
        "mean_direction_error",
        "mean_direction_coefficient",
        "mean_direction_cosine",
    ):
        arrays[name] = np.zeros(len(delta))
    arrays["training_mean_shift"] = np.zeros((5, delta.shape[1]))
    report = {"n": len(delta), "dimension": delta.shape[1], "spectra": {}, "folds": []}
    for mode, energy in zip(MODES, values, strict=True):
        arrays[f"{mode}_energy"] = energy
        report["spectra"][mode] = spectrum_summary(energy)
    for fold in range(5):
        tr, te = np.flatnonzero(membership != fold), np.flatnonzero(membership == fold)
        if min(len(tr), len(te)) < 2:
            raise ValueError("insufficient fold coverage")
        train = gram[np.ix_(tr, tr)]
        cross = gram[np.ix_(te, tr)]
        train_center = train - train.mean(0)[None, :] - train.mean(1)[:, None] + train.mean()
        cross_center = cross - cross.mean(1)[:, None] - train.mean(0)[None, :] + train.mean()
        norm = np.diag(gram)[te]
        train_constant = np.array_equal(delta[tr], np.broadcast_to(delta[tr[0]], delta[tr].shape))
        mean = delta[tr[0]] if train_constant else delta[tr].mean(0)
        if train_constant:
            train_center, cross_center = np.zeros_like(train), np.zeros_like(cross)
        mean_energy = float(mean @ mean)
        if mean_energy <= 0 or np.any(norm <= 0):
            raise ValueError("zero shift makes mean-direction alignment undefined")
        coefficient = (delta[te] @ mean) / mean_energy
        if constant:
            coefficient = np.ones(len(te))
        residual_norm = np.square(delta[te] - mean).sum(1)
        direction_error = np.square(delta[te] - coefficient[:, None] * mean).sum(1)
        np.testing.assert_allclose(
            residual_norm,
            direction_error + (coefficient - 1) ** 2 * mean_energy,
            rtol=1e-10,
            atol=1e-8,
        )
        arrays["training_mean_shift"][fold] = mean
        arrays["constant_error"][te] = residual_norm
        arrays["mean_direction_error"][te] = direction_error
        arrays["mean_direction_coefficient"][te] = coefficient
        arrays["mean_direction_cosine"][te] = (delta[te] @ mean) / np.sqrt(norm * mean_energy)
        np.testing.assert_allclose(
            residual_norm, norm - 2 * cross.mean(1) + train.mean(), rtol=1e-10, atol=1e-8
        )
        energy, vectors, tolerances = gram_spectra(np.stack([train, train_center]), delta.shape[1])
        record = {"fold": fold, "n_train": len(tr), "n_test": len(te), "modes": {}}
        arrays[f"fold{fold}_shift_error"] = residual_norm
        for j, (mode, test_cross, test_norm) in enumerate(
            [("raw", cross, norm), ("centered", cross_center, residual_norm)]
        ):
            keep = energy[j] > tolerances[j]
            coordinates = (test_cross @ vectors[j][:, keep]) / np.sqrt(energy[j, keep])
            projected = np.r_[0.0, np.cumsum(np.square(coordinates).sum(0))]
            denominator = float(test_norm.sum())
            if denominator < 0 or projected[-1] > denominator * (1 + 1e-8):
                raise ValueError("invalid orthogonal projection energy")
            arrays[f"fold{fold}_{mode}_train_energy"] = energy[j]
            arrays[f"fold{fold}_{mode}_projection_energy"] = projected
            record["modes"][mode] = {
                "numerically_resolved_directions": int(keep.sum()),
                "gram_eigenvalue_tolerance": float(tolerances[j]),
                "test_energy": denominator,
                "raw_test_energy": float(norm.sum()),
            }
        report["folds"].append(record)
    report["query_constancy"] = {
        "constant_vector_fraction": float(1 - arrays["constant_error"].sum() / np.trace(gram)),
        "variable_amplitude_mean_direction_fraction": float(
            1 - arrays["mean_direction_error"].sum() / np.trace(gram)
        ),
        "fraction_of_constant_residual_perpendicular_to_mean": float(
            arrays["mean_direction_error"].sum() / arrays["constant_error"].sum()
        )
        if arrays["constant_error"].sum() > 0
        else None,
        "cosine_quantiles_10_50_90": np.quantile(
            arrays["mean_direction_cosine"], [0.1, 0.5, 0.9]
        ).tolist(),
        "coefficient_quantiles_10_50_90": np.quantile(
            arrays["mean_direction_coefficient"], [0.1, 0.5, 0.9]
        ).tolist(),
        "shift_norm_cv": float(arrays["shift_norm"].std() / arrays["shift_norm"].mean()),
    }
    report["heldout"] = {}
    for mode in MODES:
        count = min(len(arrays[f"fold{f}_{mode}_projection_energy"]) for f in range(5))
        captured = sum(arrays[f"fold{f}_{mode}_projection_energy"][:count] for f in range(5))
        denominator = sum(f["modes"][mode]["test_energy"] for f in report["folds"])
        raw_energy = float(np.trace(gram))
        arrays[f"{mode}_heldout_fraction"] = (
            captured / denominator if denominator > 0 else np.full(count, np.nan)
        )
        # For centered mode, include the trained mean in the total-shift approximation.
        arrays[f"{mode}_heldout_total_fraction"] = 1 - (denominator - captured) / raw_energy
        curve = arrays[f"{mode}_heldout_fraction"]
        if denominator == 0:
            report["heldout"][mode] = {
                "status": "zero_residual_energy",
                "max_common_rank": 0,
                "fraction_at_rank": {"0": None},
                "r50": 0,
                "r90": 0,
                "r95": 0,
                "max_fraction": None,
                "shift_only_total_fraction": 1.0,
            }
            continue
        report["heldout"][mode] = {
            "max_common_rank": count - 1,
            "fraction_at_rank": {
                str(k): float(curve[k]) for k in [0, 1, 3, 10, 30, 100, 300] if k < count
            },
            "r50": int(np.searchsorted(curve, 0.5)) if curve[-1] >= 0.5 else None,
            "r90": int(np.searchsorted(curve, 0.9)) if curve[-1] >= 0.9 else None,
            "r95": int(np.searchsorted(curve, 0.95)) if curve[-1] >= 0.95 else None,
            "max_fraction": float(curve[-1]),
            "shift_only_total_fraction": float(1 - denominator / raw_energy)
            if mode == "centered"
            else 0.0,
        }
    return report, arrays


def mean_shift_spectrum(character_means):
    """Use a common query cohort so the four-character rank-three bound is exact."""
    means = np.asarray(character_means, dtype=np.float64)
    if means.ndim != 2 or len(means) != 4:
        raise ValueError("expected four character means")
    centered = means - means.mean(0)
    pairs = np.stack([means[j] - means[i] for i in range(4) for j in range(i + 1, 4)])
    energy = np.linalg.svd(centered, compute_uv=False) ** 2
    pair_energy = np.linalg.svd(pairs, compute_uv=False) ** 2
    np.testing.assert_allclose(pair_energy[:3], 4 * energy[:3], rtol=1e-10, atol=1e-10)
    if energy[3] > energy[0] * 1e-20 or pair_energy[3:].sum() > pair_energy[0] * 1e-20:
        raise ValueError("common-cohort mean shifts violate rank-three bound")
    return spectrum_summary(energy), {
        "character_means": means,
        "centroid_energy": energy,
        "pair_energy": pair_energy,
    }


def fingerprint():
    """Pin the implementation and reused input loader."""
    return {
        "script_sha256": base.sha(__file__),
        "loader_source": inspect.getsource(base.load_panel),
        "parent_results_sha256": affine.PARENT_SHA,
    }


def fit(out, inputs, model):
    """Audit cached banks, checkpoint each character pair, then analyze mean shifts."""
    parent, previous = affine.parent_report(out)
    refs = {
        r["path"]: r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]
    }
    panel = base.load_panel(model, refs)
    indexes = {cell: {cid: i for i, cid in enumerate(p["ids"])} for cell, p in panel.items()}
    old_rows = [
        p
        for p in previous["pairs"]
        if p["model"] == model
        and p["source_index"] in CHARACTERS
        and p["target_index"] in CHARACTERS
    ]
    if len(old_rows) != 12:
        raise RuntimeError("expected six character pairs and two representation summaries")
    cohorts = []
    for old in old_rows:
        started = time.monotonic()
        i, j, arm = old["source_index"], old["target_index"], old["arm"]
        source, target = old["source"], old["target"]
        prior = parent / "pairs" / f"{model}__{i}_{j}__{arm}.npz"
        expected = [
            refs[f"{base.PREFIX}/production_v1/k5/{c}.npz"]["sha256"] for c in (source, target)
        ]
        if base.sha(prior) != old["array_sha256"] or expected != old["input_sha256"]:
            raise RuntimeError("parent array or input bank identity changed")
        with np.load(prior, allow_pickle=False) as z:
            ids, folds, prior_error = z["conv_id"], z["fold"], z["residual_energy"]
        cohorts.append(set(ids))
        ia, ib = [np.array([indexes[c][str(cid)] for cid in ids]) for c in (source, target)]
        if len(ids) != old["n_paired"] or not all(
            np.array_equal(panel[c]["membership"][idx], folds)
            for c, idx in [(source, ia), (target, ib)]
        ):
            raise RuntimeError("query cohort or global fold changed")
        key = "x" if arm == "context" else "y"
        result, arrays = analyze_difference(panel[target][key][ib] - panel[source][key][ia], folds)
        for f in range(5):
            np.testing.assert_allclose(
                arrays[f"fold{f}_shift_error"], prior_error[folds == f], rtol=1e-10, atol=1e-8
            )
        np.testing.assert_allclose(
            result["heldout"]["centered"]["shift_only_total_fraction"],
            old["constant_fraction"],
            atol=1e-12,
        )
        arrays["conv_id"] = ids
        path = out / "pairs" / f"{model}__{i}_{j}__{arm}.json"
        if path.exists():
            raise RuntimeError("fresh output path required")
        base.save_npz(path.with_suffix(".npz"), arrays)
        result.update(
            model=model,
            arm=arm,
            source_index=i,
            target_index=j,
            source=source,
            target=target,
            array_sha256=base.sha(path.with_suffix(".npz")),
            parent_array_sha256=old["array_sha256"],
            input_sha256=expected,
            fingerprint=fingerprint(),
            wall_seconds=time.monotonic() - started,
        )
        base.atomic_json(path, result)
        print(
            f"[phase=pair] {model} {i}-{j} {arm} seconds={result['wall_seconds']:.2f} centered_r90={result['spectra']['centered']['r90']} heldout_top10={result['heldout']['centered']['fraction_at_rank']['10']:.4f}",
            flush=True,
        )
    common = sorted(set.intersection(*cohorts))
    if len(common) < 2:
        raise RuntimeError("no common query cohort for character means")
    cells = [
        next(p["source"] for p in old_rows if p["source_index"] == i)
        if i < 5
        else next(p["target"] for p in old_rows if p["target_index"] == i)
        for i in CHARACTERS
    ]
    for arm, key in [("context", "x"), ("answer", "y")]:
        means = np.stack(
            [panel[c][key][[indexes[c][str(cid)] for cid in common]].mean(0) for c in cells]
        )
        record, arrays = mean_shift_spectrum(means)
        arrays["conv_id"] = np.array(common)
        path = out / "means" / f"{model}__{arm}.json"
        if path.exists():
            raise RuntimeError("fresh mean-shift output path required")
        base.save_npz(path.with_suffix(".npz"), arrays)
        record.update(
            model=model,
            arm=arm,
            n_common_queries=len(common),
            characters=list(CHARACTERS),
            array_sha256=base.sha(path.with_suffix(".npz")),
            fingerprint=fingerprint(),
        )
        base.atomic_json(path, record)
        print(f"[phase=means] {model} {arm} n={len(common)} top1={record['top1']:.4f}", flush=True)


def collect(out, inputs):
    """Verify every expected spectrum and persist an explicit analysis contract."""
    parent, previous = affine.parent_report(out)
    refs = [r for r in json.loads(inputs.read_text()) if "/production_v1/k5/" in r["path"]]
    for ref in refs:
        if base.sha(ref["local"]) != ref["sha256"]:
            raise RuntimeError("input bank changed during analysis")
    rows, means = [], []
    expected = [
        p
        for p in previous["pairs"]
        if p["source_index"] in CHARACTERS and p["target_index"] in CHARACTERS
    ]
    old_by_key = {
        f"{p['model']}__{p['source_index']}_{p['target_index']}__{p['arm']}": p for p in expected
    }
    current_hashes = {Path(r["path"]).stem: r["sha256"] for r in refs}
    for folder, keys, dest in [
        (
            "pairs",
            [
                f"{p['model']}__{p['source_index']}_{p['target_index']}__{p['arm']}"
                for p in expected
            ],
            rows,
        ),
        ("means", [f"{m}__{a}" for m in base.MODELS for a in ("context", "answer")], means),
    ]:
        if len(list((out / folder).glob("*.json"))) != len(keys):
            raise RuntimeError("unexpected output coverage")
        for key in keys:
            path = out / folder / f"{key}.json"
            row = json.loads(path.read_text())
            if row["fingerprint"] != fingerprint() or row["array_sha256"] != base.sha(
                path.with_suffix(".npz")
            ):
                raise RuntimeError("changed result provenance")
            if folder == "pairs":
                old = old_by_key[key]
                if (
                    row["input_sha256"] != old["input_sha256"]
                    or row["input_sha256"]
                    != [current_hashes[c] for c in (old["source"], old["target"])]
                    or row["parent_array_sha256"] != old["array_sha256"]
                    or row["parent_array_sha256"] != base.sha(parent / "pairs" / f"{key}.npz")
                    or row["n"] != old["n_paired"]
                ):
                    raise RuntimeError(
                        "collected pair no longer binds the pinned parent and current inputs"
                    )
            dest.append(row)
    base.atomic_json(out / "inputs.json", refs)
    base.atomic_json(
        out / "results.json",
        {
            "status": "complete",
            "pairs": rows,
            "mean_shifts": means,
            "coverage": {
                "representation_panels": 24,
                "fold_evaluations": 120,
                "mean_shift_panels": 4,
            },
            "parent_results_sha256": affine.PARENT_SHA,
            "cohort": previous["cohort"],
            "seed": 137,
            "method": "D has one row per literal matched query and columns for representation coordinates; D=target-source. Raw and mean-centered spectra use squared singular values. All six unordered character pairs, two models, context and K5-mean answers. Full-cohort descriptive spectra; train-only subspaces projected onto observed held-out differences under five global conversation folds.",
            "limitations": [
                "Held-out projection uses the observed target difference: subspace coverage, not prediction or zero-shot transfer.",
                "Pair cohorts differ; common-cohort mean-shift analysis separately uses the intersection across all four characters.",
                "Four character means have contrast rank at most three by construction.",
                "Story scaffolds differ across characters; finite-five-rollout answer noise is not corrected.",
                "Ranks are measured in the observed sample and bounded by sample count, not estimates of a noise-free population rank.",
            ],
            "metadata": base.as_metadata_dict(base.git_provenance(cwd=REPO), phase="matched_rank"),
        },
    )


def main():
    """Run one checkpoint worker or verify the complete character-shift analysis."""
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
