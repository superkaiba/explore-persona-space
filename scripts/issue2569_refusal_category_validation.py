"""Saved-data, category-stratified validation of the fixed L19 refusal decomposition.

No model calls or fitted map. Manifest orientation is retained for behavioral
tests; refusal axes are constructed exclusively from other families/classes.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

SEED = 256919
N_BOOT = 2000  # inherited leg9; finite saved-bank exploratory uncertainty
BASE = Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory")
MANIFEST = BASE / "leg9_dl/leg9_manifest.json"
FACTOR = BASE / "smoke-final/weights_prod/leg1/factor_L19.pt"
MAP = Path(
    "/mnt/eps-data/thomasjiralerspong/issue2094_r2decomp/banked_maps/issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
)
CURRENT = Path(
    "/home/thomasjiralerspong/explore-persona-space/eval_results/issue_2617/svmp_verbharm/perpair.jsonl"
)
VALENCE = ("obj_flip", "verb_flip")


def source(path: Path) -> dict:
    """Content-pin a consumed artifact without modifying it."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def dump(path: Path, value: object) -> None:
    """Persist standards-compliant JSON; nonfinite values must be explicit nulls."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rho_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Spearman, preserving undefined constant-input correlations."""
    a = rankdata(x, axis=-1, nan_policy="omit")
    b = rankdata(y, axis=-1, nan_policy="omit")
    a -= np.nanmean(a, axis=-1, keepdims=True)
    b -= np.nanmean(b, axis=-1, keepdims=True)
    den = np.sqrt(np.nansum(a * a, axis=-1) * np.nansum(b * b, axis=-1))
    out = np.full(den.shape, np.nan)
    np.divide(np.nansum(a * b, axis=-1), den, out=out, where=den > 0)
    return out


def cluster_indices(groups: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Resample whole named families into padded vectorized bootstrap indices."""
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    padded = np.full((len(unique), max(map(len, members))), -1, dtype=int)
    for i, rows in enumerate(members):
        padded[i, : len(rows)] = rows
    draw = np.random.default_rng(seed).integers(len(unique), size=(N_BOOT, len(unique)))
    return padded[draw].reshape(N_BOOT, -1)


def corr(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Rank association and cluster-bootstrap CI; no iid pair p-value."""
    point = float(rho_rows(x, y))
    result = {
        "n": len(x),
        "n_families": len(np.unique(groups)),
        "rho": point if np.isfinite(point) else None,
    }
    if not np.isfinite(point):
        return result | {"ci95": None, "reason": "constant input", "valid_bootstrap": 0}
    if len(np.unique(groups)) < 2:
        return result | {"ci95": None, "reason": "fewer than two independent families"}
    idx = cluster_indices(groups)
    bx, by = np.where(idx >= 0, x[idx], np.nan), np.where(idx >= 0, y[idx], np.nan)
    boot = rho_rows(bx, by)
    good = np.isfinite(boot)
    return result | {
        "ci95": np.quantile(boot[good], [0.025, 0.975]).tolist(),
        "valid_bootstrap": int(good.sum()),
        "undefined_bootstrap": int((~good).sum()),
    }


def describe(x: np.ndarray, groups: np.ndarray) -> dict:
    """Pair-weighted descriptives and whole-family bootstrap median interval."""
    idx = cluster_indices(groups)
    boot = np.nanmedian(np.where(idx >= 0, x[idx], np.nan), axis=1)
    return {
        "n": len(x),
        "n_families": len(np.unique(groups)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "sd": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
        "median_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def crossfit_axes(
    da: np.ndarray, gap: np.ndarray, groups: np.ndarray, primary: np.ndarray
) -> tuple[np.ndarray, list[dict]]:
    """Leave-whole-group-out, positively oriented mean observed refusal direction."""
    axes = np.empty_like(da)
    folds = []
    for group in np.unique(groups):
        test = groups == group
        train = (~test) & primary & (np.abs(gap) >= 0.5)
        if train.sum() < 2:
            raise ValueError(f"Insufficient disjoint axis members: {group}")
        axis = np.mean(da[train] * np.sign(gap[train, None]), axis=0)
        if not np.linalg.norm(axis) > 0:
            raise ValueError("Zero training refusal direction")
        axes[test] = axis / np.linalg.norm(axis)
        folds.append(
            {
                "held_out": str(group),
                "test_indices": np.flatnonzero(test).tolist(),
                "train_axis_indices": np.flatnonzero(train).tolist(),
            }
        )
    return axes, folds


def matched_contrast(
    values: np.ndarray, classes: np.ndarray, families: np.ndarray, control: str
) -> dict:
    """Equal-family contrast with exact sign-flip reference, not randomized causality."""
    shared = sorted(set(families[np.isin(classes, VALENCE)]) & set(families[classes == control]))
    diffs = np.array(
        [
            values[(families == f) & np.isin(classes, VALENCE)].mean()
            - values[(families == f) & (classes == control)].mean()
            for f in shared
        ]
    )
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(diffs))))
    null = (signs * diffs).mean(axis=1)
    obs = float(diffs.mean())
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-14))
    draw = np.random.default_rng(SEED).integers(len(diffs), size=(N_BOOT, len(diffs)))
    return {
        "families": shared,
        "family_differences": diffs.tolist(),
        "mean_difference": obs,
        "ci95": np.quantile(diffs[draw].mean(axis=1), [0.025, 0.975]).tolist(),
        "p_exact_signflip": p,
        "assumption": "independent symmetric family differences; exploratory, nonrandomized categories",
    }


def load_data() -> tuple:
    """Load only frozen bank, judge, saved captures, and map factors; audit joins."""
    paths = json.loads(MANIFEST.read_text())
    keys = [
        "issue2617_svmp/manifests/svmp_bank.json",
        "issue2617_svmp/raw_completions/judge/judge_scores.json",
        "issue2617_svmp/analysis_tensors/vc/vc_langow_bank.pt",
        "issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt",
    ]
    files = [Path(paths[k]) for k in keys]
    bank, judge = [json.loads(p.read_text()) for p in files[:2]]
    vc, va = [torch.load(p, map_location="cpu", weights_only=False, mmap=True) for p in files[2:]]
    assert vc["position"] == "context_end_last_token"
    assert vc["layers"] == va["layers"] and not va["empty_rows"]
    lay = list(vc["layers"]).index(19)
    ids = list(vc["context_ids"])
    row = {k: i for i, k in enumerate(ids)}
    assert len(row) == 248 == len(bank["contexts"]) and len(bank["pairs"]) == 124
    assert set(row) == {r["id"] for r in bank["contexts"]} == set(judge["per_context"])
    contexts = {r["id"]: r for r in bank["contexts"]}
    c = vc["vc"][:, lay].double().numpy()
    count = np.zeros(len(ids))
    a = np.zeros_like(c)
    ar = va["va_tail_incl"][:, lay].double().numpy()
    ari = np.array([row[r["context_id"]] for r in va["index"]])
    assert len({(r["context_id"], r["draw"]) for r in va["index"]}) == 2480
    np.add.at(a, ari, ar)
    np.add.at(count, ari, 1)
    assert np.all(count == 10)
    a /= count[:, None]
    current = {r["pair_id"]: r for r in map(json.loads, CURRENT.read_text().splitlines())}
    assert set(current) == {r["pair_id"] for r in bank["pairs"]}
    rows, differences = [], []
    for p in bank["pairs"]:
        ra, rb = [judge["per_context"][p[k]] for k in ("a", "b")]
        assert ra["n_valid"] == rb["n_valid"] == 10
        gap = float(ra["refusal_rate"] - rb["refusal_rate"])
        old = current[p["pair_id"]]
        if (
            old["refusal_rate_a"] != ra["refusal_rate"]
            or old["refusal_rate_b"] != rb["refusal_rate"]
        ):
            differences.append(
                {
                    "pair_id": p["pair_id"],
                    "frozen_rates": [ra["refusal_rate"], rb["refusal_rate"]],
                    "current_rates": [old["refusal_rate_a"], old["refusal_rate_b"]],
                }
            )
        rows.append(
            p
            | {
                "signed_refusal_gap": gap,
                "absolute_refusal_gap": abs(gap),
                "observed_flip": abs(gap) >= 0.5,
                "rate_a": ra["refusal_rate"],
                "rate_b": rb["refusal_rate"],
                "question_a": contexts[p["a"]]["user"],
                "question_b": contexts[p["b"]]["user"],
                "graded_gap": float(
                    np.mean(list(ra["draw_scores"].values()))
                    - np.mean(list(rb["draw_scores"].values()))
                ),
            }
        )
    ia, ib = [np.array([row[p[k]] for p in rows]) for k in ("a", "b")]
    dc, da = c[ia] - c[ib], a[ia] - a[ib]
    factor = torch.load(FACTOR, map_location="cpu", weights_only=False, mmap=True)
    u, v, s = [
        factor[k].double().numpy() for k in ("read_input_u_fp32", "write_output_v_fp32", "sigma")
    ]
    m = torch.load(MAP, map_location="cpu", weights_only=False, mmap=True)
    assert m["layer"] == 19 and m["kind"] == m["fitter"] == "ridge"
    w = m["W"].double().numpy() / m["xsd"].double().numpy()[:, None]
    assert u.shape == v.shape == w.shape == (3584, 3584) and np.all(np.diff(s) <= 0)
    assert all(np.isfinite(x).all() for x in (dc, da, u, v, s, w))
    z = dc @ u
    pred = dc @ w
    error = np.linalg.norm((z * s) @ v.T - pred) / np.linalg.norm(pred)
    assert error < 2e-6, error
    audit = {
        "sources": [source(p) for p in [MANIFEST, *files, FACTOR, MAP, CURRENT, Path(__file__)]],
        "frozen_snapshot": "74bb871a5edf1afe777ac9b64a4e2fec5e9947c2",
        "factor_map_relative_prediction_error": float(error),
        "current_label_differences": differences,
        "judge_model": judge["judge_model"],
        "n_contexts": len(ids),
        "n_answer_draws": len(ar),
        "n_pairs": len(rows),
    }
    return rows, dc, da, z, pred, u, v, s, audit


def main() -> None:
    """Run bounded batched saved-data analysis and persist each completed phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = datetime.now(UTC).isoformat()
    rows, dc, da, z, pred, u, v, s, audit = load_data()
    dump(args.out / "input_provenance.json", audit)
    print("loaded and validated frozen inputs", flush=True)
    classes = np.array([r["pair_class"] for r in rows])
    families = np.array([r["artifact_family_id"] for r in rows])
    # XSTest family IDs are unique item IDs, not independent semantic families.
    # Hold the entire established benchmark out as one family for axis fitting.
    axis_groups = np.where(classes == "xstest", "ALL_XSTEST", families)
    gap = np.array([r["signed_refusal_gap"] for r in rows])
    primary = classes != "verb_harm"
    axes, folds = crossfit_axes(da, gap, axis_groups, primary)
    class_axes, class_folds = crossfit_axes(da, gap, classes, primary)
    dump(args.out / "crossfit_folds.json", {"family": folds, "class": class_folds})
    metrics = {
        "context_norm": np.linalg.norm(dc, axis=1),
        "normalized_gain": np.linalg.norm(pred, axis=1) / np.linalg.norm(dc, axis=1),
        "signed_refusal_gap": gap,
        "absolute_refusal_gap": abs(gap),
        "predicted_refusal_LOFO": np.sum(pred * axes, axis=1),
        "identity_refusal_LOFO": np.sum(dc * axes, axis=1),
        "observed_refusal_LOFO": np.sum(da * axes, axis=1),
        "predicted_refusal_LOCO": np.sum(pred * class_axes, axis=1),
    }
    masses = {}
    for mass in (0.9, 0.99, 0.999):
        rank = int(np.searchsorted(np.cumsum(s * s) / np.sum(s * s), mass) + 1)
        read = z[:, :rank] @ u[:, :rank].T
        kernel = dc - read
        high_pred = (z[:, :rank] * s[:rank]) @ v[:, :rank].T
        low_pred = pred - high_pred
        share = np.sum(kernel * kernel, axis=1) / np.sum(dc * dc, axis=1)
        components = {
            "kernel_share": share,
            "read_norm": np.linalg.norm(read, axis=1),
            "kernel_norm": np.linalg.norm(kernel, axis=1),
            "read_refusal_LOFO": np.sum(high_pred * axes, axis=1),
            "kernel_refusal_LOFO": np.sum(low_pred * axes, axis=1),
            "kernel_output_energy_share": np.sum(low_pred * low_pred, axis=1)
            / np.sum(pred * pred, axis=1),
        }
        masses[str(mass)] = {
            "read_rank": rank,
            "kernel_rank": 3584 - rank,
            "all_rho_read_vs_gap": corr(components["read_refusal_LOFO"], gap, axis_groups),
            "all_rho_kernel_vs_gap": corr(components["kernel_refusal_LOFO"], gap, axis_groups),
            "median_kernel_share": float(np.median(share)),
            "kernel_vs_gap_by_class": {
                c: corr(share[classes == c], abs(gap[classes == c]), families[classes == c])
                for c in np.unique(classes)
            },
        }
        if mass == 0.99:
            metrics.update(components)
    dump(args.out / "cutoff_sensitivity.json", masses)
    for i, row in enumerate(rows):
        row.update({k: float(value[i]) for k, value in metrics.items()})
    (args.out / "perpair.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    print("saved per-pair geometry, behavior, and held-out axis scores", flush=True)
    category = {}
    for c in np.unique(classes):
        mask = classes == c
        category[c] = {
            "n": int(mask.sum()),
            "n_observed_flips": int(np.sum(abs(gap[mask]) >= 0.5)),
            "descriptive": {
                k: describe(value[mask], families[mask]) for k, value in metrics.items()
            },
            "associations": {
                k: corr(
                    value[mask], gap[mask] if "refusal_" in k else abs(gap[mask]), families[mask]
                )
                for k, value in metrics.items()
                if k
                in (
                    "kernel_share",
                    "normalized_gain",
                    "predicted_refusal_LOFO",
                    "read_refusal_LOFO",
                    "kernel_refusal_LOFO",
                    "identity_refusal_LOFO",
                    "predicted_refusal_LOCO",
                )
            },
        }
    contrasts = {
        f"{control}:{metric}": matched_contrast(metrics[metric], classes, families, control)
        for control in ("subj_ctl", "verb_harm")
        for metric in ("kernel_share", "normalized_gain")
    }
    ordered = sorted(contrasts, key=lambda k: contrasts[k]["p_exact_signflip"])
    previous = 0.0
    for i, k in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - i) * contrasts[k]["p_exact_signflip"]))
        contrasts[k]["p_holm_four_primary_contrasts"] = previous
    flip = primary & (abs(gap) >= 0.5)
    oriented = dc[flip] * np.sign(gap[flip, None])
    mean_context = oriented.mean(axis=0)
    mean_z = mean_context @ u
    rank = masses["0.99"]["read_rank"]
    mean_kernel = mean_context - mean_z[:rank] @ u[:, :rank].T
    direction_stability = {
        "n_primary_flips": int(flip.sum()),
        "mean_direction_kernel_share": float(np.sum(mean_kernel**2) / np.sum(mean_context**2)),
        "pair_kernel_share": describe(metrics["kernel_share"][flip], axis_groups[flip]),
        "pair_cosine_to_pooled_mean": describe(
            (oriented @ mean_context)
            / (np.linalg.norm(oriented, axis=1) * np.linalg.norm(mean_context)),
            axis_groups[flip],
        ),
        "interpretation": "Descriptive concentration, not held-out semantic validation.",
    }
    benign = np.char.endswith(classes, "_benign")
    valence = np.isin(classes, VALENCE)
    benign_contrasts = {}
    for key in ("kernel_share", "normalized_gain"):
        lv, rv = metrics[key][valence], metrics[key][benign]
        li = cluster_indices(families[valence])
        # Distinct reproducible seed avoids artificially paired bootstrap arms.
        ri = cluster_indices(families[benign], seed=SEED + 1)
        boot = np.nanmean(np.where(li >= 0, lv[li], np.nan), axis=1) - np.nanmean(
            np.where(ri >= 0, rv[ri], np.nan), axis=1
        )
        benign_contrasts[key] = {
            "mean_difference": float(lv.mean() - rv.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "n_valence": int(valence.sum()),
            "n_benign": int(benign.sum()),
            "limitation": "Different topic families across arms; descriptive, not domain-matched.",
        }
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "started_utc": start,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": SEED,
        "bootstrap_draws": N_BOOT,
        "primary_mass": 0.99,
        "categories": category,
        "matched_family_contrasts": contrasts,
        "unmatched_benign_contrasts": benign_contrasts,
        "mean_vs_pair_direction_stability": direction_stability,
        "refusal_projection_amplitudes": {
            k: {
                "rms": float(np.sqrt(np.mean(metrics[k] ** 2))),
                "median_absolute": float(np.median(abs(metrics[k]))),
                "rms_relative_to_full_map": float(
                    np.linalg.norm(metrics[k]) / np.linalg.norm(metrics["predicted_refusal_LOFO"])
                ),
            }
            for k in (
                "predicted_refusal_LOFO",
                "read_refusal_LOFO",
                "kernel_refusal_LOFO",
                "identity_refusal_LOFO",
                "observed_refusal_LOFO",
            )
        },
        "pooled": {
            k: corr(value, gap if "refusal_" in k else abs(gap), axis_groups)
            for k, value in metrics.items()
            if k
            in (
                "kernel_share",
                "normalized_gain",
                "predicted_refusal_LOFO",
                "read_refusal_LOFO",
                "kernel_refusal_LOFO",
                "identity_refusal_LOFO",
            )
        },
        "counts": {
            "all": len(rows),
            "primary": int(primary.sum()),
            "primary_flip": int(np.sum(primary & (abs(gap) >= 0.5))),
            "harmharm_flip": int(np.sum((~primary) & (abs(gap) >= 0.5))),
        },
        "limitations": [
            "Exploratory secondary analysis on one saved bank, not prospective validation.",
            "Categories are not randomized; domain-matched contrasts do not isolate semantic harmfulness causally.",
            "XSTest is held out together for axis construction; within-XSTest intervals resample items, not unknown semantic families.",
            "OOF correlation intervals condition on fitted disjoint axes; axes are not re-estimated in bootstrap.",
            "Binary rates and means use ten archived draws per endpoint; no new judging or graded calibration.",
            "Kernel is low-gain, not zero-gain: a small mapped component can correlate with refusal.",
            "No framing manipulation: this cannot validate harmful-request versus jailbreak-framing semantics.",
            "Identity plus learned bias reduces to identity for pair differences. No new map/readout is fit.",
        ],
    }
    dump(args.out / "summary.json", summary)
    lines = [
        "# Existing refusal directions: category validation",
        "",
        "Frozen L19 map and saved 124 pairs; no model calls. Manifest a-minus-b orientation; no selection on evaluated refusal flips.",
        "",
        "| Category | n | Observed flips | Median kernel share | Median normalized gain | LOFO predicted/refusal rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, d in category.items():
        rr = d["associations"]["predicted_refusal_LOFO"]["rho"]
        lines.append(
            f"| {c} | {d['n']} | {d['n_observed_flips']} | {d['descriptive']['kernel_share']['median']:.3f} | {d['descriptive']['normalized_gain']['median']:.3f} | {rr if rr is None else round(rr, 3)} |"
        )
    lines += [
        "",
        "## Domain-matched exploratory contrasts",
        "",
        "Valence = object + verb harmful/benign swaps. Equal weight per shared semantic family.",
    ]
    for k, r in contrasts.items():
        lines.append(
            f"- {k}: {len(r['families'])} shared families; difference {r['mean_difference']:.4f}, family-bootstrap CI {r['ci95']}, exact sign-flip p={r['p_exact_signflip']:.4f}, Holm p={r['p_holm_four_primary_contrasts']:.4f}."
        )
    lines += [
        "",
        "## Refusal association versus projection amplitude",
        "",
        "Scores use a refusal axis estimated without the evaluated semantic family; XSTest is held out as one corpus. Intervals condition on the fitted axes and resample whole held-out groups.",
        "",
        "| Component | Spearman rho | Cluster-bootstrap 95% CI | Projection RMS | RMS / full map |",
        "|---|---:|---|---:|---:|",
    ]
    for k in (
        "predicted_refusal_LOFO",
        "read_refusal_LOFO",
        "kernel_refusal_LOFO",
        "identity_refusal_LOFO",
    ):
        association = summary["pooled"][k]
        amplitude = summary["refusal_projection_amplitudes"][k]
        lines.append(
            f"| {k} | {association['rho']:.3f} | {association['ci95']} | {amplitude['rms']:.4f} | {amplitude['rms_relative_to_full_map']:.5f} |"
        )
    lines += [
        "",
        "The map's projected answer changes track refusal within object swaps, verb swaps, subject swaps, harmful-to-harmful swaps, and the held-out XSTest corpus. All benign-control gaps are zero, so their behavioral correlations are undefined.",
        "The low-gain component also covaries with refusal, despite its small mapped amplitude. This does not support a simple safety-in-read versus nonsafety-in-kernel dichotomy. The map is not superior to identity on this one-dimensional rank-association metric; no test establishes a difference between those correlated correlations.",
        "The two kernel-share contrasts have the same direction in all seven shared families, but Holm-adjusted p=0.0625 across the four primary geometry contrasts. Treat these as exploratory estimates, not conventionally significant confirmatory results.",
    ]
    lines += ["", "## Limitations", ""] + ["- " + s for s in summary["limitations"]]
    lines += [
        "",
        "See summary.json for all class-specific uncertainty, including undefined constant strata; crossfit_folds.json records each held-out/training index. input_provenance.json pins every source and enumerates current-versus-frozen label differences.",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    dump(
        args.out / "completion.json",
        {
            "finished_utc": datetime.now(UTC).isoformat(),
            "exit_code": 0,
            "n_pairs": len(rows),
            "outputs": [source(p) for p in sorted(args.out.iterdir()) if p.is_file()],
        },
    )
    print(json.dumps({"counts": summary["counts"], "pooled": summary["pooled"]}), flush=True)


if __name__ == "__main__":
    main()
