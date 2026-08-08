"""How deep into the R^2 ranking do the judged binary label contrasts separate?

The full-panel Spearman is ~null (abstraction rho = -0.057, n=16,328) while the
tail contrasts are strong (Set A/B OR ~3-4). This quantifies WHERE the signal
lives: for a grid of tail widths k, Delta_k = frac(label | top-k by R^2) -
frac(label | bottom-k), against an activity-decile-stratified permutation null
(pointwise band + a scan-corrected max-over-k band, since nested k are
dependent), plus a 20-bin prevalence-vs-rank profile over the full panel.

Inputs all banked/local (#1773 axis labels + recovery replacements; #1482
sae_ctx per-feature R^2 + activity). 0 GPU, seconds on VM CPU.
"""

from __future__ import annotations

import json
import sys

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

LABELS = "eval_results/issue_1773/labels/axis_labels.jsonl"
RECOVERED = "eval_results/issue_1773/recovery_1934/axis_labels_recovered.jsonl"
PERFEATURE = "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
OUT = "eval_results/issue_1482/feature_correlates/tail_depth_sweep.json"
FIG_DIR = "figures/issue_1482/feature_correlates"

K_GRID = (25, 50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 8000)
N_PERM = 2000
N_DECILES = 10
N_BINS = 20
SEED = 1482

# coding -> (source axis, positive-class predicate over the modal label, rows to drop)
AXES = {
    "abstraction_high": ("abstraction", lambda s: s == "abstract_contextual", ()),
    "speaker_any": (
        "speaker_property",
        lambda s: s in ("language", "register_style", "identity_disposition"),
        ("unclear",),
    ),
    "speaker_language": ("speaker_property", lambda s: s == "language", ("unclear",)),
    "speaker_register": ("speaker_property", lambda s: s == "register_style", ("unclear",)),
}


def _modal_labels(axis: str) -> dict[int, str]:
    """Modal label per feat_id for one axis; recovery rows REPLACE originals."""
    out: dict[int, str] = {}
    for path in (LABELS, RECOVERED):
        p = PROJECT_ROOT / path
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["axis"] == axis:
                    out[int(r["feat_id"])] = r["label"]
    return out


def _decile_of(activity: np.ndarray) -> np.ndarray:
    edges = np.quantile(activity, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    return np.searchsorted(edges, activity, side="right")


def _perm_within_strata(lab: np.ndarray, strata: np.ndarray, rng) -> np.ndarray:
    """(n, N_PERM) int8 matrix of labels permuted independently within strata."""
    n = len(lab)
    P = np.empty((n, N_PERM), dtype=np.int8)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        vals = lab[idx]
        for p in range(N_PERM):
            P[idx, p] = vals[rng.permutation(len(idx))]
    return P


def _wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.959964
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return (float(c - h), float(c + h))


def main() -> None:
    rng = np.random.default_rng(SEED)
    z = np.load(PROJECT_ROOT / PERFEATURE)
    feat_ids = np.asarray(z["feat_ids"], dtype=int)
    r2 = np.asarray(z["r2"], dtype=np.float64)
    activity = np.asarray(z["activity"], dtype=np.float64)

    doc: dict = {
        "design": {
            "question": (
                "How deep into the per-feature R^2 ranking does each judged binary "
                "label separate? Delta_k = frac(label|top-k) - frac(label|bottom-k) "
                "over a grid of tail widths, vs an activity-decile-stratified "
                "permutation null (pointwise 2.5/97.5 band + scan-corrected "
                "max-over-k band), plus a 20-bin prevalence-vs-rank profile."
            ),
            "r2_source": PERFEATURE,
            "labels": "modal per-feature #1773 axis labels, recovery_1934 replacements applied",
            "n_perm": N_PERM,
            "k_grid": list(K_GRID),
            "note_dependence": (
                "nested k are dependent; the pointwise band is descriptive per-k, "
                "the max-over-k band controls for scanning the grid"
            ),
        },
        "axes": {},
    }

    for coding, (axis, pos_fn, drop) in AXES.items():
        labels = _modal_labels(axis)
        keep_i, lab_list = [], []
        for i, fid in enumerate(feat_ids):
            s = labels.get(int(fid))
            if s is None or s in drop or not np.isfinite(r2[i]):
                continue
            keep_i.append(i)
            lab_list.append(1 if pos_fn(s) else 0)
        keep = np.asarray(keep_i, dtype=int)
        lab = np.asarray(lab_list, dtype=np.int8)
        r2_a, act_a = r2[keep], activity[keep]
        n = len(keep)
        order = np.argsort(-r2_a)  # descending: rank 0 = best-predicted
        lab_sorted = lab[order]
        strata = _decile_of(act_a)

        # observed Delta_k for every k at once (one cumsum from each end)
        cs_top = np.cumsum(lab_sorted, dtype=np.int64)
        cs_bot = np.cumsum(lab_sorted[::-1], dtype=np.int64)
        ks = np.asarray([k for k in K_GRID if 2 * k <= n], dtype=int)
        d_obs = cs_top[ks - 1] / ks - cs_bot[ks - 1] / ks

        # permutation null: (n, N_PERM) labels shuffled within activity decile,
        # reordered by the FIXED R^2 sort, then the same two cumsums
        P = _perm_within_strata(lab, strata, rng)[order]
        cs_top_p = np.cumsum(P, axis=0, dtype=np.int64)
        cs_bot_p = np.cumsum(P[::-1], axis=0, dtype=np.int64)
        d_perm = cs_top_p[ks - 1] / ks[:, None] - cs_bot_p[ks - 1] / ks[:, None]  # (K, N_PERM)

        lo = np.percentile(d_perm, 2.5, axis=1)
        hi = np.percentile(d_perm, 97.5, axis=1)
        # scan correction: STUDENTIZED max-T (Westfall-Young) — raw |Delta| is
        # not comparable across k (small-k Deltas are far noisier and would
        # dominate a raw max), so standardize each k by its permutation
        # mean/sd, then take max_k |z| per permutation.
        mu_k = d_perm.mean(axis=1)
        sd_k = d_perm.std(axis=1)
        z_obs = (d_obs - mu_k) / sd_k
        z_perm = (d_perm - mu_k[:, None]) / sd_k[:, None]
        scan_thresh_z = float(np.percentile(np.abs(z_perm).max(axis=0), 95))
        outside_pw = (d_obs < lo) | (d_obs > hi)
        outside_scan = np.abs(z_obs) > scan_thresh_z
        p_per_k = ((np.abs(d_perm) >= np.abs(d_obs)[:, None]).sum(axis=1) + 1) / (N_PERM + 1)

        def _depth(mask: np.ndarray) -> int | None:
            hit = ks[mask]
            return int(hit.max()) if len(hit) else None

        # prevalence-vs-rank profile (equal-count bins, best-predicted first)
        edges = np.linspace(0, n, N_BINS + 1).astype(int)
        prof = []
        for b in range(N_BINS):
            seg = lab_sorted[edges[b] : edges[b + 1]]
            ci = _wilson(int(seg.sum()), len(seg))
            prof.append(
                {
                    "bin": b,
                    "n": int(len(seg)),
                    "prevalence": float(seg.mean()),
                    "wilson95": ci,
                    "r2_range": [
                        float(r2_a[order][edges[b + 1] - 1]),
                        float(r2_a[order][edges[b]]),
                    ],
                }
            )

        doc["axes"][coding] = {
            "source_axis": axis,
            "n": n,
            "n_positive": int(lab.sum()),
            "marginal_prevalence": float(lab.mean()),
            "k": [int(x) for x in ks],
            "delta_obs": [float(x) for x in d_obs],
            "perm_band_pointwise_2p5": [float(x) for x in lo],
            "perm_band_pointwise_97p5": [float(x) for x in hi],
            "perm_p_per_k": [float(x) for x in p_per_k],
            "z_obs": [float(x) for x in z_obs],
            "scan_corrected_threshold_z": scan_thresh_z,
            "scan_band_delta_lo": [
                float(mu_k[j] - scan_thresh_z * sd_k[j]) for j in range(len(ks))
            ],
            "scan_band_delta_hi": [
                float(mu_k[j] + scan_thresh_z * sd_k[j]) for j in range(len(ks))
            ],
            "outside_pointwise": [bool(x) for x in outside_pw],
            "outside_scan_corrected": [bool(x) for x in outside_scan],
            "separation_depth_pointwise": _depth(outside_pw),
            "separation_depth_scan_corrected": _depth(outside_scan),
            "prevalence_profile": prof,
        }

    out_path = PROJECT_ROOT / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"[out] {out_path}")

    set_paper_style()
    import matplotlib.pyplot as plt

    codings = list(AXES)
    colors = paper_palette(len(codings))
    fig, axes_f = plt.subplots(len(codings), 2, figsize=(12.5, 4.2 * len(codings)))
    for row, coding in enumerate(codings):
        a = doc["axes"][coding]
        ks = np.asarray(a["k"])
        ax = axes_f[row, 0]
        ax.fill_between(
            ks,
            a["perm_band_pointwise_2p5"],
            a["perm_band_pointwise_97p5"],
            color="#98a2b3",
            alpha=0.35,
            label="stratified permutation band (pointwise 95%)",
        )
        ax.plot(ks, a["scan_band_delta_hi"], color="#98a2b3", ls="--", lw=1.2)
        ax.plot(ks, a["scan_band_delta_lo"], color="#98a2b3", ls="--", lw=1.2)
        ax.plot(ks, a["delta_obs"], "-o", ms=4.5, color=colors[row], label="observed Δ_k")
        ax.set_xscale("log")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("tail width k (per arm, log)")
        ax.set_ylabel("Δ_k = frac(top-k) − frac(bottom-k)")
        ax.set_title(f"{coding}: tail-contrast depth (dashed = scan-corrected)", loc="left")
        ax.legend(frameon=False, fontsize=8)

        ax = axes_f[row, 1]
        prof = a["prevalence_profile"]
        xs = np.arange(len(prof))
        pv = [p["prevalence"] for p in prof]
        err = np.array(
            [
                [p["prevalence"] - p["wilson95"][0] for p in prof],
                [p["wilson95"][1] - p["prevalence"] for p in prof],
            ]
        )
        ax.errorbar(xs, pv, yerr=err, fmt="o-", ms=4, color=colors[row], lw=1.4, capsize=2)
        ax.axhline(a["marginal_prevalence"], color="black", lw=0.8, ls=":")
        ax.set_xlabel("R² rank bin (0 = best-predicted 5%)")
        ax.set_ylabel("label prevalence")
        ax.set_title(f"{coding}: prevalence across the full ranking", loc="left")
    for a_ in axes_f.ravel():
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "tail_depth_sweep", dir=PROJECT_ROOT / FIG_DIR)

    for coding in AXES:
        a = doc["axes"][coding]
        print(
            f"\n[{coding}] n={a['n']} marginal={a['marginal_prevalence']:.3f} "
            f"depth(pointwise)={a['separation_depth_pointwise']} "
            f"depth(scan)={a['separation_depth_scan_corrected']}"
        )
        for j, k in enumerate(a["k"]):
            flag = (
                "**"
                if a["outside_scan_corrected"][j]
                else ("*" if a["outside_pointwise"][j] else "")
            )
            print(
                f"   k={k:5d}  Δ={a['delta_obs'][j]:+.4f}  "
                f"band [{a['perm_band_pointwise_2p5'][j]:+.4f}, {a['perm_band_pointwise_97p5'][j]:+.4f}]  "
                f"p={a['perm_p_per_k'][j]:.4f} {flag}"
            )


if __name__ == "__main__":
    main()
