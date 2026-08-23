"""P-Analysis-ext figures + ``ladder_ext_summary.json`` for #823 origin-ladder extension.

Consumes the P-Fit-ext eval artifacts (all under ONE ``--results-dir``, the fits
driver's eval_dir == out_root; schemas from ``scripts/issue823_ladder_ext_fits.py``):

- ``ladder_ext_r2.json``          per-rung estimator-hygiene blocks (``primary`` /
                                  ``companion`` -> label -> cells keyed ``{arm}:L{layer}``
                                  with fold_lambdas/fold_dofs/n_train_per_fold/pooled_r2/
                                  identity_bias_pooled_r2), gates, estimator fingerprint.
- ``shared_persona_paired_{rung|rand_rung}{label}.json``
                                  paired reads: ``arms.k16.per_layer.L{layer}`` with
                                  mean_paired_diff(+ci95), full_ratio.rho_point,
                                  rho_ci95, rho_ci95_unstable, n_negligible_E_draws.
- ``percontext_{rung|rand_rung}{label}.npz``
                                  per-context OOF ss rows (arm_names, context_ids,
                                  p1_ss_res, p1_ss_tot, ...identity...) at 28 layers.
- ``p2_ext_boundary.json``        boundary ladder cells ``L{layer}:n{n}:seed{s}``.
- ``mask_ext.json``               rung masks + bridge ids + per-(arm x persona) refusal.
- ``rung_{label}/mixture_diffs.npz`` (+ ``rand_rung_{label}/``)
                                  per-context k16 difference vectors at the read-out
                                  layers (keys: layers, k16_diffs, k16_personas,
                                  k16_n_persona0) -> correlated-offset floor.

Produces (plan section 6, deliverable stems reconciled to the section 9/10 committed
glob ``figures/issue_823/ladder_ext_*.png``):

- fig_ext1..fig_ext10 as ``ladder_ext_fig{N}_<slug>`` via ``savefig_paper`` (no
  caption/provenance blocks on canvas — axes + ticks + legend + panel titles only),
- ``ladder_ext_summary.json`` (``--summary-out``): headline per-rung rho per layer +
  rho_ci95 for BOTH ladders, realized n/d, lambda/dof medians, G2 verdicts, the
  section-3 lattice label + guard atoms, and the registered analyzer diagnostics
  (fixed-banked-subset rho re-slice, correlated-offset floor).

Zero GPU; small JSON/npz reads only (plan section 9 P-Analysis-ext row). Fail-loud:
missing inputs raise; empty shared-persona subsets raise; no silent defaults.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE numpy/matplotlib import (#847)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# ── Registered constants (plan sections 3/4/6; parent-round conventions) ───────
READ_OUT_LAYERS = (14, 26, 17)
LAYER_TITLE = {
    14: "layer 14 (evil read-out)",
    26: "layer 26 (sycophancy read-out)",
    17: "layer 17 (hallucination read-out)",
}
POOLED_K = 16
POOLED_ARM = "k16"
REF_ARM = "k1"
OFFSET_BAND_UPPER = 2.0 / POOLED_K  # 0.125 — decisively-artifact ⇔ ci_hi < this
FULL_ENERGY_LOWER = 0.5  # decisively-real ⇔ ci_lo > this
K_REFERENCE = 1.0 / POOLED_K  # 0.0625 reference line (hero)
BANKED_BAND = (0.73, 0.80)  # parent pure-GCV / 4,629-mask regime — HISTORICAL
PARENT_SURVIVAL = 0.937  # parent-measured 2-arm survival (plan section 7 Gate A)
DOF_CAP = 0.9  # fits driver DOF_CAP (dof cap = 0.9 * n_train)
P2_WITHHELD_N_TRAIN = 3336  # plan section 6: the withheld rung highlighted in fig_ext5

# (ladder tag, paired/percontext file suffix prefix, rung-dir prefix)
LADDERS = (("primary", "rung", "rung_"), ("companion", "rand_rung", "rand_rung_"))

LAYER_COLORS = dict(zip(READ_OUT_LAYERS, paper_palette(3)))
_PAL5 = paper_palette(5)
ARM_COLORS = {REF_ARM: _PAL5[3], POOLED_ARM: _PAL5[4]}


def sha256_file(path: pathlib.Path) -> str:
    """Chunked sha256 of a file (provenance for the summary's source_artifacts map)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def band_class(ci: list | None, unstable: bool) -> str:
    """MF-S1/MF-A1 band-class of one (ladder, rung, layer) cell from its rho_ci95.

    decisively-artifact ⇔ ci wholly below 0.125; decisively-real ⇔ ci wholly above
    0.5; 'neither' otherwise — a missing CI or the rho_ci95_unstable flag is
    'neither' by fiat (plan section 3).
    """
    if unstable or ci is None:
        return "neither"
    lo, hi = float(ci[0]), float(ci[1])
    if hi < OFFSET_BAND_UPPER:
        return "decisively-artifact"
    if lo > FULL_ENERGY_LOWER:
        return "decisively-real"
    return "neither"


def err_offsets(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Non-negative (2, n) errorbar offsets from CI bounds (gotchas xerr/yerr rule)."""
    return np.vstack([np.maximum(0.0, vals - lo), np.maximum(0.0, hi - vals)])


# ── Loading ─────────────────────────────────────────────────────────────────────


def _read_json(path: pathlib.Path) -> dict:
    """Fail-loud JSON read (FileNotFoundError names the missing artifact)."""
    return json.loads(path.read_text())


def _paired_cell(paired: dict, layer: int, path: pathlib.Path) -> dict:
    """The arms.k16.per_layer.L{layer} cell; requires the full_ratio block."""
    try:
        cell = paired["arms"][POOLED_ARM]["per_layer"][f"L{layer}"]
    except KeyError as exc:
        raise RuntimeError(f"{path}: missing arms.{POOLED_ARM}.per_layer.L{layer}") from exc
    if "full_ratio" not in cell:
        raise RuntimeError(
            f"{path}: L{layer} has no full_ratio block — the paired read ran without "
            "the mixture_diffs sidecar (pipeline fault, plan section 4.2)"
        )
    return cell


def _load_percontext(path: pathlib.Path) -> dict:
    """Materialize the percontext npz arrays needed downstream."""
    with np.load(path, allow_pickle=False) as z:
        return {
            "arm_names": [str(a) for a in z["arm_names"]],
            "context_ids": np.asarray(z["context_ids"], dtype=np.int64),
            "p1_ss_res": np.asarray(z["p1_ss_res"], dtype=np.float64),
            "p1_ss_tot": np.asarray(z["p1_ss_tot"], dtype=np.float64),
        }


def shared_paired_diff(pc: dict, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """(shared context ids, pooled−reference per-context ss_res diff) at one layer."""
    ids = pc["context_ids"]
    pos = np.flatnonzero(ids % POOLED_K == 0)
    if pos.size == 0:
        raise RuntimeError("no shared-persona contexts (id % 16 == 0) in percontext npz")
    ip = pc["arm_names"].index(POOLED_ARM)
    ir = pc["arm_names"].index(REF_ARM)
    diff = pc["p1_ss_res"][ip, layer, pos] - pc["p1_ss_res"][ir, layer, pos]
    return ids[pos], diff


def correlated_offset_floor(mix_path: pathlib.Path) -> dict[int, dict]:
    """floor_r = ||weighted mean over personas p != 0 of per-persona mean-shift
    vectors||^2 / E_r, per read-out layer (plan section 6 registered diagnostic).

    Weighted mean uses weights n_p / sum(n_p) over personas p != 0; E_r is the
    registered mixture energy sum_p n_p ||m_p||^2 / n_tot (issue823_ladder_common
    formula), recomputed here from the SAME persisted difference matrices.
    """
    with np.load(mix_path, allow_pickle=False) as z:
        layers = [int(v) for v in z["layers"]]
        diffs = np.asarray(z["k16_diffs"], dtype=np.float64)
        personas = np.asarray(z["k16_personas"], dtype=np.int64)
        n_persona0 = int(z["k16_n_persona0"])
    if diffs.shape[0] != personas.shape[0]:
        raise RuntimeError(f"{mix_path}: k16_diffs rows != k16_personas rows")
    out: dict[int, dict] = {}
    for j, layer in enumerate(layers):
        n_tot = n_persona0
        n_nonzero = 0
        wsum = np.zeros(diffs.shape[2], dtype=np.float64)
        between = 0.0
        for p in np.unique(personas):
            rows = diffs[personas == p, j, :]
            n_p = rows.shape[0]
            m_p = rows.mean(axis=0)
            between += n_p * float(m_p @ m_p)
            wsum += n_p * m_p
            n_tot += n_p
            n_nonzero += n_p
        e_point = between / max(n_tot, 1)
        w = wsum / max(n_nonzero, 1)
        floor_raw = float(w @ w)
        out[layer] = {
            "floor_raw": floor_raw,
            "e_point_from_diffs": e_point,
            "floor_ratio": (floor_raw / e_point) if e_point > 0.0 else None,
        }
    return out


def load_all(results_dir: pathlib.Path) -> dict:
    """Load + index every consumed artifact (fail-loud on any missing piece)."""
    src: dict[str, str] = {}

    def track(path: pathlib.Path) -> pathlib.Path:
        src[str(path.relative_to(results_dir))] = sha256_file(path)
        return path

    r2 = _read_json(track(results_dir / "ladder_ext_r2.json"))
    labels = sorted(r2["primary"], key=int)
    if not labels:
        raise RuntimeError("ladder_ext_r2.json: empty primary ladder")
    if sorted(r2["companion"], key=int) != labels:
        raise RuntimeError("ladder_ext_r2.json: companion rung labels != primary labels")

    mask = _read_json(track(results_dir / "mask_ext.json"))
    p2 = _read_json(track(results_dir / "p2_ext_boundary.json"))

    nd: dict[str, dict[str, float]] = {}
    paired: dict[str, dict[str, dict[int, dict]]] = {}
    pc: dict[str, dict[str, dict]] = {}
    floor: dict[str, dict[str, dict[int, dict]]] = {}
    for tag, suffix, dirprefix in LADDERS:
        nd[tag] = {lab: float(r2[tag][lab]["n_over_d_ratio"]) for lab in labels}
        paired[tag] = {}
        pc[tag] = {}
        floor[tag] = {}
        for lab in labels:
            ppath = track(results_dir / f"shared_persona_paired_{suffix}{lab}.json")
            pobj = _read_json(ppath)
            paired[tag][lab] = {
                layer: _paired_cell(pobj, layer, ppath) for layer in READ_OUT_LAYERS
            }
            pc[tag][lab] = _load_percontext(track(results_dir / f"percontext_{suffix}{lab}.npz"))
            floor[tag][lab] = correlated_offset_floor(
                track(results_dir / f"{dirprefix}{lab}" / "mixture_diffs.npz")
            )
    banked_shared = sorted(int(i) for i in mask["bridge"]["ids"] if int(i) % POOLED_K == 0)
    if not banked_shared:
        raise RuntimeError("mask_ext.json: bridge ids contain no shared-persona contexts")
    return {
        "r2": r2,
        "labels": labels,
        "nd": nd,
        "paired": paired,
        "pc": pc,
        "floor": floor,
        "p2": p2,
        "mask": mask,
        "banked_shared": np.asarray(banked_shared, dtype=np.int64),
        "source_artifacts": src,
    }


# ── Per-ladder series helpers ───────────────────────────────────────────────────


def rho_series(data: dict, tag: str, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rho, ci_lo, ci_hi) arrays over rung labels for one ladder x layer (nan where
    the paired read is degenerate/unstable — plotted as gaps, never coerced)."""
    rho, lo, hi = [], [], []
    for lab in data["labels"]:
        cell = data["paired"][tag][lab][layer]
        point = cell["full_ratio"]["rho_point"]
        rho.append(np.nan if point is None else float(point))
        ci = cell.get("rho_ci95")
        if ci is None or cell.get("rho_ci95_unstable"):
            lo.append(np.nan)
            hi.append(np.nan)
        else:
            lo.append(float(ci[0]))
            hi.append(float(ci[1]))
    return np.asarray(rho), np.asarray(lo), np.asarray(hi)


def band_classes(data: dict, tag: str) -> dict[str, dict[int, str]]:
    """band_class per rung label x read-out layer for one ladder."""
    out: dict[str, dict[int, str]] = {}
    for lab in data["labels"]:
        out[lab] = {}
        for layer in READ_OUT_LAYERS:
            cell = data["paired"][tag][lab][layer]
            out[lab][layer] = band_class(cell.get("rho_ci95"), bool(cell.get("rho_ci95_unstable")))
    return out


def lattice_verdict(data: dict) -> dict:
    """Plan section-3 lattice: label + guard atoms (MF-S1 heterogeneity +
    CI-decisive-band guard, MF-A1 stream-vs-randomized ladder agreement)."""
    classes = {tag: band_classes(data, tag) for tag, _s, _d in LADDERS}
    top = data["labels"][-1]
    top_classes = classes["primary"][top]
    n_art = sum(1 for c in top_classes.values() if c == "decisively-artifact")
    n_real = sum(1 for c in top_classes.values() if c == "decisively-real")
    n_disagree = 0
    for layer in READ_OUT_LAYERS:
        opposite = False
        for lab in data["labels"]:
            a = classes["primary"][lab][layer]
            b = classes["companion"][lab][layer]
            if {a, b} == {"decisively-artifact", "decisively-real"}:
                opposite = True
        n_disagree += int(opposite)
    if n_art >= 2 and n_real == 0 and n_disagree == 0:
        label = "Interpolation-artifact"
    elif n_real >= 2 and n_art == 0 and n_disagree == 0:
        label = "Origin-effect-real"
    else:
        label = "Partial-attenuation/mixed"
    return {
        "label": label,
        "top_rung": top,
        "n_artifact_layers": n_art,
        "n_real_layers": n_real,
        "n_ladder_disagree_layers": n_disagree,
        "bands": {
            "offset_band_upper": OFFSET_BAND_UPPER,
            "full_energy_lower": FULL_ENERGY_LOWER,
        },
        "band_class": {
            tag: {lab: {f"L{ly}": cls[lab][ly] for ly in READ_OUT_LAYERS} for lab in cls}
            for tag, cls in classes.items()
        },
    }


def cell_lambda_dof(rung_block: dict) -> tuple[list[float], list[float]]:
    """All per-fit (lambda, dof/n_train) pairs of one rung block (both arms, 28 layers)."""
    lams: list[float] = []
    ratios: list[float] = []
    for cell in rung_block["cells"].values():
        for lam, dof, n_tr in zip(
            cell["fold_lambdas"], cell["fold_dofs"], cell["n_train_per_fold"]
        ):
            lams.append(float(lam))
            ratios.append(float(dof) / float(n_tr))
    return lams, ratios


# ── Figures ─────────────────────────────────────────────────────────────────────


def fig_ext1(data: dict) -> plt.Figure:
    """HERO: rho vs realized n/d (log-x), per-layer lines + mean, rho_ci95 bands,
    1/k reference, banked band (historical), companion trajectory (dashed)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for tag, style, alpha in (("primary", "-", 0.18), ("companion", "--", 0.10)):
        x = np.asarray([data["nd"][tag][lab] for lab in data["labels"]])
        per_layer = []
        for layer in READ_OUT_LAYERS:
            rho, lo, hi = rho_series(data, tag, layer)
            per_layer.append(rho)
            c = LAYER_COLORS[layer]
            ax.plot(x, rho, style, color=c, marker="o" if tag == "primary" else "s", ms=4)
            ax.fill_between(x, lo, hi, color=c, alpha=alpha, linewidth=0)
        mean = np.nanmean(np.vstack(per_layer), axis=0)
        ax.plot(x, mean, style, color="black", lw=2.0 if tag == "primary" else 1.2, alpha=0.8)
    ax.axhline(K_REFERENCE, color="grey", ls=":", lw=1.0)
    ax.axhspan(*BANKED_BAND, color="grey", alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("realized n/d (log scale)")
    ax.set_ylabel(r"$\rho$ = measured excess / E (bands: 95% CI)")
    handles = [
        Line2D([], [], color=LAYER_COLORS[ly], marker="o", label=LAYER_TITLE[ly])
        for ly in READ_OUT_LAYERS
    ]
    handles += [
        Line2D([], [], color="black", lw=2.0, label="mean over read-out layers"),
        Line2D([], [], color="grey", ls="--", label="randomized-subset companion"),
        Line2D([], [], color="grey", ls=":", label=f"1/k = {K_REFERENCE:.4f}"),
        matplotlib.patches.Patch(
            color="grey", alpha=0.15, label="parent 0.73–0.80 (pure-GCV, 4,629-mask; historical)"
        ),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    return fig


def fig_ext2(data: dict) -> plt.Figure:
    """Per-rung paired-diff forest: mean paired diff + numerator-only 95% CI,
    one row per rung x layer; primary filled, companion open."""
    labels = data["labels"]
    rows = [(lab, layer) for lab in labels for layer in READ_OUT_LAYERS]
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(rows) + 1.6))
    for k, (tag, mfc_open) in enumerate((("primary", False), ("companion", True))):
        y_off = -0.18 if k else 0.18
        for i, (lab, layer) in enumerate(rows):
            cell = data["paired"][tag][lab][layer]
            v = float(cell["mean_paired_diff"])
            lo, hi = (float(b) for b in cell["mean_paired_diff_ci95"])
            xerr = err_offsets(np.asarray([v]), np.asarray([lo]), np.asarray([hi]))
            c = LAYER_COLORS[layer]
            ax.errorbar(
                [v],
                [-(i + y_off)],
                xerr=xerr,
                fmt="s" if mfc_open else "o",
                color=c,
                mfc="white" if mfc_open else c,
                ms=4,
                capsize=2,
                lw=1.0,
            )
    ax.axvline(0.0, color="grey", ls=":", lw=1.0)
    ax.set_yticks([-i for i in range(len(rows))])
    ax.set_yticklabels([f"rung {lab} · L{layer}" for lab, layer in rows], fontsize=7)
    ax.set_xlabel("mean paired diff, pooled − reference ss_res (>0 = pooled worse)")
    handles = [
        Line2D([], [], color="grey", marker="o", ls="", label="stream-prefix ladder"),
        Line2D([], [], color="grey", marker="s", mfc="white", ls="", label="randomized subset"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    return fig


def fig_ext3(data: dict) -> plt.Figure:
    """Pooled OOF R² per arm vs n/d with the identity+bias baseline overlay
    (stream-prefix ladder; per read-out layer panels)."""
    labels = data["labels"]
    x = np.asarray([data["nd"]["primary"][lab] for lab in labels])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, layer in zip(axes, READ_OUT_LAYERS):
        for arm in (REF_ARM, POOLED_ARM):
            r2s = [
                data["r2"]["primary"][lab]["cells"][f"{arm}:L{layer}"]["pooled_r2"]
                for lab in labels
            ]
            idb = [
                data["r2"]["primary"][lab]["cells"][f"{arm}:L{layer}"]["identity_bias_pooled_r2"]
                for lab in labels
            ]
            ax.plot(x, r2s, "-o", color=ARM_COLORS[arm], ms=4, label=f"{arm} ridge")
            ax.plot(x, idb, ":", color=ARM_COLORS[arm], lw=1.2, label=f"{arm} identity+bias")
        ax.set_xscale("log")
        ax.set_title(LAYER_TITLE[layer], fontsize=8)
        ax.set_xlabel("realized n/d")
    axes[0].set_ylabel("pooled OOF R²")
    axes[0].legend(fontsize=7, loc="best")
    return fig


def fig_ext4(data: dict) -> plt.Figure:
    """Estimator hygiene: per-fit selected lambda (left) and dof/n_train (right)
    vs n/d — scatter over all arms x 28 layers x 5 folds + per-ladder medians."""
    rng = np.random.default_rng(0)
    grid = (
        data["r2"]["estimator"].get("grid") if isinstance(data["r2"]["estimator"], dict) else None
    )
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    med_handles = []
    for tag, style in (("primary", "-"), ("companion", "--")):
        xs = np.asarray([data["nd"][tag][lab] for lab in data["labels"]])
        lam_med, dof_med = [], []
        for i, lab in enumerate(data["labels"]):
            lams, ratios = cell_lambda_dof(data["r2"][tag][lab])
            lam_med.append(float(np.median(lams)))
            dof_med.append(float(np.median(ratios)))
            if tag == "primary":
                jitter = np.exp(rng.normal(0.0, 0.02, size=len(lams)))
                ax_l.scatter(xs[i] * jitter, lams, s=4, color="grey", alpha=0.15, linewidths=0)
                ax_r.scatter(xs[i] * jitter, ratios, s=4, color="grey", alpha=0.15, linewidths=0)
        ax_l.plot(xs, lam_med, style, color=_PAL5[0], marker="o", ms=4)
        ax_r.plot(xs, dof_med, style, color=_PAL5[0], marker="o", ms=4)
        med_handles.append(
            Line2D(
                [],
                [],
                color=_PAL5[0],
                ls=style,
                marker="o",
                label=f"median ({'stream-prefix' if tag == 'primary' else 'randomized subset'})",
            )
        )
    if grid and grid[0] == "logspace":
        for edge in (10.0 ** float(grid[1]), 10.0 ** float(grid[2])):
            ax_l.axhline(edge, color="grey", ls=":", lw=1.0)
    ax_l.set_yscale("log")
    ax_l.set_xscale("log")
    ax_l.set_xlabel("realized n/d")
    ax_l.set_ylabel(r"selected $\lambda$ per fit (grid edges dotted)")
    ax_r.axhline(DOF_CAP, color="grey", ls=":", lw=1.0)
    ax_r.set_xscale("log")
    ax_r.set_xlabel("realized n/d")
    ax_r.set_ylabel("dof / n_train per fit (cap 0.9 dotted)")
    ax_l.legend(handles=med_handles, fontsize=7, loc="best")
    return fig


def fig_ext5(data: dict) -> plt.Figure:
    """P2-ext boundary: held-out R² vs n_train per read-out layer, per-seed spread
    shown, d marked, the withheld 3,336 rung highlighted when present."""
    p2 = data["p2"]
    grid = [int(n) for n in p2["n_train_grid"]]
    seeds = [int(s) for s in p2["draw_seeds"]]
    d = int(p2["d"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, layer in zip(axes, READ_OUT_LAYERS):
        means, id_means = [], []
        for n in grid:
            r2s = [p2["cells"][f"L{layer}:n{n}:seed{s}"]["r2"] for s in seeds]
            idb = [p2["cells"][f"L{layer}:n{n}:seed{s}"]["identity_bias_r2"] for s in seeds]
            ax.scatter([n] * len(r2s), r2s, s=8, color=LAYER_COLORS[layer], alpha=0.45)
            means.append(float(np.mean(r2s)))
            id_means.append(float(np.mean(idb)))
        ax.plot(grid, means, "-o", color=LAYER_COLORS[layer], ms=4, label="ridge (seed mean)")
        ax.plot(grid, id_means, ":", color=LAYER_COLORS[layer], lw=1.2, label="identity+bias")
        ax.axvline(d, color="grey", ls="--", lw=1.0, label=f"d = {d:,}")
        if P2_WITHHELD_N_TRAIN in grid:
            ax.axvline(
                P2_WITHHELD_N_TRAIN,
                color=_PAL5[3],
                ls=":",
                lw=1.2,
                label=f"withheld rung ({P2_WITHHELD_N_TRAIN:,})",
            )
        ax.set_xscale("log")
        ax.set_title(LAYER_TITLE[layer], fontsize=8)
        ax.set_xlabel("n_train")
    axes[0].set_ylabel("held-out R² (fixed holdout)")
    axes[0].legend(fontsize=7, loc="best")
    return fig


def fig_ext6(data: dict) -> plt.Figure:
    """kNN retrieval vs rung (stream-prefix ladder, OOF fold pools): acc@1 and
    acc@5 per arm (euclidean) with the k/n_pool chance lines."""
    labels = data["labels"]
    x = np.asarray([data["nd"]["primary"][lab] for lab in labels])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, layer in zip(axes, READ_OUT_LAYERS):
        chance: dict[str, list[float]] = {"1": [], "5": []}
        for arm in (REF_ARM, POOLED_ARM):
            acc1, acc5 = [], []
            for lab in labels:
                knn = data["r2"]["primary"][lab]["knn_read_out"]
                fold_vals = [
                    v["euclidean"] for k, v in knn.items() if k.startswith(f"{arm}:L{layer}:")
                ]
                if not fold_vals:
                    raise RuntimeError(f"knn_read_out: no folds for {arm}:L{layer} rung {lab}")
                acc1.append(float(np.mean([fv["acc_at_k"]["1"] for fv in fold_vals])))
                acc5.append(float(np.mean([fv["acc_at_k"]["5"] for fv in fold_vals])))
                if arm == REF_ARM:
                    chance["1"].append(float(np.mean([1.0 / fv["n_pool"] for fv in fold_vals])))
                    chance["5"].append(float(np.mean([5.0 / fv["n_pool"] for fv in fold_vals])))
            ax.plot(x, acc1, "-o", color=ARM_COLORS[arm], ms=4, label=f"{arm} acc@1")
            ax.plot(x, acc5, "--s", color=ARM_COLORS[arm], ms=3, alpha=0.7, label=f"{arm} acc@5")
        ax.plot(x, chance["1"], ":", color="black", lw=1.0, label="chance@1")
        ax.plot(x, chance["5"], ":", color="grey", lw=1.0, label="chance@5")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LAYER_TITLE[layer], fontsize=8)
        ax.set_xlabel("realized n/d")
    axes[0].set_ylabel("retrieval accuracy (euclidean, OOF fold pools)")
    axes[0].legend(fontsize=6, loc="best")
    return fig


def fig_ext7(data: dict, parent_refusal: dict | None) -> plt.Figure:
    """Per-persona refusal attribution on the extension rows (k16 arm), with the
    parent-measured overall rate as reference; optional parent per-persona overlay."""
    per_p = data["mask"]["refusal_fraction_by_arm_persona"][str(POOLED_K)]
    personas = sorted(per_p, key=int)
    vals = [float(per_p[p]) for p in personas]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(range(len(personas)), vals, color=_PAL5[4], label="extension rows (k16 arm)")
    ax.axhline(
        1.0 - PARENT_SURVIVAL,
        color="grey",
        ls="--",
        lw=1.0,
        label=f"parent banked overall ({1.0 - PARENT_SURVIVAL:.1%})",
    )
    k1_overall = float(data["mask"]["ext_arm_stats"]["1"]["refusal_fraction"])
    ax.axhline(k1_overall, color=_PAL5[3], ls=":", lw=1.0, label="k1 arm overall (extension)")
    if parent_refusal is not None:
        px = [i for i, p in enumerate(personas) if p in parent_refusal]
        py = [float(parent_refusal[personas[i]]) for i in px]
        ax.plot(px, py, "o", mfc="white", color="black", ms=5, label="parent per-persona")
    ax.set_xticks(range(len(personas)))
    ax.set_xticklabels(personas, fontsize=7)
    ax.set_xlabel("persona (context id mod 16)")
    ax.set_ylabel("refusal fraction")
    ax.legend(fontsize=7, loc="best")
    return fig


def fig_ext8(data: dict) -> plt.Figure:
    """Per-rung paired-diff ECDFs (stream-prefix ladder), per read-out layer."""
    labels = data["labels"]
    cmap = matplotlib.colormaps["viridis"]
    rung_colors = {lab: cmap(i / max(len(labels) - 1, 1)) for i, lab in enumerate(labels)}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, layer in zip(axes, READ_OUT_LAYERS):
        for lab in labels:
            _, diff = shared_paired_diff(data["pc"]["primary"][lab], layer)
            xs = np.sort(diff)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, color=rung_colors[lab], lw=1.2, label=f"rung {lab}")
        ax.axvline(0.0, color="grey", ls=":", lw=1.0)
        ax.set_title(LAYER_TITLE[layer], fontsize=8)
        ax.set_xlabel("paired diff (pooled − reference ss_res)")
    axes[0].set_ylabel("ECDF over shared contexts")
    axes[0].legend(fontsize=7, loc="best")
    return fig


def fig_ext9(data: dict) -> plt.Figure:
    """Stream-prefix vs randomized-subset overlay per layer with band-class
    annotations (marker shape encodes the MF-S1 band class of each rung's CI)."""
    marker_for = {"decisively-artifact": "v", "decisively-real": "^", "neither": "o"}
    classes = {tag: band_classes(data, tag) for tag, _s, _d in LADDERS}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, layer in zip(axes, READ_OUT_LAYERS):
        for tag, style in (("primary", "-"), ("companion", "--")):
            x = np.asarray([data["nd"][tag][lab] for lab in data["labels"]])
            rho, lo, hi = rho_series(data, tag, layer)
            c = LAYER_COLORS[layer]
            alpha = 1.0 if tag == "primary" else 0.6
            ax.plot(x, rho, style, color=c, lw=1.4, alpha=alpha)
            ax.fill_between(
                x, lo, hi, color=c, alpha=0.12 if tag == "primary" else 0.07, linewidth=0
            )
            for i, lab in enumerate(data["labels"]):
                m = marker_for[classes[tag][lab][layer]]
                ax.plot(
                    [x[i]],
                    [rho[i]],
                    m,
                    color=c,
                    mfc=c if tag == "primary" else "white",
                    ms=6,
                    alpha=alpha,
                )
        ax.axhline(OFFSET_BAND_UPPER, color="grey", ls=":", lw=1.0)
        ax.axhline(FULL_ENERGY_LOWER, color="grey", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_title(LAYER_TITLE[layer], fontsize=8)
        ax.set_xlabel("realized n/d")
    axes[0].set_ylabel(r"$\rho$ = measured excess / E")
    handles = [
        Line2D([], [], color="grey", ls="-", label="stream-prefix"),
        Line2D([], [], color="grey", ls="--", label="randomized subset"),
        Line2D([], [], color="grey", marker="v", ls="", label="CI wholly < 0.125 (artifact band)"),
        Line2D([], [], color="grey", marker="^", ls="", label="CI wholly > 0.5 (real band)"),
        Line2D([], [], color="grey", marker="o", ls="", label="neither / unstable"),
    ]
    axes[0].legend(handles=handles, fontsize=6, loc="best")
    return fig


def fig_ext10(data: dict) -> plt.Figure:
    """Correlated-offset floor vs rung with rho overlaid (stream-prefix ladder):
    floor_r solid, rho dashed, per read-out layer; 0.125 offset edge dotted."""
    labels = data["labels"]
    x = np.asarray([data["nd"]["primary"][lab] for lab in labels])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for layer in READ_OUT_LAYERS:
        c = LAYER_COLORS[layer]
        fl = [data["floor"]["primary"][lab][layer]["floor_ratio"] for lab in labels]
        fl = np.asarray([np.nan if v is None else float(v) for v in fl])
        rho, _lo, _hi = rho_series(data, "primary", layer)
        ax.plot(x, fl, "-o", color=c, ms=4)
        ax.plot(x, rho, "--s", color=c, ms=3, alpha=0.6)
    ax.axhline(OFFSET_BAND_UPPER, color="grey", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("realized n/d")
    ax.set_ylabel("ratio over E")
    handles = [
        Line2D([], [], color=LAYER_COLORS[ly], label=LAYER_TITLE[ly]) for ly in READ_OUT_LAYERS
    ] + [
        Line2D([], [], color="grey", ls="-", marker="o", label="correlated-offset floor"),
        Line2D([], [], color="grey", ls="--", marker="s", label=r"$\rho$ (measured excess / E)"),
        Line2D([], [], color="grey", ls=":", label="offset band edge (2/k = 0.125)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best")
    return fig


FIGURES = (
    ("ladder_ext_fig1_excess_ratio_ladder", fig_ext1),
    ("ladder_ext_fig2_paired_diff_forest", fig_ext2),
    ("ladder_ext_fig3_pooled_r2_arms", fig_ext3),
    ("ladder_ext_fig4_lambda_dof_hygiene", fig_ext4),
    ("ladder_ext_fig5_p2_boundary", fig_ext5),
    ("ladder_ext_fig6_knn_retrieval", fig_ext6),
    ("ladder_ext_fig7_refusal_by_persona", fig_ext7),
    ("ladder_ext_fig8_paired_diff_ecdf", fig_ext8),
    ("ladder_ext_fig9_ladder_agreement", fig_ext9),
    ("ladder_ext_fig10_offset_floor", fig_ext10),
)


# ── Summary ─────────────────────────────────────────────────────────────────────


def fixed_banked_subset(data: dict) -> dict:
    """Registered diagnostic: paired read re-sliced to the FIXED banked
    shared-persona subset, intersected with each rung's realized mask/E_eval."""
    banked = set(int(i) for i in data["banked_shared"])
    out: dict = {"n_banked_shared": len(banked)}
    for tag, _s, _d in LADDERS:
        out[tag] = {}
        for lab in data["labels"]:
            out[tag][lab] = {}
            for layer in READ_OUT_LAYERS:
                ids, diff = shared_paired_diff(data["pc"][tag][lab], layer)
                keep = np.asarray([int(i) in banked for i in ids], dtype=bool)
                n = int(keep.sum())
                cell = data["paired"][tag][lab][layer]
                e_point = float(cell["full_ratio"]["e_point_from_diffs"])
                mean_fixed = float(diff[keep].mean()) if n else None
                out[tag][lab][f"L{layer}"] = {
                    "n": n,
                    "mean_excess_fixed": mean_fixed,
                    "rho_fixed": (mean_fixed / e_point) if (n and e_point > 0.0) else None,
                }
    return out


def build_summary(data: dict, fig_paths: dict[str, str], results_dir: pathlib.Path) -> dict:
    """ladder_ext_summary.json: headline table + estimator hygiene + gates +
    lattice label/guard atoms + registered diagnostics (plan P-Analysis-ext)."""
    r2 = data["r2"]
    rungs: dict = {}
    for lab in data["labels"]:
        rungs[lab] = {}
        for tag, _s, _d in LADDERS:
            block = r2[tag][lab]
            lams, ratios = cell_lambda_dof(block)
            per_layer = {}
            for layer in READ_OUT_LAYERS:
                cell = data["paired"][tag][lab][layer]
                pooled_key = "pooled_r2" if tag == "primary" else "pooled_r2_eval"
                per_layer[f"L{layer}"] = {
                    "rho_point": cell["full_ratio"]["rho_point"],
                    "rho_ci95": cell.get("rho_ci95"),
                    "rho_ci95_unstable": bool(cell.get("rho_ci95_unstable")),
                    "n_negligible_E_draws": cell.get("n_negligible_E_draws"),
                    "mean_paired_diff": cell["mean_paired_diff"],
                    "mean_paired_diff_ci95": cell["mean_paired_diff_ci95"],
                    "mean_excess_point": cell["full_ratio"]["mean_excess_point"],
                    "e_point_from_diffs": cell["full_ratio"]["e_point_from_diffs"],
                    "band_class": band_class(
                        cell.get("rho_ci95"), bool(cell.get("rho_ci95_unstable"))
                    ),
                    "pooled_r2": {
                        arm: block["cells"][f"{arm}:L{layer}"][pooled_key]
                        for arm in (REF_ARM, POOLED_ARM)
                    },
                    "identity_bias_pooled_r2": {
                        arm: block["cells"][f"{arm}:L{layer}"]["identity_bias_pooled_r2"]
                        for arm in (REF_ARM, POOLED_ARM)
                    },
                }
            rungs[lab][tag] = {
                "n_mask": block["n_mask"],
                "n_eval": block["n_eval"],
                "n_over_d": block["n_over_d_ratio"],
                "solver": block["solver"],
                "g2_verdict": block["g2_verdict"],
                "estimator_degenerate": block["estimator_degenerate"],
                "lambda_edge_fraction": block["lambda_edge_fraction"],
                "lambda_median": float(np.median(lams)),
                "dof_over_ntrain_median": float(np.median(ratios)),
                "per_layer": per_layer,
            }
    prov = as_metadata_dict(git_provenance(), phase="panalysis-ext")
    return {
        "metadata": {
            "script": "scripts/issue823_ladder_ext_figures.py",
            "task": 823,
            "followup_label": "origin-ladder-more-contexts",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results_dir": str(results_dir),
            "source_artifacts": data["source_artifacts"],
            **prov,
        },
        "read_out_layers": {str(ly): LAYER_TITLE[ly] for ly in READ_OUT_LAYERS},
        "estimator": r2["estimator"],
        "lambda_edge_fraction_trigger": r2.get("lambda_edge_fraction_trigger"),
        "capture_drops": r2.get("capture_drops"),
        "gates": {
            **r2.get("gates", {}),
            "gate_f_mask_integrity": data["mask"].get("integrity_gate"),
        },
        "rungs": rungs,
        "lattice": lattice_verdict(data),
        "diagnostics": {
            "fixed_banked_subset": fixed_banked_subset(data),
            "correlated_offset_floor": {
                tag: {
                    lab: {f"L{ly}": data["floor"][tag][lab][ly] for ly in READ_OUT_LAYERS}
                    for lab in data["labels"]
                }
                for tag, _s, _d in LADDERS
            },
        },
        "refusal": {
            "ext_arm_stats": data["mask"]["ext_arm_stats"],
            "refusal_fraction_by_arm_persona": data["mask"]["refusal_fraction_by_arm_persona"],
            "parent_overall_survival_reference": PARENT_SURVIVAL,
        },
        "figures": fig_paths,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    """P-Analysis-ext CLI (figure-generation script; single-shot, no phase registry)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", required=True, help="fits eval dir (out_root)")
    ap.add_argument("--out-dir", default="figures/issue_823", help="figure output dir")
    ap.add_argument(
        "--summary-out",
        default=None,
        help="ladder_ext_summary.json path (default: <results-dir>/ladder_ext_summary.json)",
    )
    ap.add_argument("--formats", default="png,pdf", help="comma-separated savefig formats")
    ap.add_argument(
        "--parent-refusal-json",
        default=None,
        help="optional {persona: refusal_fraction} JSON for the fig_ext7 parent overlay",
    )
    ap.add_argument("--import-check", action="store_true", help="argcheck + exit 0")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Render fig_ext1–fig_ext10 + write ladder_ext_summary.json. Returns 0."""
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    parent_refusal = None
    if args.parent_refusal_json:
        parent_refusal = _read_json(pathlib.Path(args.parent_refusal_json))

    set_paper_style()
    t0 = time.monotonic()
    data = load_all(results_dir)
    print(
        f"[panalysis-ext] loaded {len(data['source_artifacts'])} artifacts "
        f"({time.monotonic() - t0:.1f}s)",
        flush=True,
    )

    fig_paths: dict[str, str] = {}
    for i, (stem, fn) in enumerate(FIGURES, start=1):
        fig = fn(data, parent_refusal) if fn is fig_ext7 else fn(data)
        written = savefig_paper(fig, stem, dir=out_dir, formats=formats)
        plt.close(fig)
        fig_paths[stem] = str(written.get("png", next(iter(written.values()))))
        print(f"[panalysis-ext] figure {i}/{len(FIGURES)} {stem}", flush=True)

    summary = build_summary(data, fig_paths, results_dir)
    summary_out = (
        pathlib.Path(args.summary_out)
        if args.summary_out
        else results_dir / "ladder_ext_summary.json"
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[panalysis-ext] summary -> {summary_out} | lattice: {summary['lattice']['label']}",
        flush=True,
    )
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
