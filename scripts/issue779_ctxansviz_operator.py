#!/usr/bin/env python
"""Issue #779 ctxansviz round — operator dissection of the banked n1m context→answer ridge map.

User-chat inline free-analysis leg (0 GPU-h, local artifacts only; dispatch record: the
`file-set claim ... round=ctxansviz-r1` epm:progress note on #779). Dissects the INTERNAL
structure of the 963,444-context ridge map (the `mixed_1m` point of #779
fitter-fair-comparison-n1m) at its three banked layers L14/L19/L26 — the "theoretical
analysis of the mapping" TODO of the context→answer paper line.

Object under study: the RAW-SPACE affine form of the registered prediction path
``vhat = ((v_C - xmu)/xsd) @ W + ymu`` (row-vector convention, identical in
``issue2474_analysis.predict_answer`` and ``issue779_ffc_n1m_fits.apply_map``), i.e.

    v  ↦  v @ A + c      with  A = diag(1/xsd) @ W,   c = ymu - (xmu/xsd) @ W

(a numeric self-check against the registered formula runs before any analysis). The map's
own identity+bias companion is ``v ↦ v + (ymu - xmu)`` (issue2474_n1m_map.py convention).

Analyses per layer (all on A, fp64): singular-value spectrum + participation-ratio
effective rank; distance from identity; nonsymmetric eigenvalue spectrum (contraction /
rotation picture, departure from normality); transport of the four #779 r_B trait
directions vs a random-direction null; principal angles between the top-k right (output)
singular subspace and the trait span. Cross-layer (L14↔L19↔L26): raw direction-aware
vec-cosine with the random-orthogonal rotation null + the one-sided Procrustes-aligned
cosine with a rotation null + the rotation-invariant spectrum cosine (descriptive ceiling
only, per the #1310 convention) + principal angles.

Reuse: ``raw_cosine_with_rotation_null`` / ``spectrum_cosine`` are IMPORTED from
scripts/issue1345_operator_comparison.py; Haar rotations from
scripts/issue825_map_alignment.py::_random_orthogonal; principal-angle math follows
scripts/issue825_crossmodel_map_transfer.py::principal_angles (equivalence-asserted fast
path on precomputed SVD bases — the helper recomputes two full operator SVDs per call).
The activation-fitted ``_procrustes_cosine_null`` needs the fit data matrices, which are
not local; its aligned-cosine + rotation-null convention is adapted here to
operator-only inputs (closed-form one-sided Procrustes).

Checkpointing: every stage persists to ``eval_results/issue_779/ctxansviz/
operator_stats.json`` (atomic replace) the moment it completes; re-runs skip completed
stages (resume key = generating parameters, never recomputed floats). VM launch carries
the shared-VM thread-cap prefix (OMP/MKL/OPENBLAS/NUMEXPR=8, MALLOC_ARENA_MAX=2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_crossmodel_map_transfer as cm  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import issue1345_operator_comparison as oc  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

SEED = 42
D = 3584
LAYERS = (14, 19, 26)
TRAITS = ("apathetic", "humorous", "impolite", "optimistic")
SUBSPACE_KS = (16, 64)
N_ROT_DRAWS = 16  # Haar-rotation null draws per cross-layer statistic (analytic sd ~1/d)
N_NULL_DIRS = 200  # random unit directions for the trait-transport null band
WEIGHTS_DIR = _REPO_ROOT / (
    "data/issue_2094/joint_transport/banked_maps/issue779_monitoring/n1m_readout/weights"
)
RB_DIR = _REPO_ROOT / "data/issue_779/r_b"
OUT_JSON = _REPO_ROOT / "eval_results/issue_779/ctxansviz/operator_stats.json"
FIG_SUBDIR = "issue_779"
LAYER_STAGES = ("spectrum", "identity", "eig", "traits", "subspace")
CROSS_PAIRS = ((14, 19), (19, 26), (14, 26))

N1M_PROVENANCE = {
    "fit_point": "mixed_1m",
    "n_train": 963444,
    "selected_lambda": 0.001,
    "whole_map_r2_L19": 0.7541708417500051,
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "source": "eval_results/issue_779/fitter-fair-comparison-n1m/n1m_fits.json",
}


def _params() -> dict:
    """Generating parameters — the resume key (machine-stable ints/strings only)."""
    return {
        "seed": SEED,
        "d": D,
        "layers": list(LAYERS),
        "traits": list(TRAITS),
        "subspace_ks": list(SUBSPACE_KS),
        "n_rot_draws": N_ROT_DRAWS,
        "n_null_dirs": N_NULL_DIRS,
        "cross_pairs": [list(p) for p in CROSS_PAIRS],
    }


def _r6(x) -> float:
    return float(f"{float(x):.6g}")


def _r6_list(a) -> list[float]:
    return [_r6(v) for v in np.asarray(a).ravel()]


# ---------------------------------------------------------------------------
# Loading + the raw-space affine form
# ---------------------------------------------------------------------------
def load_affine(layer: int) -> dict[str, np.ndarray]:
    """Load the banked ridge at ``layer`` and build the raw-space affine map (A, c).

    Asserts payload identity (ridge fitter, requested layer, square d) and then
    numerically self-checks the affine form against the registered prediction path
    ``vhat = ((v - xmu)/xsd) @ W + ymu`` on random vectors before returning.
    """
    path = WEIGHTS_DIR / f"L{layer}" / "ridge.pt"
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p.get("kind") != "ridge" or int(p.get("layer", -1)) != layer:
        raise RuntimeError(f"{path}: expected ridge payload at layer {layer}, got {p.get('layer')}")
    W = np.asarray(p["W"], dtype=np.float64)
    xmu, xsd, ymu = (np.asarray(p[k], dtype=np.float64) for k in ("xmu", "xsd", "ymu"))
    assert W.shape == (D, D), W.shape
    assert xsd.min() > 0, "xsd must be strictly positive"
    A = W / xsd[:, None]  # diag(1/xsd) @ W — scales the ROWS of W (row-vector convention)
    c = ymu - (xmu / xsd) @ W
    rng = np.random.default_rng(0)
    v = rng.standard_normal((3, D))
    registered = ((v - xmu) / xsd) @ W + ymu
    mine = v @ A + c
    assert np.allclose(registered, mine, rtol=1e-10, atol=1e-8), (
        f"affine self-check failed at L{layer}: max abs err {np.abs(registered - mine).max()}"
    )
    return {"A": A, "c": c, "xmu": xmu, "xsd": xsd, "ymu": ymu}


def load_trait_basis(layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit-normalized r_B trait directions at ``layer``: (U (4, D), Q (D, 4) orthonormal)."""
    rows = []
    for t in TRAITS:
        d = torch.load(RB_DIR / f"{t}.pt", map_location="cpu", weights_only=False)
        assert d["trait"] == t, d["trait"]
        rb = np.asarray(d["r_b"], dtype=np.float64)
        assert rb.shape[1] == D and layer < rb.shape[0], rb.shape
        u = rb[layer]
        rows.append(u / np.linalg.norm(u))
    U = np.stack(rows)
    smin = np.linalg.svd(U, compute_uv=False).min()
    assert smin > 1e-6, f"trait directions near-collinear at L{layer} (min sv {smin})"
    Q, _ = np.linalg.qr(U.T)  # (D, 4) orthonormal basis of the trait span
    return U, Q


def principal_angle_cosines(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    """cos of principal angles between column spans of two orthonormal bases.

    Same math as scripts/issue825_crossmodel_map_transfer.py::principal_angles
    (svdvals of Qa^T Qb, clamped to [0, 1]) on PRECOMPUTED bases — the helper
    recomputes two full operator SVDs per call, paid once per layer here.
    Equivalence vs the helper is asserted in ``self_check``.
    """
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def self_check() -> None:
    """Fast startup checks: vendored principal-angle path == cm.principal_angles."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((96, 96))
    b = rng.standard_normal((96, 96))
    ref = cm.principal_angles(torch.as_tensor(a), torch.as_tensor(b), k=8)
    _, _, vha = np.linalg.svd(a)
    _, _, vhb = np.linalg.svd(b)
    mine = principal_angle_cosines(vha[:8].T, vhb[:8].T)
    assert np.allclose(np.sort(ref), np.sort(mine), atol=1e-8), (ref, mine)
    print("[operator] self-check ok: principal-angle fast path matches cm.principal_angles")


# ---------------------------------------------------------------------------
# Per-layer stages
# ---------------------------------------------------------------------------
def stage_spectrum(A: np.ndarray, svd: tuple) -> dict:
    _, s, _ = svd
    s2 = s**2
    cum = np.cumsum(s2) / s2.sum()
    return {
        "singular_values": _r6_list(s),
        "effective_rank_participation_ratio": _r6(s.sum() ** 2 / s2.sum()),
        "n_sv_for_energy": {
            str(frac): int(np.searchsorted(cum, frac) + 1) for frac in (0.5, 0.9, 0.99)
        },
        "top_sv": _r6(s[0]),
        "median_sv": _r6(np.median(s)),
    }


def stage_identity(A: np.ndarray, c: np.ndarray, xmu: np.ndarray, ymu: np.ndarray) -> dict:
    fro = np.linalg.norm(A)
    return {
        "fro_dist_identity_rel": _r6(np.linalg.norm(A - np.eye(D)) / fro),
        "cos_vec_A_identity_direction_aware": _r6(np.trace(A) / (np.sqrt(D) * fro)),
        "fro_norm_A": _r6(fro),
        "norm_c": _r6(np.linalg.norm(c)),
        "norm_ymu_minus_xmu": _r6(np.linalg.norm(ymu - xmu)),
        # mean-vector norm proxies (per-row activation norms are not local to this round)
        "norm_xmu": _r6(np.linalg.norm(xmu)),
        "norm_ymu": _r6(np.linalg.norm(ymu)),
    }


def stage_eig(A: np.ndarray) -> dict:
    w = scipy.linalg.eigvals(A)  # nonsymmetric spectrum; values only (no vectors)
    mag = np.abs(w)
    comm = A @ A.T - A.T @ A
    anti = 0.5 * (A - A.T)
    fro2 = np.linalg.norm(A) ** 2
    return {
        "eigenvalues_re_im": [[_r6(x.real), _r6(x.imag)] for x in w],
        "frac_abs_lt_1": _r6((mag < 1.0).mean()),
        "spectral_radius": _r6(mag.max()),
        "median_abs_eig": _r6(np.median(mag)),
        "frac_complex": _r6((np.abs(w.imag) > 1e-12).mean()),
        "departure_from_normality_rel": _r6(np.linalg.norm(comm) / fro2),
        "antisymmetric_energy_share": _r6(np.linalg.norm(anti) ** 2 / fro2),
    }


def stage_traits(A: np.ndarray, U: np.ndarray, Q: np.ndarray) -> dict:
    T = U @ A  # (4, D)
    gains = np.linalg.norm(T, axis=1)
    cosines = (T * U).sum(1) / (gains + 1e-12)  # U rows are unit
    inspan = ((T @ Q) ** 2).sum(1) / (gains**2 + 1e-12)
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal((N_NULL_DIRS, D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    TN = R @ A
    ngains = np.linalg.norm(TN, axis=1)
    ncos = (TN * R).sum(1) / (ngains + 1e-12)
    ninspan = ((TN @ Q) ** 2).sum(1) / (ngains**2 + 1e-12)
    pct = (2.5, 50.0, 97.5)
    return {
        "per_trait": {
            t: {
                "gain": _r6(gains[i]),
                "cos_mapped_vs_own_direction_aware": _r6(cosines[i]),
                "share_in_trait_span": _r6(inspan[i]),
            }
            for i, t in enumerate(TRAITS)
        },
        "null_random_unit_dirs": {
            "n": N_NULL_DIRS,
            "gain_pct": {str(p): _r6(np.percentile(ngains, p)) for p in pct},
            "cos_pct": {str(p): _r6(np.percentile(ncos, p)) for p in pct},
            "share_in_trait_span_pct": {str(p): _r6(np.percentile(ninspan, p)) for p in pct},
        },
    }


def stage_subspace(svd: tuple, Q_traits: np.ndarray, xsd: np.ndarray) -> dict:
    U_sv, _, Vh = svd
    out: dict = {}
    for k in SUBSPACE_KS:
        right = Vh[:k].T  # top-k right (output) singular subspace, (D, k)
        left = U_sv[:, :k]  # top-k left (input-sensitivity) singular subspace
        out[f"k{k}"] = {
            "right_vs_trait_span_cos": _r6_list(principal_angle_cosines(right, Q_traits)),
            "left_vs_trait_span_cos": _r6_list(principal_angle_cosines(left, Q_traits)),
            # DIAGONAL input second-moment proxy only: xsd is the per-coordinate sd of the
            # fit corpus; the full input covariance is not local to this round (disclosed).
            "right_vs_topk_xsd_coords_cos_diag_proxy": _r6_list(
                principal_angle_cosines(right, np.eye(D)[:, np.argsort(xsd)[::-1][:k]])
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Cross-layer stage
# ---------------------------------------------------------------------------
def procrustes_aligned_cosine_with_null(
    ta: torch.Tensor, tb: torch.Tensor, *, n_draws: int, seed: int
) -> dict:
    """One-sided orthogonal-Procrustes-aligned cosine between two given operators + null.

    observed = max over orthogonal Q of cos(vec(Aa), vec(Ab @ Q))
             = sum(svdvals(Aa^T Ab)) / (||Aa||_F ||Ab||_F).
    Convention adapted from scripts/issue825_map_alignment.py::_procrustes_cosine_null
    (aligned cosine vs a Q1^T M Q2 rotation null); that helper fits maps from activation
    data, whereas here the operators are given, so the alignment is closed-form. Each
    null draw wraps a random two-sided rotation around Ab; svdvals(Aa^T Q1^T Ab Q2) ==
    svdvals((Q1 Aa)^T Ab) (right-orthogonal invariance), so one Haar Q per draw suffices.
    """
    na = float(torch.linalg.norm(ta))
    nb = float(torch.linalg.norm(tb))
    observed = float(torch.linalg.svdvals(ta.T @ tb).sum() / (na * nb + 1e-12))
    gen = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(n_draws):
        q = ma._random_orthogonal(ta.shape[0], gen)
        draws.append(float(torch.linalg.svdvals((q @ ta).T @ tb).sum() / (na * nb + 1e-12)))
    arr = np.asarray(draws)
    return {
        "observed_aligned_cosine": _r6(observed),
        "n_draws": n_draws,
        "null_mean": _r6(arr.mean()),
        "null_std": _r6(arr.std()),
        "null_min": _r6(arr.min()),
        "null_max": _r6(arr.max()),
        "z_observed_vs_null": _r6((observed - arr.mean()) / (arr.std() + 1e-12)),
    }


def stage_cross(la: int, lb: int, Aa: np.ndarray, Ab: np.ndarray, svd_cache: dict) -> dict:
    ta = torch.as_tensor(Aa)
    tb = torch.as_tensor(Ab)
    raw = oc.raw_cosine_with_rotation_null(ta, tb, n_draws=N_ROT_DRAWS, seed=SEED)
    proc = procrustes_aligned_cosine_with_null(ta, tb, n_draws=N_ROT_DRAWS, seed=SEED + 1)
    spec = oc.spectrum_cosine(ta, tb)  # rotation-invariant DESCRIPTIVE ceiling only
    angles = {}
    for k in SUBSPACE_KS:
        qa = svd_cache[la][2][:k].T
        qb = svd_cache[lb][2][:k].T
        angles[f"k{k}"] = _r6_list(principal_angle_cosines(qa, qb))
    return {
        "raw_cosine_direction_aware": _r6(raw["raw_cosine"]),
        "rotation_null": {kk: _r6(vv) for kk, vv in raw["rotation_null"].items()},
        "procrustes_aligned_output_rotation": proc,
        "spectrum_cosine_rotation_invariant_descriptive": _r6(spec),
        "principal_angle_cos_right_subspaces": angles,
    }


# ---------------------------------------------------------------------------
# Runner (checkpoint per stage, resume keyed on generating parameters)
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n")
    os.replace(tmp, path)


def run_stats(fresh: bool) -> dict:
    params = _params()
    doc: dict = {"params": params, "stages": {}}
    if OUT_JSON.exists() and not fresh:
        prior = json.loads(OUT_JSON.read_text())
        if prior.get("params") != params:
            raise SystemExit(
                f"{OUT_JSON}: existing params differ from this run's — pass --fresh to recompute"
            )
        doc = prior
    doc["metadata"] = {
        **as_metadata_dict(git_provenance()),
        "n1m_map": N1M_PROVENANCE,
        "round": "user-chat inline free analysis (ctxansviz operator dissection)",
    }

    units = [f"{st}_L{layer}" for layer in LAYERS for st in LAYER_STAGES]
    units += [f"cross_L{a}_L{b}" for a, b in CROSS_PAIRS]
    todo = [u for u in units if u not in doc["stages"]]
    print(f"[operator] {len(units)} units total, {len(todo)} to compute", flush=True)

    affine: dict[int, dict] = {}
    svd_cache: dict[int, tuple] = {}
    traits_cache: dict[int, tuple] = {}

    def _affine(layer: int) -> dict:
        if layer not in affine:
            affine[layer] = load_affine(layer)
        return affine[layer]

    def _svd(layer: int) -> tuple:
        if layer not in svd_cache:
            t0 = time.time()
            svd_cache[layer] = np.linalg.svd(_affine(layer)["A"])
            print(f"[operator] svd L{layer} done in {time.time() - t0:.0f}s", flush=True)
        return svd_cache[layer]

    def _traits(layer: int) -> tuple:
        if layer not in traits_cache:
            traits_cache[layer] = load_trait_basis(layer)
        return traits_cache[layer]

    for i, unit in enumerate(units):
        if unit in doc["stages"]:
            continue
        t0 = time.time()
        if unit.startswith("cross_L"):
            la, lb = (int(x) for x in unit[len("cross_L") :].split("_L"))
            _svd(la), _svd(lb)
            res = stage_cross(la, lb, _affine(la)["A"], _affine(lb)["A"], svd_cache)
        else:
            stage, loc = unit.rsplit("_L", 1)
            layer = int(loc)
            f = _affine(layer)
            if stage == "spectrum":
                res = stage_spectrum(f["A"], _svd(layer))
            elif stage == "identity":
                res = stage_identity(f["A"], f["c"], f["xmu"], f["ymu"])
            elif stage == "eig":
                res = stage_eig(f["A"])
            elif stage == "traits":
                res = stage_traits(f["A"], *_traits(layer))
            elif stage == "subspace":
                res = stage_subspace(_svd(layer), _traits(layer)[1], f["xsd"])
            else:
                raise ValueError(stage)
        doc["stages"][unit] = res
        _atomic_write(OUT_JSON, doc)
        print(
            f"[operator] unit {i + 1}/{len(units)} {unit} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return doc


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(doc: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    pal = paper_palette(7)
    layer_colors = {14: pal[0], 19: pal[1], 26: pal[2]}
    trait_colors = {t: pal[3 + i] for i, t in enumerate(TRAITS)}
    stages = doc["stages"]

    # --- 1. singular-value spectrum -------------------------------------------------
    fig, ax = plt.subplots()
    for layer in LAYERS:
        s = np.asarray(stages[f"spectrum_L{layer}"]["singular_values"])
        ax.plot(np.arange(1, len(s) + 1), s, color=layer_colors[layer], label=f"Layer {layer}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("singular value rank (largest first)")
    ax.set_ylabel("singular value of the map")
    ax.legend()
    set_title_subtitle(
        ax,
        "Singular-value spectrum of the context→answer map",
        "n=963,444-context ridge map, one line per banked layer",
    )
    savefig_paper(fig, f"{FIG_SUBDIR}/ctxansviz_operator_spectrum", dir="figures/")
    plt.close(fig)

    # --- 2. eigenvalue complex plane + magnitude histogram --------------------------
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    eig = {
        layer: np.asarray(stages[f"eig_L{layer}"]["eigenvalues_re_im"], dtype=float)
        for layer in LAYERS
    }
    rmax = 1.05 * max(np.abs(e).max() for e in eig.values())
    theta = np.linspace(0, 2 * np.pi, 256)
    hbs = []
    for j, layer in enumerate(LAYERS):
        axp = axes[0, j]
        e = eig[layer]
        hb = axp.hexbin(
            e[:, 0],
            e[:, 1],
            gridsize=55,
            cmap="viridis",
            mincnt=1,
            extent=(-rmax, rmax, -rmax, rmax),
        )
        hbs.append(hb)
        axp.plot(np.cos(theta), np.sin(theta), ls="--", lw=0.8, color="0.55")
        axp.set_aspect("equal")
        axp.set_title(f"Layer {layer}")
        axp.set_xlabel("eigenvalue real part")
        if j == 0:
            axp.set_ylabel("eigenvalue imaginary part")
        axh = axes[1, j]
        mags = np.hypot(e[:, 0], e[:, 1])
        axh.hist(mags, bins=60, color=layer_colors[layer])
        axh.axvline(1.0, ls="--", lw=0.8, color="0.55")
        axh.set_xlabel("eigenvalue magnitude |λ|")
        if j == 0:
            axh.set_ylabel("number of eigenvalues")
    gmax = max(hb.get_array().max() for hb in hbs)
    for hb in hbs:
        hb.set_norm(LogNorm(vmin=1, vmax=gmax))
    fig.colorbar(hbs[-1], ax=axes[0, :].tolist(), label="eigenvalues per bin", shrink=0.9)
    savefig_paper(fig, f"{FIG_SUBDIR}/ctxansviz_operator_eigplane", dir="figures/")
    plt.close(fig)

    # --- 3. trait-direction transport vs random-direction null ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    panels = [
        ("gain", "gain_pct", "stretch of a unit trait direction (‖u·A‖)"),
        (
            "cos_mapped_vs_own_direction_aware",
            "cos_pct",
            "direction-aware cosine, mapped vs original",
        ),
        (
            "share_in_trait_span",
            "share_in_trait_span_pct",
            "share of mapped energy\nin the 4-trait span",
        ),
    ]
    xs = np.arange(len(LAYERS))
    for ax, (key, nkey, ylabel) in zip(axes, panels):
        lo = [stages[f"traits_L{la}"]["null_random_unit_dirs"][nkey]["2.5"] for la in LAYERS]
        hi = [stages[f"traits_L{la}"]["null_random_unit_dirs"][nkey]["97.5"] for la in LAYERS]
        md = [stages[f"traits_L{la}"]["null_random_unit_dirs"][nkey]["50.0"] for la in LAYERS]
        ax.fill_between(xs, lo, hi, color="0.85", label="random-direction null (95% band)")
        ax.plot(xs, md, color="0.55", lw=1.0)
        for i, t in enumerate(TRAITS):
            vals = [stages[f"traits_L{la}"]["per_trait"][t][key] for la in LAYERS]
            ax.plot(
                xs + (i - 1.5) * 0.05,
                vals,
                marker="o",
                ls="-",
                lw=1.0,
                color=trait_colors[t],
                label=t,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{la}" for la in LAYERS])
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, fontsize=8)
    savefig_paper(fig, f"{FIG_SUBDIR}/ctxansviz_operator_trait_transport", dir="figures/")
    plt.close(fig)

    # --- 4. cross-layer operator similarity -----------------------------------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    pair_keys = [f"cross_L{a}_L{b}" for a, b in CROSS_PAIRS]
    pair_labels = [f"L{a} ↔ L{b}" for a, b in CROSS_PAIRS]
    xs = np.arange(len(pair_keys))
    raw = [stages[k]["raw_cosine_direction_aware"] for k in pair_keys]
    ali = [
        stages[k]["procrustes_aligned_output_rotation"]["observed_aligned_cosine"]
        for k in pair_keys
    ]
    spe = [stages[k]["spectrum_cosine_rotation_invariant_descriptive"] for k in pair_keys]
    nm = [stages[k]["procrustes_aligned_output_rotation"]["null_mean"] for k in pair_keys]
    ns = [stages[k]["procrustes_aligned_output_rotation"]["null_std"] for k in pair_keys]
    axa.errorbar(
        xs,
        nm,
        yerr=2 * np.asarray(ns),
        fmt="_",
        color="0.6",
        capsize=3,
        markeredgewidth=1.2,
        label="rotation null, aligned (mean ± 2 s.d.)",
    )
    axa.plot(xs - 0.12, raw, "o", color="0.1", label="raw vec-cosine (direction-aware)")
    axa.plot(xs, ali, "s", color="0.35", label="Procrustes-aligned (output rotation)")
    axa.plot(
        xs + 0.12,
        spe,
        "D",
        mfc="none",
        mec="0.2",
        markeredgewidth=1.2,
        ls="none",
        label="spectrum cosine (rotation-invariant ceiling)",
    )
    axa.set_xticks(xs)
    axa.set_xticklabels(pair_labels)
    axa.set_ylabel("cosine similarity between layer maps")
    axa.set_ylim(bottom=0)
    axa.legend(fontsize=8)
    styles = ["-", "--", ":"]
    shades = ["0.1", "0.4", "0.65"]
    for k, lab, st, sh in zip(pair_keys, pair_labels, styles, shades):
        cs = np.asarray(stages[k]["principal_angle_cos_right_subspaces"]["k64"])
        axb.plot(np.arange(1, len(cs) + 1), cs, ls=st, color=sh, label=lab)
    axb.set_xlabel("principal angle rank")
    axb.set_ylabel("cosine of principal angle,\ntop-64 output subspaces")
    axb.set_ylim(0, 1.02)
    axb.legend(fontsize=8)
    savefig_paper(fig, f"{FIG_SUBDIR}/ctxansviz_operator_crosslayer", dir="figures/")
    plt.close(fig)
    print("[operator] figures written to figures/issue_779/", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default="stats,figures", help="comma subset of stats,figures")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint JSON")
    args = ap.parse_args()
    wanted = set(args.stages.split(","))
    self_check()
    doc = None
    if "stats" in wanted:
        doc = run_stats(fresh=args.fresh)
    if "figures" in wanted:
        if doc is None:
            doc = json.loads(OUT_JSON.read_text())
        make_figures(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
