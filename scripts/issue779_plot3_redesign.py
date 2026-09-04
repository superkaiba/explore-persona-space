"""Issue #779 inline free-analysis — Plot 3 redesign: single-panel useful-directions spectrum.

Rebuilds the paper "useful directions" figure (`c3_persona_direction_spectrum`)
as ONE single panel at layer 19 (Qwen2.5-7B-Instruct): x = 1-based variance rank
(log), y = held-out per-direction R² of the context->answer map; backdrop = the
answer-PCA per-direction R² spectrum curve + the 50-random-direction band
(mean ± SD), exactly the banked
``eval_results/issue_779/fitter-fair-comparison/perdirection_single_layer.json``
recipe (pass_b 5000-context pool, fold 0 of 5-fold seed 0, GCV ridge
``PR._ridge_fit_predict_fast``, k_lead 200 / tail_step 20 / n_random 50); every
named "useful" direction is then overlaid as a labeled point at
(its equivalent variance rank + 1, its held-out per-direction R²).

Directions (all read at layer 19 = post-block-19 residual, d=3584):
  evil / sycophancy / hallucination — #779 persona vectors (28-layer ``r_b``);
  refusal — Arditi-style diff-of-means (refuse − engage) over #2356's banked
    armA consolidated context store (29 hidden states; block 19 = index 20);
  assistant axis — #2203 per-layer axis (``axis_by_layer["19"]``);
  casualness — #1434 writing-style persona vector (28-layer ``r_b``);
  impoliteness — #1482 rb4 persona vector (28-layer ``r_b``).
  harmful compliance — #658 primary persona-vector ``diffmeans`` recipe;
  correctness (math / MMLU-Pro / code) — #2388 matched within-context
    mean(correct answer t1) minus mean(incorrect answer t1), then equal-weight
    averaged over spread contexts.  The three surface directions stay separate
    because #2388 did not establish a universal cross-surface correctness axis.

Equality gate (BEFORE any new read is trusted): the refit must reproduce the
banked evil ``heldout_r2``, the full 370-point spectrum, and the random band
to <= 1e-6 abs; a gate FAIL raises and nothing is written.

0-GPU, no model forwards, no LLM calls. Inputs staged under
``/mnt/eps-data/thomasjiralerspong/issue779_plot3/hf_dl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_identity_baseline as IB  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue779_plot3_redesign")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue779_plot3/hf_dl")
HF_REPO = "superkaiba1/explore-persona-space-data"
BANKED_SPECTRUM = (
    PROJECT_ROOT / "eval_results/issue_779/fitter-fair-comparison/perdirection_single_layer.json"
)
LABELS_2356 = PROJECT_ROOT / ".claude/worktrees/issue-2356/eval_results/issue_2356/armA/labels.json"
HIDDEN = 3584

# (name, plot label, provenance one-liner) in plot order.
ROSTER = [
    ("evil", "evil", "#779 persona vector r_B (HF issue779_monitoring/r_b/evil.pt)"),
    (
        "sycophancy",
        "sycophancy",
        "#779 persona vector r_B (HF issue779_monitoring/r_b/sycophancy.pt)",
    ),
    (
        "hallucination",
        "hallucination",
        "#779 persona vector r_B (HF issue779_monitoring/r_b/hallucination.pt)",
    ),
    (
        "refusal",
        "refusal",
        "diff-of-means refuse−engage over #2356 armA consolidated v_C at block 19 "
        "(HF issue2356_refusalpred/analysis_tensors/consolidated/armA__v_C.npy)",
    ),
    (
        "assistant_axis",
        "assistant axis",
        "#2203 assistant axis, default-assistant − role-play diff-of-means "
        "(HF issue2203_ctx_capping/axis/qwen25_7b_axis_per_layer.pt)",
    ),
    (
        "casualness",
        "casualness",
        "#1434 writing-style persona vector (casual-vs-formal; "
        "HF issue1434_writingstyle/analysis_tensors/rb_writing_style.pt)",
    ),
    (
        "impoliteness",
        "impoliteness",
        "#1482 rb4 persona vector (local data/issue_779/r_b/impolite.pt, #779-recipe extraction)",
    ),
    (
        "harmful_compliance",
        "harmful compliance",
        "#658 persona-vector primary diffmeans recipe "
        "(HF issue658_theory_assumptions/store/r_b.pt)",
    ),
    (
        "correctness_math",
        "correctness (math)",
        "#2388 matched correct−incorrect t1 direction over math spread contexts",
    ),
    (
        "correctness_mcq",
        "correctness (MMLU-Pro)",
        "#2388 matched correct−incorrect t1 direction over MMLU-Pro spread contexts",
    ),
    (
        "correctness_code",
        "correctness (code)",
        "#2388 matched correct−incorrect t1 direction over code spread contexts",
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_rb28(path: Path, layer: int) -> np.ndarray:
    """Load a 28-layer {r_b, layers} direction bundle; return the block-``layer`` row."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    rb = d["r_b"]
    layers = list(d["layers"])
    assert tuple(rb.shape) == (28, HIDDEN), (path, tuple(rb.shape))
    assert layers == list(range(28)), (path, layers[:5])
    return rb[layers.index(layer)].to(torch.float32).numpy()


def _load_axis(path: Path, layer: int) -> np.ndarray:
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert "axis_by_layer" in d, list(d)
    v = d["axis_by_layer"][str(layer)]
    assert tuple(v.shape) == (HIDDEN,), tuple(v.shape)
    return v.to(torch.float32).numpy()


def _load_harmful_compliance(path: Path, layer: int) -> tuple[np.ndarray, dict]:
    """Load #658's pre-designated primary persona-vector recipe."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert bundle["capture_layers"] == list(range(28)), bundle["capture_layers"]
    behavior = bundle["r_b"]["harmful_compliance"]
    direction = behavior["diffmeans"][layer].to(torch.float32).numpy()
    return direction, {
        "recipe": "diffmeans",
        "n_db": int(behavior["n_db"]),
        "n_dbbar": int(behavior["n_dbbar"]),
    }


def _load_correctness(path: Path, layer: int) -> tuple[dict[str, np.ndarray], dict]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert int(bundle["layer"]) == layer, (bundle["layer"], layer)
    assert int(bundle["hidden_dim"]) == HIDDEN, bundle["hidden_dim"]
    directions = {
        f"correctness_{surface}": bundle["directions"][surface].to(torch.float32).numpy()
        for surface in ("math", "mcq", "code")
    }
    return directions, bundle


def _refusal_direction(stage_root: Path, layer: int) -> tuple[np.ndarray, dict]:
    """Arditi-style diff-of-means (refuse − engage) on #2356 armA context vectors.

    The armA store holds 29 hidden states per row (``LAYERS = [-1, *range(28)]``
    in issue2356_pod.py), so block ``layer`` sits at axis index ``layer + 1``.
    Labels join on the full 64-hex consolidated row sha.
    """
    cons = stage_root / "issue2356_refusalpred/analysis_tensors/consolidated"
    rows = json.load(open(cons / "armA.rows.json"))["rows"]
    lab = json.load(open(LABELS_2356))
    lab_rows = lab["rows"]
    vc = np.load(cons / "armA__v_C.npy", mmap_mode="r")
    assert vc.shape == (len(rows), 29, HIDDEN), vc.shape
    idx: dict[str, list[int]] = {"refuse": [], "engage": []}
    for i, sha in enumerate(rows):
        r = lab_rows.get(sha)
        if r is None or r.get("drop_reason") is not None:
            continue
        if r["label"] in idx:
            idx[r["label"]].append(i)
    counts = lab["counts"]
    assert len(idx["refuse"]) == counts["n_refuse"], (len(idx["refuse"]), counts)
    assert len(idx["engage"]) == counts["n_engage"], (len(idx["engage"]), counts)
    x = np.asarray(vc[:, layer + 1, :], dtype=np.float32)
    d = x[idx["refuse"]].mean(0) - x[idx["engage"]].mean(0)
    return d, {"n_refuse": len(idx["refuse"]), "n_engage": len(idx["engage"])}


def load_directions(stage_root: Path, layer: int) -> tuple[dict[str, np.ndarray], dict]:
    """Every roster direction at block ``layer``, plus per-direction provenance."""
    prov: dict[str, dict] = {}
    dirs: dict[str, np.ndarray] = {}
    rb_hf = stage_root / "issue779_monitoring/r_b"
    for t in ("evil", "sycophancy", "hallucination"):
        p = rb_hf / f"{t}.pt"
        dirs[t] = _load_rb28(p, layer)
        prov[t] = {"path": f"issue779_monitoring/r_b/{t}.pt", "sha256": _sha256(p)}
    d_ref, ref_counts = _refusal_direction(stage_root, layer)
    dirs["refusal"] = d_ref
    prov["refusal"] = {
        "path": "issue2356_refusalpred/analysis_tensors/consolidated/armA__v_C.npy",
        "labels": str(LABELS_2356.relative_to(PROJECT_ROOT)),
        **ref_counts,
    }
    p_axis = stage_root / "issue2203_ctx_capping/axis/qwen25_7b_axis_per_layer.pt"
    dirs["assistant_axis"] = _load_axis(p_axis, layer)
    prov["assistant_axis"] = {
        "path": "issue2203_ctx_capping/axis/qwen25_7b_axis_per_layer.pt",
        "sha256": _sha256(p_axis),
    }
    p_ws = stage_root / "issue1434_writingstyle/analysis_tensors/rb_writing_style.pt"
    dirs["casualness"] = _load_rb28(p_ws, layer)
    prov["casualness"] = {
        "path": "issue1434_writingstyle/analysis_tensors/rb_writing_style.pt",
        "sha256": _sha256(p_ws),
    }
    p_imp = PROJECT_ROOT / "data/issue_779/r_b/impolite.pt"
    dirs["impoliteness"] = _load_rb28(p_imp, layer)
    prov["impoliteness"] = {
        "path": "data/issue_779/r_b/impolite.pt (local, #1482 rb4)",
        "sha256": _sha256(p_imp),
    }
    p_harm = stage_root / "issue658_theory_assumptions/store/r_b.pt"
    dirs["harmful_compliance"], harm_meta = _load_harmful_compliance(p_harm, layer)
    prov["harmful_compliance"] = {
        "path": "issue658_theory_assumptions/store/r_b.pt",
        "sha256": _sha256(p_harm),
        **harm_meta,
        "caveat": (
            "#658 reports a severely depleted judged positive pole for harmful compliance; "
            "interpret this point as lower-confidence"
        ),
    }
    p_correct = (
        stage_root / "issue2388_correctness/derived" / f"correctness_directions_L{layer:02d}.pt"
    )
    correct_dirs, correct_bundle = _load_correctness(p_correct, layer)
    dirs.update(correct_dirs)
    for surface in ("math", "mcq", "code"):
        prov[f"correctness_{surface}"] = {
            "path": str(p_correct.relative_to(stage_root)),
            "sha256": _sha256(p_correct),
            "surface": surface,
            "recipe": correct_bundle["recipe"],
            "recipe_source": correct_bundle["recipe_source"],
            "source_commit": correct_bundle["source_commit"],
            **correct_bundle["surfaces"][surface],
        }
    for k, v in dirs.items():
        assert v.shape == (HIDDEN,) and np.isfinite(v).all(), k
    return dirs, prov


def compute(
    X: np.ndarray,
    Y: np.ndarray,
    dirs: dict[str, np.ndarray],
    test_idx: np.ndarray,
    *,
    k_lead: int,
    tail_step: int,
    n_random: int,
    seed: int,
) -> dict:
    """The banked ``analysis_d_layer`` computation, factored to fit/PCA ONCE and
    evaluate MANY named directions (op-for-op identical to
    issue779_identity_baseline.analysis_d_layer for the shared parts, so the
    equality gate can pin the refit against the banked JSON)."""
    n, h_dim = X.shape
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    Xtr, Ytr = X[mask], Y[mask]
    Xte, Yte = X[test_idx], Y[test_idx]
    logger.info("fitting GCV ridge: n_train=%d n_test=%d d=%d", Xtr.shape[0], Xte.shape[0], h_dim)
    pred = PR._ridge_fit_predict_fast(Xtr, Ytr, Xte)

    Ytr_c = Ytr - Ytr.mean(0)
    n_tr = Ytr.shape[0]
    logger.info("SVD of train-fold targets (%d, %d) fp64", n_tr, h_dim)
    _u, s, vh = torch.linalg.svd(torch.as_tensor(Ytr_c, dtype=torch.float64), full_matrices=False)
    vh_np = vh.numpy()
    var_spectrum = (s.numpy() ** 2) / (n_tr - 1)
    total_var = float(var_spectrum.sum())
    d_full = vh_np.shape[0]

    ranks = list(range(min(k_lead, d_full))) + list(range(k_lead, d_full, tail_step))
    dirs_pca = vh_np[ranks].T
    r2_by_rank = IB._per_direction_r2(Yte, pred, dirs_pca)
    var_by_rank = var_spectrum[ranks]

    rng = np.random.default_rng(seed + 779)
    rand = rng.standard_normal((h_dim, n_random))
    rand /= np.linalg.norm(rand, axis=0, keepdims=True) + 1e-12
    r2_rand = IB._per_direction_r2(Yte, pred, rand)

    per_dir: dict[str, dict] = {}
    for name, v in dirs.items():
        u = v / (np.linalg.norm(v) + 1e-12)
        r2 = float(IB._per_direction_r2(Yte, pred, u[:, None])[0])
        var_u = float(np.var(Ytr_c @ u, ddof=1))
        pctile = float(np.mean(var_spectrum < var_u) * 100.0)
        eq_rank = int(np.sum(var_spectrum > var_u))
        log_dist = np.abs(np.log(var_by_rank + 1e-30) - np.log(var_u + 1e-30))
        nearest = np.argsort(log_dist)[:5]
        per_dir[name] = {
            "heldout_r2": r2,
            "train_variance": var_u,
            "variance_percentile_of_pca_spectrum": pctile,
            "equivalent_variance_rank": eq_rank,
            "plotted_rank_1based": eq_rank + 1,
            "pca_r2_at_matched_variance": {
                "ranks": [int(ranks[i]) for i in nearest],
                "r2_mean": float(np.nanmean(r2_by_rank[nearest])),
            },
        }
        logger.info("dir %-16s R2=%.4f eq-rank=%d var-pctile=%.2f", name, r2, eq_rank, pctile)

    return {
        "n_train": int(n_tr),
        "n_test": int(len(test_idx)),
        "ranks_evaluated": [int(r) for r in ranks],
        "r2_by_rank": [float(v) for v in r2_by_rank],
        "variance_share_by_rank": [float(v) for v in (var_by_rank / total_var)],
        "random_directions": {
            "n": int(n_random),
            "r2_mean": float(np.nanmean(r2_rand)),
            "r2_sd": float(np.nanstd(r2_rand)),
        },
        "directions": per_dir,
    }


def equality_gate(res: dict, banked_path: Path, tol: float = 1e-6) -> dict:
    """Pin the refit against the banked single-layer spectrum JSON (fail-loud)."""
    b = json.load(open(banked_path))
    assert b["layer"] == 19, b["layer"]
    diffs = {
        "evil_heldout_r2": abs(
            res["directions"]["evil"]["heldout_r2"] - b["r_b_by_trait"]["evil"]["heldout_r2"]
        ),
        "spectrum_max_abs": float(
            np.nanmax(np.abs(np.array(res["r2_by_rank"]) - np.array(b["r2_by_rank"])))
        ),
        "random_mean": abs(res["random_directions"]["r2_mean"] - b["random_directions"]["r2_mean"]),
    }
    for k, v in diffs.items():
        if not (v <= tol):
            raise RuntimeError(f"equality gate FAIL: {k} abs diff {v:.3e} > {tol:.0e}")
    logger.info("equality gate PASS: %s", {k: f"{v:.2e}" for k, v in diffs.items()})
    return diffs


# Per-label annotation placement ((dx, dy) offset points, ha, va); tuned on the
# rendered PNG.  New points get conservative defaults, then are adjusted after
# the actual evaluation rather than from their outcomes in advance.
LABEL_OFFSETS = {
    "evil": (0, -12, "center", "top"),
    "sycophancy": (-10, 12, "right", "bottom"),
    "hallucination": (-8, -2, "right", "center"),
    "refusal": (16, 12, "left", "bottom"),
    "assistant axis": (28, 7, "left", "bottom"),
    "casualness": (0, 12, "center", "bottom"),
    "impoliteness": (-5, -12, "right", "top"),
    "harmful compliance": (-10, -10, "right", "top"),
    "correctness (math)": (15, 8, "left", "bottom"),
    "correctness (MMLU-Pro)": (15, -8, "left", "top"),
    "correctness (code)": (10, -10, "left", "top"),
}


def make_figure(res: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_full,
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("iclr")
    fig, ax = plt.subplots(figsize=figsize_iclr_full(0.45))
    ranks1 = np.array(res["ranks_evaluated"]) + 1
    r2 = np.array(res["r2_by_rank"])
    rd = res["random_directions"]
    ax.axhspan(
        rd["r2_mean"] - rd["r2_sd"],
        rd["r2_mean"] + rd["r2_sd"],
        alpha=0.3,
        color=paper_color("null"),
        lw=0,
        label="random directions (mean ± SD)",
    )
    ax.plot(ranks1, r2, lw=0.9, color=paper_color("instruct"), label="answer-PCA direction")
    ax.axhline(0.0, lw=0.6, color=paper_color("reference"))

    # The point color carries no meaning (each point is identified by its own
    # text label), so one shared color — the old figure's persona-direction
    # color — keeps the expanded roster visually coherent.
    point_color = paper_color("persona_vector")
    for name, lab, _ in ROSTER:
        e = res["directions"][name]
        x, y = e["plotted_rank_1based"], e["heldout_r2"]
        ax.scatter([x], [y], s=26, zorder=7, color=point_color, edgecolors="black", linewidths=0.4)
        dx, dy, ha, va = LABEL_OFFSETS.get(lab, (5, 3, "left", "bottom"))
        # A long offset (the crowded upper-right cluster) gets a thin leader line.
        arrow = (
            dict(arrowstyle="-", lw=0.5, color="#5A5A5A", shrinkA=1, shrinkB=2)
            if abs(dx) + abs(dy) > 20
            else None
        )
        ax.annotate(
            lab,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            color="black",
            ha=ha,
            va=va,
            zorder=6,
            arrowprops=arrow,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.55),
        )
    ax.set_xscale("log")
    ax.set_xlabel("variance rank (log)")
    ax.set_ylabel("per-direction $R^2$")
    ax.legend(loc="lower left", fontsize=6, frameon=True)
    savefig_paper(fig, "c3_persona_direction_spectrum_redesign", dir=str(fig_dir) + "/")
    plt.close(fig)
    logger.info("wrote %s/c3_persona_direction_spectrum_redesign.{png,pdf,meta.json}", fig_dir)


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, else absolute (sidecar provenance)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


# Offsets tuned on the c2a wide canvas (fontsize-13 labels); the legacy
# LABEL_OFFSETS above were tuned for the print-size iclr canvas and do not
# transfer. (dx, dy) in points, then ha/va.
PAPER_OFFSETS = {
    "evil": (0, -14, "center", "top"),
    "sycophancy": (0, 12, "center", "bottom"),
    "hallucination": (-12, -2, "right", "center"),
    "refusal": (12, -2, "left", "center"),
    "assistant axis": (128, 18, "left", "bottom"),
    "casualness": (30, 16, "left", "bottom"),
    "impoliteness": (-8, -20, "right", "top"),
    "harmful compliance": (-10, -28, "right", "top"),
    "correctness (math)": (20, 10, "left", "bottom"),
    "correctness (MMLU-Pro)": (20, -10, "left", "top"),
    "correctness (code)": (16, -12, "left", "top"),
}


def draw_spectrum_panel(
    ax, res: dict, *, offsets: dict | None = None, legend_frame: bool = False
) -> None:
    """Draw the useful-directions spectrum panel (c2a-v2) onto ``ax``.

    Everything inside the axes: the random-direction band, the answer-PCA
    spectrum curve, the labeled direction points with leader lines, log x
    scale, axis labels, and the in-axes legend.  Panel furniture (kicker and
    title) stays with the caller, so the same panel serves the standalone
    ``c3_persona_direction_spectrum`` figure and combined layouts
    (``scripts/make_paper_section42_figures.py``).  ``offsets`` overrides
    :data:`PAPER_OFFSETS` per plot label for hosts whose panel geometry
    differs from the standalone wide canvas.  ``legend_frame`` puts a white
    knockout behind the legend for hosts whose narrower axes let the curve
    descend through the legend area.
    """
    from explore_persona_space.analysis.c2a_plot_style import (
        INK,
        METRIC_LABELS,
        MUTED,
        ROLES,
        better_label,
        style_axis,
    )

    ranks1 = np.array(res["ranks_evaluated"]) + 1
    r2 = np.array(res["r2_by_rank"])
    rd = res["random_directions"]
    ax.axhspan(
        rd["r2_mean"] - rd["r2_sd"],
        rd["r2_mean"] + rd["r2_sd"],
        alpha=0.3,
        color=ROLES["control"].color,
        lw=0,
        label="random directions (mean ± SD)",
    )
    ax.plot(ranks1, r2, lw=1.8, color=ROLES["linear"].color, label="answer-PCA direction")
    ax.axhline(0.0, lw=0.8, color=MUTED)

    label_offsets = dict(PAPER_OFFSETS)
    label_offsets.update(offsets or {})
    # Headroom for the top label row (the labeled cluster sits at R2 0.84-0.91).
    ax.set_ylim(-0.32, 1.05)
    for name, lab, _ in ROSTER:
        e = res["directions"][name]
        x, y = e["plotted_rank_1based"], e["heldout_r2"]
        ax.scatter([x], [y], s=64, zorder=7, color=INK, marker="o", linewidths=0)
        dx, dy, ha, va = label_offsets.get(lab, (8, 6, "left", "bottom"))
        # A long offset (the crowded upper-right cluster) gets a thin leader line.
        arrow = (
            dict(arrowstyle="-", lw=0.8, color=MUTED, shrinkA=1, shrinkB=3)
            if abs(dx) + abs(dy) > 27
            else None
        )
        ax.annotate(
            lab,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=13,
            color=INK,
            ha=ha,
            va=va,
            zorder=6,
            arrowprops=arrow,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Variance rank (log)")
    ax.set_ylabel(better_label(METRIC_LABELS["r2"]))
    style_axis(ax)
    legend = ax.legend(loc="lower left")
    if legend_frame:
        legend.set_frame_on(True)
        frame = legend.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.85)
        frame.set_edgecolor("none")


def make_paper_figure(res: dict, paper_out: Path, source: Path) -> None:
    """Paper-standard (c2a-v2) render -> ``figures/paper/c3_persona_direction_spectrum``.

    Same plotted values as :func:`make_figure` (a restyle, not a recompute);
    style comes entirely from ``analysis/c2a_plot_style`` per
    ``docs/paper_context_answer_map/figure_standard.md`` (wide = 0.75 include
    width; teal answer-PCA curve, muted random band, ink labeled points).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.c2a_plot_style import (
        STYLE_VERSION,
        c2a_figure,
        panel_header,
        save_c2a_figure,
        set_c2a_style,
    )

    set_c2a_style()
    fig, frac = c2a_figure("wide", aspect=0.48)
    ax = fig.add_subplot(111)
    draw_spectrum_panel(ax, res)
    panel_header(
        ax,
        "",
        "Qwen2.5-7B-Instruct · layer 19 · 5,000 contexts (4,000 train, 1,000 test)",
        title="Per-direction held-out $R^2$ against variance rank",
    )

    paper_out.mkdir(parents=True, exist_ok=True)
    stem = paper_out / "c3_persona_direction_spectrum"
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Useful directions on the answer-PCA spectrum",
        subject=(
            "Per-direction held-out R2 of the context-to-answer map at layer 19 "
            "(Qwen2.5-7B-Instruct), answer-PCA spectrum + labeled behavior/persona directions"
        ),
        creator="scripts/issue779_plot3_redesign.py",
        include_width=frac,
    )
    plt.close(fig)

    sidecar = stem.with_suffix(".meta.json")
    payload = {
        "figure": "c3_persona_direction_spectrum",
        "status": "manuscript figure (c2a-v2 restyle of the #779 plot3 redesign)",
        "style_version": STYLE_VERSION,
        "plotting_script": "scripts/issue779_plot3_redesign.py",
        "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
        "reproduction_command": ("uv run python scripts/issue779_plot3_redesign.py --figure-only"),
        "git": as_metadata_dict(git_provenance()),
        "sources": {"plot3_redesign": {"path": _display_path(source), "sha256": _sha256(source)}},
        "record": outputs["record"],
        "data": {
            "layer": res.get("layer", 19),
            "ranks_evaluated": res["ranks_evaluated"],
            "r2_by_rank": res["r2_by_rank"],
            "random_directions": res["random_directions"],
            "directions": {
                name: {
                    "plotted_rank_1based": res["directions"][name]["plotted_rank_1based"],
                    "heldout_r2": res["directions"][name]["heldout_r2"],
                }
                for name, _lab, _t in ROSTER
            },
        },
        "output_sha256": {k: _sha256(p) for k, p in outputs.items() if k != "record"},
    }
    sidecar.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info("wrote %s.{pdf,png,meta.json} (+ grayscale)", stem)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--k-lead", type=int, default=200)
    ap.add_argument("--tail-step", type=int, default=20)
    ap.add_argument("--n-random", type=int, default=50)
    ap.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    ap.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_779/plot3_redesign"
    )
    ap.add_argument(
        "--fig-dir", type=Path, default=PROJECT_ROOT / "figures/issue_779/plot3_redesign"
    )
    ap.add_argument("--paper-out", type=Path, default=PROJECT_ROOT / "figures/paper")
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help=(
            "skip the recompute: render the paper (c2a-v2) figure from the existing "
            "--out-dir plot3_redesign.json; the legacy issue-folder figure is untouched"
        ),
    )
    args = ap.parse_args()

    out_json = args.out_dir / "plot3_redesign.json"
    if args.figure_only:
        res = json.loads(out_json.read_text())
        make_paper_figure(res, args.paper_out, out_json)
        print("[phase=done]", flush=True)
        return 0

    dirs, prov = load_directions(args.stage_root, args.layer)

    pass_b_path = (
        args.stage_root / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
    )
    bundle = F.load_pass_b(pass_b_path)
    layers = list(bundle["layers"])
    li = layers.index(args.layer)
    X = bundle["cx_last"][:, li, :].to(dtype=torch.float32).numpy()
    Y = bundle["v_x"][:, li, :].to(dtype=torch.float32).numpy()
    n = X.shape[0]
    test_idx = PR._cv_folds(n, args.n_folds, args.seed)[0]

    res = compute(
        X,
        Y,
        dirs,
        test_idx,
        k_lead=args.k_lead,
        tail_step=args.tail_step,
        n_random=args.n_random,
        seed=args.seed,
    )
    gate = equality_gate(res, BANKED_SPECTRUM)

    out = {
        "layer": args.layer,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "recipe": (
            "banked perdirection_single_layer recipe: pass_b 5000-context pool, fold 0 of "
            "5-fold (seed 0), GCV ridge PR._ridge_fit_predict_fast, train-fold answer PCA; "
            "per-direction R2 of held-out map predictions projected onto each unit direction"
        ),
        "equality_gate": {
            "banked": str(BANKED_SPECTRUM.relative_to(PROJECT_ROOT)),
            **{k: float(v) for k, v in gate.items()},
        },
        "roster": [
            {"name": name, "label": lab, "provenance": text, **prov[name]}
            for name, lab, text in ROSTER
        ],
        "skipped": [
            {
                "name": "misalignment_em",
                "reason": "no banked Qwen2.5-7B-Instruct EM direction found — needs minting",
            },
            {
                "name": "truthfulness",
                "reason": "no banked mass-mean truth direction found — needs minting",
            },
        ],
        **res,
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "script": "issue779_plot3_redesign",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "k_lead": args.k_lead,
            "tail_step": args.tail_step,
            "n_random": args.n_random,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    logger.info("wrote %s", out_json)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    make_figure(res, args.fig_dir)
    make_paper_figure(res, args.paper_out, out_json)
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
