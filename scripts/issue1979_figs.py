"""#1979 F3 figures — the plan §6 figure list from the race outputs.

Reads ``eval_results/issue_1979/race/`` (summary, champions, frames, verdict
JSONs) and renders to ``figures/issue_1979/`` via the paper-plots conventions
(``analysis/paper_plots.py``: ``set_paper_style`` + ``savefig_paper`` — PNG +
PDF + ``.meta.json`` sidecar with commit pin + per-point data).

Body-round revision (clean-result pass): all rendered text uses plain-English
predictor + arm labels (no ``p1``/slug codes on ticks), panel titles state what
is plotted (never a verdict), and three per-unit scatters back the race
headlines (change vs the through-map predicted-answer similarity; level vs the
read-out projection + base propensity; marker Δ log P vs nearest-training-rows).
``--only stem1,stem2`` renders a subset.

Smoke: run against a fixture race output dir with ``--out-dir`` pointed at a
scratch dir (never the canonical ``figures/issue_1979/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1979.figs")

RACE_DIR = REPO_ROOT / "eval_results/issue_1979/race"
FIG_DIR = REPO_ROOT / "figures/issue_1979"
GATE_BAND = (0.3, 0.7)
GATE_PER_QUERY_ANCHOR = 0.14
A6_CRITERION = 0.6

# Plain-English rendered-text labels (paper-plots §3.5: no config codes on ticks).
CAND_LABELS = {
    "p1": "context similarity",
    "p2": "answer similarity",
    "p3a": "through-map context sim",
    "p3b": "through-map predicted-answer sim",
    "p4": "whitened gate",
    "p5": "read-out projection (direct)",
    "p6": "read-out projection (through map)",
    "p7": "base propensity",
    "p8a": "write forecast (size)",
    "p8b": "write forecast (alignment)",
    "p9": "nearest training rows (context)",
    "p10": "nearest training rows (answer)",
}
ARM_LABELS = {
    "cas-bare-con-lr1e5-s42": "casual bare-context contrastive",
    "cas-pers-con-lr1e5-s42": "casual persona contrastive",
    "cas-pers-ft-con-s42": "casual persona full fine-tune",
    "cas-pers-po-lr1e5-s42": "casual persona positive-only",
    "imp-pers-con-lr3e5-s137": "impolite persona contrastive (seed 137)",
    "imp-pers-con-lr3e5-s42": "impolite persona contrastive (seed 42)",
    "imp-pers-ft-con-s42": "impolite persona full fine-tune",
    "imp-pers-po-lr1e5-s42": "impolite persona positive-only",
    "syc-bare-con-lr1e5-s42": "sycophancy bare-context contrastive",
    "syc-conv-con-lr1e5-s42": "sycophancy conversation contrastive",
    "syc-pers-ft-con-s42": "sycophancy persona full fine-tune",
    "syc-pers-po-lr1e5-s42": "sycophancy persona positive-only",
    "mk-bare-con-lr5e6-s42": "marker bare-context contrastive",
    "mk-conv-con-lr5e6-s42": "marker conversation contrastive",
    "mk-icl-con-lr5e6-s42": "marker ICL contrastive",
    "mk-pers-con-lr5e6-s42": "marker persona contrastive",
    "mk-pers-ft-con-s42": "marker persona full fine-tune",
    "mk-pers-po-lr5e6-s42": "marker persona positive-only",
}
DV_TITLES = {
    "dv_level": "leakage LEVEL (judge score) — within-arm Spearman rho",
    "dv_change": "leakage CHANGE (trained − base) — within-arm Spearman rho",
    "dv_dlogp": "Δ log P(marker) — within-arm Spearman rho",
}


def _cand_label(c: str) -> str:
    return CAND_LABELS.get(c, c)


def _arm_label(a: str) -> str:
    return ARM_LABELS.get(a, a)


def _load(race_dir: Path, name: str) -> dict:
    p = race_dir / name
    assert p.exists(), f"race output missing: {p}"
    return json.loads(p.read_text())


def _arm_jsons(race_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(race_dir.glob("arm_*.json")):
        pl = json.loads(p.read_text())
        out[pl["arm_id"]] = pl
    assert out, f"no arm_*.json under {race_dir}"
    return out


def _frames(race_dir: Path) -> dict[str, dict]:
    return {
        p.stem.removeprefix("frame_"): json.loads(p.read_text())
        for p in sorted(race_dir.glob("frame_*.json"))
    }


def _save(fig, stem: str, out_dir: Path) -> Path:
    paths = savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    png = paths.get("png")
    assert png is not None and png.exists() and png.stat().st_size > 0, (stem, paths)
    logger.info("[figs] %s (%d bytes)", png, png.stat().st_size)
    return png


# ── hero heatmap (content + marker replicate) ─────────────────────────────────


def hero_heatmap(
    arms: dict[str, dict],
    champ: dict,
    kind: str,
    dv_pair: tuple[str, str] | tuple[str],
    crossgrain: dict,
    stem: str,
    out_dir: Path,
) -> None:
    ids = [a for a, pl in arms.items() if pl["kind"] == kind]
    if not ids:
        return
    prim = champ["prefix_resample_PRIMARY"]
    cands = prim["panel_candidates"]
    n_dv = len(dv_pair)
    fig, axes = plt.subplots(
        1,
        n_dv + 1,
        figsize=(4.6 * n_dv + 3.4, 0.42 * len(cands) + 2.2),
        gridspec_kw={"width_ratios": [*([4] * n_dv), 2]},
    )
    axes = np.atleast_1d(axes)
    ghost = (crossgrain.get("i1900", {}) or {}).get(
        "champion_content" if kind == "content" else "champion_marker", {}
    )
    for j, dv in enumerate(dv_pair):
        M = np.full((len(cands), len(ids)), np.nan)
        for ai, a in enumerate(ids):
            obs = arms[a]["observed_rho"].get(dv, {})
            for ci, c in enumerate(cands):
                if c in obs:
                    M[ci, ai] = obs[c]
        ax = axes[j]
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels([_arm_label(a) for a in ids], rotation=45, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(cands)))
        if j == 0:
            ax.set_yticklabels([_cand_label(c) for c in cands], fontsize=6.5)
            ax.tick_params(axis="y", pad=32)
        else:
            ax.set_yticklabels([])
        med = np.nanmedian(M, axis=1)
        for ci, c in enumerate(cands):
            ax.text(len(ids) - 0.3, ci, f" med {med[ci]:+.2f}", va="center", fontsize=6)
            if isinstance(ghost, dict) and c in ghost and isinstance(ghost[c], int | float):
                ax.text(
                    -0.7,
                    ci,
                    f"{ghost[c]:+.2f}",
                    va="center",
                    ha="right",
                    fontsize=5,
                    color="gray",
                )
        ax.set_title(DV_TITLES.get(dv, dv), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.16)
    axb = axes[-1]
    pw = prim["p_win"]
    order = list(cands)
    axb.barh(range(len(order)), [pw.get(c, 0.0) for c in order], color=paper_palette(1)[0])
    axb.set_yticks(range(len(order)))
    axb.set_yticklabels([])
    axb.invert_yaxis()
    axb.set_xlabel("P(winner) per draw")
    ci = prim["selection_inherited_ci_max_median"]
    axb.set_title(
        f"winner probability (rows as left panel)\n"
        f"selection-inherited CI on max median [{ci[0]:+.2f},{ci[1]:+.2f}]",
        fontsize=7,
    )
    fig.suptitle(
        f"{kind} race — gray left marks = per-query-grain medians (parent run, span-mean position)",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, stem, out_dir)


# ── per-unit companions ───────────────────────────────────────────────────────


def _scatter_panels(
    frame: dict,
    dv_key: str,
    dv_label: str,
    xcols: list[tuple[str, str]],
    title: str,
    stem: str,
    out_dir: Path,
) -> None:
    """Per-prefix scatter: DV vs each named predictor column, family-colored,
    trained + negative prefixes labeled, Spearman rho + p annotated per panel."""
    dv = np.asarray(frame[dv_key], dtype=float)
    fams = frame["family"]
    fam_list = sorted(set(fams))
    colors = dict(zip(fam_list, paper_palette(len(fam_list)), strict=True))
    fig, axes = plt.subplots(1, len(xcols), figsize=(4.6 * len(xcols) + 1.2, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, xlabel) in zip(axes, xcols, strict=True):
        x = np.asarray(frame[col], dtype=float)
        for fam in fam_list:
            m = np.asarray([ff == fam for ff in fams])
            ax.scatter(x[m], dv[m], s=20, color=colors[fam], label=fam)
        for i, pid in enumerate(frame["prefix_id"]):
            if fams[i] in ("trained", "negatives"):
                ax.annotate(pid, (x[i], dv[i]), fontsize=5)
        rho, p = stats.spearmanr(x, dv)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(f"rho = {rho:+.2f}, p = {p:.3g} (n = {len(dv)} prefixes)", fontsize=8)
    axes[0].set_ylabel(dv_label, fontsize=8)
    axes[-1].legend(fontsize=5.5, loc="best", title="prefix family", title_fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    _save(fig, stem, out_dir)


def scatter_change_p3b(frames: dict[str, dict], out_dir: Path) -> None:
    """CHANGE-DV per-unit view: the strong impolite-contrastive arm beside the
    sycophancy bare-context sign-reversal arm, DV vs the change champion."""
    for aid, stem_suffix in (
        ("imp-pers-con-lr3e5-s42", "imp"),
        ("syc-bare-con-lr1e5-s42", "syc"),
    ):
        assert aid in frames, aid
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, aid in zip(axes, ("imp-pers-con-lr3e5-s42", "syc-bare-con-lr1e5-s42"), strict=True):
        f = frames[aid]["frame"]
        x = np.asarray(f["p3b_tc"], dtype=float)
        dv = np.asarray(f["dv_change"], dtype=float)
        fams = f["family"]
        fam_list = sorted(set(fams))
        colors = dict(zip(fam_list, paper_palette(len(fam_list)), strict=True))
        for fam in fam_list:
            m = np.asarray([ff == fam for ff in fams])
            ax.scatter(x[m], dv[m], s=20, color=colors[fam], label=fam)
        for i, pid in enumerate(f["prefix_id"]):
            if fams[i] in ("trained", "negatives"):
                ax.annotate(pid, (x[i], dv[i]), fontsize=5)
        rho, p = stats.spearmanr(x, dv)
        ax.set_title(
            f"{_arm_label(aid)}\nrho = {rho:+.2f}, p = {p:.3g} (n = 50 prefixes)",
            fontsize=8,
        )
        ax.set_xlabel("through-map predicted-answer similarity", fontsize=8)
    axes[0].set_ylabel("per-prefix leakage change\n(trained − base, judge points)", fontsize=8)
    axes[1].legend(
        fontsize=5.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="prefix family",
        title_fontsize=6,
    )
    fig.tight_layout()
    _save(fig, "scatter_change_p3b", out_dir)


def scatter_level_p6_p7(frames: dict[str, dict], out_dir: Path) -> None:
    """LEVEL-DV per-unit view for one strong content arm: DV vs the two tied
    level candidates (through-map read-out projection; base propensity)."""
    aid = "imp-pers-po-lr1e5-s42"
    _scatter_panels(
        frames[aid]["frame"],
        "dv_level",
        "per-prefix leakage level\n(judge score 0-100, query mean)",
        [
            ("p6", "read-out projection (through map)"),
            ("p7", "base propensity (judge score 0-100)"),
        ],
        f"leakage level per prefix — {_arm_label(aid)}",
        "scatter_level_p6_p7",
        out_dir,
    )


def scatter_marker_p9(frames: dict[str, dict], out_dir: Path) -> None:
    """Marker per-unit view: Δ log P(marker) vs the marker champion
    (nearest-training-rows context similarity) and the propensity incumbent."""
    aid = "mk-pers-con-lr5e6-s42"
    _scatter_panels(
        frames[aid]["frame"],
        "dv_dlogp",
        "per-prefix Δ log P(marker)\n(trained − base, nats)",
        [
            ("p9_k8", "nearest training rows, context side (top-8 cosine)"),
            ("p7", "base propensity (base slot log P)"),
        ],
        f"marker leakage change per prefix — {_arm_label(aid)}",
        "scatter_marker_p9",
        out_dir,
    )


def scatter_per_arm(frames: dict[str, dict], out_dir: Path) -> None:
    for aid, fr in frames.items():
        f = fr["frame"]
        dv_name = "dv_level" if "dv_level" in f else "dv_dlogp"
        dv_label = (
            "per-prefix leakage level (judge 0-100)"
            if dv_name == "dv_level"
            else "per-prefix Δ log P(marker) (nats)"
        )
        _scatter_panels(
            f,
            dv_name,
            dv_label,
            [
                ("p7", "base propensity"),
                ("p1_tc", "context similarity"),
                ("p2_tc", "answer similarity"),
            ],
            f"per-prefix DV — {_arm_label(aid)} (points = prefixes)",
            f"scatter_{aid}",
            out_dir,
        )


H5_GROUP_LABELS = {
    "con": "contrastive",
    "po": "positive-only placebo",
    "bare-placebo": "bare-context placebo",
}


def h5_dots(h5: dict, out_dir: Path) -> None:
    per = {a: v for a, v in h5["per_arm"].items() if "median_signed_residual" in v}
    if not per:
        return
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(per) + 1.8))
    ids = sorted(per, key=lambda a: (per[a]["h5_group"], a))
    pal = paper_palette(3)
    gcol = {"con": pal[0], "po": pal[1], "bare-placebo": pal[2]}
    seen: set[str] = set()
    for i, a in enumerate(ids):
        v = per[a]
        g = v["h5_group"]
        ax.scatter(v["residuals"], [i] * len(v["residuals"]), s=14, color=gcol[g])
        ax.scatter(
            [v["median_signed_residual"]],
            [i],
            marker="D",
            s=42,
            color=gcol[g],
            edgecolor="black",
            zorder=3,
            label=H5_GROUP_LABELS[g] if g not in seen else None,
        )
        seen.add(g)
    ax.axvline(0.0, color="gray", lw=0.8)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(
        [f"{_arm_label(a)} ({H5_GROUP_LABELS[per[a]['h5_group']]})" for a in ids],
        fontsize=6,
    )
    ax.set_xlabel("signed residual of the 5 trained-negative prefixes vs within-arm geometry fit")
    ax.set_title(
        "trained-negative prefix residuals per arm (dots = prefixes, diamonds = medians)",
        fontsize=8,
    )
    ax.legend(fontsize=6, title="arm group", title_fontsize=6)
    fig.tight_layout()
    _save(fig, "h5_residuals", out_dir)


def gate_dots(battery: dict, out_dir: Path) -> None:
    cells = battery["A7"].get("cells", {})
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(9, 0.22 * len(cells) + 1.8))
    keys = sorted(cells)
    ax.axvspan(*GATE_BAND, color="green", alpha=0.12, label="registered band [0.3, 0.7]")
    ax.axvline(GATE_PER_QUERY_ANCHOR, color="gray", ls="--", lw=0.9, label="per-query anchor 0.14")
    ax.scatter([cells[k] for k in keys], range(len(keys)), s=16, color=paper_palette(1)[0])
    ax.set_yticks(range(len(keys)))
    labels = []
    for k in keys:
        arm, layer = k.split("/")[:2]  # "<arm>/L<layer>/span_mean_context"
        labels.append(f"{_arm_label(arm)} · layer {layer.removeprefix('L')}")
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Spearman rho(predicted gate, realized matched-text write coefficient)")
    ax.set_title(
        "whitened-gate prediction vs realized per-prefix write, per (arm, layer)", fontsize=8
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "a7_gate", out_dir)


def a6_dots(battery: dict, out_dir: Path) -> None:
    per = battery["A6"]["per_arm_tree_layer"]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(per) + 1.8))
    ids = sorted(per)
    pal = paper_palette(2)
    for i, a in enumerate(ids):
        for tree, col, mk, lbl in (
            ("matched", pal[0], "o", "matched-text tree"),
            ("onpolicy", pal[1], "s", "on-policy tree"),
        ):
            xs = [per[a][f"{tree}/L{li}"]["top1_var_share"] for li in (14, 19, 25)]
            ax.scatter(xs, [i] * 3, s=18, color=col, marker=mk, label=lbl if i == 0 else None)
    ax.axvline(A6_CRITERION, color="black", ls="--", lw=0.9, label="criterion 0.6")
    ref = battery["A6"]["per_query_reference"]
    ax.axvline(ref["matched"], color="gray", ls=":", lw=0.8, label="per-query matched 0.09")
    ax.axvline(ref["onpolicy"], color="gray", ls="-.", lw=0.8, label="per-query on-policy 0.29")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([_arm_label(a) for a in ids], fontsize=6)
    ax.set_xlabel("top-1 SVD variance share of the centered 50-prefix write matrix")
    ax.set_title("write-matrix rank read (both trees, layers 14/19/25)", fontsize=8)
    ax.legend(fontsize=5)
    fig.tight_layout()
    _save(fig, "a6_top1_share", out_dir)


def a5_dots(battery: dict, out_dir: Path) -> None:
    per = battery["A5"]
    if not per:
        return
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(per) + 1.8))
    ids = sorted(per)
    pal = paper_palette(2)
    for i, a in enumerate(ids):
        v = per[a]
        b = v["null_bands_vs_pooled_delta"]["corpus_cov"]
        ax.plot([b["p2_5"], b["p97_5"]], [i, i], color="lightgray", lw=5, zorder=1)
        ax.scatter(
            [v["pooled_cos_disjoint"]],
            [i],
            s=36,
            color=pal[0],
            zorder=3,
            label="pooled cos (disjoint halves)" if i == 0 else None,
        )
        ax.scatter(
            [v["median_per_prefix_cos_disjoint"]],
            [i],
            s=22,
            marker="s",
            color=pal[1],
            zorder=3,
            label="median per-prefix cos" if i == 0 else None,
        )
        ax.scatter(
            [v["pooled_cos_sharedB_record_only"]],
            [i],
            s=18,
            marker="x",
            color="gray",
            zorder=2,
            label="shared-baseline read (record-only, known-invalid)" if i == 0 else None,
        )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([_arm_label(a) for a in ids], fontsize=6)
    ax.set_xlabel("cos(matched-text write, on-policy delta) — layer-19 span mean")
    ax.set_title(
        "write-delta alignment per arm (gray bands = corpus-covariance norm-matched null 95%)",
        fontsize=8,
    )
    ax.legend(fontsize=5)
    fig.tight_layout()
    _save(fig, "a5_alignment", out_dir)


def a5_decomposition_panel(decomp: dict, out_dir: Path) -> None:
    """Weights-vs-text decomposition of the on-policy delta per marker arm
    (amendment marker-a5-weights-vs-text, plan v6 §6 hero figure).

    Left: per-arm pooled cosines vs the matched-text write — open gray circle =
    the parent full-delta read, colored dots = the weights-carried vs
    different-text components, light-gray bars = each component's
    corpus-covariance norm-matched null 95% band. Right: per-prefix cosine
    strips per component (points = prefixes — the low-level per-unit panel).
    """
    per = decomp["arms"]
    if not per:
        return
    ids = sorted(per)
    pal = paper_palette(2)
    comp_meta = (
        ("weights", pal[0], -0.18, "weights-carried component"),
        ("text", pal[1], +0.18, "different-text component"),
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 0.42 * len(ids) + 2.4), sharey=True, constrained_layout=True
    )
    ax = axes[0]
    for i, a in enumerate(ids):
        v = per[a]
        ax.scatter(
            [v["parent_leg_cos"]],
            [i],
            s=46,
            facecolors="none",
            edgecolors="dimgray",
            zorder=2,
            label="full on-policy delta (parent read)" if i == 0 else None,
        )
        for comp, color, dy, lbl in comp_meta:
            c = v["primary"][comp]
            b = c["null_bands"]["corpus_cov"]
            ax.plot([b["p2_5"], b["p97_5"]], [i + dy, i + dy], color="lightgray", lw=4, zorder=1)
            ax.scatter(
                [c["pooled_cos"]],
                [i + dy],
                s=30,
                color=color,
                zorder=3,
                label=lbl if i == 0 else None,
            )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([_arm_label(a) for a in ids], fontsize=6)
    ax.set_xlabel("cos(matched-text write, delta component) — layer-19 span mean")
    ax.set_title(
        "pooled write-alignment per delta component\n"
        "(gray bars = corpus-covariance norm-matched null 95%)",
        fontsize=8,
    )
    ax.legend(fontsize=5.5, loc="best")
    ax2 = axes[1]
    for i, a in enumerate(ids):
        for comp, color, dy, lbl in comp_meta:
            xs = per[a]["primary"][comp]["per_prefix_cos"]
            ax2.scatter(
                xs,
                [i + dy] * len(xs),
                s=7,
                alpha=0.45,
                color=color,
                label=lbl if i == 0 else None,
            )
    ax2.axvline(0, color="black", lw=0.6)
    ax2.set_xlabel("per-prefix cos(matched-text write, delta component)")
    ax2.set_title("per-prefix alignment strips (one point per destination prefix)", fontsize=8)
    ax2.legend(fontsize=5.5, loc="best")
    _save(fig, "a5_decomposition", out_dir)


MED_KEYS = [
    ("r_p1_given_p7", "context sim | propensity"),
    ("r_p1_given_p2_p7", "context sim | answer sim + propensity"),
    ("r_p2_given_p7", "answer sim | propensity"),
    ("r_p2_given_p1_p7", "answer sim | context sim + propensity"),
]


def mediation_forest(mediation: dict, out_dir: Path) -> None:
    per = mediation["per_arm"]
    if not per:
        return
    ids = sorted(per)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(ids) + 1.8))
    pal = paper_palette(4)
    for j, (key, lbl) in enumerate(MED_KEYS):
        xs, ys = [], []
        for i, a in enumerate(ids):
            v = per[a].get(key)
            if v is not None:
                xs.append(v)
                ys.append(i + (j - 1.5) * 0.15)
        ax.scatter(xs, ys, s=22, color=pal[j], label=lbl)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([_arm_label(a) for a in ids], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("partial Spearman rho with per-prefix leakage level")
    ax.set_title(
        "context vs answer similarity partials, base propensity always conditioned", fontsize=8
    )
    ax.legend(fontsize=5.5, loc="best")
    fig.tight_layout()
    _save(fig, "mediation_forest", out_dir)


def mapping_bars(mapping: dict, out_dir: Path) -> None:
    fits = mapping.get("fits", {})
    if not fits:
        return
    keys = sorted(fits)

    def _key_label(k: str) -> str:
        fam, layer, pos = k.split("/")
        pos_lbl = {"last_prefix": "last prefix token", "prefix_span": "prefix span mean"}.get(
            pos, pos
        )
        return f"{fam} · layer {layer.removeprefix('L')} · {pos_lbl}"

    fig, ax = plt.subplots(figsize=(0.75 * len(keys) + 3, 4.0))
    x = np.arange(len(keys))
    pal = paper_palette(2)
    ax.bar(
        x - 0.2,
        [fits[k]["lofo_r2"] for k in keys],
        width=0.4,
        color=pal[0],
        label="leave-one-family-out ridge R2",
    )
    ax.bar(
        x + 0.2,
        [fits[k]["identity_bias_lofo_r2"] for k in keys],
        width=0.4,
        color=pal[1],
        label="identity + learned-bias baseline R2",
    )
    for i, k in enumerate(keys):
        acc = fits[k]["knn"]["fitted_cos"]["acc_at_k"]
        first = next(iter(acc.values()))
        ax.text(i, max(fits[k]["lofo_r2"], 0) + 0.15, f"acc@1={first:.2f}", fontsize=5, ha="center")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_key_label(k) for k in keys], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("held-out R2 (symlog axis)")
    ax.set_title(
        "prefix-to-answer mapping fits (exploratory; n=50 < d=3584 regularization-limit regime)",
        fontsize=8,
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "mapping_arm", out_dir)


def dump_heatmap(dump: dict, out_dir: Path) -> None:
    arms = dump.get("arms", {})
    if not arms:
        return
    cells = sorted(next(iter(arms.values())).keys())
    cands = sorted({c for a in arms.values() for cell in a.values() for c in cell})
    M = np.full((len(cands), len(cells)), np.nan)
    for j, cell in enumerate(cells):
        for i, c in enumerate(cands):
            vals = [a[cell][c] for a in arms.values() if cell in a and c in a[cell]]
            if vals:
                M[i, j] = float(np.median(vals))
    fig, ax = plt.subplots(figsize=(0.9 * len(cells) + 2, 0.28 * len(cands) + 1.6))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(cands)))
    ax.set_yticklabels([_cand_label(c) for c in cands], fontsize=6)
    ax.set_title("exploratory dump: across-arm median rho per (layer x position)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    _save(fig, "dump_grid", out_dir)


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--race-dir", type=Path, default=RACE_DIR)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated figure stems to render (default: all)",
    )
    args = ap.parse_args(argv)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def want(stem: str) -> bool:
        return not only or stem in only

    set_paper_style("generic")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = _arm_jsons(args.race_dir)
    frames = _frames(args.race_dir)
    crossgrain = _load(args.race_dir, "crossgrain.json")
    if want("hero_content_race") and (args.race_dir / "champion_level.json").exists():
        hero_heatmap(
            arms,
            _load(args.race_dir, "champion_level.json"),
            "content",
            ("dv_level", "dv_change"),
            crossgrain,
            "hero_content_race",
            args.out_dir,
        )
    if want("hero_marker_race") and (args.race_dir / "champion_marker.json").exists():
        hero_heatmap(
            arms,
            _load(args.race_dir, "champion_marker.json"),
            "marker",
            ("dv_dlogp",),
            crossgrain,
            "hero_marker_race",
            args.out_dir,
        )
    if want("scatter_change_p3b"):
        scatter_change_p3b(frames, args.out_dir)
    if want("scatter_level_p6_p7"):
        scatter_level_p6_p7(frames, args.out_dir)
    if want("scatter_marker_p9"):
        scatter_marker_p9(frames, args.out_dir)
    if not only:
        scatter_per_arm(frames, args.out_dir)
    if want("h5_residuals"):
        h5_dots(_load(args.race_dir, "h5_residuals.json"), args.out_dir)
    battery = _load(args.race_dir, "battery_verdicts.json")
    if want("a7_gate"):
        gate_dots(battery, args.out_dir)
    if want("a6_top1_share"):
        a6_dots(battery, args.out_dir)
    if want("a5_alignment"):
        a5_dots(battery, args.out_dir)
    if want("a5_decomposition") and (args.race_dir / "a5_decomposition.json").exists():
        a5_decomposition_panel(_load(args.race_dir, "a5_decomposition.json"), args.out_dir)
    if want("mediation_forest"):
        mediation_forest(_load(args.race_dir, "mediation.json"), args.out_dir)
    if want("mapping_arm"):
        mapping_bars(_load(args.race_dir, "mapping_arm.json"), args.out_dir)
    if want("dump_grid"):
        dump_heatmap(_load(args.race_dir, "dump_grid.json"), args.out_dir)
    n = len(list(args.out_dir.glob("*.png")))
    print(f"[phase=done] figs n_png={n} out={args.out_dir}", flush=True)
    assert n >= 5, f"too few figures rendered: {n}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
