"""#1979 F3 figures — the plan §6 figure list from the race outputs.

Reads ``eval_results/issue_1979/race/`` (summary, champions, frames, verdict
JSONs) and renders to ``figures/issue_1979/`` via the paper-plots conventions
(``analysis/paper_plots.py``: ``set_paper_style`` + ``savefig_paper`` — PNG +
PDF + ``.meta.json`` sidecar with commit pin + per-point data).

HERO: candidates x content-arms heatmap of within-arm rho (level + change side
by side) with the across-arm median column + winner-probability bars
(selection-inherited CI labeled) and the per-query #1900 medians as ghost
marks (cross-grain read, span-mean<->span-mean position-matched). Low-level
per-unit companions: per-arm DV scatters vs {P7, P1, P2} (points = prefixes,
colored by family, trained + negative prefixes labeled), H5 residual dots,
A7 gate dots vs the [0.3, 0.7] band (0.14 per-query anchor drawn), A6 top-1
share dots (both trees) vs 0.6, A5 alignment dots with null bands, mediation
forest, marker-panel hero replicate, mapping-arm LOFO bars vs identity+bias.
Exploratory dump: the (layer x position) grid heatmap.

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
        figsize=(4.2 * n_dv + 2.6, 0.42 * len(cands) + 1.8),
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
        ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(cands)))
        ax.set_yticklabels(cands, fontsize=7)
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
        ax.set_title(f"{dv} (within-arm Spearman rho)")
        fig.colorbar(im, ax=ax, fraction=0.03)
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
        f"winner={prim['winner_observed']}\nsel-inherited CI [{ci[0]:+.2f},{ci[1]:+.2f}]",
        fontsize=7,
    )
    fig.suptitle(
        f"{kind} race — gray left marks = per-query #1900 medians (span-mean cross-grain)",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, stem, out_dir)


# ── per-unit companions ───────────────────────────────────────────────────────


def scatter_per_arm(frames: dict[str, dict], out_dir: Path) -> None:
    for aid, fr in frames.items():
        f = fr["frame"]
        dv_name = "dv_level" if "dv_level" in f else "dv_dlogp"
        dv = np.asarray(f[dv_name], dtype=float)
        fams = f["family"]
        fam_list = sorted(set(fams))
        colors = dict(zip(fam_list, paper_palette(len(fam_list)), strict=True))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        for ax, cand in zip(axes, ("p7", "p1_tc", "p2_tc"), strict=True):
            x = np.asarray(f[cand], dtype=float)
            for fam in fam_list:
                m = np.asarray([ff == fam for ff in fams])
                ax.scatter(x[m], dv[m], s=18, color=colors[fam], label=fam)
            for i, pid in enumerate(f["prefix_id"]):
                if fams[i] in ("trained", "negatives"):
                    ax.annotate(pid, (x[i], dv[i]), fontsize=5)
            ax.set_xlabel(cand)
        axes[0].set_ylabel(dv_name)
        axes[0].legend(fontsize=5, loc="best")
        fig.suptitle(f"{aid}: per-prefix DV vs {{P7, P1, P2}} (points = prefixes)", fontsize=8)
        fig.tight_layout()
        _save(fig, f"scatter_{aid}", out_dir)


def h5_dots(h5: dict, out_dir: Path) -> None:
    per = {a: v for a, v in h5["per_arm"].items() if "median_signed_residual" in v}
    if not per:
        return
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(per) + 1.6))
    ids = sorted(per, key=lambda a: (per[a]["h5_group"], a))
    pal = paper_palette(3)
    gcol = {"con": pal[0], "po": pal[1], "bare-placebo": pal[2]}
    for i, a in enumerate(ids):
        v = per[a]
        ax.scatter(v["residuals"], [i] * len(v["residuals"]), s=14, color=gcol[v["h5_group"]])
        ax.scatter(
            [v["median_signed_residual"]],
            [i],
            marker="D",
            s=42,
            color=gcol[v["h5_group"]],
            edgecolor="black",
            zorder=3,
        )
    ax.axvline(0.0, color="gray", lw=0.8)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"{a} [{per[a]['h5_group']}]" for a in ids], fontsize=6)
    ax.set_xlabel("signed residual of NEGATIVE prefixes vs within-arm geometry fit")
    ax.set_title(f"H5 — verdict: {h5['verdict']}", fontsize=8)
    fig.tight_layout()
    _save(fig, "h5_residuals", out_dir)


def gate_dots(battery: dict, out_dir: Path) -> None:
    cells = battery["A7"].get("cells", {})
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(cells) + 1.6))
    keys = sorted(cells)
    ax.axvspan(*GATE_BAND, color="green", alpha=0.12, label="registered band [0.3, 0.7]")
    ax.axvline(GATE_PER_QUERY_ANCHOR, color="gray", ls="--", lw=0.9, label="per-query anchor 0.14")
    ax.scatter([cells[k] for k in keys], range(len(keys)), s=16, color=paper_palette(1)[0])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=5)
    ax.set_xlabel("Spearman rho(g_pred, g_hat) per (arm, layer)")
    ax.set_title(f"A7 gate — verdict: {battery['A7'].get('verdict')}", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    _save(fig, "a7_gate", out_dir)


def a6_dots(battery: dict, out_dir: Path) -> None:
    per = battery["A6"]["per_arm_tree_layer"]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(per) + 1.6))
    ids = sorted(per)
    pal = paper_palette(2)
    for i, a in enumerate(ids):
        for tree, col, mk in (("matched", pal[0], "o"), ("onpolicy", pal[1], "s")):
            xs = [per[a][f"{tree}/L{li}"]["top1_var_share"] for li in (14, 19, 25)]
            ax.scatter(xs, [i] * 3, s=18, color=col, marker=mk, label=tree if i == 0 else None)
    ax.axvline(A6_CRITERION, color="black", ls="--", lw=0.9, label="criterion 0.6")
    ref = battery["A6"]["per_query_reference"]
    ax.axvline(ref["matched"], color="gray", ls=":", lw=0.8, label="per-query matched 0.09")
    ax.axvline(ref["onpolicy"], color="gray", ls="-.", lw=0.8, label="per-query onpolicy 0.29")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=6)
    ax.set_xlabel("top-1 SVD variance share of the centered 50-prefix write matrix")
    ax.set_title("A6 rank read (both trees, layers 14/19/25)", fontsize=8)
    ax.legend(fontsize=5)
    fig.tight_layout()
    _save(fig, "a6_top1_share", out_dir)


def a5_dots(battery: dict, out_dir: Path) -> None:
    per = battery["A5"]
    if not per:
        return
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(per) + 1.6))
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
            label="shared-B (record-only, known-invalid)" if i == 0 else None,
        )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=6)
    ax.set_xlabel("cos(matched-text write, on-policy delta) — L19 span")
    ax.set_title("A5 write-delta alignment (bands = corpus-cov norm-matched null 95%)", fontsize=8)
    ax.legend(fontsize=5)
    fig.tight_layout()
    _save(fig, "a5_alignment", out_dir)


def mediation_forest(mediation: dict, out_dir: Path) -> None:
    per = mediation["per_arm"]
    if not per:
        return
    rows = []
    for a, v in per.items():
        rows.append((a, "p1|p7", v["r_p1_given_p7"]))
        rows.append((a, "p1|p2,p7", v["r_p1_given_p2_p7"]))
        rows.append((a, "p2|p7", v["r_p2_given_p7"]))
        rows.append((a, "p2|p1,p7", v["r_p2_given_p1_p7"]))
    fig, ax = plt.subplots(figsize=(7, 0.16 * len(rows) + 1.6))
    pal = paper_palette(4)
    kcol = dict(zip(["p1|p7", "p1|p2,p7", "p2|p7", "p2|p1,p7"], pal, strict=True))
    labels = []
    for i, (a, k, v) in enumerate(rows):
        if v is not None:
            ax.scatter([v], [i], s=20, color=kcol[k])
        labels.append(f"{a} {k}")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("partial Spearman rho")
    ax.set_title(f"mediation lattice — {mediation['lattice']['verdict']} (provisional)", fontsize=8)
    fig.tight_layout()
    _save(fig, "mediation_forest", out_dir)


def mapping_bars(mapping: dict, out_dir: Path) -> None:
    fits = mapping.get("fits", {})
    if not fits:
        return
    keys = sorted(fits)
    fig, ax = plt.subplots(figsize=(0.6 * len(keys) + 3, 3.4))
    x = np.arange(len(keys))
    pal = paper_palette(2)
    ax.bar(
        x - 0.2, [fits[k]["lofo_r2"] for k in keys], width=0.4, color=pal[0], label="LOFO ridge R2"
    )
    ax.bar(
        x + 0.2,
        [fits[k]["identity_bias_lofo_r2"] for k in keys],
        width=0.4,
        color=pal[1],
        label="identity+bias R2",
    )
    for i, k in enumerate(keys):
        acc = fits[k]["knn"]["fitted_cos"]["acc_at_k"]
        first = next(iter(acc.values()))
        ax.text(i, max(fits[k]["lofo_r2"], 0) + 0.02, f"acc@1={first:.2f}", fontsize=5, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("pooled LOFO R2")
    ax.set_title(
        "prefix-based v_P->v_A mapping arm (EXPLORATORY; n<d reg-limit regime)", fontsize=8
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
    ax.set_yticklabels(cands, fontsize=6)
    ax.set_title("exploratory dump: across-arm median rho per (layer x position)", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    _save(fig, "dump_grid", out_dir)


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--race-dir", type=Path, default=RACE_DIR)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args(argv)
    set_paper_style("generic")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = _arm_jsons(args.race_dir)
    frames = _frames(args.race_dir)
    crossgrain = _load(args.race_dir, "crossgrain.json")
    if (args.race_dir / "champion_level.json").exists():
        hero_heatmap(
            arms,
            _load(args.race_dir, "champion_level.json"),
            "content",
            ("dv_level", "dv_change"),
            crossgrain,
            "hero_content_race",
            args.out_dir,
        )
    if (args.race_dir / "champion_marker.json").exists():
        hero_heatmap(
            arms,
            _load(args.race_dir, "champion_marker.json"),
            "marker",
            ("dv_dlogp",),
            crossgrain,
            "hero_marker_race",
            args.out_dir,
        )
    scatter_per_arm(frames, args.out_dir)
    h5_dots(_load(args.race_dir, "h5_residuals.json"), args.out_dir)
    battery = _load(args.race_dir, "battery_verdicts.json")
    gate_dots(battery, args.out_dir)
    a6_dots(battery, args.out_dir)
    a5_dots(battery, args.out_dir)
    mediation_forest(_load(args.race_dir, "mediation.json"), args.out_dir)
    mapping_bars(_load(args.race_dir, "mapping_arm.json"), args.out_dir)
    dump_heatmap(_load(args.race_dir, "dump_grid.json"), args.out_dir)
    n = len(list(args.out_dir.glob("*.png")))
    print(f"[phase=done] figs n_png={n} out={args.out_dir}", flush=True)
    assert n >= 5, f"too few figures rendered: {n}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
