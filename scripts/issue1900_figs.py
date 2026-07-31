"""#1900 figures — plan §6 figure set over the P3 race outputs (VM, CPU).

Paper-plots conventions (`analysis/paper_plots.py`): `set_paper_style("blog")`
rcParams, colorblind-safe Wong-derived palette, `savefig_paper` sidecars, NO
annotations/arrows/effect-size overlays (project rule). One color = one
meaning: a FIXED candidate->color map shared by every figure. All labels are
plain English (slug->name maps below).

Figures (plan §6): HERO candidates x content-arms heatmap + across-arm median
column; winner-probability bar; per-arm DV-vs-{P7,P1,P2} scatters (points =
contexts); mediation forest (raw vs partial); marker-panel hero replicate;
commonality stacked bars; P3-recovers-P2 structural scatter; anchor-comparison
paired dots; exploratory dump (layers x anchors grid, M-panel heatmap +
M4-residual-channel caption note, binary-rate hero variant, per-behavior
split, LOFO combination reads, band-vs-ceiling display).

Caption notes that must ride the artifacts (registered line (5) M4 prose, DV
provenance) land in `<fig-dir>/figures_notes.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps before numpy/matplotlib

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1900.figs")

CANDIDATE_NAMES = {
    "p1": "Context similarity (pre)",
    "p2": "Answer similarity (pre)",
    "p3a": "Through-map context sim",
    "p3b": "Through-map pred-answer sim",
    "p4": "Whitened gate similarity",
    "p5": "Read-out projection (direct)",
    "p6": "Read-out projection (map)",
    "p7": "Base behavioral propensity",
    "p8a": "Write prediction (size)",
    "p8b": "Write prediction (alignment)",
    "p9": "Nearest-training-rows sim",
}
M_NAMES = {
    "m1": "Post-FT context sim",
    "m2": "Post-FT answer sim",
    "m3": "Delta-context sim",
    "m4": "Delta-answer sim (matched text)",
    "m5": "Through-map sim (post)",
    "m6": "Write magnitude",
}
BEH_NAMES = {
    "cas": "casual writing",
    "imp": "impoliteness",
    "syc": "sycophancy",
    "mk": "marker",
}
CTX_NAMES = {"pers": "persona", "bare": "bare", "conv": "conversation", "icl": "ICL"}


def arm_plain(arm_id: str) -> str:
    """`cas-pers-con-lr1e5-s42` -> 'casual writing / persona / contrastive (s42)'."""
    parts = arm_id.split("-")
    beh = BEH_NAMES.get(parts[0], parts[0])
    ctx = CTX_NAMES.get(parts[1], parts[1])
    regime = "contrastive" if "con" in parts[2:] else "positive-only"
    ft = " full-FT" if "ft" in parts else ""
    seed = parts[-1]
    return f"{beh} / {ctx} / {regime}{ft} ({seed})"


def _cand_colors(cands: list[str]) -> dict[str, str]:
    from explore_persona_space.analysis.paper_plots import paper_palette

    order = list(CANDIDATE_NAMES)  # STABLE global order -> one color = one meaning
    pal = paper_palette(len(order))
    return {c: pal[order.index(c)] for c in cands if c in order}


def _save(fig, stem: str, fig_dir: Path) -> Path:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    paths = savefig_paper(fig, stem, dir=fig_dir, formats=("png",))
    plt.close(fig)
    return paths["png"]


def _load_arm_jsons(race_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(race_dir.glob("arm_*.json"))]


def _heatmap(
    arm_payloads: list[dict], dv_index_name: str, stem: str, fig_dir: Path, title: str
) -> Path | None:
    arms = [p for p in arm_payloads if dv_index_name in p["observed_rho"]]
    if not arms:
        return None
    cands = sorted(
        set.intersection(*[set(p["observed_rho"][dv_index_name]) for p in arms]),
        key=lambda c: list(CANDIDATE_NAMES).index(c) if c in CANDIDATE_NAMES else 99,
    )
    mat = np.array(
        [[p["observed_rho"][dv_index_name][c] for p in arms] for c in cands], dtype=float
    )
    med = np.median(mat, axis=1, keepdims=True)
    full = np.concatenate([mat, med], axis=1)
    fig, ax = plt.subplots(
        figsize=(max(6.5, 0.62 * full.shape[1] + 3.2), 0.42 * len(cands) + 1.6),
        layout="constrained",
    )
    vmax = float(np.nanmax(np.abs(full))) or 1.0
    im = ax.imshow(full, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(
        range(full.shape[1]),
        [arm_plain(p["arm_id"]) for p in arms] + ["across-arm median"],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(cands)), [CANDIDATE_NAMES.get(c, c) for c in cands])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Spearman rho (per context)")
    return _save(fig, stem, fig_dir)


def fig_winner_probability(champ: dict, stem: str, fig_dir: Path) -> Path:
    p_win = champ["p_win"]
    cands = sorted(p_win, key=lambda c: p_win[c], reverse=True)
    colors = _cand_colors(cands)
    fig, ax = plt.subplots(layout="constrained")
    ax.bar(
        range(len(cands)),
        [p_win[c] for c in cands],
        color=[colors.get(c, "#888888") for c in cands],
    )
    ax.set_xticks(
        range(len(cands)), [CANDIDATE_NAMES.get(c, c) for c in cands], rotation=45, ha="right"
    )
    ax.set_ylabel("P(winner across bootstrap draws)")
    ax.set_title(f"Winner probability — {champ['dv']}")
    ax.axhline(0.5, color="#555555", linewidth=0.8, linestyle="--")
    return _save(fig, stem, fig_dir)


def fig_dv_scatters(
    assemblies: list[dict], cand: str, cand_col: str | None, stem: str, fig_dir: Path
) -> Path:
    n = len(assemblies)
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow), layout="constrained", squeeze=False
    )
    for i, asm in enumerate(assemblies):
        ax = axes[i // ncol][i % ncol]
        f = asm["frame"]
        x = (f["p7"] if cand == "p7" else f[cand_col]).to_numpy(float)
        y = f[asm["dv_names"][0]].to_numpy(float)
        ax.scatter(x, y, s=5, alpha=0.35, color=_cand_colors([cand]).get(cand, "#333"))
        ax.set_title(arm_plain(asm["arm"]["arm_id"]), fontsize=8)
        ax.set_xlabel(CANDIDATE_NAMES.get(cand, cand), fontsize=7)
        ax.set_ylabel("leakage DV", fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    return _save(fig, stem, fig_dir)


def fig_mediation_forest(mediation: dict, stem: str, fig_dir: Path) -> Path:
    per_arm = mediation["per_arm"]
    arms = sorted(per_arm)
    measures = [
        ("r_p1_given_p7", "context sim, given propensity"),
        ("r_p1_given_p2_p7", "context sim, given answer sim + propensity"),
        ("r_p2_given_p7", "answer sim, given propensity"),
        ("r_p2_given_p1_p7", "answer sim, given context sim + propensity"),
    ]
    colors = _cand_colors(["p1", "p2"])
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(arms) + 1.8), layout="constrained")
    offs = np.linspace(-0.3, 0.3, len(measures))
    markers = ["o", "s", "o", "s"]
    for k, (key, label) in enumerate(measures):
        ys, xs = [], []
        for i, a in enumerate(arms):
            v = per_arm[a].get(key)
            if v is not None:
                ys.append(i + offs[k])
                xs.append(v)
        ax.scatter(
            xs,
            ys,
            s=22,
            marker=markers[k],
            facecolors="none" if "p2_p7" in key or "p1_p7" in key else None,
            color=colors["p1"] if key.startswith("r_p1") else colors["p2"],
            label=label,
            # blog style zeroes scatter edge widths; open markers need an
            # explicit width or they render invisible (#536 pitfall)
            linewidths=1.2,
        )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(range(len(arms)), [arm_plain(a) for a in arms])
    ax.set_xlabel("partial Spearman rho (rank residuals)")
    ax.set_title("Mediation: context vs answer similarity, base propensity partialled")
    # legend outside the axes so no per-arm marker is ever occluded
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, stem, fig_dir)


def fig_commonality(mediation: dict, stem: str, fig_dir: Path) -> Path:
    per_arm = mediation["per_arm"]
    arms = sorted(per_arm)
    comps = [
        ("unique_p1", "unique: context sim"),
        ("unique_p2", "unique: answer sim"),
        ("unique_p7", "unique: base propensity"),
        ("common_p1_p2", "shared: ctx+ans"),
        ("common_p1_p7", "shared: ctx+base"),
        ("common_p2_p7", "shared: ans+base"),
        ("common_p1_p2_p7", "shared: all three"),
    ]
    from explore_persona_space.analysis.paper_plots import paper_palette

    pal = paper_palette(len(comps))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    x = np.arange(len(arms))
    pos_bottom = np.zeros(len(arms))
    neg_bottom = np.zeros(len(arms))
    for k, (key, label) in enumerate(comps):
        vals = np.array([per_arm[a]["commonality"][key] for a in arms])
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        ax.bar(x, pos, bottom=pos_bottom, color=pal[k], label=label, width=0.7)
        ax.bar(x, neg, bottom=neg_bottom, color=pal[k], width=0.7)
        pos_bottom += pos
        neg_bottom += neg
    ax.set_xticks(x, [arm_plain(a) for a in arms], rotation=45, ha="right")
    ax.set_ylabel("rank-R^2 commonality component")
    ax.set_title("Commonality decomposition of leakage rank-R^2 over {P1, P2, P7}")
    ax.legend(fontsize=6, ncols=2)
    return _save(fig, stem, fig_dir)


def fig_p3_recovers_p2(mediation: dict, stem: str, fig_dir: Path) -> Path:
    per_arm = mediation["per_arm"]
    fig, ax = plt.subplots(layout="constrained")
    from explore_persona_space.analysis.paper_plots import paper_palette

    pal = paper_palette(2)
    for k, p3 in enumerate(("p3a", "p3b")):
        xs, ys = [], []
        for a, m in per_arm.items():
            s = m.get(f"structural_{p3}")
            if s:
                xs.append(s[f"rank_agreement_rho({p3},p1)"])
                ys.append(s[f"rank_agreement_rho({p3},p2)"])
        if xs:
            ax.scatter(xs, ys, s=26, color=pal[k], label=CANDIDATE_NAMES[p3])
    lim = [-1.02, 1.02]
    ax.plot(lim, lim, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("rank agreement with context sim (P1)")
    ax.set_ylabel("rank agreement with answer sim (P2)")
    ax.set_title("Does the through-map read recover answer-side ranking?")
    ax.legend(fontsize=7)
    return _save(fig, stem, fig_dir)


def fig_anchor_comparison(exploration: dict, stem: str, fig_dir: Path) -> Path | None:
    rows = [
        r
        for r in exploration["rows"]
        if r["layer"] == (19 if r["kind"] == "content" else 25) and r["rho"] is not None
    ]
    pairs = {}
    for r in rows:
        key = (r["arm_id"], r["candidate"])
        pairs.setdefault(key, {})[r["anchor"]] = r["rho"]
    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if not both:
        return None
    cands = sorted({k[1] for k in both})
    colors = _cand_colors(cands)
    fig, ax = plt.subplots(layout="constrained")
    for k, v in both.items():
        ax.plot(
            [0, 1],
            [v["training_centroid"], v["panel_source"]],
            marker="o",
            markersize=3,
            linewidth=0.7,
            alpha=0.6,
            color=colors.get(k[1], "#888888"),
        )
    ax.set_xticks([0, 1], ["training-centroid anchor", "panel-source anchor"])
    ax.set_ylabel("Spearman rho (per context)")
    ax.set_title("Anchor comparison (one line per arm x candidate)")
    handles = [
        plt.Line2D([0], [0], color=colors[c], label=CANDIDATE_NAMES.get(c, c)) for c in cands
    ]
    ax.legend(handles=handles, fontsize=6)
    return _save(fig, stem, fig_dir)


def fig_layers_grid(exploration: dict, stem: str, fig_dir: Path) -> Path | None:
    rows = [r for r in exploration["rows"] if r["rho"] is not None]
    if not rows:
        return None
    layers = sorted({r["layer"] for r in rows})
    anchors = sorted({r["anchor"] for r in rows})
    cands = sorted(
        {r["candidate"] for r in rows},
        key=lambda c: list(CANDIDATE_NAMES).index(c) if c in CANDIDATE_NAMES else 99,
    )
    fig, axes = plt.subplots(
        1,
        len(anchors),
        figsize=(4.6 * len(anchors), 0.38 * len(cands) + 1.6),
        layout="constrained",
        squeeze=False,
    )
    for j, anc in enumerate(anchors):
        mat = np.full((len(cands), len(layers)), np.nan)
        for ci, c in enumerate(cands):
            for li, layer in enumerate(layers):
                vals = [
                    r["rho"]
                    for r in rows
                    if r["candidate"] == c and r["layer"] == layer and r["anchor"] == anc
                ]
                if vals:
                    mat[ci, li] = float(np.median(vals))
        ax = axes[0][j]
        vmax = np.nanmax(np.abs(mat)) or 1.0
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(layers)), [f"layer {la}" for la in layers])
        ax.set_yticks(range(len(cands)), [CANDIDATE_NAMES.get(c, c) for c in cands])
        ax.set_title(f"{anc.replace('_', ' ')} anchor (across-arm median)")
        fig.colorbar(im, ax=ax, label="median Spearman rho")
    return _save(fig, stem, fig_dir)


def fig_m_panel(arm_payloads: list[dict], stem: str, fig_dir: Path) -> Path | None:
    arms = [p for p in arm_payloads if p.get("m_panel_rho_primary")]
    if not arms:
        return None
    ms = sorted({m for p in arms for m in p["m_panel_rho_primary"]})
    mat = np.array(
        [[p["m_panel_rho_primary"].get(m, np.nan) for p in arms] for m in ms], dtype=float
    )
    fig, ax = plt.subplots(
        figsize=(max(6.5, 0.6 * len(arms) + 3), 0.45 * len(ms) + 1.5), layout="constrained"
    )
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(arms)), [arm_plain(p["arm_id"]) for p in arms], rotation=45, ha="right")
    ax.set_yticks(range(len(ms)), [M_NAMES.get(m, m) for m in ms])
    ax.set_title("Mechanistic panel (explains, never the headline race)")
    fig.colorbar(im, ax=ax, label="Spearman rho (per context)")
    return _save(fig, stem, fig_dir)


def fig_combination(comb: dict, stem: str, fig_dir: Path) -> Path | None:
    loao = comb.get("loao_spearman_per_held_arm", {})
    if not isinstance(loao, dict) or not loao:
        return None
    arms = sorted(loao)
    vals = [loao[a] if loao[a] is not None else np.nan for a in arms]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    ax.bar(range(len(arms)), vals, color="#4477AA")
    ax.set_xticks(range(len(arms)), [arm_plain(a) for a in arms], rotation=45, ha="right")
    ax.set_ylabel("held-out-arm Spearman rho")
    ax.set_title("Fitted combination predictor — leave-one-arm-out")
    ax.axhline(0, color="#555555", linewidth=0.8)
    return _save(fig, stem, fig_dir)


def fig_band_vs_ceiling(arm_payloads: list[dict], stem: str, fig_dir: Path) -> Path:
    arms = arm_payloads
    x = np.arange(len(arms))
    band = np.array([p["perm_band"]["p975_max_selected"] for p in arms])
    obs_max = np.array(
        [max(v for v in p["observed_rho"][p["regime"]["dv_names"][0]].values()) for p in arms]
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    ax.scatter(x, obs_max, s=26, color="#EE6677", label="observed max-selected rho")
    ax.scatter(
        x, band, s=26, marker="_", color="#4477AA", label="permutation max-selected band (p97.5)"
    )
    ax.axhline(
        1.0, color="#555555", linewidth=0.8, linestyle="--", label="achievable ceiling (rho = 1)"
    )
    ax.set_xticks(x, [arm_plain(p["arm_id"]) for p in arms], rotation=45, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Selection-symmetric null band vs achievable ceiling, per arm")
    ax.legend(fontsize=7)
    return _save(fig, stem, fig_dir)


def render_all(race_dir: Path, fig_dir: Path, assemblies: list[dict] | None) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    arm_payloads = _load_arm_jsons(race_dir)
    content = [p for p in arm_payloads if p["kind"] == "content"]
    marker = [p for p in arm_payloads if p["kind"] == "marker"]

    def add(p: Path | None) -> None:
        if p is not None:
            made.append(p)

    add(
        _heatmap(
            content,
            "dv_level",
            "hero_content_race",
            fig_dir,
            "Leakage-predictor race — content arms (graded DV, per-context rho)",
        )
    )
    champ_path = race_dir / "champion_content.json"
    if champ_path.exists():
        champ = json.loads(champ_path.read_text())
        add(fig_winner_probability(champ["primary"], "winner_probability_content", fig_dir))
        add(
            fig_winner_probability(
                champ["change_companion"], "winner_probability_change_companion", fig_dir
            )
        )
    if marker:
        add(
            _heatmap(
                marker,
                "dv_dlogp",
                "hero_marker_race",
                fig_dir,
                "Marker replication panel (delta logP DV, per-context rho)",
            )
        )
    add(
        _heatmap(
            content,
            "dv_binary",
            "hero_content_binary_companion",
            fig_dir,
            "Binary-rate companion (fraction of draws >= 50)",
        )
    )
    med_path = race_dir / "mediation.json"
    if med_path.exists():
        mediation = json.loads(med_path.read_text())
        add(fig_mediation_forest(mediation, "mediation_forest", fig_dir))
        add(fig_commonality(mediation, "commonality_bars", fig_dir))
        add(fig_p3_recovers_p2(mediation, "p3_recovers_p2", fig_dir))
    expl_path = race_dir / "exploration.json"
    if expl_path.exists():
        exploration = json.loads(expl_path.read_text())
        add(fig_anchor_comparison(exploration, "anchor_comparison", fig_dir))
        add(fig_layers_grid(exploration, "exploration_layers_grid", fig_dir))
        # per-behavior split of the hero (content arms only)
        behs = sorted({p["beh_key"] for p in content})
        for b in behs:
            add(
                _heatmap(
                    [p for p in content if p["beh_key"] == b],
                    "dv_level",
                    f"hero_content_race_{b}",
                    fig_dir,
                    f"Race — {BEH_NAMES.get(b, b)} arms only",
                )
            )
    add(fig_m_panel(arm_payloads, "m_panel", fig_dir))
    comb_path = race_dir / "combination.json"
    if comb_path.exists():
        add(fig_combination(json.loads(comb_path.read_text()), "combination_loao", fig_dir))
    if arm_payloads:
        add(fig_band_vs_ceiling(arm_payloads, "band_vs_ceiling", fig_dir))
    if assemblies:
        for cand in ("p7", "p1", "p2"):
            col = {"p7": None, "p1": "p1_tc", "p2": "p2_tc"}[cand]
            add(fig_dv_scatters(assemblies, cand, col, f"dv_scatter_{cand}", fig_dir))
    notes = {
        "m_panel": "M4 (matched-text delta-answer similarity) retains a residual "
        "text-borne channel: the weight delta interacts with behavior-laden completion "
        "text even in matched-text form (registered line (5)); the mechanistic panel "
        "explains and never carries the headline.",
        "hero_content_race": "per-arm Spearman of each pre-FT candidate vs the graded "
        "on-policy leakage DV (mean of 3 judge draws); arms never raw-pooled.",
        "winner_probability_content": "winner re-selected inside every bootstrap draw "
        "(selection-symmetric); CI at the winner is selection-inherited "
        "(champion_content.json; frozen CI persisted alongside, labeled).",
        "band_vs_ceiling": "permutation band = per-draw max over the raced candidates "
        "(1,000 within-arm DV permutations); ceiling |rho| <= 1 shown; the "
        "champion-vs-P7 contrast's conditional ceiling interval is persisted in "
        "champion_content.json.",
        "generated_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = fig_dir / "figures_notes.json.tmp"
    tmp.write_text(json.dumps(notes, indent=1))
    os.replace(tmp, fig_dir / "figures_notes.json")
    return made


def main() -> None:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--race-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/race")
    ap.add_argument("--fig-dir", type=Path, default=REPO_ROOT / "figures/issue_1900")
    ap.add_argument("--config-dir", type=Path, default=REPO_ROOT / "data/issue_1900/config")
    ap.add_argument("--p1-root", type=Path, default=REPO_ROOT / "data/issue_1900/out")
    ap.add_argument("--judge-dir", type=Path, default=REPO_ROOT / "eval_results/issue_1900/judge")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-root", type=Path, default=REPO_ROOT / "data/issue_1900/race_smoke")
    ap.add_argument("--no-scatters", action="store_true", help="skip per-context scatters")
    args = ap.parse_args()

    set_paper_style("blog")
    _t0 = time.time()
    import issue1900_race as R

    if args.smoke:
        race_dir = args.smoke_root / "race_out"
        fig_dir = args.smoke_root / "figs"  # SCRATCH — never canonical figures/
        arms = json.loads((args.smoke_root / "arms_smoke.json").read_text())["arms"]
        tables_dir = args.smoke_root / "tables"
        judge_dir = args.smoke_root / "judge"
        marker_dir = args.smoke_root / "marker_tf"
    else:
        race_dir = args.race_dir
        fig_dir = args.fig_dir
        arms = R.J.load_arms(args.config_dir)
        tables_dir = args.p1_root / "predictor_tables"
        judge_dir = args.judge_dir
        marker_dir = args.p1_root / "marker_tf"

    assemblies = None
    if not args.no_scatters:
        assemblies = []
        for arm in arms:
            if arm["kind"] != "content":
                continue
            try:
                assemblies.append(R.assemble_content_arm(arm, tables_dir, judge_dir))
            except (AssertionError, FileNotFoundError) as e:
                logger.info("[figs] scatter assembly %s skipped: %s", arm["arm_id"], e)
        del marker_dir
    made = render_all(race_dir, fig_dir, assemblies)
    for p in made:
        print(f"[figs] wrote {p}", flush=True)
    print(f"[phase=done] figures={len(made)} elapsed={time.time() - _t0:.1f}s", flush=True)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
