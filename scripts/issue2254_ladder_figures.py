"""Figures for issue #2254 follow-up `transpose_ladder` (plan v14 §6).

Hero: per behavior, the pinv→transpose ladder — grouped Δscore bars for
{parent pinv (reference), rl1, rl2, rl3, tr} at each arm's best cell, the
parent band edge dashed, the directly-measured context direction's decisive
value as a reference line, floor/ceiling markers, frozen (black) +
selection-aware (gray) whiskers — the parent hero1 grammar.

Exploratory dump (over-produced, plan §6): all-44 per-cell bars; Δ vs λ
curves per (behavior, layer_config); per-question dot plots for clearing
cells; cap-hit/CJK degradation panel; cos-structure heatmap; alignment-
concentration spectra.

Conventions: paper style (`set_paper_style`); reader-facing labels only (the
no-opaque-condition-codes rule); NO caption blocks inside the plot (standing
figure-conciseness directive) — provenance + the `fresh_nulls: false` scope
note live in the .meta.json sidecar each `_save` writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_repo_root_on_syspath()

from scripts.issue2254_figures import DIR_LABELS, _err, _save  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_ROOT = _REPO_ROOT / "eval_results" / "issue_2254"

LADDER_ORDER = ("rl1", "rl2", "rl3", "tr")
LADDER_COLORS = {
    "tr": "#8c564b",
    "rl1": "#17becf",
    "rl2": "#0e7c8c",
    "rl3": "#065361",
}
LADDER_LABELS = {
    "tr": "transpose pullback",
    "rl1": "near-pseudo-inverse pullback",
    "rl2": "median-λ pullback",
    "rl3": "near-transpose pullback",
}
# Reader-facing layer-config names (Lens 3: no config codes in figures).
LAYER_LABELS = {
    "L14": "layer 14",
    "L17": "layer 17",
    "mid": "mid layers",
    "all": "all layers",
}
_LAYER_ORDER = {"L14": 0, "L17": 1, "mid": 2, "all": 3}
FRESH_NULLS_SCOPE_NOTE = (
    "fresh_nulls: false — clears are read against the parent's REUSED context null band "
    "(measured for other directions at matched injected norm; plan v14 §5 inference-scope "
    "caveat)"
)

REQUIRED_FIGURES = ("hero_ladder",)


def _style() -> None:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()


def _load(rroot: Path, rel: str):
    """Load a reduce/report JSON; None when absent (⇒ figure skips loudly)."""
    path = rroot / rel
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _save_meta(fig, fig_dir: Path, name: str, inputs: list[str]) -> str:
    """`issue2254_figures._save` + the fresh-nulls scope note in the sidecar."""
    out = _save(fig, fig_dir, name, inputs)
    meta_path = fig_dir / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["scope_note"] = FRESH_NULLS_SCOPE_NOTE
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out


def _save_meta_paper(fig, fig_dir: Path, name: str, inputs: list[str]) -> str:
    """`savefig_paper` save (PNG + PDF + sidecar with embedded points/text) plus
    the `inputs` + fresh-nulls scope-note keys `_save_meta` writes.

    Used where the sidecar must carry the rendered tick/legend text for the
    mechanical opaque-code scan (verify_task_body checks 24/28/34)."""
    from explore_persona_space.analysis.paper_plots import savefig_paper

    paths = savefig_paper(fig, name, dir=fig_dir)
    plt.close(fig)
    meta_path = paths["meta"]
    meta = json.loads(meta_path.read_text())
    meta["figure"] = name
    meta["inputs"] = inputs
    meta["scope_note"] = FRESH_NULLS_SCOPE_NOTE
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return name


def _behaviors(percell: dict) -> list[str]:
    return sorted(percell["behaviors"])


def _best_row(cells_b: dict, slug: str) -> tuple[str, dict] | None:
    """Best coherence-passing DEFINED cell for one ladder arm by margin."""
    best: tuple[str, dict] | None = None
    for cid, row in cells_b.items():
        if row["cell"]["direction"] != slug or row.get("margin") is None:
            continue
        if not row.get("coherence_pass", False):
            continue
        if best is None or row["margin"] > best[1]["margin"]:
            best = (cid, row)
    return best


def _parent_refs(behavior: str) -> dict:
    """Parent decisive reference values (committed artifacts): pre-image Δ,
    measured-context-direction Δ, band p97.5, ceiling Δ."""
    verd = json.loads((INPUTS_ROOT / "decisive" / "verdicts.json").read_text())["behaviors"][
        behavior
    ]
    base = json.loads((INPUTS_ROOT / "baseline_ceiling" / "judged_percell.json").read_text())[
        "behaviors"
    ][behavior]
    band = float(verd["null_band_context"]["p975"])
    margins = verd.get("margins", {})
    out = {"band": band, "ceiling_delta": float(base["ceiling_delta"])}
    if "E_pre" in margins:
        out["pre_delta"] = float(margins["E_pre"]["value"]) + band
    if "E_ctxdir" in margins:
        out["ctxdir_delta"] = float(margins["E_ctxdir"]["value"]) + band
    return out


# ---------------------------------------------------------------------------
# hero — the pinv→transpose ladder (plan §6)
# ---------------------------------------------------------------------------


def fig_hero_ladder(rroot: Path, fig_dir: Path):
    percell = _load(rroot, "reduce/delta_score_percell.json")
    verdicts = _load(rroot, "reduce/verdicts.json")
    if percell is None or verdicts is None:
        return "skip:reduce outputs absent (run --phases reduce first)"
    behaviors = _behaviors(percell)
    fig, axes = plt.subplots(1, len(behaviors), figsize=(6.0 * len(behaviors), 4.2), squeeze=False)
    for ax, b in zip(axes[0], behaviors, strict=True):
        cells_b = percell["behaviors"][b]
        refs = _parent_refs(b)
        band = refs["band"]
        xs, vals, los, his, colors, labels = [], [], [], [], [], []
        sel_lo, sel_hi = [], []
        x = 0
        if "pre_delta" in refs:
            xs.append(x)
            vals.append(refs["pre_delta"])
            los.append(refs["pre_delta"])
            his.append(refs["pre_delta"])
            sel_lo.append(np.nan)
            sel_hi.append(np.nan)
            colors.append("#bbbbbb")
            labels.append(f"{DIR_LABELS['pre']}\n(parent, ref)")
            x += 1
        for slug in LADDER_ORDER:
            best = _best_row(cells_b, slug)
            if best is None:
                continue
            cid, row = best
            xs.append(x)
            vals.append(row["delta_score"])
            los.append(row["ci_frozen"][0])
            his.append(row["ci_frozen"][1])
            sa = verdicts["selection_aware"]["behavior_arm"].get(f"{b}__{slug}")
            if sa is None:
                sel_lo.append(np.nan)
                sel_hi.append(np.nan)
            else:  # selection-aware CI in margin space -> shift back to Δ space
                sel_lo.append(sa["ci"][0] + band)
                sel_hi.append(sa["ci"][1] + band)
            colors.append(LADDER_COLORS[slug])
            labels.append(LADDER_LABELS[slug])
            x += 1
        if not xs:
            continue
        ax.bar(xs, vals, color=colors, width=0.62, zorder=2)
        ax.errorbar(
            xs, vals, yerr=_err(vals, los, his), fmt="none", ecolor="black", lw=1.2, zorder=3
        )
        sel_ok = ~np.isnan(np.asarray(sel_lo))
        if sel_ok.any():
            xs_a = np.asarray(xs, dtype=float)[sel_ok] + 0.18
            mid = (np.asarray(sel_lo)[sel_ok] + np.asarray(sel_hi)[sel_ok]) / 2.0
            ax.errorbar(
                xs_a,
                mid,
                yerr=_err(mid, np.asarray(sel_lo)[sel_ok], np.asarray(sel_hi)[sel_ok]),
                fmt="none",
                ecolor="#888888",
                lw=1.0,
                zorder=3,
            )
        ax.axhline(band, color="#d62728", ls="--", lw=1.2, zorder=1)
        if "ctxdir_delta" in refs:
            ax.axhline(refs["ctxdir_delta"], color="#2ca02c", ls="-.", lw=1.2, zorder=1)
        ax.axhline(0.0, color="black", lw=0.8, zorder=1)
        ax.plot(
            [len(xs) - 0.5],
            [refs["ceiling_delta"]],
            marker="^",
            color="#9467bd",
            ms=7,
            ls="none",
            zorder=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_title(b)
        ax.set_ylabel("Δ graded score vs unsteered floor")
    handles = [
        Line2D([0], [0], color="#d62728", ls="--", label="parent null band p97.5"),
        Line2D([0], [0], color="#2ca02c", ls="-.", label=f"{DIR_LABELS['ctxext']} (parent)"),
        Line2D([0], [0], color="black", lw=1.2, label="frozen 95% CI"),
        Line2D([0], [0], color="#888888", lw=1.0, label="selection-aware 95% CI"),
        Line2D(
            [0], [0], marker="^", color="#9467bd", ls="none", label="donor-swap ceiling (parent)"
        ),
    ]
    axes[0][-1].legend(handles=handles, fontsize=6, loc="upper left")
    fig.tight_layout()
    return _save_meta(
        fig,
        fig_dir,
        "hero_ladder",
        ["reduce/delta_score_percell.json", "reduce/verdicts.json", "parent decisive artifacts"],
    )


# ---------------------------------------------------------------------------
# exploratory dump (plan §6, over-produced)
# ---------------------------------------------------------------------------


def fig_expl_all_cells(rroot: Path, fig_dir: Path):
    from matplotlib.patches import Patch

    percell = _load(rroot, "reduce/delta_score_percell.json")
    if percell is None:
        return "skip:reduce outputs absent"
    behaviors = _behaviors(percell)
    fig, axes = plt.subplots(len(behaviors), 1, figsize=(11.0, 3.4 * len(behaviors)), squeeze=False)
    for ax, b in zip(axes[:, 0], behaviors, strict=True):
        cells_b = percell["behaviors"][b]
        # Group by pullback construction (ladder order), then layer config, then dose.
        rows = sorted(
            cells_b.items(),
            key=lambda kv: (
                LADDER_ORDER.index(kv[1]["cell"]["direction"]),
                _LAYER_ORDER.get(kv[1]["cell"]["layer_config"], 99),
                kv[1]["cell"]["c"],
            ),
        )
        xs = np.arange(len(rows))
        vals = [r.get("delta_score") if r.get("delta_score") is not None else 0.0 for _, r in rows]
        undef = [r.get("delta_score") is None for _, r in rows]
        colors = [
            "#dddddd" if u else LADDER_COLORS[r["cell"]["direction"]]
            for u, (_, r) in zip(undef, rows, strict=True)
        ]
        ax.bar(xs, vals, color=colors, width=0.7)
        for (_, r), xv in zip(rows, xs, strict=True):
            if r.get("ci_frozen"):
                ax.errorbar(
                    [xv],
                    [r["delta_score"]],
                    yerr=_err([r["delta_score"]], [r["ci_frozen"][0]], [r["ci_frozen"][1]]),
                    fmt="none",
                    ecolor="black",
                    lw=0.8,
                )
        dirs_seq = [r["cell"]["direction"] for _, r in rows]
        for i in range(1, len(rows)):
            if dirs_seq[i] != dirs_seq[i - 1]:
                ax.axvline(i - 0.5, color="#bbbbbb", lw=0.6)
        band = rows[0][1]["band_p975"] if rows else 0.0
        ax.axhline(band, color="#d62728", ls="--", lw=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                f"{LAYER_LABELS[r['cell']['layer_config']]} · dose {r['cell']['c']:g}"
                + (" (undefined)" if u else "")
                for u, (_, r) in zip(undef, rows, strict=True)
            ],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        ax.set_title(b)
        ax.set_ylabel("Δ graded score")
    handles = [Patch(facecolor=LADDER_COLORS[s], label=LADDER_LABELS[s]) for s in LADDER_ORDER]
    handles.append(
        Line2D([0], [0], color="#d62728", ls="--", lw=1.0, label="reused parent band edge")
    )
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return _save_meta_paper(fig, fig_dir, "expl_all_cells", ["reduce/delta_score_percell.json"])


def fig_expl_delta_vs_lambda(rroot: Path, fig_dir: Path):
    import scripts.issue2254_preimage as i2254  # function-local by convention; hoisted from loop

    percell = _load(rroot, "reduce/delta_score_percell.json")
    report = _load(rroot, "ladder_report.json")
    if percell is None or report is None:
        return "skip:reduce or ladder_report outputs absent"
    behaviors = _behaviors(percell)
    fig, axes = plt.subplots(1, len(behaviors), figsize=(5.6 * len(behaviors), 4.0), squeeze=False)
    for ax, b in zip(axes[0], behaviors, strict=True):
        cells_b = percell["behaviors"][b]
        combos = sorted({(r["cell"]["layer_config"], r["cell"]["c"]) for r in cells_b.values()})
        for lc, c in combos:
            pts = []
            for slug in ("rl1", "rl2", "rl3"):
                row = next(
                    (
                        r
                        for r in cells_b.values()
                        if r["cell"]["direction"] == slug
                        and r["cell"]["layer_config"] == lc
                        and r["cell"]["c"] == c
                        and r.get("delta_score") is not None
                    ),
                    None,
                )
                if row is None:
                    continue
                # per-config λ: single layer -> that layer's λ; bands -> the
                # config's member-layer median (display placement only). Filter
                # to layers PRESENT in the report — a partial/smoke report
                # (subset of the 28 production layers) must skip gracefully
                # rather than KeyError on an absent band member.

                lys = [ly for ly in i2254.LAYER_CONFIGS[lc] if str(ly) in report["layers"]]
                if not lys:
                    continue
                lam = float(np.median([report["layers"][str(ly)]["lambdas"][slug] for ly in lys]))
                pts.append((lam, row["delta_score"]))
            tr_row = next(
                (
                    r
                    for r in cells_b.values()
                    if r["cell"]["direction"] == "tr"
                    and r["cell"]["layer_config"] == lc
                    and r["cell"]["c"] == c
                    and r.get("delta_score") is not None
                ),
                None,
            )
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            (line,) = ax.plot(xs, ys, marker="o", ms=4, lw=1.2, label=f"{lc}, c={c:g}")
            if tr_row is not None:
                ax.plot(
                    [max(xs) * 30.0],
                    [tr_row["delta_score"]],
                    marker="s",
                    ms=5,
                    color=line.get_color(),
                )
        ax.set_xscale("log")
        ax.set_xlabel("ridge λ (square marker = transpose limit)")
        ax.set_ylabel("Δ graded score")
        ax.set_title(b)
        ax.legend(fontsize=6)
    fig.tight_layout()
    return _save_meta(
        fig,
        fig_dir,
        "expl_delta_vs_lambda",
        ["reduce/delta_score_percell.json", "ladder_report.json"],
    )


def fig_expl_perq_clouds(rroot: Path, fig_dir: Path):
    percell = _load(rroot, "reduce/delta_score_percell.json")
    verdicts = _load(rroot, "reduce/verdicts.json")
    if percell is None or verdicts is None:
        return "skip:reduce outputs absent"
    clearing = verdicts.get("h1_clearing_cells", [])
    if not clearing:
        return "skip:no clearing cells (per-question clouds render for clears only)"
    import scripts.issue2254_transpose_ladder as ladder

    n = len(clearing)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.6 * n), squeeze=False)
    for ax, cid in zip(axes[:, 0], clearing, strict=True):
        judged = json.loads((rroot / "judge" / "judged" / f"{cid}.json").read_text())
        b = judged["cell"]["behavior"]
        floor_q, _fm, _cd = ladder.load_parent_floor(b)
        cell_q = [v for v in judged["per_question_mean_score"]]
        nq = len(cell_q)
        xs = np.arange(nq)
        ax.plot(xs, floor_q[:nq], "o", color="#999999", ms=4, label="unsteered floor")
        ax.plot(xs, cell_q, "o", color="#d62728", ms=4, label="steered cell")
        ax.set_title(cid, fontsize=8)
        ax.set_xlabel("question index")
        ax.set_ylabel("mean graded score")
        ax.legend(fontsize=6)
    fig.tight_layout()
    return _save_meta(
        fig, fig_dir, "expl_perq_clouds", ["reduce/verdicts.json", "judge/judged/*.json"]
    )


def fig_expl_degradation(rroot: Path, fig_dir: Path):
    percell = _load(rroot, "reduce/delta_score_percell.json")
    if percell is None:
        return "skip:reduce outputs absent"
    behaviors = _behaviors(percell)
    fig, axes = plt.subplots(len(behaviors), 1, figsize=(11.0, 2.8 * len(behaviors)), squeeze=False)
    for ax, b in zip(axes[:, 0], behaviors, strict=True):
        rows = sorted(percell["behaviors"][b].items())
        xs = np.arange(len(rows))
        cap = [float(r.get("cap_hit_fraction") or 0.0) for _, r in rows]
        cjk = [float(r["sensitivity"].get("cjk_common") or 0.0) for _, r in rows]
        ax.bar(xs - 0.18, cap, width=0.36, color="#1f77b4", label="cap-hit fraction")
        ax.bar(xs + 0.18, cjk, width=0.36, color="#ff7f0e", label="CJK-intrusion fraction")
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                f"{r['cell']['layer_config']} c{r['cell']['c']:g} {r['cell']['direction']}"
                for _, r in rows
            ],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        ax.set_title(b)
        ax.set_ylabel("fraction")
        ax.legend(fontsize=6)
    fig.tight_layout()
    return _save_meta(fig, fig_dir, "expl_degradation", ["reduce/delta_score_percell.json"])


def fig_expl_cos_heatmap(rroot: Path, fig_dir: Path):
    report = _load(rroot, "ladder_report.json")
    if report is None:
        return "skip:ladder_report.json absent"
    layers = sorted(int(ly) for ly in report["layers"])
    behaviors = sorted({k.split("__")[0] for ly in report["layers"].values() for k in ly["arms"]})
    metrics = ("cos_vs_parent_pre", "cos_vs_ctxext", "cos_vs_rb")
    fig, axes = plt.subplots(
        len(behaviors),
        len(LADDER_ORDER),
        figsize=(3.2 * len(LADDER_ORDER), 2.6 * len(behaviors)),
        squeeze=False,
    )
    for bi, b in enumerate(behaviors):
        for si, slug in enumerate(LADDER_ORDER):
            ax = axes[bi][si]
            mat = np.full((len(metrics), len(layers)), np.nan)
            for lj, ly in enumerate(layers):
                arm = report["layers"][str(ly)]["arms"].get(f"{b}__{slug}")
                if arm is None:
                    continue
                for mi, m in enumerate(metrics):
                    if m in arm:
                        mat[mi, lj] = arm[m]
            im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_yticks(range(len(metrics)))
            ax.set_yticklabels(
                ["vs parent pre-image", "vs measured ctx dir", "vs persona vector"], fontsize=6
            )
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, fontsize=5)
            ax.set_title(f"{b} — {LADDER_LABELS[slug]}", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="cosine")
    return _save_meta(fig, fig_dir, "expl_cos_heatmap", ["ladder_report.json"])


def fig_expl_alignment_spectra(rroot: Path, fig_dir: Path):
    report = _load(rroot, "ladder_report.json")
    if report is None:
        return "skip:ladder_report.json absent"
    layers = sorted(int(ly) for ly in report["layers"])
    behaviors = sorted({k.split("__")[0] for ly in report["layers"].values() for k in ly["arms"]})
    fig, axes = plt.subplots(1, len(behaviors), figsize=(5.4 * len(behaviors), 3.6), squeeze=False)
    for ax, b in zip(axes[0], behaviors, strict=True):
        for key, label, color in (
            ("k10", "top 10 modes", "#1f77b4"),
            ("k100", "top 100 modes", "#ff7f0e"),
            ("kstar", "top k* modes", "#2ca02c"),
        ):
            ys = [
                report["layers"][str(ly)]["arms"][f"{b}__tr"]["alignment_concentration"][key]
                for ly in layers
            ]
            ax.plot(layers, ys, marker="o", ms=3, lw=1.1, color=color, label=label)
        ax.set_xlabel("layer")
        ax.set_ylabel("‖c[:k]‖ / ‖c‖  (c = Umᵀ r_B)")
        ax.set_ylim(0, 1.02)
        ax.set_title(b)
        ax.legend(fontsize=6)
    fig.tight_layout()
    return _save_meta(fig, fig_dir, "expl_alignment_spectra", ["ladder_report.json"])


_BUILDERS = (
    fig_hero_ladder,
    fig_expl_all_cells,
    fig_expl_delta_vs_lambda,
    fig_expl_perq_clouds,
    fig_expl_degradation,
    fig_expl_cos_heatmap,
    fig_expl_alignment_spectra,
)


def render_all(rroot: Path, fig_dir: Path, *, require: tuple[str, ...] = ()) -> dict:
    """Render every figure whose inputs exist; return {'rendered', 'skipped'}.
    Missing INPUTS skip with a named reason; real errors propagate (fail
    fast). Any figure named in `require` that skips raises RuntimeError."""
    _style()
    rroot = Path(rroot)
    fig_dir = Path(fig_dir)
    rendered: list[str] = []
    skipped: dict[str, str] = {}
    for builder in _BUILDERS:
        name = builder.__name__.removeprefix("fig_")
        res = builder(rroot, fig_dir)
        if isinstance(res, str) and res.startswith("skip:"):
            skipped[name] = res.removeprefix("skip:")
        else:
            rendered.append(res)
    missing = [n for n in require if n not in rendered]
    if missing:
        raise RuntimeError(f"required figures not rendered: {missing} (skipped={skipped})")
    return {"rendered": rendered, "skipped": skipped}
