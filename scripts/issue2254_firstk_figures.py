"""Issue #2254 follow-up `first-k-answer-token-steering` — figures (plan v10 §6).

Renders the two hero figures + the exploratory dump from the reduce's
committed JSONs under ``<out-root>/first-k-answer-token-steering/`` (plus the
judged per-question arrays for the point clouds). ``render_all`` is the single
entrypoint ``issue2254_first_k_steering.py --phases figures`` calls. Pure
json+numpy+matplotlib: importable without torch/HF (the parent
``issue2254_figures`` convention).

Conventions: paper-plots rcParams via ``set_paper_style()``; one color per
direction across every figure (reused ``DIR_COLORS``); axes + ticks + legend +
panel titles only (no in-canvas caption blocks — standing user directive
2026-08-12; the #2333 67% reference line goes in the CAPTION, never on the
hero-2 canvas, plan §6); matplotlib yerr = NON-NEGATIVE offsets via the
parent's clamped ``_err``; PNG dpi=200 + ``.meta.json`` provenance sidecars
via the parent's ``_save``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports: shared-VM thread caps bind in-process (#847)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _ensure_repo_root_on_syspath() -> None:
    """Repo root on sys.path so `import scripts.<mod>` resolves (#823)."""
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_repo_root_on_syspath()

from scripts.issue2254_figures import (  # noqa: E402
    BREADTH_LABELS,
    DIR_COLORS,
    DIR_LABELS,
    _err,
    _load,
    _save,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PERCELL = _REPO_ROOT / "eval_results/issue_2254/baseline_ceiling/judged_percell.json"

# The 8 position arms in §4.1 accrual order; reader-facing tick labels
# (no internal slugs on rendered axes — no-opaque-condition-codes rule).
POS_ORDER = ("lastctx", "tok1", "tok2", "tok3", "span13", "span15", "combined", "allans")
POS_LABELS = {
    "lastctx": "last ctx token",
    "tok1": "answer tok 1",
    "tok2": "answer tok 2",
    "tok3": "answer tok 3",
    "span13": "answer toks 1–3",
    "span15": "answer toks 1–5",
    "combined": "ctx + toks 1–3",
    "allans": "all answer toks",
}
H3_CHAIN = ("lastctx", "tok1", "span13", "span15", "allans")
STRONG_DIRECTIONS = ("rb", "pre")  # plan §3 lattice scope
HERO_DIRECTIONS = ("rb", "pre", "ctxext")  # bar directions on hero 1
BREADTH_ORDER = ("single", "mid")

REQUIRED_FIGURES = (
    "hero1_position_bars",
    "hero2_recovery_fraction",
    "expl_accrual_curves",
    "expl_h3_adjacent_forest",
    "expl_h4_pre_vs_shuffled",
    "expl_rd_lattice",
    "expl_perq_clouds",
)


def _style() -> None:
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()


def _cell_index(percell_b: dict) -> dict[tuple[str, str, str], dict]:
    """{(direction, breadth, position): percell row} for one behavior."""
    out: dict[tuple[str, str, str], dict] = {}
    for _cid, row in percell_b.items():
        c = row["cell"]
        out[(c["direction"], c["breadth"], c["position"])] = row
    return out


def _row_valid(row: dict | None) -> bool:
    """True when the per-cell validity gate (coherence + rule-29 completeness,
    written by the reduce phase) passed. Rows lacking the block (legacy /
    partial artifacts) are treated as valid so old smoke fixtures still plot;
    the reduce phase always writes it for this round's outputs."""
    if row is None:
        return False
    v = row.get("validity")
    return True if v is None else bool(v.get("valid", True))


def _panels(percell: dict) -> tuple[list[str], list[str]]:
    """(behaviors, breadths) actually present in the percell rows."""
    behaviors = sorted(percell["behaviors"])
    present = {
        row["cell"]["breadth"] for b in behaviors for row in percell["behaviors"][b].values()
    }
    breadths = [br for br in BREADTH_ORDER if br in present]
    assert behaviors and breadths, (behaviors, breadths)
    return behaviors, breadths


# ---------------------------------------------------------------------------
# hero 1 — Δscore across the 8 positions + cap-hit/CJK strip (plan §6)
# ---------------------------------------------------------------------------


def fig_hero1_position_bars(rroot: Path, fig_dir: Path):
    percell = _load(rroot, "steer/delta_score_percell.json")
    if percell is None:
        return "skip:steer/delta_score_percell.json missing"
    foc = _load(rroot, "reads/fraction_of_ceiling.json")
    behaviors, breadths = _panels(percell)
    nrows, ncols = len(behaviors), len(breadths)
    fig = plt.figure(figsize=(7.2 * ncols, 4.8 * nrows))
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3.0, 1.1] * nrows, hspace=0.42)
    x = np.arange(len(POS_ORDER), dtype=np.float64)
    width = 0.26
    for r, beh in enumerate(behaviors):
        idx = _cell_index(percell["behaviors"][beh])
        ceil = None
        if foc is not None and foc["behaviors"].get(beh):
            ceil = next(iter(foc["behaviors"][beh].values()))["denominator_point"]
        for cix, br in enumerate(breadths):
            ax = fig.add_subplot(gs[2 * r, cix])
            axs = fig.add_subplot(gs[2 * r + 1, cix], sharex=ax)
            for k, d in enumerate(HERO_DIRECTIONS):
                xs, vals, los, his, caps, cjks = [], [], [], [], [], []
                for i, pos in enumerate(POS_ORDER):
                    row = idx.get((d, br, pos))
                    if not _row_valid(row):  # invalid cells excluded (plan rule 29 gate)
                        continue
                    xs.append(x[i] + (k - 1) * width)
                    vals.append(row["delta_score"])
                    los.append(row["ci"][0])
                    his.append(row["ci"][1])
                    caps.append(row["horizons"]["caphit_common"])
                    cjks.append(row["horizons"]["cjk_common"])
                if not vals:
                    continue
                ax.bar(
                    xs,
                    vals,
                    width=width,
                    color=DIR_COLORS[d],
                    yerr=_err(vals, los, his),
                    capsize=2,
                    error_kw={"lw": 0.8},
                )
                axs.bar(xs, caps, width=width, color=DIR_COLORS[d])
                axs.bar(
                    xs,
                    cjks,
                    width=width,
                    bottom=caps,
                    color=DIR_COLORS[d],
                    hatch="///",
                    edgecolor="white",
                    linewidth=0,
                )
            # shuffled-map twin: per-position null band at the pre-image's point
            sxs, sv, slo, shi = [], [], [], []
            for i, pos in enumerate(POS_ORDER):
                row = idx.get(("preshuf", br, pos))
                if not _row_valid(row):
                    continue
                sxs.append(x[i])
                sv.append(row["delta_score"])
                slo.append(row["ci"][0])
                shi.append(row["ci"][1])
            if sv:
                ax.fill_between(sxs, slo, shi, color=DIR_COLORS["preshuf"], alpha=0.25, lw=0)
                ax.plot(sxs, sv, color=DIR_COLORS["preshuf"], lw=1.0)
            # random control: its OWN operating point (diagnostic floor, §3 H4)
            rxs, rv, rlo, rhi = [], [], [], []
            for i, pos in enumerate(POS_ORDER):
                row = idx.get(("random", br, pos))
                if not _row_valid(row):
                    continue
                rxs.append(x[i])
                rv.append(row["delta_score"])
                rlo.append(row["ci"][0])
                rhi.append(row["ci"][1])
            if rv:
                ax.errorbar(
                    rxs,
                    rv,
                    yerr=_err(rv, rlo, rhi),
                    color=DIR_COLORS["random"],
                    fmt="o--",
                    ms=3,
                    lw=0.9,
                    capsize=2,
                )
            if ceil is not None:
                ax.axhline(ceil, color="black", ls="--", lw=1.0)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(f"{beh} — {BREADTH_LABELS[br]}")
            ax.set_ylabel("Δ graded score vs α=0")
            plt.setp(ax.get_xticklabels(), visible=False)
            # Stack of cap-hit + CJK fractions ranges over [0, 2] (both can be 1).
            axs.set_ylim(0, 2.05)
            axs.axhline(1.0, color="0.75", lw=0.5)
            axs.set_ylabel("degraded frac (cap+CJK, 0-2)")
            axs.set_xticks(x)
            axs.set_xticklabels([POS_LABELS[p] for p in POS_ORDER], rotation=35, ha="right")
    handles = [Patch(color=DIR_COLORS[d], label=DIR_LABELS[d]) for d in HERO_DIRECTIONS]
    handles += [
        Line2D([], [], color=DIR_COLORS["preshuf"], lw=4, alpha=0.4, label=DIR_LABELS["preshuf"]),
        Line2D(
            [],
            [],
            color=DIR_COLORS["random"],
            ls="--",
            marker="o",
            ms=3,
            label=DIR_LABELS["random"],
        ),
        Line2D([], [], color="black", ls="--", label="donor-swap ceiling"),
        Patch(facecolor="0.55", label="cap-hit fraction (strip)"),
        Patch(facecolor="0.85", hatch="///", label="CJK-intrusion fraction (strip)"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    return _save(
        fig,
        fig_dir,
        "hero1_position_bars",
        ["steer/delta_score_percell.json", "reads/fraction_of_ceiling.json"],
    )


# ---------------------------------------------------------------------------
# hero 2 — recovery fraction R per lattice cell (plan §6; 67% line caption-only)
# ---------------------------------------------------------------------------


def _lattice_blocks(lat: dict) -> list[dict]:
    return [
        v
        for _k, v in sorted(lat["lattice"].items())
        # startswith covers "not-computable" AND "not-computable pending
        # remediation" (invalid core arm, plan validity gates).
        if isinstance(v, dict)
        and not str(v.get("verdict", "")).startswith("not-computable")
        and "R" in v
    ]


def fig_hero2_recovery_fraction(rroot: Path, fig_dir: Path):
    lat = _load(rroot, "reads/verdict_lattice.json")
    if lat is None:
        return "skip:reads/verdict_lattice.json missing"
    blocks = _lattice_blocks(lat)
    if not blocks:
        return "skip:no computable lattice blocks"
    groups = sorted(
        {(b["behavior"], b["breadth"]) for b in blocks},
        key=lambda g: (g[0], BREADTH_ORDER.index(g[1])),
    )
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.32
    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 2.4, 3.8))
    for k, d in enumerate(STRONG_DIRECTIONS):
        xs, vals, los, his = [], [], [], []
        for gi, (beh, br) in enumerate(groups):
            blk = next(
                (
                    b
                    for b in blocks
                    if (b["behavior"], b["breadth"], b["direction"]) == (beh, br, d)
                ),
                None,
            )
            if blk is None or blk["R"]["point"] is None:
                continue
            xs.append(x[gi] + (k - 0.5) * width)
            vals.append(blk["R"]["point"])
            los.append(blk["R"].get("lo") if blk["R"].get("lo") is not None else blk["R"]["point"])
            his.append(blk["R"].get("hi") if blk["R"].get("hi") is not None else blk["R"]["point"])
        if not vals:
            continue
        ax.bar(
            xs,
            vals,
            width=width,
            color=DIR_COLORS[d],
            label=DIR_LABELS[d],
            yerr=_err(vals, los, his),
            capsize=3,
            error_kw={"lw": 0.9},
        )
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{beh}\n{BREADTH_LABELS[br]}" for beh, br in groups])
    ax.set_ylabel("recovery R (toks 1–3 / all answer)")
    ax.legend()
    return _save(fig, fig_dir, "hero2_recovery_fraction", ["reads/verdict_lattice.json"])


# ---------------------------------------------------------------------------
# exploratory dump (plan §6 + unit-3 brief)
# ---------------------------------------------------------------------------


def fig_expl_accrual_curves(rroot: Path, fig_dir: Path):
    percell = _load(rroot, "steer/delta_score_percell.json")
    if percell is None:
        return "skip:steer/delta_score_percell.json missing"
    behaviors, breadths = _panels(percell)
    fig, axes = plt.subplots(
        len(behaviors),
        len(breadths),
        figsize=(5.4 * len(breadths), 3.6 * len(behaviors)),
        squeeze=False,
    )
    x = np.arange(len(H3_CHAIN), dtype=np.float64)
    for r, beh in enumerate(behaviors):
        idx = _cell_index(percell["behaviors"][beh])
        for cix, br in enumerate(breadths):
            ax = axes[r][cix]
            for d in ("rb", "pre", "ctxext", "random", "preshuf"):
                xs, vals, los, his = [], [], [], []
                for i, pos in enumerate(H3_CHAIN):
                    row = idx.get((d, br, pos))
                    if not _row_valid(row):  # invalid cells excluded (plan rule 29 gate)
                        continue
                    xs.append(x[i])
                    vals.append(row["delta_score"])
                    los.append(row["ci"][0])
                    his.append(row["ci"][1])
                if not vals:
                    continue
                ax.errorbar(
                    xs,
                    vals,
                    yerr=_err(vals, los, his),
                    color=DIR_COLORS[d],
                    marker="o",
                    ms=3,
                    lw=1.2,
                    capsize=2,
                    label=DIR_LABELS[d],
                )
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(f"{beh} — {BREADTH_LABELS[br]}")
            ax.set_ylabel("Δ graded score vs α=0")
            ax.set_xticks(x)
            ax.set_xticklabels([POS_LABELS[p] for p in H3_CHAIN], rotation=25, ha="right")
            if r == 0 and cix == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, fig_dir, "expl_accrual_curves", ["steer/delta_score_percell.json"])


def _pair_label(pair: str) -> str:
    late, early = pair.split("-minus-")
    return f"{POS_LABELS[late]} − {POS_LABELS[early]}"


def fig_expl_h3_adjacent_forest(rroot: Path, fig_dir: Path):
    lat = _load(rroot, "reads/verdict_lattice.json")
    if lat is None:
        return "skip:reads/verdict_lattice.json missing"
    blocks = _lattice_blocks(lat)
    if not blocks:
        return "skip:no computable lattice blocks"
    rows = []  # (label, point, lo, hi, color, inverted)
    for blk in blocks:
        head = f"{blk['behavior']} {BREADTH_LABELS[blk['breadth']]} {DIR_LABELS[blk['direction']]}"
        for con in blk["h3_adjacent_contrasts"]:
            rows.append(
                (
                    f"{head}: {_pair_label(con['pair'])}",
                    con["point"],
                    con["ci"][0],
                    con["ci"][1],
                    DIR_COLORS[blk["direction"]],
                    bool(con["inverted"]),
                )
            )
    if not rows:
        return "skip:no adjacent contrasts in lattice"
    y = np.arange(len(rows), dtype=np.float64)[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 0.28 * len(rows) + 1.6))
    for yi, (_label, pt, lo, hi, color, inverted) in zip(y, rows, strict=True):
        ax.errorbar(
            [pt],
            [yi],
            xerr=_err([pt], [lo], [hi]),
            color=color,
            marker="v" if inverted else "o",
            ms=5 if inverted else 4,
            lw=1.1,
            capsize=2,
        )
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlabel("adjacent paired difference (later − earlier), Δ graded score")
    ax.legend(
        handles=[
            Line2D([], [], color="0.3", marker="o", ls="", label="adjacent contrast"),
            Line2D([], [], color="0.3", marker="v", ls="", label="inverted (CI < 0)"),
        ],
        fontsize=7,
    )
    return _save(fig, fig_dir, "expl_h3_adjacent_forest", ["reads/verdict_lattice.json"])


def fig_expl_h4_pre_vs_shuffled(rroot: Path, fig_dir: Path):
    lat = _load(rroot, "reads/verdict_lattice.json")
    if lat is None:
        return "skip:reads/verdict_lattice.json missing"
    h4 = lat.get("h4") or {}
    if not h4:
        return "skip:no h4 rows (pre/preshuf arms absent)"
    keys = sorted(h4)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 3.6), squeeze=False)
    for cix, key in enumerate(keys):
        ax = axes[0][cix]
        beh, br = key.split("__")
        per_pos = h4[key]["per_position"]
        pos_present = [p for p in POS_ORDER if p in per_pos]
        x = np.arange(len(pos_present), dtype=np.float64)
        vals = [per_pos[p]["pre_minus_preshuf"]["point"] for p in pos_present]
        los = [per_pos[p]["pre_minus_preshuf"]["ci"][0] for p in pos_present]
        his = [per_pos[p]["pre_minus_preshuf"]["ci"][1] for p in pos_present]
        ax.errorbar(
            x,
            vals,
            yerr=_err(vals, los, his),
            color=DIR_COLORS["pre"],
            marker="o",
            ms=4,
            lw=1.2,
            capsize=2,
            label="pre-image − shuffled-map twin",
        )
        diag = [
            (i, per_pos[p]["pre_minus_random_diagnostic"])
            for i, p in enumerate(pos_present)
            if "pre_minus_random_diagnostic" in per_pos[p]
        ]
        if diag:
            dx = [i for i, _ in diag]
            dv = [d["point"] for _, d in diag]
            dlo = [d["ci"][0] for _, d in diag]
            dhi = [d["ci"][1] for _, d in diag]
            ax.errorbar(
                dx,
                dv,
                yerr=_err(dv, dlo, dhi),
                color=DIR_COLORS["random"],
                marker="s",
                ms=3,
                lw=0.9,
                ls="--",
                capsize=2,
                label="pre-image − random (diagnostic)",
            )
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(f"{beh} — {BREADTH_LABELS[br]}")
        ax.set_ylabel("paired Δ graded score difference")
        ax.set_xticks(x)
        ax.set_xticklabels([POS_LABELS[p] for p in pos_present], rotation=25, ha="right")
        if cix == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, fig_dir, "expl_h4_pre_vs_shuffled", ["reads/verdict_lattice.json"])


def fig_expl_rd_lattice(rroot: Path, fig_dir: Path):
    lat = _load(rroot, "reads/verdict_lattice.json")
    if lat is None:
        return "skip:reads/verdict_lattice.json missing"
    blocks = _lattice_blocks(lat)
    pts = [b for b in blocks if b["R"]["point"] is not None]
    if not pts:
        return "skip:no lattice blocks with a defined R point"
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    seen: set[str] = set()
    for blk in pts:
        d = blk["direction"]
        ax.scatter(
            [blk["D"]["point"]],
            [blk["R"]["point"]],
            color=DIR_COLORS[d],
            s=28,
            label=DIR_LABELS[d] if d not in seen else None,
        )
        seen.add(d)
        ax.annotate(
            f"{blk['behavior']} {BREADTH_LABELS[blk['breadth']]}",
            (blk["D"]["point"], blk["R"]["point"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6.5,
        )
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("D = deg(all answer) − 2·deg(answer toks 1–3)")
    ax.set_ylabel("recovery R point estimate")
    ax.legend(fontsize=7)
    return _save(fig, fig_dir, "expl_rd_lattice", ["reads/verdict_lattice.json"])


def fig_expl_perq_clouds(rroot: Path, fig_dir: Path):
    percell = _load(rroot, "steer/delta_score_percell.json")
    if percell is None:
        return "skip:steer/delta_score_percell.json missing"
    judged_dir = rroot / "judge" / "judged"
    if not judged_dir.is_dir() or not any(judged_dir.glob("*.json")):
        return "skip:judge/judged/*.json absent (per-question arrays unavailable)"
    assert BASELINE_PERCELL.is_file(), f"{BASELINE_PERCELL} missing (committed parent input)"
    base = json.loads(BASELINE_PERCELL.read_text())
    behaviors, breadths = _panels(percell)
    rng = np.random.default_rng(2254)
    width = 0.36
    fig, axes = plt.subplots(
        len(behaviors),
        len(breadths),
        figsize=(7.0 * len(breadths), 3.8 * len(behaviors)),
        squeeze=False,
    )
    x = np.arange(len(POS_ORDER), dtype=np.float64)
    plotted = 0
    for r, beh in enumerate(behaviors):
        a0_full = np.array(
            [
                np.nan if v is None else float(v)
                for v in base["behaviors"][beh]["alpha0"]["per_question_mean_score"]
            ],
            dtype=np.float64,
        )
        rows_by_cid = {cid: row for cid, row in percell["behaviors"][beh].items()}
        for cix, br in enumerate(breadths):
            ax = axes[r][cix]
            for k, d in enumerate(STRONG_DIRECTIONS):
                off = (k - 0.5) * width
                for i, pos in enumerate(POS_ORDER):
                    hit = next(
                        (
                            (cid, row)
                            for cid, row in rows_by_cid.items()
                            if (
                                row["cell"]["direction"],
                                row["cell"]["breadth"],
                                row["cell"]["position"],
                            )
                            == (d, br, pos)
                        ),
                        None,
                    )
                    if hit is None or not _row_valid(hit[1]):
                        continue  # missing or validity-gated cell (plan rule 29)
                    cid, row = hit
                    jf = judged_dir / f"{cid}.json"
                    if not jf.is_file():
                        continue
                    j = json.loads(jf.read_text())
                    q = np.array(
                        [np.nan if v is None else float(v) for v in j["per_question_mean_score"]],
                        dtype=np.float64,
                    )
                    dq = q - a0_full[: len(q)]
                    dq = dq[np.isfinite(dq)]
                    ax.bar(
                        [x[i] + off],
                        [row["delta_score"]],
                        width=width,
                        color=DIR_COLORS[d],
                        alpha=0.45,
                        label=DIR_LABELS[d] if (i == 0 and r == 0 and cix == 0) else None,
                    )
                    jit = rng.uniform(-width / 3.2, width / 3.2, size=len(dq))
                    ax.scatter(
                        np.full(len(dq), x[i] + off) + jit,
                        dq,
                        s=6,
                        color=DIR_COLORS[d],
                        alpha=0.55,
                        linewidths=0,
                    )
                    plotted += 1
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(f"{beh} — {BREADTH_LABELS[br]}")
            ax.set_ylabel("per-question Δ graded score vs α=0")
            ax.set_xticks(x)
            ax.set_xticklabels([POS_LABELS[p] for p in POS_ORDER], rotation=35, ha="right")
            if r == 0 and cix == 0:
                ax.legend(fontsize=7)
    if not plotted:
        plt.close(fig)
        return "skip:no strong-direction judged cells found"
    fig.tight_layout()
    return _save(
        fig,
        fig_dir,
        "expl_perq_clouds",
        ["steer/delta_score_percell.json", "judge/judged/*.json", str(BASELINE_PERCELL)],
    )


_BUILDERS = (
    fig_hero1_position_bars,
    fig_hero2_recovery_fraction,
    fig_expl_accrual_curves,
    fig_expl_h3_adjacent_forest,
    fig_expl_h4_pre_vs_shuffled,
    fig_expl_rd_lattice,
    fig_expl_perq_clouds,
)


def render_all(rroot: Path, fig_dir: Path, *, require: tuple[str, ...] = ()) -> dict:
    """Render every figure whose inputs exist; return {'rendered', 'skipped'}.

    Missing INPUTS skip with a named reason; real errors propagate (fail
    fast). Any figure named in `require` that skips raises RuntimeError.
    """
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
