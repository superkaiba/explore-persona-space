"""Issue #2254 — figures from the reduce outputs (off-pod VM; CPU-only).

Renders every figure whose reduce inputs exist under the out-root and skips
(with a named reason) the ones whose inputs are absent — `render_all` is the
single entrypoint `issue2254_preimage.py --phase figures` calls. Pure
json+numpy+matplotlib: importable without torch/HF.

Conventions: one color per direction across every figure; axes + ticks +
legend + panel titles only (no in-canvas caption blocks — standing user
directive 2026-08-12); matplotlib yerr = NON-NEGATIVE offsets, so CI whiskers
are clamped `max(0, v - lo)` / `max(0, hi - v)` (gotchas.md xerr/yerr entry);
PNG dpi=200 + a `.meta.json` sidecar carrying git provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports: shared-VM thread caps bind in-process (#847)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR_COLORS = {
    "pre": "#1f77b4",
    "rb": "#d62728",
    "ctxext": "#2ca02c",
    "random": "#7f7f7f",
    "preshuf": "#bcbd22",
    # ctxext-subspace-split amendment arms (plan v7)
    "par": "#9467bd",
    "perp": "#ff7f0e",
}
# Reader-facing names (standing no-opaque-condition-codes rule): internal
# direction slugs never appear on rendered axes/legends.
DIR_LABELS = {
    "pre": "map pre-image",
    "rb": "persona vector",
    "ctxext": "measured context direction",
    "random": "random control",
    "preshuf": "shuffled-map pre-image",
    "par": "retained-subspace component",
    "perp": "map-invisible complement",
}
POS_LABELS = {"context": "context vector", "answer": "answer tokens"}
OP_LABELS = {"proj": "patch", "ablate": "ablation"}
BREADTH_LABELS = {"single": "one layer", "mid": "layer band"}
_SINGLE_CONFIGS = ("L14", "L17", "L20", "L26")
_CONFIG_ORDER = ("L14", "L17", "L20", "L26", "mid", "all")
HERO_ARMS = (("pre", "context"), ("ctxext", "context"), ("rb", "answer"))


def _load(out_root: Path, rel: str):
    """Load a reduce JSON; None when absent (=> figure skipped with reason)."""
    path = out_root / rel
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _err(vals, los, his):
    """Clamped asymmetric yerr offsets (non-negative by contract)."""
    v = np.asarray(vals, dtype=np.float64)
    lo = np.asarray(los, dtype=np.float64)
    hi = np.asarray(his, dtype=np.float64)
    return np.stack([np.maximum(0.0, v - lo), np.maximum(0.0, hi - v)])


def _save(fig, fig_dir: Path, name: str, inputs: list[str]) -> str:
    """Write PNG (dpi=200) + a .meta.json provenance sidecar; return name."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    meta = {"figure": name, "inputs": inputs, **as_metadata_dict(git_provenance())}
    (fig_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return name


def _best_cells(cells_b: dict) -> dict[tuple[str, str], tuple[str, dict]]:
    """Best coherence-passing cell per (direction, position) by delta_score."""
    best: dict[tuple[str, str], tuple[str, dict]] = {}
    for cid, rec in cells_b.items():
        cell = rec["cell"]
        key = (cell["direction"], cell["position"])
        if not rec.get("coherence_pass", True):
            continue
        if key not in best or rec["delta_score"] > best[key][1]["delta_score"]:
            best[key] = (cid, rec)
    return best


# ---------------------------------------------------------------------------
# figure builders (each returns the saved name, or a skip-reason string
# prefixed 'skip:'; input-missing only — real errors propagate)
# ---------------------------------------------------------------------------


def fig_hero1(out_root: Path, fig_dir: Path):
    percell = _load(out_root, "decisive/delta_score_percell.json")
    verdicts = _load(out_root, "decisive/verdicts.json")
    base = _load(out_root, "baseline_ceiling/judged_percell.json")
    if percell is None or verdicts is None:
        return "skip:decisive percell/verdicts not present"
    behaviors = sorted(percell["behaviors"])
    if not behaviors:
        return "skip:no behaviors in decisive percell"
    fig, ax = plt.subplots(figsize=(1.0 + 2.6 * len(behaviors), 4.2))
    width = 0.22
    for bi, b in enumerate(behaviors):
        best = _best_cells(percell["behaviors"][b])
        vb = verdicts["behaviors"].get(b, {})
        sel = vb.get("selection_inherited", {})
        band = vb.get("null_band_context")
        for ai, (d, p) in enumerate(HERO_ARMS):
            x = bi + (ai - 1) * width
            hit = best.get((d, p))
            if hit is None:
                continue
            _cid, rec = hit
            v = rec["delta_score"]
            lo, hi = rec["ci_frozen"]
            ax.bar(
                x,
                v,
                width=width * 0.9,
                color=DIR_COLORS[d],
                label=f"{DIR_LABELS[d]} @ {POS_LABELS[p]}" if bi == 0 else None,
            )
            ax.errorbar([x], [v], yerr=_err([v], [lo], [hi]), fmt="none", ecolor="black", capsize=3)
            si = sel.get(f"{d}__{p}")
            if si is not None:
                slo, shi = si["ci"]
                ax.errorbar(
                    [x + width * 0.28],
                    [v],
                    yerr=_err([v], [slo], [shi]),
                    fmt="none",
                    ecolor="0.45",
                    elinewidth=0.9,
                    capsize=2,
                )
        if band is not None:
            ax.hlines(band["p975"], bi - 0.4, bi + 0.4, color="0.3", linestyle="--", linewidth=1.0)
        if base is not None and b in base.get("behaviors", {}):
            bb = base["behaviors"][b]
            # Star = the plan-§6 ACHIEVABLE ceiling on the Δscore scale
            # (100 − α0 mean graded score — the gate-3 registered quantity;
            # review blocker g3 sibling). Computed from alpha0.mean_score so
            # pre-fix reduce outputs render too.
            a0_mean = bb["alpha0"]["mean_score"]
            if a0_mean is not None:
                ax.plot(
                    [bi + 0.38],
                    [100.0 - float(a0_mean)],
                    "*",
                    color="goldenrod",
                    markersize=11,
                    label="achievable ceiling (100−α0)" if bi == 0 else None,
                )
            # Donor-swap ceiling Δ kept as labeled CONTEXT (the §4.3 patch-
            # fraction denominator), no longer mislabeled as "ceiling".
            clo, chi = bb["ceiling_ci"]
            ax.errorbar(
                [bi + 0.30],
                [bb["ceiling_delta"]],
                yerr=_err([bb["ceiling_delta"]], [clo], [chi]),
                fmt="D",
                color="darkgoldenrod",
                markersize=5,
                capsize=2,
                label="donor-swap ceiling Δ" if bi == 0 else None,
            )
    ax.set_xticks(range(len(behaviors)))
    ax.set_xticklabels(behaviors)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylabel("Δ judge score vs α=0")
    ax.set_title(
        "Decisive steering effect — best cell per arm over breadths\n"
        "(frozen CI; thin gray = selection-inherited)"
    )
    ax.legend(frameon=False, fontsize=8)
    return _save(
        fig,
        fig_dir,
        "hero1_decisive_bars",  # plan §6.5 primary_deliverable filename
        [
            "decisive/delta_score_percell.json",
            "decisive/verdicts.json",
            "baseline_ceiling/judged_percell.json",
        ],
    )


def fig_hero2(out_root: Path, fig_dir: Path):
    pvc = _load(out_root, "patch/patch_vs_ceiling.json")
    if pvc is None or not pvc.get("cells"):
        return "skip:patch/patch_vs_ceiling.json not present or empty"
    cells = pvc["cells"]
    order = sorted(cells, key=lambda c: (cells[c]["cell"]["behavior"], cells[c]["cell"]["op"]))
    fig, ax = plt.subplots(figsize=(1.5 + 0.42 * len(order), 4.4))
    xs, labels = [], []
    for i, cid in enumerate(order):
        rec = cells[cid]
        cell = rec["cell"]
        v = rec["fraction_point"]
        if v is None:
            continue
        lo, hi = rec["fraction_ci"]
        ax.bar(i, v, color=DIR_COLORS[cell["direction"]], width=0.8)
        if lo is not None and hi is not None:  # all-degenerate cells persist null CI edges
            ax.errorbar([i], [v], yerr=_err([v], [lo], [hi]), fmt="none", ecolor="black", capsize=2)
        xs.append(i)
        labels.append(
            f"{cell['behavior']}: {DIR_LABELS[cell['direction']]}\n"
            f"{OP_LABELS[cell['op']]}, {BREADTH_LABELS[cell['breadth']]}"
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=75, fontsize=6.5)
    ax.axhline(1.0, color="goldenrod", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylabel("effect / donor-swap ceiling")
    ax.set_title("Projection-patch and ablation effects as fraction of ceiling")
    return _save(fig, fig_dir, "hero2_patch_fraction", ["patch/patch_vs_ceiling.json"])


def fig_result0(out_root: Path, fig_dir: Path):
    geo = _load(out_root, "directions/geometry_cosines.json")
    if geo is None:
        return "skip:directions/geometry_cosines.json not present"
    gg = geo["geometry"]
    behaviors = sorted(gg)
    fig, axes = plt.subplots(1, len(behaviors), figsize=(4.0 * len(behaviors), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    series = {
        "cos_pre_ctxext": "pre-image vs measured context direction",
        "cos_pre_rb": "pre-image vs persona vector",
        "cos_ctxext_rb": "context direction vs persona vector",
        "cos_pre_preshuf": "pre-image vs shuffled-map twin",
    }
    for ax, b in zip(axes, behaviors, strict=True):
        layers = gg[b]["layers"]
        for s, lab in series.items():
            ax.plot(layers, gg[b][s], marker="o", markersize=3, label=lab)
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_title(b)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("cosine")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Result 0: direction-family cosines per layer")
    return _save(fig, fig_dir, "result0_geometry", ["directions/geometry_cosines.json"])


def _dose_grid(dose: dict, value_key: str, name: str, fig_dir: Path, title: str, config: str):
    behaviors = sorted(dose["behaviors"])
    fig, axes = plt.subplots(
        len(behaviors), 2, figsize=(9.0, 2.9 * len(behaviors)), sharex=True, squeeze=False
    )
    for bi, b in enumerate(behaviors):
        cells = dose["behaviors"][b]["cells"]
        for pi, pos in enumerate(("context", "answer")):
            ax = axes[bi][pi]
            per_dir: dict[str, list[tuple[float, float]]] = {}
            for rec in cells.values():
                cell = rec["cell"]
                if cell["position"] != pos or cell["layer_config"] != config:
                    continue
                v = rec.get(value_key)
                if v is None:
                    continue
                per_dir.setdefault(cell["direction"], []).append((cell["c"], v))
            for d, pts in sorted(per_dir.items()):
                pts.sort()
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    markersize=3,
                    color=DIR_COLORS[d],
                    label=d,
                )
            ax.axhline(0.0, color="0.6", linewidth=0.8)
            ax.set_title(f"{b} — {pos}", fontsize=9)
            if bi == len(behaviors) - 1:
                ax.set_xlabel("dose c")
            if pi == 0:
                ax.set_ylabel(value_key)
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title} (layer config: {config})")
    fig.tight_layout()
    return _save(fig, fig_dir, name, ["localize/dose_response.json"])


def fig_dose_response(out_root: Path, fig_dir: Path):
    dose = _load(out_root, "localize/dose_response.json")
    if dose is None:
        return "skip:localize/dose_response.json not present"
    return _dose_grid(
        dose, "delta_score", "dose_response", fig_dir, "Dose-response, Δ score", "mid"
    )


def fig_rate_companion(out_root: Path, fig_dir: Path):
    dose = _load(out_root, "localize/dose_response.json")
    if dose is None:
        return "skip:localize/dose_response.json not present"
    return _dose_grid(
        dose, "delta_rate", "rate_companion", fig_dir, "Dose-response, Δ rate (score≥50)", "mid"
    )


def fig_layer_dose_heatmap(out_root: Path, fig_dir: Path):
    dose = _load(out_root, "localize/dose_response.json")
    if dose is None:
        return "skip:localize/dose_response.json not present"
    behaviors = sorted(dose["behaviors"])
    fig, axes = plt.subplots(1, len(behaviors), figsize=(4.6 * len(behaviors), 3.6), squeeze=False)
    for bi, b in enumerate(behaviors):
        ax = axes[0][bi]
        cells = dose["behaviors"][b]["cells"]
        doses = sorted({rec["cell"]["c"] for rec in cells.values()})
        configs = [
            lc
            for lc in _CONFIG_ORDER
            if any(rec["cell"]["layer_config"] == lc for rec in cells.values())
        ]
        grid = np.full((len(configs), len(doses)), np.nan)
        for rec in cells.values():
            cell = rec["cell"]
            if cell["direction"] != "pre" or cell["position"] != "context":
                continue
            if cell["layer_config"] not in configs:
                continue
            grid[configs.index(cell["layer_config"]), doses.index(cell["c"])] = rec["delta_score"]
        vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(doses)))
        ax.set_xticklabels([f"{c:g}" for c in doses], fontsize=7)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=8)
        ax.set_title(f"{b} — pre @ context", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Δ score by layer config × dose")
    fig.tight_layout()
    return _save(fig, fig_dir, "layer_dose_heatmap", ["localize/dose_response.json"])


def fig_coherence(out_root: Path, fig_dir: Path):
    dose = _load(out_root, "localize/dose_response.json")
    if dose is None:
        return "skip:localize/dose_response.json not present"
    return _dose_grid(
        dose, "coherence_rate", "coherence_vs_dose", fig_dir, "Coherence rate vs dose", "mid"
    )


def fig_per_question(out_root: Path, fig_dir: Path):
    percell = _load(out_root, "decisive/delta_score_percell.json")
    judged_dir = out_root / "judge" / "decisive" / "judged"
    if percell is None or not judged_dir.is_dir():
        return "skip:decisive percell or judged dir not present"
    behaviors = sorted(percell["behaviors"])
    if not behaviors:
        return "skip:no behaviors survived the gates"
    fig, axes = plt.subplots(1, len(behaviors), figsize=(4.4 * len(behaviors), 3.4), squeeze=False)
    for bi, b in enumerate(behaviors):
        ax = axes[0][bi]
        best = _best_cells(percell["behaviors"][b])
        targets = [("baseline (α=0)", judged_dir / f"{b}__a0.json", "0.4")]
        for d, p in HERO_ARMS:
            hit = best.get((d, p))
            if hit is not None:
                # Reader-facing tick labels (no internal arm slugs on axes).
                label = f"{DIR_LABELS[d]} @ {POS_LABELS[p]}"
                targets.append((label, judged_dir / f"{hit[0]}.json", DIR_COLORS[d]))
        for ti, (label, path, color) in enumerate(targets):
            if not path.is_file():
                continue
            j = json.loads(path.read_text())
            ys = [v for v in j["per_question_mean_score"] if v is not None]
            xs = np.full(len(ys), ti) + np.linspace(-0.15, 0.15, max(len(ys), 1))
            ax.plot(xs, ys, "o", markersize=3.5, color=color, alpha=0.75)
            if ys:
                ax.hlines(float(np.mean(ys)), ti - 0.25, ti + 0.25, color=color, linewidth=1.6)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([t[0] for t in targets], fontsize=7, rotation=20, ha="right")
        ax.set_title(b, fontsize=9)
        ax.set_ylabel("per-question mean judge score" if bi == 0 else "")
    fig.suptitle("Per-question scores at the decisive operating points")
    fig.tight_layout()
    return _save(
        fig,
        fig_dir,
        "per_question_dots",
        ["decisive/delta_score_percell.json", "judge/decisive/judged/*.json"],
    )


def fig_margin_scatter(out_root: Path, fig_dir: Path):
    margin = _load(out_root, "margin/margin_percell.json")
    percell = _load(out_root, "decisive/delta_score_percell.json")
    if margin is None or percell is None:
        return "skip:margin percell or decisive percell not present"
    cells = margin.get("cells", {})
    a0 = {rec["behavior"]: rec["margin_mean"] for rec in cells.values() if rec["c"] == 0.0}
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    markers = {"evil": "o", "sycophancy": "s", "hallucination": "^"}
    n_pts = 0
    for key, rec in cells.items():
        if rec["c"] == 0.0 or rec["behavior"] not in a0:
            continue
        cid = key.removesuffix("__mg")
        dec = percell["behaviors"].get(rec["behavior"], {}).get(cid)
        if dec is None:
            continue
        ax.plot(
            rec["margin_mean"] - a0[rec["behavior"]],
            dec["delta_score"],
            markers.get(rec["behavior"], "o"),
            color=DIR_COLORS[rec["direction"]],
            markersize=7,
        )
        n_pts += 1
    if n_pts == 0:
        plt.close(fig)
        return "skip:no margin cell joined a decisive judged cell"
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.axvline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("Δ teacher-forced pos−neg margin vs α=0")
    ax.set_ylabel("Δ judge score vs α=0")
    ax.set_title(
        "Continuous companion DV vs judged effect\n"
        "(single-layer operating points; marker = behavior, color = direction)"
    )
    from matplotlib.lines import Line2D

    dirs_present = sorted({rec["direction"] for rec in cells.values() if rec["c"] != 0.0})
    behs_present = sorted({rec["behavior"] for rec in cells.values() if rec["c"] != 0.0})
    handles = [
        Line2D([], [], marker="o", linestyle="", color=DIR_COLORS[d], label=DIR_LABELS[d])
        for d in dirs_present
    ] + [
        Line2D([], [], marker=markers.get(b, "o"), linestyle="", color="0.3", label=b)
        for b in behs_present
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="center right")
    return _save(
        fig,
        fig_dir,
        "margin_scatter",
        ["margin/margin_percell.json", "decisive/delta_score_percell.json"],
    )


def fig_offdesign_positives(out_root: Path, fig_dir: Path):
    """Off-design arms (round-2 critic request): persona vector @ context and
    pre-image @ answer, with per-arm null-band edges; right panel = the
    per-question dots behind the two clean sycophancy positives.

    Guarded for REDUCED/partial trees (single-behavior runs, pre-fix reduce
    outputs): a bar whose per-cell record or null-band key is absent — and a
    right-panel target whose judged file is absent — is dropped (the sibling
    builders' per-element `.get()`/`continue` contract) rather than
    KeyError-ing; on a full tree every element is present and the render is
    unchanged.
    """
    percell = _load(out_root, "decisive/delta_score_percell.json")
    verdicts = _load(out_root, "decisive/verdicts.json")
    gates = _load(out_root, "localize/gates.json")
    if percell is None or verdicts is None or gates is None:
        return "skip:decisive percell/verdicts or localize gates not present"

    def _band(tree: dict, beh: str, *keys: str):
        """Nested tree['behaviors'][beh][k0][k1]... lookup; None when any level is absent."""
        node = tree.get("behaviors", {}).get(beh)
        for k in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
        return node

    specs = [  # (cell_id, behavior, direction, band) — absent cell/band drops the bar
        (
            "evil__rb__ctx__mid__c2",
            "evil",
            "rb",
            _band(verdicts, "evil", "null_band_context", "p975"),
        ),
        (
            "evil__pre__ans__L17__c1",
            "evil",
            "pre",
            _band(gates, "evil", "gate2", "answer_band_p975"),
        ),
        (
            "sycophancy__rb__ctx__L17__c4",
            "sycophancy",
            "rb",
            _band(verdicts, "sycophancy", "null_band_context", "p975"),
        ),
        (
            "sycophancy__pre__ans__L14__c1",
            "sycophancy",
            "pre",
            _band(gates, "sycophancy", "gate2", "answer_band_p975"),
        ),
    ]
    bars = []  # (cell_id, behavior, direction, band, percell record)
    for cid, beh, d, band in specs:
        rec = percell.get("behaviors", {}).get(beh, {}).get(cid)
        if rec is None or band is None:
            continue
        bars.append((cid, beh, d, band, rec))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), width_ratios=[1.2, 1.0])
    ticklabels = []
    pos_of_cell = {"ctx": "context", "ans": "answer"}
    for i, (cid, beh, d, band, rec) in enumerate(bars):
        v = rec["delta_score"]
        lo, hi = rec["ci_frozen"]
        ax.bar(i, v, color=DIR_COLORS[d], width=0.7)
        ax.errorbar([i], [v], yerr=_err([v], [lo], [hi]), fmt="none", ecolor="black", capsize=3)
        ax.hlines(band, i - 0.42, i + 0.42, color="0.3", linestyle="--", linewidth=1.0)
        pos = pos_of_cell[cid.split("__")[2]]
        ticklabels.append(f"{beh}\n{DIR_LABELS[d]}\n@ {POS_LABELS[pos]}")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylabel("Δ judge score vs α=0")
    ax.set_title("Off-design arms vs their null bands (dashes)", fontsize=10)
    # right panel: per-question dots behind the two clean sycophancy positives
    judged_dir = out_root / "judge" / "decisive" / "judged"
    targets = [
        (label, fname, color)
        for label, fname, color in (
            ("baseline (α=0)", "sycophancy__a0.json", "0.5"),
            (
                "persona vector\n@ context vector",
                "sycophancy__rb__ctx__L17__c4.json",
                DIR_COLORS["rb"],
            ),
            (
                "pre-image\n@ answer tokens",
                "sycophancy__pre__ans__L14__c1.json",
                DIR_COLORS["pre"],
            ),
        )
        if (judged_dir / fname).is_file()
    ]
    rng = np.random.default_rng(0)
    for ti, (label, fname, color) in enumerate(targets):
        j = json.loads((judged_dir / fname).read_text())
        qs = [v for v in j["per_question_mean_score"] if v is not None]
        xj = ti + rng.uniform(-0.12, 0.12, size=len(qs))
        ax2.plot(xj, qs, "o", color=color, markersize=4, alpha=0.7)
        if qs:
            ax2.hlines(float(np.mean(qs)), ti - 0.25, ti + 0.25, color=color, linewidth=2)
    ax2.set_xticks(range(len(targets)))
    ax2.set_xticklabels([t[0] for t in targets], fontsize=7)
    ax2.set_ylabel("per-question mean judge score")
    ax2.set_title("Per-question view (sycophancy)", fontsize=10)
    fig.tight_layout()
    return _save(
        fig,
        fig_dir,
        "offdesign_positives",
        [
            "decisive/delta_score_percell.json",
            "decisive/verdicts.json",
            "localize/gates.json",
            "judge/decisive/judged/",
        ],
    )


def fig_map_quality(out_root: Path, fig_dir: Path):
    rep = _load(out_root, "maps/fit_report.json")
    if rep is None:
        return "skip:maps/fit_report.json not present"
    rows = sorted(rep["per_layer"], key=lambda r: r["layer"])
    layers = [r["layer"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax1.plot(
        layers,
        [r["heldout"]["map"]["r2"] for r in rows],
        marker="o",
        markersize=3,
        color="#1f77b4",
        label="ridge map",
    )
    ax1.plot(
        layers,
        [r["heldout"]["identity_bias"]["r2"] for r in rows],
        marker="s",
        markersize=3,
        color="#7f7f7f",
        label="identity + bias",
    )
    ax1.set_xlabel("layer")
    ax1.set_ylabel("held-out R²")
    ax1.set_title("Map quality per layer")
    ax1.legend(frameon=False, fontsize=8)
    knn = [r["heldout"]["knn"]["cosine"]["acc_at_k"] for r in rows]
    ax2.plot(
        layers,
        [k.get("10", k.get(10)) for k in knn],
        marker="o",
        markersize=3,
        color="#2ca02c",
        label="acc@10 (cosine)",
    )
    ax2.axhline(
        rows[0]["heldout"]["knn_chance_at_10"],
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label="chance@10",
    )
    ax2.set_xlabel("layer")
    ax2.set_ylabel("retrieval acc@10")
    ax2.set_title("kNN retrieval per layer")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, fig_dir, "map_quality", ["maps/fit_report.json"])


def fig_ctxext_split_hero(out_root: Path, fig_dir: Path):
    """Plan v7 §6 hero: per behavior, best split-arm bars per breadth beside
    the parent's persisted d_ctxext + d_pre operating cells, with the
    RESTRICTED pooled null band (dashed) + frozen CIs (thin gray =
    selection-inherited on the split arms)."""
    percell = _load(out_root, "ctxext_split/decisive/delta_score_percell.json")
    verdicts = _load(out_root, "ctxext_split/decisive/verdicts.json")
    parent_percell = _load(out_root, "decisive/delta_score_percell.json")
    parent_verdicts = _load(out_root, "decisive/verdicts.json")
    if percell is None or verdicts is None:
        return "skip:ctxext_split decisive percell/verdicts not present"
    behaviors = sorted(percell["behaviors"])
    if not behaviors:
        return "skip:no behaviors in ctxext_split decisive percell"
    fig, ax = plt.subplots(figsize=(1.6 + 3.2 * len(behaviors), 4.4))
    width = 0.14
    breadth_order = ("single", "mid")
    slots: list[tuple[str, str]] = [
        (d, br) for d in ("par", "perp") for br in breadth_order
    ]  # 4 split bars
    for bi, b in enumerate(behaviors):
        cells_b = percell["behaviors"][b]
        vb = verdicts["behaviors"].get(b, {})
        sel = vb.get("selection_inherited", {})
        band = vb.get("null_band_context_restricted")
        for si, (d, br) in enumerate(slots):
            hits = [
                rec
                for rec in cells_b.values()
                if rec["cell"]["direction"] == d
                and rec["cell"]["layer_config"] in (("mid",) if br == "mid" else ("L14", "L17"))
                and rec.get("coherence_pass", True)
            ]
            if not hits:
                continue
            rec = max(hits, key=lambda r: r["delta_score"])
            x = bi + (si - 1.5) * width
            v = rec["delta_score"]
            lo, hi = rec["ci_frozen"]
            ax.bar(
                x,
                v,
                width=width * 0.9,
                color=DIR_COLORS[d],
                alpha=1.0 if br == "single" else 0.55,
                label=(f"{DIR_LABELS[d]} ({BREADTH_LABELS[br]})" if bi == 0 else None),
            )
            ax.errorbar([x], [v], yerr=_err([v], [lo], [hi]), fmt="none", ecolor="black", capsize=3)
            si_ci = sel.get(f"{d}__context")
            if si_ci is not None:
                slo, shi = si_ci["ci"]
                ax.errorbar(
                    [x + width * 0.3],
                    [v],
                    yerr=_err([v], [slo], [shi]),
                    fmt="none",
                    ecolor="0.45",
                    elinewidth=0.9,
                    capsize=2,
                )
        # parent comparators: persisted operating cells (paired reference)
        if parent_percell is not None and parent_verdicts is not None:
            pv = parent_verdicts["behaviors"].get(b, {}).get("margins", {})
            for pi_, (mkey, d) in enumerate((("E_ctxdir", "ctxext"), ("E_pre", "pre"))):
                cid = pv.get(mkey, {}).get("cell_id")
                rec = parent_percell["behaviors"].get(b, {}).get(cid) if cid else None
                if rec is None:
                    continue
                x = bi + (2.5 + pi_) * width
                v = rec["delta_score"]
                lo, hi = rec["ci_frozen"]
                ax.bar(
                    x,
                    v,
                    width=width * 0.9,
                    color=DIR_COLORS[d],
                    label=f"{DIR_LABELS[d]} (parent op. cell)" if bi == 0 else None,
                )
                ax.errorbar(
                    [x], [v], yerr=_err([v], [lo], [hi]), fmt="none", ecolor="black", capsize=3
                )
        if band is not None:
            ax.hlines(
                band["p975"], bi - 0.45, bi + 0.62, color="0.3", linestyle="--", linewidth=1.0
            )
    ax.set_xticks(range(len(behaviors)))
    ax.set_xticklabels(behaviors)
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylabel("Δ judge score vs α=0")
    ax.set_title(
        "Split-direction steering — best cell per arm × breadth\n"
        "(dashed = restricted pooled null band; thin gray = selection-inherited)"
    )
    ax.legend(frameon=False, fontsize=7)
    return _save(
        fig,
        fig_dir,
        "ctxext_split_hero",  # plan v7 §6.5 primary_deliverable filename
        [
            "ctxext_split/decisive/delta_score_percell.json",
            "ctxext_split/decisive/verdicts.json",
            "decisive/delta_score_percell.json",
            "decisive/verdicts.json",
        ],
    )


def fig_ctxext_split_dose(out_root: Path, fig_dir: Path):
    """Plan v7 §6 exploratory dump: split-arm dose-response per behavior x
    layer-config (context position only), coherence-gated cells included as
    plotted points regardless (gating shown by the operating-point choice)."""
    dose = _load(out_root, "ctxext_split/localize/dose_response.json")
    if dose is None:
        return "skip:ctxext_split/localize/dose_response.json not present"
    behaviors = sorted(dose["behaviors"])
    if not behaviors:
        return "skip:no behaviors in split dose_response"
    configs_of = {
        b: sorted(
            {rec["cell"]["layer_config"] for rec in dose["behaviors"][b]["cells"].values()},
            key=lambda lc: _CONFIG_ORDER.index(lc) if lc in _CONFIG_ORDER else 99,
        )
        for b in behaviors
    }
    ncols = max(len(v) for v in configs_of.values())
    fig, axes = plt.subplots(
        len(behaviors), ncols, figsize=(4.6 * ncols, 3.0 * len(behaviors)), squeeze=False
    )
    for bi, b in enumerate(behaviors):
        cells = dose["behaviors"][b]["cells"]
        band = dose["behaviors"][b].get("null_band_context_restricted")
        for ci_, lc in enumerate(configs_of[b]):
            ax = axes[bi][ci_]
            per_dir: dict[str, list[tuple[float, float]]] = {}
            for rec in cells.values():
                cell = rec["cell"]
                if cell["layer_config"] != lc:
                    continue
                per_dir.setdefault(cell["direction"], []).append((cell["c"], rec["delta_score"]))
            for d, pts in sorted(per_dir.items()):
                pts.sort()
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    markersize=3,
                    color=DIR_COLORS[d],
                    label=DIR_LABELS[d],
                )
            if band is not None:
                ax.axhline(band["p975"], color="0.3", linestyle="--", linewidth=0.9)
            ax.axhline(0.0, color="0.6", linewidth=0.8)
            ax.set_title(f"{b} — {lc}", fontsize=9)
            if bi == len(behaviors) - 1:
                ax.set_xlabel("dose c")
            if ci_ == 0:
                ax.set_ylabel("Δ judge score vs α=0")
    if axes[0][0].get_legend_handles_labels()[0]:
        axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Split-direction dose-response (restricted sub-grid, context position)")
    fig.tight_layout()
    return _save(
        fig, fig_dir, "ctxext_split_dose_response", ["ctxext_split/localize/dose_response.json"]
    )


_BUILDERS = (
    fig_hero1,
    fig_hero2,
    fig_result0,
    fig_dose_response,
    fig_rate_companion,
    fig_layer_dose_heatmap,
    fig_coherence,
    fig_per_question,
    fig_margin_scatter,
    fig_offdesign_positives,
    fig_map_quality,
    fig_ctxext_split_hero,
    fig_ctxext_split_dose,
)


def render_all(out_root: Path, fig_dir: Path, *, require: tuple[str, ...] = ()) -> dict:
    """Render every figure whose inputs exist; return {'rendered', 'skipped'}.

    Missing INPUTS skip with a named reason; real errors propagate (fail
    fast). Any figure named in `require` that skips raises RuntimeError.
    """
    out_root = Path(out_root)
    fig_dir = Path(fig_dir)
    rendered: list[str] = []
    skipped: dict[str, str] = {}
    for builder in _BUILDERS:
        name = builder.__name__.removeprefix("fig_")
        res = builder(out_root, fig_dir)
        if isinstance(res, str) and res.startswith("skip:"):
            skipped[name] = res.removeprefix("skip:")
        else:
            rendered.append(res)
    missing = [
        n for n in require if n not in {Path(r).stem for r in rendered} and n not in rendered
    ]
    if missing:
        raise RuntimeError(f"required figures not rendered: {missing} (skipped={skipped})")
    return {"rendered": rendered, "skipped": skipped}
