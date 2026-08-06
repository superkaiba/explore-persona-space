"""issue #2091 unit D — render ALL plan §6 figures from the unit-C §6.5 JSONs.

Inputs (produced by ``issue2091_analysis.py`` under ``eval_results/issue_2091/``):
``r1_dispersion.json``, ``r2_delta.json``, ``r3_moderators_<behavior>.json``,
``r4_grids.json``, ``r5_polarization.json``, ``capture_parity.json``,
``greedy_dv/<behavior>.json`` — plus two BANKED inputs replotted per plan §6:
#1738 ``kresample/floor_summary.json`` (R1-b) and #1073
``heldout_recon_percontext.json`` + ``adequacy_tail_characterization.json`` (R1-c).

Contract (unit-D brief):
- every figure gets a sidecar ``<name>.caption.md`` carrying the plan-mandated
  caption duties (per-arm provenance; A1 WildChat-headline note; A2
  disjoint-half approximation; A3 per-column ceilings; A4 median-split
  within-column caveat; W2 expectation band; W3 layer-set naming; K/top_p
  deviations) — the report splices these as the figure captions;
- a MISSING input JSON skips that figure with a warning (production renders
  after P3/P4 land) — never a crash, never a placeholder figure;
- evil floor-censored cells render as labeled UNINFORMATIVE (greyed/hatched
  panel with the label), never as bars;
- ``--phase lengths`` computes the R1 median answer-length annotations from
  packed completion shards IN CODE ONLY (char counts; no row text is ever
  printed — trigger-dense digest-only discipline);
- ``--phase synth-fixtures`` writes shape-faithful synthetic §6.5 JSONs so the
  smoke exercises EVERY figure function without staged/judged data.

Style: /paper-plots conventions — ``set_paper_style("blog")``, colorblind-safe
Wong palette with ONE color = ONE meaning across the whole set (families ->
palette[0:4], layers -> palette[4:7], regimes -> black/grey/palette[7]), error
bars as NON-NEGATIVE offsets (gotchas.md xerr/yerr), no interpretive overlays.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue2091_figures")

# ── constants (mirror issue2091_analysis.py) ──────────────────────────────────
LAYERS: tuple[int, ...] = (14, 19, 26)
HEADLINE_LAYER = 19
REGIMES: tuple[str, ...] = ("greedy", "avg_k5", "single")
BEHAVIORS: tuple[str, ...] = ("sycophancy", "hallucination", "evil")
SEVERE_TAIL = -0.02

SETTING_ORDER: tuple[str, ...] = (
    "wildchat",
    "lmsys",
    "syc_train",
    "syc_aita",
    "hal_train",
    "hal_nqopen",
    "hal_simpleqa",
    "evil_train",
    "evil_hhrt",
    "evil_toxicchat",
)
SETTING_LABELS: dict[str, str] = {
    "wildchat": "WildChat (generic)",
    "lmsys": "LMSYS (context only)",
    "syc_train": "Sycophancy train",
    "syc_aita": "Sycophancy OOD (AITA)",
    "hal_train": "Hallucination train",
    "hal_nqopen": "Hallucination OOD (NQ-Open)",
    "hal_simpleqa": "Hallucination OOD (SimpleQA)",
    "evil_train": "Evil train",
    "evil_hhrt": "Evil OOD (hh-rlhf)",
    "evil_toxicchat": "Evil OOD (ToxicChat)",
}
FAMILY_OF_SETTING: dict[str, str] = {
    "wildchat": "generic",
    "lmsys": "generic",
    "syc_train": "sycophancy",
    "syc_aita": "sycophancy",
    "hal_train": "hallucination",
    "hal_nqopen": "hallucination",
    "hal_simpleqa": "hallucination",
    "evil_train": "evil",
    "evil_hhrt": "evil",
    "evil_toxicchat": "evil",
}
REGIME_LABELS: dict[str, str] = {
    "greedy": "Greedy (deterministic)",
    "avg_k5": "Five-draw average",
    "single": "Single stochastic draw",
}
# dedicated method-family palette (Tol-muted subset) — disjoint from the family /
# layer / regime color meanings above (one color = one meaning across the set).
METHOD_COLORS: dict[str, str] = {
    "pv_projection": "#332288",
    "supervised_context": "#88ccee",
    "map_pv_projection": "#ddcc77",
    "map_supervised_answer": "#999933",
    "oracle_answer": "#882255",
}
METHOD_LABELS: dict[str, str] = {
    "pv_projection": "Persona-vector projection",
    "supervised_context": "Supervised context probe",
    "map_pv_projection": "Mapped persona-vector projection",
    "map_supervised_answer": "Mapped supervised answer probe",
    "oracle_answer": "Oracle answer probe",
    "disjoint_half": "Disjoint-half DV reference",
}

_PAL = paper_palette(8)
FAMILY_COLORS: dict[str, str] = {
    "generic": _PAL[0],
    "sycophancy": _PAL[1],
    "hallucination": _PAL[2],
    "evil": _PAL[3],
}
LAYER_COLORS: dict[int, str] = {14: _PAL[4], 19: _PAL[5], 26: _PAL[6]}
# regimes get colors OUTSIDE the family/layer palette slices (one color = one meaning):
# black (Wong idx 7, otherwise unused) + a colorblind-safe purple + grey.
REGIME_COLORS: dict[str, str] = {"greedy": "#000000", "avg_k5": "#7570b3", "single": "#8a8a8a"}

PROVENANCE = (
    "Per-arm provenance: greedy = deterministic temperature-0 decode (1 completion/context, "
    "max_tokens=1024); averaged = mean over K=5 temperature-1.0 on-policy rollouts (banked "
    "#1739); single = one temperature-1.0 draw (fixed-seed pick of the banked k). All "
    "completions are on-policy Qwen-2.5-7B-Instruct."
)
LMSYS_DEVIATIONS = (
    "LMSYS values use the banked #1073 K 5-of-10 fixed-seed subsample generated at "
    "top_p=0.95 (vs 1.0 everywhere else) — both stated deviations; LMSYS is shown for "
    "context only and never carries the generic-vs-trait headline (plan A1)."
)
W3_LAYERS = (
    "Read-out layers frozen at L14/L19/L26 (headline L19, #1738); this set differs from "
    "#1073's per-trait read-out layers (plan W3)."
)


# ── small helpers ─────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict | None:
    """Load a JSON input; None + warning when missing (skip-with-warning contract)."""
    if not path.is_file():
        logger.warning("[figures] missing input %s — figure(s) depending on it are skipped", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_entry(block: dict | None, layer: int) -> dict | None:
    """Fetch a per-layer entry tolerating 'L19' / '19' / int keys."""
    if not isinstance(block, dict):
        return None
    for key in (f"L{layer}", str(layer), layer):
        if key in block:
            return block[key]
    return None


def _err_from_ci(vals: list[float], cis: list) -> np.ndarray:
    """(2, n) NON-NEGATIVE error offsets from [lo, hi] CIs (gotchas.md xerr/yerr)."""
    lo = np.array(
        [
            max(0.0, v - c[0]) if (c and c[0] is not None and np.isfinite(v)) else 0.0
            for v, c in zip(vals, cis)
        ]
    )
    hi = np.array(
        [
            max(0.0, c[1] - v) if (c and c[1] is not None and np.isfinite(v)) else 0.0
            for v, c in zip(vals, cis)
        ]
    )
    return np.vstack([lo, hi])


def _fig_dir(args) -> Path:
    return Path(args.out_root) / "issue_2091"


def _write_caption(args, name: str, text: str) -> None:
    dest = _fig_dir(args) / f"{name}.caption.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text.strip() + "\n", encoding="utf-8")


def _save(args, fig, name: str) -> None:
    savefig_paper(fig, f"issue_2091/{name}", dir=args.out_root)
    plt.close(fig)
    logger.info("[figures] rendered %s", name)


def _settings_present(settings: dict) -> list[str]:
    ordered = [s for s in SETTING_ORDER if settings.get(s)]
    ordered += [s for s in settings if s not in SETTING_ORDER and settings.get(s)]
    return ordered


# ── R1: dispersion ────────────────────────────────────────────────────────────
def fig_r1_hero(args, layer: int = HEADLINE_LAYER, suffix: str = "") -> bool:
    r1 = _load_json(Path(args.results_root) / "r1_dispersion.json")
    if r1 is None:
        return False
    lengths = _load_json(Path(args.results_root) / "r1_answer_lengths.json")
    settings = r1.get("settings") or {}
    names, meds, cis, colors, alphas, clouds = [], [], [], [], [], []
    for s in _settings_present(settings):
        entry = _layer_entry(settings[s], layer)
        if not entry:
            continue
        summ = entry.get("summary") or {}
        med = summ.get("median")
        if med is None:
            continue
        names.append(s)
        meds.append(float(med))
        cis.append(entry.get("boot_ci_median"))
        colors.append(FAMILY_COLORS[FAMILY_OF_SETTING.get(s, "generic")])
        alphas.append(0.45 if s == "lmsys" else 0.9)
        pc = entry.get("percontext") or {}
        clouds.append(np.asarray(pc.get("dispersion") or [], dtype=np.float64))
    if not names:
        logger.warning("[figures] r1_hero%s: no settings with L%d data — skipped", suffix, layer)
        return False
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(names))
    rng = np.random.default_rng(20910)
    for i, cloud in enumerate(clouds):
        if cloud.size:
            jit = rng.uniform(-0.18, 0.18, size=cloud.size)
            ax.scatter(
                np.full(cloud.size, x[i]) + jit,
                cloud,
                s=6,
                color=colors[i],
                alpha=0.22,
                linewidths=0,
                zorder=1,
            )
    for i in range(len(names)):
        ax.bar(x[i], meds[i], width=0.62, color=colors[i], alpha=alphas[i], zorder=2)
    ax.errorbar(
        x,
        meds,
        yerr=_err_from_ci(meds, cis),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=3,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS.get(s, s) for s in names], rotation=20, ha="right")
    ax.set_ylabel(f"answer-vector dispersion\n(mean pairwise cosine distance, K=5, L{layer})")
    ax.set_title(f"Per-setting answer-vector dispersion (L{layer})", loc="left")
    name = f"r1_dispersion_bars{suffix}"
    _save(args, fig, name)
    len_lines = []
    if lengths:
        for s in names:
            row = (lengths.get("settings") or {}).get(s)
            if row:
                g = row.get("greedy_median_chars")
                b = row.get("banked_median_chars")
                len_lines.append(
                    f"{SETTING_LABELS.get(s, s)}: greedy median {g} chars, "
                    f"banked-K5 median {b} chars (n={row.get('n_greedy')}/{row.get('n_banked')})"
                )
    lmsys_len = None
    lm = _layer_entry(settings.get("lmsys"), layer)
    if lm and lm.get("median_answer_len") is not None:
        lmsys_len = lm["median_answer_len"]
    _write_caption(
        args,
        name,
        f"Median per-context answer-vector dispersion per setting at L{layer} (bars; cluster-"
        "bootstrap 95% CI on the median; light points = the per-context dispersion cloud "
        "behind each bar). HEADLINE: the generic-vs-trait comparison reads off the matched-"
        "instrument WildChat bar — LMSYS bars are context only and never carry the headline "
        f"(plan A1). {LMSYS_DEVIATIONS} {PROVENANCE} {W3_LAYERS}\n\n"
        "Median answer lengths (length confound named, plan R1): "
        + (
            "; ".join(len_lines)
            if len_lines
            else "per-setting packed-shard lengths unavailable at render time "
            "(r1_answer_lengths.json absent — run --phase lengths after staging)"
        )
        + (
            f". LMSYS banked median answer span length: {lmsys_len} tokens."
            if lmsys_len is not None
            else ""
        ),
    )
    return True


def fig_r1b_floor_depth(args) -> bool:
    fs = _load_json(Path(args.banked_1738))
    if fs is None:
        return False
    per_layer = fs.get("per_layer") or {}
    depth_keys = ["2-2", "3-4", ">=5"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(depth_keys))
    for layer in LAYERS:
        row = per_layer.get(str(layer)) or per_layer.get(f"L{layer}")
        if not row:
            continue
        by_depth = row.get("floor_share_median_by_depth") or {}
        y = [by_depth.get(k) for k in depth_keys]
        ax.plot(
            x,
            [np.nan if v is None else v for v in y],
            marker="o",
            color=LAYER_COLORS[layer],
            label=f"layer {layer}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["2", "3–4", "≥5"])
    ax.set_xlabel("conversation depth (turns)")
    ax.set_ylabel("answer-sampling floor share (median)")
    ax.set_title("Answer-sampling floor share vs multi-turn depth", loc="left")
    ax.legend()
    name = "r1b_floor_share_vs_depth"
    _save(args, fig, name)
    n = (per_layer.get("19") or {}).get("n")
    _write_caption(
        args,
        name,
        f"Median answer-sampling floor share by conversation depth per frozen read-out layer "
        f"(banked #1738 kresample summary, n={n}, k=4 resamples; REPLOTTED as-is — the banked "
        "summary carries medians only, no CI). The flat profile is the banked answer to the "
        f"multi-turn-depth question (plan R1-b). {W3_LAYERS}",
    )
    return True


def fig_r1c_r2_vs_dispersion(args) -> bool:
    pc = _load_json(Path(args.banked_1073_percontext))
    adeq = _load_json(Path(args.banked_1073_adequacy))
    if pc is None or adeq is None:
        return False
    # the adequacy percontext arrays are at the PARENT's own headline layer (L27 for
    # #1073) — read the matching last_L<layer> arms, never assume this task's L19
    # (W3: the layer sets differ; the caption names both).
    ph = adeq.get("percontext_headline") or {}
    banked_layer = int(ph.get("layer") or HEADLINE_LAYER)
    arms = (pc.get("readout_percontext_last") or {}).get(f"last_L{banked_layer}") or {}
    disp = np.asarray(ph.get("rollout_dispersion") or [], dtype=np.float64)
    if not arms or disp.size == 0:
        logger.warning("[figures] r1c: banked #1073 arms or dispersion missing — skipped")
        return False
    # index alignment contract: both banked files cover the SAME canonical context
    # order (heldout common_set.common_index is the identity with zero drops —
    # verified 2026-08-05); a size mismatch below skips rather than mis-joins.

    def _r2(arm: str) -> np.ndarray | None:
        a = arms.get(arm)
        if not a:
            return None
        sse = np.asarray(a.get("sse") or [], dtype=np.float64)
        sst = np.asarray(a.get("sst") or [], dtype=np.float64)
        if sse.size == 0 or sse.size != sst.size:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return 1.0 - sse / sst

    # file-NATIVE arm names (banked #1073 readout_percontext_last semantics — the
    # analyzer interprets them against #1073's body; this figure only replots).
    show = {
        "greedy": "banked arm: greedy",
        "stoch1_new": "banked arm: single stochastic draw",
        "avg10": "banked arm: 10-draw mean",
    }
    arm_color = {
        "greedy": REGIME_COLORS["greedy"],
        "stoch1_new": REGIME_COLORS["single"],
        "avg10": REGIME_COLORS["avg_k5"],
    }
    r2s = {k: _r2(k) for k in show}
    r2s = {k: v for k, v in r2s.items() if v is not None and v.size == disp.size}
    if not r2s:
        logger.warning("[figures] r1c: per-context R2 arrays absent or misaligned — skipped")
        return False
    q = np.quantile(disp, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(disp, q)  # 0..4 quintiles
    y_lo = -4.0  # display clip: per-context R2 has an extreme negative tail (fraction reported)
    clipped = {}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm, r2v in r2s.items():
        clipped[arm] = float(np.mean(r2v < y_lo))
        ax.scatter(
            disp,
            np.clip(r2v, y_lo, None),
            s=5,
            color=arm_color[arm],
            alpha=0.12,
            linewidths=0,
            zorder=1,
        )
        med = [
            float(np.nanmedian(r2v[bins == b])) if (bins == b).any() else np.nan for b in range(5)
        ]
        centers = [
            float(np.median(disp[bins == b])) if (bins == b).any() else np.nan for b in range(5)
        ]
        ax.plot(centers, med, marker="o", color=arm_color[arm], label=show[arm], zorder=3)
    floor = _r2("mean_floor")
    if floor is not None:
        ax.axhline(float(np.nanmedian(floor)), color="#666666", ls="--", lw=1.2, zorder=2)
    ax.set_ylim(y_lo, 1.1)
    ax.set_xlabel("per-context rollout dispersion (mean pairwise cosine distance)")
    ax.set_ylabel(f"per-context held-out R² (banked #1073, L{banked_layer})")
    ax.set_title("Held-out reconstruction vs dispersion (banked #1073)", loc="left")
    ax.legend()
    name = "r1c_heldout_r2_vs_dispersion"
    _save(args, fig, name)
    clip_note = "; ".join(
        f"{show[a]}: {100 * frac:.1f}% of points below the display clip (R² < {y_lo})"
        for a, frac in clipped.items()
    )
    _write_caption(
        args,
        name,
        f"Per-context held-out reconstruction R² (1 − sse/sst, banked #1073 percontext, "
        f"L{banked_layer} — the PARENT's own headline layer, not this task's frozen L19) vs "
        "per-context rollout dispersion; light points = raw per-context values (display "
        f"clipped at R² = {y_lo}: {clip_note}), lines = dispersion-quintile medians per "
        "target arm; grey dashed line = the mean-floor reference (median per-context R² of "
        "the corpus-mean predictor — the entropy floor). Generic LMSYS n=5,000, K=10 banked "
        "rollouts. Arm names are the banked file's OWN (readout_percontext_last) — their "
        "exact estimator semantics are defined in #1073, and this replot asserts nothing "
        "beyond the file's numbers (the 'avg10' arm reads negative per-context R² in this "
        f"artifact; interpretation belongs to #1073's body, not this caption). {W3_LAYERS}",
    )
    return True


# ── R2: matched-reference Delta ───────────────────────────────────────────────
def _r2_layer_stats(settings: dict, s: str, layer: int) -> dict | None:
    entry = _layer_entry(settings.get(s), layer)
    if not entry:
        return None
    if "median_delta" in entry:
        return {
            "median": entry["median_delta"],
            # lmsys entries key the delta CI "delta_boot_ci_median"; behavior blocks key it
            # "boot_ci_median" (and never carry both) — prefer the delta-named key.
            "ci": entry.get("delta_boot_ci_median") or entry.get("boot_ci_median"),
            "jack": (entry.get("jackknife") or {}).get("band"),
            "p_hat": entry.get("common_language_p"),
            "severe": entry.get("severe_tail_rate"),
            "exch": entry.get("exchangeability"),
            "percontext": entry.get("percontext"),
            "quintile_curve": entry.get("quintile_curve"),
        }
    return None


def fig_r2_hero(args, layer: int = HEADLINE_LAYER, suffix: str = "") -> bool:
    r2 = _load_json(Path(args.results_root) / "r2_delta.json")
    if r2 is None:
        return False
    settings = r2.get("settings") or {}
    rows = []
    for s in _settings_present(settings):
        st = _r2_layer_stats(settings, s, layer)
        if st and st["median"] is not None:
            rows.append((s, st))
    if not rows:
        logger.warning("[figures] r2_hero%s: no settings with L%d delta — skipped", suffix, layer)
        return False
    contrasts = r2.get("generic_vs_trait_contrasts_L19") or {}
    ncols = 2 if contrasts and layer == HEADLINE_LAYER else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols + 1.5, 4.4))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    x = np.arange(len(rows))
    meds = [st["median"] for _, st in rows]
    cis = [st["ci"] for _, st in rows]
    colors = [FAMILY_COLORS[FAMILY_OF_SETTING.get(s, "generic")] for s, _ in rows]
    ax.axhline(0.0, color="#444444", lw=1.0)
    for i, (s, st) in enumerate(rows):
        ax.bar(x[i], meds[i], width=0.6, color=colors[i], alpha=0.45 if s == "lmsys" else 0.9)
        band = st.get("jack")
        if band and band[0] is not None:
            ax.vlines(x[i] + 0.36, band[0], band[1], color="#555555", lw=2.6, alpha=0.9)
    ax.errorbar(
        x,
        meds,
        yerr=_err_from_ci(meds, cis),
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS.get(s, s) for s, _ in rows], rotation=20, ha="right")
    ax.set_ylabel(f"median Δ (greedy-vs-draw matched-reference, L{layer})")
    ax.set_title("Greedy-vs-draw matched-reference Δ per setting", loc="left")
    if ncols == 2:
        axc = axes[1]
        cn = list(contrasts)
        cm = [contrasts[k].get("median") for k in cn]
        cc = [contrasts[k].get("ci_holm") or contrasts[k].get("ci95") for k in cn]
        axc.axhline(0.0, color="#444444", lw=1.0)
        xc = np.arange(len(cn))
        fam = [k.split("_train_minus")[0] for k in cn]
        axc.bar(xc, cm, width=0.55, color=[FAMILY_COLORS.get(f, "#777777") for f in fam])
        axc.errorbar(
            xc,
            cm,
            yerr=_err_from_ci(cm, cc),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.2,
            capsize=3,
        )
        axc.set_xticks(xc)
        axc.set_xticklabels(
            [k.replace("_train_minus_wildchat", " train − WildChat") for k in cn],
            rotation=20,
            ha="right",
        )
        axc.set_ylabel("median Δ difference (Holm-adjusted CI)")
        axc.set_title("Generic-vs-trait contrasts", loc="left")
    name = f"r2_delta_bars{suffix}"
    _save(args, fig, name)
    exch_lines = []
    p_lines = []
    for s, st in rows:
        ex = st.get("exch") or {}
        if ex.get("mean_rank") is not None:
            exch_lines.append(
                f"{SETTING_LABELS.get(s, s)}: mean rank {ex['mean_rank']:.2f} "
                f"(uniform-null expectation {ex.get('expected_mean')})"
            )
        if st.get("p_hat") is not None:
            p_lines.append(f"{SETTING_LABELS.get(s, s)}: P̂ = {st['p_hat']:.2f}")
    holm_lines = [
        f"{k}: median {v.get('median'):+.4f}, Holm-adjusted CI {v.get('ci_holm')}, "
        f"p_holm={v.get('p_holm')}"
        for k, v in contrasts.items()
    ]
    _write_caption(
        args,
        name,
        f"Median per-context Δ (greedy leg minus typical-draw leg against shared LOO "
        f"references, L{layer}) per setting; cluster-bootstrap 95% CI; the grey tick beside "
        "each bar is the drop-one-draw jackknife band (values inside it are indistinguishable "
        "from a draw at the K=5 grain); zero line = the exchangeability expectation. "
        "Right panel (where present): generic-vs-trait differences tested directly with "
        "Holm-adjusted CIs. Exchangeability rank test (greedy ranked among the K+1=6 "
        "rollouts): " + ("; ".join(exch_lines) if exch_lines else "n/a") + ". "
        "Per-bar common-language effect size (fraction of contexts with Δ>0): "
        + ("; ".join(p_lines) if p_lines else "n/a")
        + ". Holm contrasts: "
        + ("; ".join(holm_lines) if holm_lines else "n/a")
        + f". {PROVENANCE} {LMSYS_DEVIATIONS} {W3_LAYERS} "
        "MF-2: any negative trait-side Δ is read against capture_parity.json + the WildChat-"
        "vs-LMSYS triangulation BEFORE any #1073 correction posts (report, never a gate).",
    )
    return True


def fig_r2b_ecdf(args) -> bool:
    r2 = _load_json(Path(args.results_root) / "r2_delta.json")
    if r2 is None:
        return False
    settings = r2.get("settings") or {}
    panels: dict[str, list[tuple[str, np.ndarray, float | None]]] = {}
    for s in _settings_present(settings):
        st = _r2_layer_stats(settings, s, HEADLINE_LAYER)
        if not st or not st.get("percontext"):
            continue
        delta = np.asarray(st["percontext"].get("delta") or [], dtype=np.float64)
        if delta.size == 0:
            continue
        panels.setdefault(FAMILY_OF_SETTING.get(s, "generic"), []).append(
            (s, delta, st.get("severe"))
        )
    if not panels:
        logger.warning("[figures] r2b: no per-context deltas — skipped")
        return False
    fams = [f for f in ("generic", *BEHAVIORS) if f in panels]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.0 * len(fams), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    styles = ["-", "--", ":", "-."]
    severe_lines = []
    for ax, fam in zip(axes, fams):
        for j, (s, delta, severe) in enumerate(panels[fam]):
            xs = np.sort(delta)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(
                xs,
                ys,
                color=FAMILY_COLORS[fam],
                ls=styles[j % len(styles)],
                label=SETTING_LABELS.get(s, s),
            )
            if severe is not None:
                severe_lines.append(f"{SETTING_LABELS.get(s, s)}: {100 * severe:.1f}%")
        ax.axvline(SEVERE_TAIL, color="#666666", lw=1.0, ls="--")
        ax.set_title(fam, loc="left")
        ax.set_xlabel("per-context Δ (L19)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("ECDF")
    name = "r2b_delta_ecdf"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "ECDF of per-context Δ per setting (L19), grouped by family; dashed vertical line = "
        f"the severe-tail threshold Δ < {SEVERE_TAIL}. Severe-tail rates: "
        + ("; ".join(severe_lines) if severe_lines else "n/a")
        + ". The hero reports the MEDIAN because Δ is heavy-tailed: the median reads the "
        "typical context while the severe tail is reported separately here (plan R2-b). "
        f"{PROVENANCE}",
    )
    return True


def fig_r2c_quintiles(args) -> bool:
    r2 = _load_json(Path(args.results_root) / "r2_delta.json")
    if r2 is None:
        return False
    settings = r2.get("settings") or {}
    rows = []
    for s in _settings_present(settings):
        st = _r2_layer_stats(settings, s, HEADLINE_LAYER)
        if st and st.get("quintile_curve"):
            rows.append((s, st["quintile_curve"]))
    if not rows:
        logger.warning("[figures] r2c: no quintile curves — skipped")
        return False
    ncols = min(5, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, qc) in zip(axes, rows):
        q = qc.get("quintile") or list(range(1, 6))
        gk = [np.nan if v is None else v for v in (qc.get("cos_g_kmean_median") or [])]
        dh = [np.nan if v is None else v for v in (qc.get("disjoint_half_median") or [])]
        fam = FAMILY_OF_SETTING.get(s, "generic")
        ax.plot(q, gk, marker="o", color=FAMILY_COLORS[fam], label="greedy vs K-mean")
        ax.plot(q, dh, marker="s", color="#777777", ls="--", label="disjoint-half reference")
        ax.set_title(SETTING_LABELS.get(s, s), loc="left", fontsize=9)
        ax.set_xticks(list(range(1, 6)))
    for ax in axes[len(rows) :]:
        ax.set_visible(False)
    axes[0].set_ylabel("median cosine agreement")
    for ax in axes[max(0, len(rows) - ncols) : len(rows)]:
        ax.set_xlabel("dispersion quintile")
    axes[0].legend(fontsize=7)
    name = "r2c_quintile_agreement"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "Greedy-vs-K-mean cosine agreement (solid, family color) and the disjoint-half "
        "matched-noise reference (grey dashed) by within-setting dispersion quintile, L19. "
        "CAVEAT (plan A2): the disjoint-half reference at K=5 is a 2-vs-3 split — "
        "approximately, NOT exactly, noise-matched; gaps between the curves are weighed "
        "against the drop-one-draw jackknife band (hero figure), never read at face value. "
        f"{PROVENANCE}",
    )
    return True


# ── R3: moderators ────────────────────────────────────────────────────────────
def fig_r3_hero(args, behavior: str) -> bool:
    r3 = _load_json(Path(args.results_root) / f"r3_moderators_{behavior}.json")
    if r3 is None:
        return False
    settings = r3.get("settings") or {}
    setting = next(
        (s for s in _settings_present(settings) if (settings[s] or {}).get("commonality")), None
    )
    if setting is None:
        logger.warning("[figures] r3_hero_%s: no setting carries a commonality block", behavior)
        return False
    comm = settings[setting]["commonality"]
    sig_names = [k for k in ("sigma_a_total", "sigma_a_proj") if k in comm]
    if not sig_names:
        logger.warning("[figures] r3_hero_%s: commonality has no sigma defs — skipped", behavior)
        return False
    fig, axes = plt.subplots(
        2, len(sig_names), figsize=(5.8 * len(sig_names), 6.2), height_ratios=[2.2, 1.0]
    )
    axes = np.atleast_2d(axes)
    if axes.shape[1] != len(sig_names):
        axes = axes.reshape(2, len(sig_names))
    # greyscale segments — deliberately OUTSIDE the family/layer/regime color meanings
    # (one color = one meaning across the figure set; luminance separates the stack).
    seg_colors = {"unique_sigma": "#3b3b3b", "shared": "#9e9e9e", "unique_p": "#d9d9d9"}
    supp_flags = []
    for col, sig in enumerate(sig_names):
        fams = [f for f, cm in (comm[sig] or {}).items() if cm and cm.get("r2_full") is not None]
        ax = axes[0, col]
        x = np.arange(len(fams))
        u1 = np.array([comm[sig][f].get("unique_x1", np.nan) for f in fams])
        u2 = np.array([comm[sig][f].get("unique_x2", np.nan) for f in fams])
        sh = np.array([comm[sig][f].get("shared", np.nan) for f in fams])
        ax.bar(x, u1, width=0.6, color=seg_colors["unique_sigma"], label="unique σ_A")
        ax.bar(x, sh, width=0.6, bottom=u1, color=seg_colors["shared"], label="shared")
        ax.bar(
            x,
            u2,
            width=0.6,
            bottom=u1 + np.clip(sh, 0, None),
            color=seg_colors["unique_p"],
            label="unique P",
        )
        ax.axhline(0.0, color="#444444", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS.get(f, f) for f in fams], rotation=25, ha="right", fontsize=7
        )
        ax.set_ylabel("rank-error commonality R² share")
        sig_label = "σ_A (total)" if sig == "sigma_a_total" else "σ_A (r_B projection)"
        ax.set_title(f"{behavior} — {sig_label}", loc="left")
        if col == 0:
            ax.legend(fontsize=7)
        for f in fams:
            if comm[sig][f].get("suppression"):
                supp_flags.append(f"{sig}/{f}")
        axc = axes[1, col]
        meds, cis = [], []
        for f in fams:
            comp = comm[sig][f].get("companion_unique_sigma_minus_unique_p") or {}
            meds.append(comp.get("median", np.nan))
            cis.append(comp.get("ci95"))
        axc.axhline(0.0, color="#444444", lw=0.8)
        axc.errorbar(
            np.arange(len(fams)),
            meds,
            yerr=_err_from_ci(meds, cis),
            fmt="o",
            color="#222222",
            capsize=3,
        )
        axc.set_xticks(np.arange(len(fams)))
        axc.set_xticklabels(
            [METHOD_LABELS.get(f, f) for f in fams], rotation=25, ha="right", fontsize=7
        )
        axc.set_ylabel("unique σ_A − unique P")
    name = f"r3_hero_{behavior}"
    _save(args, fig, name)
    guard = (settings[setting] or {}).get("guardrails") or {}
    _write_caption(
        args,
        name,
        f"Rank-error commonality decomposition for {behavior} ({SETTING_LABELS.get(setting, setting)} "
        "eval set): stacked unique/shared/unique R² shares per method family for each σ_A "
        "definition (top), with the companion strip unique σ_A − unique P and cluster-"
        "bootstrap 95% CI (bottom). Outcome = per-context |rank(method score) − rank(avg-K5 "
        "DV)|. Guardrails (analyzer-weighed diagnostics, not kill gates): "
        f"ρ(σ_A, P) = {guard.get('rho_sigma_a_total_vs_p')}, split-half reliabilities "
        f"{guard.get('split_half')}. Suppression (negative shared — do not stack): "
        + (", ".join(supp_flags) if supp_flags else "none flagged")
        + f". Judge-noise footnote: {r3.get('judge_var_note')}. {PROVENANCE}",
    )
    return True


def fig_r3b_components(args, behavior: str) -> bool:
    r3 = _load_json(Path(args.results_root) / f"r3_moderators_{behavior}.json")
    banked = None
    if r3 is None and args.banked_smoke and behavior == "sycophancy":
        banked = _load_json(Path(args.banked_smoke))
        if banked is None:
            return False
        logger.info("[figures] r3b_%s: rendering from banked-smoke fallback input", behavior)
    r5p = _load_json(Path(args.results_root) / "r5_polarization.json") or {}
    uninformative = set()
    for key, blk in (r5p.get("panels") or {}).items():
        if blk and blk.get("uninformative"):
            b, _, s = key.partition("::")
            if b == behavior:
                uninformative.add(s)
    rows: list[tuple[str, dict, dict]] = []
    if banked is not None:
        ceils = (banked.get("r3_plot1") or {}).get("ceilings") or {}
        comps = (banked.get("r3_plot1") or {}).get("variance_components") or {}
        rows.append(("banked smoke (sycophancy)", comps, ceils))
    elif r3 is not None:
        for s in _settings_present(r3.get("settings") or {}):
            ceil = (r3["settings"][s] or {}).get("ceilings") or {}
            comps = ceil.get("components") or {}
            rows.append((s, comps, ceil))
    if not rows:
        logger.warning("[figures] r3b_%s: no ceilings/components — skipped", behavior)
        return False
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(max(6.0, 1.5 * len(rows) + 3), 6.4), sharex=True, height_ratios=[1.4, 1.0]
    )
    x = np.arange(len(rows))
    keys = [
        ("between_sd_raw", "between-context SD (raw)"),
        ("between_sd_corrected", "between-context SD (Var/K-corrected)"),
        ("within_sd_mean", "within-context SD (mean)"),
    ]
    width = 0.26
    # greyscale luminance triple — the three SD components are one factor, and the
    # bright palette slices are reserved for families/layers (one color = one meaning).
    bar_colors = ["#252525", "#7a7a7a", "#c4c4c4"]
    for j, (k, lbl) in enumerate(keys):
        vals = [
            np.nan if (s in uninformative or comps.get(k) is None) else comps.get(k)
            for s, comps, _ in rows
        ]
        ax.bar(x + (j - 1) * width, vals, width=width, color=bar_colors[j], label=lbl)
    for i, (s, _, _) in enumerate(rows):
        if s in uninformative:
            ax.axvspan(x[i] - 0.45, x[i] + 0.45, color="#dddddd", hatch="///", alpha=0.6, lw=0)
            axc.axvspan(x[i] - 0.45, x[i] + 0.45, color="#dddddd", hatch="///", alpha=0.6, lw=0)
    ax.set_ylabel("judged-score SD (0–100 scale)")
    ax.set_title(f"{behavior}: between/within variance components per setting", loc="left")
    ax.legend(fontsize=7)
    mk = {
        "ceil_greedy": ("o", REGIME_COLORS["greedy"]),
        "ceil_avg_k5": ("s", REGIME_COLORS["avg_k5"]),
        "ceil_single": ("^", REGIME_COLORS["single"]),
    }
    for k, (m, c) in mk.items():
        vals = [
            np.nan if s in uninformative else (ceil.get(k) if ceil else None) for s, _, ceil in rows
        ]
        vals = [np.nan if v is None else v for v in vals]
        axc.scatter(
            x,
            vals,
            marker=m,
            color=c,
            label=REGIME_LABELS[k.removeprefix("ceil_")],
            zorder=3,
        )
    axc.set_ylim(0, 1.05)
    axc.set_ylabel("reliability ceiling √(var_true/var_obs)")
    labels = [
        (SETTING_LABELS.get(s, s) + (" — UNINFORMATIVE" if s in uninformative else ""))
        for s, _, _ in rows
    ]
    axc.set_xticks(x)
    axc.set_xticklabels(labels, rotation=20, ha="right")
    axc.legend(fontsize=7)
    name = f"r3b_variance_components_{behavior}"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        f"Top: between-context SD (raw and Var_within/K-corrected) and mean within-context SD "
        f"of the graded 0–100 judged DV per {behavior} setting. Bottom: per-regime-column "
        "reliability ceilings — CAVEAT (plan A3): each behavioral column is read against its "
        "OWN ceiling (greedy = 1 completion × 3 judge draws; averaged = 5-completion mean of "
        "3-draw scores; single = 1 completion × 3 draws) before narrating any regime effect. "
        "Evil floor-censored settings are rendered as hatched UNINFORMATIVE slots, never as "
        f"bars. Judge-noise: {(r3 or {}).get('judge_var_note') or 'banked-smoke fallback input'}. "
        f"{PROVENANCE}",
    )
    return True


def fig_r3c_rho(args) -> bool:
    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(4.2 * len(BEHAVIORS), 3.8))
    axes = np.atleast_1d(axes)
    any_data = False
    rho_lines = []
    for ax, behavior in zip(axes, BEHAVIORS):
        r3 = _load_json(Path(args.results_root) / f"r3_moderators_{behavior}.json")
        ax.set_title(behavior, loc="left")
        ax.set_xlabel("median σ_A (total) per setting")
        ax.set_ylabel("P spread (SD of per-context P)")
        if r3 is None:
            continue
        pts = []
        for s in _settings_present(r3.get("settings") or {}):
            mp = (r3["settings"][s] or {}).get("moderators_percontext") or {}
            sig = (mp.get("sigma_defs") or {}).get("sigma_a_total")
            p = mp.get("p")
            if sig is None or p is None:
                continue
            sig_med = float(np.nanmedian(np.asarray(sig, dtype=np.float64)))
            p_spread = float(np.nanstd(np.asarray(p, dtype=np.float64)))
            pts.append((s, sig_med, p_spread))
        if not pts:
            continue
        any_data = True
        for s, sx, sy in pts:
            ax.scatter(sx, sy, color=FAMILY_COLORS[FAMILY_OF_SETTING.get(s, "generic")], s=40)
            ax.text(sx, sy, " " + SETTING_LABELS.get(s, s), fontsize=6, va="center")
        rho = r3.get("rho_sigma_vs_p_spread_across_settings")
        rho_lines.append(f"{behavior}: ρ = {rho if rho is None else round(rho, 3)} (n={len(pts)})")
    if not any_data:
        plt.close(fig)
        logger.warning("[figures] r3c: no moderator inputs — skipped")
        return False
    name = "r3c_rho_sigma_vs_p_spread"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "Median σ_A (total answer-vector spread, L19) vs the spread of the polarization index "
        "P across settings, one point per setting (labeled). Spearman ρ across settings: "
        + ("; ".join(rho_lines) if rho_lines else "n/a")
        + f". n per behavior is the SETTING count — treat as descriptive at this n. {PROVENANCE}",
    )
    return True


# ── R4: grids ────────────────────────────────────────────────────────────────
def _grid_matrix(grid: dict, layer: int) -> tuple[np.ndarray, list[str]] | None:
    g = _layer_entry(grid.get("r2_grid"), layer)
    if not g:
        return None
    mat = np.full((len(REGIMES) + 1, len(REGIMES)), np.nan)
    for i, fr in enumerate(REGIMES):
        for j, er in enumerate(REGIMES):
            v = (g.get(fr) or {}).get(er)
            mat[i, j] = np.nan if v is None else v
    row_labels = [f"fit: {REGIME_LABELS[r]}" for r in REGIMES]
    m963 = _layer_entry((grid.get("map963k") or {}).get("r2"), layer)
    if m963:
        for j, er in enumerate(REGIMES):
            v = m963.get(er)
            mat[len(REGIMES), j] = np.nan if v is None else v
    row_labels.append("frozen 963k generic map")
    return mat, row_labels


def _heat(ax, mat: np.ndarray, row_labels: list[str], title: str) -> None:
    im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < 0.55 else "black",
                )
    ax.set_title(title, loc="left", fontsize=9)
    return im


def fig_r4_hero(args) -> bool:
    r4 = _load_json(Path(args.results_root) / "r4_grids.json")
    if r4 is None:
        return False
    settings = r4.get("settings") or {}
    names = [s for s in ("generic", *[x for x in SETTING_ORDER if x in settings]) if s in settings]
    names = list(dict.fromkeys(names))
    panels = []
    for s in names:
        gm = _grid_matrix(settings[s] or {}, HEADLINE_LAYER)
        if gm:
            panels.append((s, *gm))
    if not panels:
        logger.warning("[figures] r4_hero: no r2 grids — skipped")
        return False
    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, mat, rl) in zip(axes, panels):
        label = "Generic (LMSYS+WildChat pool)" if s == "generic" else SETTING_LABELS.get(s, s)
        _heat(ax, mat, rl, label)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    name = "r4_hero_r2_grids"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "Held-out answer-vector R² per setting at L19: rows = fit regime (which decode arm "
        "defined the training target) plus the frozen 963k generic map reference row (f_u=0 "
        "endpoint, #1739 conventions); columns = eval regime. GROUP-level held-out splits "
        "throughout; evil-train DIAGONAL cells are potentially question-interpolating (split "
        "axis is prefix-only, plan S2). CAVEAT (plan W2): the generic panel vs #1073's "
        "within-~0.01 pre-fill is an EXPECTATION BAND, not an identity — estimator and n "
        f"differ. {W3_LAYERS} {PROVENANCE}",
    )
    return True


def fig_r4_control(args) -> bool:
    r4 = _load_json(Path(args.results_root) / "r4_grids.json")
    if r4 is None:
        return False
    settings = r4.get("settings") or {}
    rows = []
    for s in _settings_present(settings):
        grid = settings[s] or {}
        ctrl = grid.get("control_r2")
        if not ctrl:
            continue
        admix = _grid_matrix(grid, HEADLINE_LAYER)
        cmat = np.full((len(REGIMES), len(REGIMES)), np.nan)
        for i, fr in enumerate(REGIMES):
            ce = _layer_entry(ctrl.get(fr) or {}, HEADLINE_LAYER) or {}
            for j, er in enumerate(REGIMES):
                v = ce.get(er)
                cmat[i, j] = np.nan if v is None else v
        if admix:
            rows.append((s, admix[0][: len(REGIMES)], cmat))
    if not rows:
        logger.warning("[figures] r4_control: no control_r2 blocks — skipped")
        return False
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.5, 3.0 * len(rows)))
    axes = np.atleast_2d(axes)
    fit_labels = [f"fit: {REGIME_LABELS[r]}" for r in REGIMES]
    for r, (s, amat, cmat) in enumerate(rows):
        _heat(
            axes[r, 0],
            amat,
            fit_labels,
            f"{'Generic (LMSYS+WildChat pool)' if s == 'generic' else SETTING_LABELS.get(s, s)} — admixed pool",
        )
        _heat(
            axes[r, 1],
            cmat,
            fit_labels,
            f"{'Generic (LMSYS+WildChat pool)' if s == 'generic' else SETTING_LABELS.get(s, s)} — matched all-generic control",
        )
    name = "r4_control_grids"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "Admixed-pool grids (left) vs the MANDATORY matched all-generic control (right; the "
        "family generic-core fit re-applied at its persisted selected λ to this setting's "
        "eval set), L19 — separates pool composition from the eval setting. Same GROUP-level "
        f"splits and caveats as the hero grids. {W3_LAYERS} {PROVENANCE}",
    )
    return True


def fig_r4b_behavioral(args, behavior: str) -> bool:
    r4 = _load_json(Path(args.results_root) / "r4_grids.json")
    if r4 is None:
        return False
    r3 = _load_json(Path(args.results_root) / f"r3_moderators_{behavior}.json") or {}
    settings = r4.get("settings") or {}
    jobs = [s for s in _settings_present(settings) if FAMILY_OF_SETTING.get(s) == behavior]
    jobs = [s for s in jobs if (settings[s] or {}).get("behavioral_rho_L19")]
    if not jobs:
        logger.warning("[figures] r4b_%s: no behavioral_rho_L19 — skipped", behavior)
        return False
    fig, axes = plt.subplots(1, len(jobs), figsize=(5.2 * len(jobs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    cols = list(REGIMES)
    for ax, s in zip(axes, jobs):
        ro = settings[s]["behavioral_rho_L19"] or {}
        fams = [f for f in METHOD_LABELS if f in ro and f != "disjoint_half"]
        width = 0.8 / max(1, len(fams))
        x = np.arange(len(cols))
        for j, f in enumerate(fams):
            vals, cis = [], []
            for c in cols:
                e = (ro.get(f) or {}).get(c) or {}
                vals.append(np.nan if e.get("rho") is None else e["rho"])
                cis.append(e.get("ci95"))
            xoff = x + (j - (len(fams) - 1) / 2) * width
            ax.bar(
                xoff,
                vals,
                width=width * 0.94,
                color=METHOD_COLORS.get(f, "#666666"),
                label=METHOD_LABELS[f],
            )
            ax.errorbar(
                xoff,
                vals,
                yerr=_err_from_ci(vals, cis),
                fmt="none",
                ecolor="#222222",
                elinewidth=0.9,
                capsize=2,
            )
        ceil = ((r3.get("settings") or {}).get(s) or {}).get("ceilings") or {}
        for j, c in enumerate(cols):
            cv = ceil.get(f"ceil_{c}")
            if cv is not None:
                ax.hlines(cv, x[j] - 0.42, x[j] + 0.42, color="#333333", ls="--", lw=1.2)
        ax.axhline(0.0, color="#444444", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[c] for c in cols], rotation=15, ha="right", fontsize=8)
        ax.set_title(SETTING_LABELS.get(s, s), loc="left", fontsize=9)
    axes[0].set_ylabel("Spearman ρ(method score, judged DV)")
    axes[0].legend(fontsize=6)
    name = f"r4b_behavioral_rho_{behavior}"
    _save(args, fig, name)
    dj = None
    for s in jobs:
        dj = ((settings[s].get("behavioral_rho_L19") or {}).get("disjoint_half") or {}).get(
            "rho_half_vs_half"
        )
        if dj is not None:
            break
    _write_caption(
        args,
        name,
        f"Behavioral-expression prediction for {behavior}: Spearman ρ between each method "
        "family's score and the judged DV, per DV regime column at L19; cluster-bootstrap "
        "95% CIs. Dashed marks: each column's OWN reliability ceiling — CAVEAT (plan A3): "
        "greedy = 1 completion × 3 judge draws, averaged = 5-completion mean of 3-draw "
        "scores, single = 1 completion × 3 draws; read each column against its own ceiling "
        "before narrating any regime effect. Disjoint-half DV noise reference "
        f"(rollouts {{0,2,4}} vs {{1,3}} means, 2-vs-3 caveat A2): ρ = {dj}. Graded 0–100 "
        "multi-draw judge (claude-sonnet-4-5-20250929, 3 draws, drop-never-coerce). "
        f"{PROVENANCE} {W3_LAYERS}",
    )
    return True


def fig_r4c_median_split(args) -> bool:
    r4 = _load_json(Path(args.results_root) / "r4_grids.json")
    if r4 is None:
        return False
    settings = r4.get("settings") or {}
    rows = []
    for s in _settings_present(settings):
        ms = (settings[s] or {}).get("median_split")
        if not ms:
            continue
        grids = ms.get("grids") or {}
        pair = []
        for half in ("low_sigma", "high_sigma"):
            g = _layer_entry(grids.get(half) or {}, HEADLINE_LAYER)
            if not g:
                pair = []
                break
            mat = np.full((len(REGIMES), len(REGIMES)), np.nan)
            for i, fr in enumerate(REGIMES):
                for j, er in enumerate(REGIMES):
                    v = (g.get(fr) or {}).get(er)
                    mat[i, j] = np.nan if v is None else v
            pair.append(mat)
        if pair:
            rows.append((s, pair[0], pair[1], ms.get("n_low"), ms.get("n_high")))
    if not rows:
        logger.warning("[figures] r4c: no median_split grids — skipped")
        return False
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.5, 3.0 * len(rows)))
    axes = np.atleast_2d(axes)
    fit_labels = [f"fit: {REGIME_LABELS[r]}" for r in REGIMES]
    for r, (s, lo, hi, n_lo, n_hi) in enumerate(rows):
        _heat(
            axes[r, 0],
            lo,
            fit_labels,
            f"{'Generic (LMSYS+WildChat pool)' if s == 'generic' else SETTING_LABELS.get(s, s)} — low σ_A (n={n_lo})",
        )
        _heat(
            axes[r, 1],
            hi,
            fit_labels,
            f"{'Generic (LMSYS+WildChat pool)' if s == 'generic' else SETTING_LABELS.get(s, s)} — high σ_A (n={n_hi})",
        )
    name = "r4c_median_split_grids"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "Held-out R² grids (L19) split by low/high σ_A halves (median split on the OOD eval "
        "set). CAVEAT (plan A4): median-split reads are WITHIN-fixed-eval-column contrasts — "
        "the high-σ_A half co-varies with answer length/truncation, so no cross-column causal "
        f"reading. {W3_LAYERS} {PROVENANCE}",
    )
    return True


# ── R5: polarization ─────────────────────────────────────────────────────────
def fig_r5_polarization(args) -> bool:
    r5 = _load_json(Path(args.results_root) / "r5_polarization.json")
    if r5 is None:
        return False
    panels = r5.get("panels") or {}
    keys = sorted(
        panels,
        key=lambda k: (
            BEHAVIORS.index(k.split("::")[0]) if k.split("::")[0] in BEHAVIORS else 9,
            k,
        ),
    )
    keys = [k for k in keys if panels[k]]
    if not keys:
        logger.warning("[figures] r5: no panels — skipped")
        return False
    ncols = min(4, len(keys))
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    mu_grid = np.linspace(0, 100, 201)
    uninf = []
    for ax, key in zip(axes, keys):
        b, _, s = key.partition("::")
        blk = panels[key]
        title = f"{b} / {SETTING_LABELS.get(s, s)}"
        ax.plot(mu_grid, np.sqrt(mu_grid * (100 - mu_grid)), color="#555555", ls="--", lw=1.0)
        if blk.get("uninformative"):
            ax.set_facecolor("#e8e8e8")
            ax.fill_between(
                mu_grid,
                0,
                np.sqrt(mu_grid * (100 - mu_grid)),
                color="#cccccc",
                hatch="///",
                alpha=0.5,
                lw=0,
            )
            ax.set_title(title + " — UNINFORMATIVE (floor-censored)", loc="left", fontsize=7)
            uninf.append(key)
            continue
        pc = blk.get("percontext") or {}
        mu = np.asarray(pc.get("mu") or [], dtype=np.float64)
        sd = np.asarray(pc.get("sd") or [], dtype=np.float64)
        color = FAMILY_COLORS.get(b, "#333333")
        if mu.size > 400:
            ax.hexbin(mu, sd, gridsize=28, cmap="Greys", mincnt=1)
        else:
            ax.scatter(mu, sd, s=8, color=color, alpha=0.5, linewidths=0)
        ax.set_title(
            title + (" (definitional)" if blk.get("definitional") else ""), loc="left", fontsize=8
        )
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    for i in range(len(keys)):
        if i % ncols == 0:
            axes[i].set_ylabel("per-context SD of judged score")
        if i >= len(keys) - ncols:
            axes[i].set_xlabel("per-context mean judged score μ")
    name = "r5_polarization_panels"
    _save(args, fig, name)
    g_lines = []
    for key in keys:
        g = (panels[key].get("g_pol") or {}) if panels[key] else {}
        if g.get("value") is not None:
            g_lines.append(f"{key}: g_pol {g['value']:+.3f} CI {g.get('ci95')}")
    _write_caption(
        args,
        name,
        "Per-context SD vs mean of the graded judged score, one panel per behavior × setting, "
        "against the √(μ(100−μ)) maximal-polarization ceiling (grey dashed). CAVEAT: K=5 "
        "rollout grain — only discrete (μ, SD) pairs are attainable (coarse lattice), so "
        "claims here are POPULATION-LEVEL only, never per-context. Floor-censored evil cells "
        "are rendered greyed/hatched as UNINFORMATIVE, never as data: "
        + (", ".join(uninf) if uninf else "none flagged")
        + ". Hallucination own-rung panels use the 3-way fabrication construct (P=1 "
        "definitional — plan A2) and are never mixed onto the 0–100 trait axis. g_pol: "
        + ("; ".join(g_lines) if g_lines else "n/a")
        + f". {PROVENANCE}",
    )
    return True


# ── exploratory dump ──────────────────────────────────────────────────────────
def fig_capture_parity(args) -> bool:
    cp = _load_json(Path(args.results_root) / "capture_parity.json")
    if cp is None:
        return False
    behaviors = cp.get("behaviors") or {}
    rows = []
    for b, blk in behaviors.items():
        if not blk:
            continue
        fc = blk.get("full_coverage") or {}
        for job, jb in fc.items():
            entry = _layer_entry((jb or {}).get("per_layer"), HEADLINE_LAYER) or {}
            if entry.get("median") is not None:
                rows.append((b, job, entry, jb.get("n_overlap")))
    if not rows:
        logger.warning("[figures] capture_parity: no full-coverage summaries — skipped")
        return False
    fig, ax = plt.subplots(figsize=(max(6.0, 1.1 * len(rows) + 2), 4.0))
    x = np.arange(len(rows))
    meds = [r[2]["median"] for r in rows]
    p5 = [r[2].get("p5") for r in rows]
    p95 = [r[2].get("p95") for r in rows]
    err = _err_from_ci(meds, [(a, b) for a, b in zip(p5, p95)])
    colors = [FAMILY_COLORS.get(r[0], "#333333") for r in rows]
    ax.scatter(x, meds, color=colors, s=42, zorder=3)
    ax.errorbar(x, meds, yerr=err, fmt="none", ecolor="#555555", elinewidth=1.1, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SETTING_LABELS.get(r[1], r[1]) for r in rows], rotation=20, ha="right")
    ax.set_ylabel("median cosine, fresh vs banked\ncontext_end (whiskers p5–p95)")
    ax.set_title("Cross-campaign capture parity per rung (L19)", loc="left")
    name = "capture_parity_per_rung"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "MF-2 full-coverage capture parity: median (whiskers p5–p95 of the distribution, not "
        "a CI) cosine between the fresh-campaign greedy capture's context_end state and the "
        "banked #1739 state on overlapping contexts, per rung, L19. Analyzer-weighed REPORT, "
        "not a kill gate: any negative trait-side Δ (R2) is read against these distributions "
        "plus the WildChat(cross-campaign)-vs-LMSYS(same-campaign) triangulation BEFORE any "
        f"#1073 correction posts. n_overlap per rung: "
        + "; ".join(f"{SETTING_LABELS.get(r[1], r[1])}: {r[3]}" for r in rows)
        + f". {W3_LAYERS}",
    )
    return True


def fig_judge_dv_coverage(args) -> bool:
    rows = []
    for b in BEHAVIORS:
        d = _load_json(Path(args.results_root) / "greedy_dv" / f"{b}.json")
        if d is None:
            continue
        n = d.get("n_contexts")
        nd = d.get("n_contexts_with_dv")
        if n:
            rows.append((b, n, nd))
    if not rows:
        logger.warning("[figures] judge_dv_coverage: no greedy_dv JSONs — skipped")
        return False
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(rows))
    ax.bar(x, [r[2] / r[1] for r in rows], width=0.55, color=[FAMILY_COLORS[r[0]] for r in rows])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of greedy contexts with a judged DV")
    ax.set_title("Greedy-arm judged-DV coverage per behavior", loc="left")
    name = "judge_dv_coverage"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "Fraction of greedy-decode contexts carrying a judged DV per behavior "
        + "; ".join(f"{r[0]}: {r[2]}/{r[1]}" for r in rows)
        + ". Per-arm content-drop vs transport-loss splits and stop_reason tallies live in "
        "eval_results/issue_2091/greedy_dv/<behavior>.json (llm-judging rules 9/18/24). "
        "Greedy arm judged at max_tokens=1024 vs the banked wave's 400/800 — stated "
        f"instrument deviation (plan §11). {PROVENANCE}",
    )
    return True


def fig_r4_knn(args) -> bool:
    r4 = _load_json(Path(args.results_root) / "r4_grids.json")
    if r4 is None:
        return False
    settings = r4.get("settings") or {}
    rows = []
    for s in _settings_present(settings):
        fits = (settings[s] or {}).get("fits") or {}
        for regime, fr in fits.items():
            for pl in fr.get("per_layer") or []:
                if pl.get("layer") != HEADLINE_LAYER:
                    continue
                knn = ((pl.get("knn") or {}).get("cosine") or {}).get("acc_at_k") or {}
                a1 = knn.get("1") if "1" in knn else knn.get(1)
                a5 = knn.get("5") if "5" in knn else knn.get(5)
                if a1 is not None:
                    rows.append((s, regime, a1, a5))
    if not rows:
        logger.warning("[figures] r4_knn: no knn diagnostics — skipped")
        return False
    fig, ax = plt.subplots(figsize=(max(6.5, 0.65 * len(rows) + 2), 4.0))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[2] for r in rows], width=0.34, color="#404040", label="acc@1 (cosine)")
    ax.bar(
        x + 0.18,
        [np.nan if r[3] is None else r[3] for r in rows],
        width=0.34,
        color="#a0a0a0",
        label="acc@5 (cosine)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{'Generic (LMSYS+WildChat pool)' if r[0] == 'generic' else SETTING_LABELS.get(r[0], r[0])}"
            f"\n{REGIME_LABELS[r[1]]}"
            for r in rows
        ],
        rotation=30,
        ha="right",
        fontsize=6,
    )
    ax.set_ylabel("kNN retrieval accuracy (held-out pool)")
    ax.set_title("kNN retrieval per fitted map (L19)", loc="left")
    ax.legend(fontsize=7)
    name = "r4_knn_retrieval"
    _save(args, fig, name)
    _write_caption(
        args,
        name,
        "kNN-retrieval read per fitted map at L19 (cosine acc@1/acc@5 among the held-out "
        "candidate pool; chance = k/n_pool, recorded per fit in r4_grids.json fits[*]."
        "per_layer[*].knn). Companion to the identity+learned-bias baseline persisted in the "
        f"same per-fit diagnostics (standing mapping-baselines rule). {W3_LAYERS}",
    )
    return True


# ── R1 answer lengths (unit-C deferred gap) ───────────────────────────────────
def _median_shard_lengths(root: Path) -> dict[str, dict]:
    """Per-subdir median completion CHAR length from packed jsonl shards.

    Digest-only: row text is measured (len) in code and never printed/logged.
    Accepts packed rows ({"src", "doc": {"completion"}}) and plain rows
    ({"completion"} / {"text"}).
    """
    out: dict[str, dict] = {}
    if not root.is_dir():
        return out
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        lengths: list[int] = []
        for shard in sorted(sub.rglob("*.jsonl")):
            with shard.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc = row.get("doc") if isinstance(row.get("doc"), dict) else row
                    text = doc.get("completion") or doc.get("text") or ""
                    if text:
                        lengths.append(len(text))
        if lengths:
            arr = np.asarray(lengths, dtype=np.float64)
            out[sub.name] = {"median_chars": float(np.median(arr)), "n": int(arr.size)}
    return out


def phase_lengths(args) -> int:
    """Compute R1 median answer-length annotations -> r1_answer_lengths.json."""
    greedy = _median_shard_lengths(Path(args.greedy_packed_root)) if args.greedy_packed_root else {}
    banked = _median_shard_lengths(Path(args.banked_packed_root)) if args.banked_packed_root else {}
    if not greedy and not banked:
        logger.warning("[lengths] no shard roots resolved — nothing written")
        return 1
    settings: dict[str, dict] = {}
    for name in set(greedy) | set(banked):
        g = greedy.get(name) or {}
        b = banked.get(name) or {}
        settings[name] = {
            "greedy_median_chars": g.get("median_chars"),
            "n_greedy": g.get("n"),
            "banked_median_chars": b.get("median_chars"),
            "n_banked": b.get("n"),
        }
    dest = Path(args.results_root) / "r1_answer_lengths.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": settings,
        "units": "characters (completion text length; digest-only, no text persisted)",
        "meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("[lengths] wrote %s (%d settings)", dest, len(settings))
    return 0


# ── synthetic fixtures (smoke leg) ────────────────────────────────────────────
def _synth_r1_block(rng, n: int) -> dict:
    out = {}
    for layer in LAYERS:
        disp = np.clip(rng.normal(0.25, 0.08, n), 0.01, 0.9)
        med = float(np.median(disp))
        out[f"L{layer}"] = {
            "percontext": {"context_id": [f"c{i}" for i in range(n)], "dispersion": disp.tolist()},
            "summary": {
                "n": n,
                "median": med,
                "p5": float(np.percentile(disp, 5)),
                "p95": float(np.percentile(disp, 95)),
                "min": float(disp.min()),
                "mean": float(disp.mean()),
            },
            "boot_ci_median": [med - 0.02, med + 0.02],
        }
    return out


def _synth_r2_block(rng, n: int) -> dict:
    out = {}
    for layer in LAYERS:
        delta = rng.normal(0.002, 0.01, n)
        med = float(np.median(delta))
        out[f"L{layer}"] = {
            "percontext": {
                "context_id": [f"c{i}" for i in range(n)],
                "delta": delta.tolist(),
                "cos_g_kmean": np.clip(rng.normal(0.93, 0.03, n), 0, 1).tolist(),
                "dispersion_quintile": rng.integers(1, 6, n).tolist(),
            },
            "median_delta": med,
            "boot_ci_median": [med - 0.003, med + 0.003],
            "severe_tail_rate": float((delta < SEVERE_TAIL).mean()),
            "common_language_p": float((delta > 0).mean()),
            "jackknife": {"drop_one_medians": [med] * 5, "band": [med - 0.001, med + 0.001]},
            "exchangeability": {
                "mean_rank": 3.2,
                "expected_mean": 3.5,
                "rank_hist": {str(r): int(n / 6) for r in range(1, 7)},
                "boot_ci_mean_rank": [3.0, 3.4],
            },
            "quintile_curve": {
                "quintile": [1, 2, 3, 4, 5],
                "cos_g_kmean_median": np.linspace(0.97, 0.85, 5).tolist(),
                "disjoint_half_median": np.linspace(0.96, 0.87, 5).tolist(),
                "note": "synthetic",
            },
        }
    return out


def _synth_grid(rng) -> dict:
    def _g():
        return {
            f"L{layer}": {
                fr: {er: float(np.clip(rng.normal(0.6, 0.08), -0.2, 0.95)) for er in REGIMES}
                for fr in REGIMES
            }
            for layer in LAYERS
        }

    per_layer = [
        {
            "layer": layer,
            "selection": {"best_lambda": 10.0},
            "knn": {"cosine": {"acc_at_k": {"1": 0.4, "5": 0.7}, "chance": {"1": 0.01, "5": 0.05}}},
        }
        for layer in LAYERS
    ]
    return {
        "setting": "synthetic",
        "pool": {"U": 40, "f_u": 0.5},
        "fits": {r: {"per_layer": per_layer} for r in REGIMES},
        "r2_grid": _g(),
        "map963k": {"r2": {f"L{la}": {er: 0.5 for er in REGIMES} for la in LAYERS}},
        # production shape (analysis phase_family): control_r2[fit_regime][L][eval_regime]
        "control_r2": {
            fr: {
                f"L{la}": {er: float(np.clip(rng.normal(0.55, 0.08), -0.2, 0.95)) for er in REGIMES}
                for la in LAYERS
            }
            for fr in REGIMES
        },
        "behavioral_rho_L19": {
            **{
                f: {
                    c: {
                        "rho": float(np.clip(rng.normal(0.3, 0.1), -1, 1)),
                        "n": 40,
                        "ci95": [0.1, 0.5],
                    }
                    for c in REGIMES
                }
                for f in (
                    "pv_projection",
                    "supervised_context",
                    "map_pv_projection",
                    "map_supervised_answer",
                    "oracle_answer",
                )
            },
            "disjoint_half": {"rho_half_vs_half": 0.6, "note": "synthetic"},
        },
        "median_split": {
            "note": "synthetic",
            "grids": {"low_sigma": _g(), "high_sigma": _g()},
            "n_low": 20,
            "n_high": 20,
        },
    }


def phase_synth_fixtures(args) -> int:
    """Write shape-faithful synthetic §6.5 JSONs so render exercises every figure."""
    rng = np.random.default_rng(20910)
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    n = 24
    meta = {"synthetic": True, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    r1 = {"meta": meta, "settings": {}, "headline_note": "synthetic"}
    r2 = {"meta": meta, "settings": {}}
    for s in SETTING_ORDER:
        if s == "lmsys":
            # production assembly splits the lmsys per_layer block by key set:
            # r1 gets summary/boot_ci_median/median_answer_len/percontext (dispersion),
            # r2 gets ONLY median_delta*/delta_boot*/severe*/percontext keys (never both
            # CI names in one dict — the reader relies on that).
            lm_r1, lm_r2 = {}, {}
            for layer in LAYERS:
                blk = _synth_r1_block(rng, n)[f"L{layer}"]
                r2blk = _synth_r2_block(rng, n)[f"L{layer}"]
                lm_r1[f"L{layer}"] = {
                    "summary": blk["summary"],
                    "boot_ci_median": blk["boot_ci_median"],
                    "median_answer_len": 180.0,
                    "percontext": blk["percontext"],
                }
                lm_r2[f"L{layer}"] = {
                    "median_delta": r2blk["median_delta"],
                    "delta_boot_ci_median": r2blk["boot_ci_median"],
                    "severe_tail_rate": r2blk["severe_tail_rate"],
                    "percontext": {
                        "idx": list(range(n)),
                        "delta": r2blk["percontext"]["delta"],
                        "dispersion": blk["percontext"]["dispersion"],
                    },
                }
            r1["settings"]["lmsys"] = lm_r1
            r2["settings"]["lmsys"] = lm_r2
        else:
            r1["settings"][s] = _synth_r1_block(rng, n)
            r2["settings"][s] = _synth_r2_block(rng, n)
    r2["generic_vs_trait_contrasts_L19"] = {
        f"{b}_train_minus_wildchat": {
            "median": float(rng.normal(0, 0.004)),
            "ci95": [-0.006, 0.006],
            "ci_holm": [-0.008, 0.008],
            "p_holm": 0.6,
        }
        for b in BEHAVIORS
    }
    (root / "r1_dispersion.json").write_text(json.dumps(r1), encoding="utf-8")
    (root / "r2_delta.json").write_text(json.dumps(r2), encoding="utf-8")

    for b in BEHAVIORS:
        settings = {}
        jobs = [s for s in SETTING_ORDER if FAMILY_OF_SETTING.get(s) == b] + ["wildchat"]
        for s in jobs:
            sig = np.abs(rng.normal(2.0, 0.5, n))
            p = np.clip(rng.normal(0.6, 0.15, n), 0, 1.2)
            cm = {
                "n": n,
                "r2_full": 0.30,
                "unique_x1": 0.12,
                "unique_x2": 0.10,
                "shared": 0.08 if s != "wildchat" else -0.02,
                "suppression": s == "wildchat",
                "signs": {"r_y_x1": 0.4, "r_y_x2": 0.35, "r_x1_x2": 0.2},
                "companion_unique_sigma_minus_unique_p": {"median": 0.02, "ci95": [-0.05, 0.09]},
            }
            settings[s] = {
                "ceilings": {
                    "ceil_greedy": 0.75,
                    "ceil_avg_k5": 0.93,
                    "ceil_single": 0.74,
                    "components": {
                        "between_sd_raw": 22.0,
                        "between_sd_corrected": 19.5,
                        "within_sd_mean": 11.0,
                    },
                },
                "guardrails": {
                    "rho_sigma_a_total_vs_p": 0.42,
                    "split_half": {"p": 0.7, "sigma_a_dispersion": 0.65},
                },
                "commonality": {
                    "sigma_a_total": {"pv_projection": cm, "oracle_answer": cm},
                    "sigma_a_proj": {"pv_projection": cm},
                },
                "moderators_percontext": {
                    "context_id": [f"c{i}" for i in range(n)],
                    "sigma_defs": {"sigma_a_total": sig.tolist()},
                    "p": p.tolist(),
                    "scores_within_sd": np.abs(rng.normal(10, 3, n)).tolist(),
                    "scores_mu": np.clip(rng.normal(50, 20, n), 0, 100).tolist(),
                },
            }
        payload = {
            "meta": meta,
            "behavior": b,
            "judge_draw_var": 30.0,
            "judge_var_note": "synthetic fixture",
            "settings": settings,
            "rho_sigma_vs_p_spread_across_settings": 0.5,
        }
        (root / f"r3_moderators_{b}.json").write_text(json.dumps(payload), encoding="utf-8")

    r4 = {"meta": meta, "settings": {"generic": _synth_grid(rng)}}
    for s in SETTING_ORDER[2:]:
        r4["settings"][s] = _synth_grid(rng)
    (root / "r4_grids.json").write_text(json.dumps(r4), encoding="utf-8")

    panels = {}
    for b in BEHAVIORS:
        for s in [x for x in SETTING_ORDER if FAMILY_OF_SETTING.get(x) == b][:2] + ["wildchat"]:
            mu = np.clip(rng.normal(45, 25, n), 0, 100)
            sd = np.clip(np.sqrt(mu * (100 - mu)) * rng.uniform(0.2, 0.9, n), 0, None)
            panels[f"{b}::{s}"] = {
                "percontext": {
                    "context_id": [f"c{i}" for i in range(n)],
                    "mu": mu.tolist(),
                    "sd": sd.tolist(),
                    "p": np.clip(rng.normal(0.6, 0.2, n), 0, 1.2).tolist(),
                },
                "n_middling": 10,
                "mean_f_mid": 0.7,
                "q_pol": -0.1,
                "g_pol": {"value": 0.2, "ci95": [0.1, 0.3]},
                "uninformative": b == "evil" and s == "wildchat",
            }
    (root / "r5_polarization.json").write_text(
        json.dumps({"meta": meta, "panels": panels}), encoding="utf-8"
    )

    cp = {
        "meta": meta,
        "behaviors": {
            b: {
                "full_coverage": {
                    s: {
                        "n_overlap": n,
                        "n_new": n,
                        "per_layer": {
                            f"L{la}": {
                                "n": n,
                                "median": 0.98,
                                "p5": 0.95,
                                "p95": 0.995,
                                "min": 0.9,
                                "mean": 0.975,
                            }
                            for la in LAYERS
                        },
                    }
                    for s in SETTING_ORDER
                    if FAMILY_OF_SETTING.get(s) == b
                },
                "probe": None,
            }
            for b in BEHAVIORS
        },
        "note": "synthetic",
    }
    (root / "capture_parity.json").write_text(json.dumps(cp), encoding="utf-8")

    gd = root / "greedy_dv"
    gd.mkdir(exist_ok=True)
    for b in BEHAVIORS:
        (gd / f"{b}.json").write_text(
            json.dumps({"behavior": b, "n_contexts": 100, "n_contexts_with_dv": 97, "meta": meta}),
            encoding="utf-8",
        )

    shard_root = root / "synthetic_shards"
    for job in ("syc_train", "wildchat"):
        d = shard_root / job
        d.mkdir(parents=True, exist_ok=True)
        with (d / "g.shard00.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(30):
                fh.write(
                    json.dumps(
                        {
                            "src": f"row{i}.json",
                            "doc": {"completion": "synthetic answer " * (i + 1)},
                        }
                    )
                    + "\n"
                )
    logger.info("[synth] fixtures written under %s", root)
    return 0


# ── render driver ─────────────────────────────────────────────────────────────
FIGURES: dict[str, object] = {}


def _register_figures() -> None:
    FIGURES.clear()
    FIGURES.update(
        {
            "r1_hero": lambda a: fig_r1_hero(a),
            "r1_hero_L14": lambda a: fig_r1_hero(a, layer=14, suffix="_L14"),
            "r1_hero_L26": lambda a: fig_r1_hero(a, layer=26, suffix="_L26"),
            "r1b": fig_r1b_floor_depth,
            "r1c": fig_r1c_r2_vs_dispersion,
            "r2_hero": lambda a: fig_r2_hero(a),
            "r2_hero_L14": lambda a: fig_r2_hero(a, layer=14, suffix="_L14"),
            "r2_hero_L26": lambda a: fig_r2_hero(a, layer=26, suffix="_L26"),
            "r2b": fig_r2b_ecdf,
            "r2c": fig_r2c_quintiles,
            **{f"r3_hero_{b}": (lambda a, b=b: fig_r3_hero(a, b)) for b in BEHAVIORS},
            **{f"r3b_{b}": (lambda a, b=b: fig_r3b_components(a, b)) for b in BEHAVIORS},
            "r3c": fig_r3c_rho,
            "r4_hero": fig_r4_hero,
            "r4_control": fig_r4_control,
            **{f"r4b_{b}": (lambda a, b=b: fig_r4b_behavioral(a, b)) for b in BEHAVIORS},
            "r4c": fig_r4c_median_split,
            "r5": fig_r5_polarization,
            "capture_parity": fig_capture_parity,
            "judge_dv_coverage": fig_judge_dv_coverage,
            "r4_knn": fig_r4_knn,
        }
    )


def phase_render(args) -> int:
    set_paper_style("blog")
    _register_figures()
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    rendered, skipped = [], []
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        try:
            ok = fn(args)
        except Exception:
            logger.exception("[figures] %s CRASHED", name)
            raise
        (rendered if ok else skipped).append(name)
    total_bytes = (
        sum(p.stat().st_size for p in _fig_dir(args).glob("*.png"))
        if _fig_dir(args).is_dir()
        else 0
    )
    print(
        f"[figures] done: rendered={len(rendered)} skipped={len(skipped)} "
        f"png_bytes={total_bytes} out={_fig_dir(args)}",
        flush=True,
    )
    if skipped:
        logger.info("[figures] skipped (missing inputs): %s", ", ".join(skipped))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        choices=["render", "synth-fixtures", "lengths"],
        default="render",
    )
    ap.add_argument("--results-root", type=Path, default=_REPO_ROOT / "eval_results" / "issue_2091")
    ap.add_argument("--out-root", type=Path, default=_REPO_ROOT / "figures")
    ap.add_argument("--only", default=None, help="comma-separated figure subset (render phase)")
    ap.add_argument(
        "--banked-1738",
        type=Path,
        default=_REPO_ROOT / "eval_results" / "issue_1738" / "kresample" / "floor_summary.json",
    )
    ap.add_argument(
        "--banked-1073-percontext",
        type=Path,
        default=_REPO_ROOT / "eval_results" / "issue_1073" / "heldout_recon_percontext.json",
    )
    ap.add_argument(
        "--banked-1073-adequacy",
        type=Path,
        default=_REPO_ROOT / "eval_results" / "issue_1073" / "adequacy_tail_characterization.json",
    )
    ap.add_argument(
        "--banked-smoke", type=Path, default=None, help="banked_smoke_<b>.json fallback for r3b"
    )
    ap.add_argument(
        "--greedy-packed-root", type=Path, default=None, help="lengths phase: greedy packed shards"
    )
    ap.add_argument(
        "--banked-packed-root", type=Path, default=None, help="lengths phase: banked packed shards"
    )
    args = ap.parse_args()
    if args.phase == "synth-fixtures":
        return phase_synth_fixtures(args)
    if args.phase == "lengths":
        return phase_lengths(args)
    return phase_render(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit — heavy C-extension atexit gotcha (gotchas.md #1689)
